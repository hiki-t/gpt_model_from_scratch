import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
from transformers import TrainingArguments, Trainer

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder
import json
from typing import List, Dict
import random
from tqdm import tqdm
import peft
import os

###

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###

class SummarizationDataProcessor:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    # ... existing code ...
    def format_prompt(self, post: str, title: str) -> str:
        return f"Title: {title}\n\nPost: {post}\n\nSummary:"
    def format_prompt_with_subreddit(self, post: str, title: str, subreddit: str) -> str:
        return f"Subreddit: r/{subreddit}\nTitle: {title}\n\nPost: {post}\n\nSummary:"
    def process_summarize_from_feedback(self, data: List[Dict]) -> List[Dict]:
        processed = []
        for item in data:
            info = item['info']
            post = info['post']
            title = info['title']
            subreddit = info.get('subreddit', '')
            summaries = item['summaries']
            choice = item['choice']
            chosen_summary = summaries[choice]['text']
            if subreddit:
                prompt = self.format_prompt_with_subreddit(post, title, subreddit)
            else:
                prompt = self.format_prompt(post, title)
            if chosen_summary.strip():
                processed.append({
                    'input_text': prompt,
                    'target_text': chosen_summary,
                    'full_text': prompt + " " + chosen_summary,
                    'post_id': info['id'],
                    'worker_id': item.get('worker', 'unknown')
                })
        return processed

    def tokenize_function(self, examples):
        # Tokenize with padding and truncation
        tokenized = self.tokenizer(
            examples['full_text'],
            truncation=True,
            max_length=self.max_length,
            padding="max_length", 
            # padding="max_length",  # Ensures all sequences are the same length
        )
        # Copy input_ids to labels
        labels = []
        for input_text, full_text, input_ids in zip(examples['input_text'], examples['full_text'], tokenized['input_ids']):
            input_tokens = self.tokenizer(
                input_text, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=self.max_length
                )['input_ids']
            input_length = len(input_tokens)
            # Mask the prompt part with -100, keep the rest
            label = [-100] * input_length + input_ids[input_length:]
            # Pad/truncate to max_length
            label = label[:self.max_length]
            if len(label) < self.max_length:
                # label += [self.tokenizer.pad_token_id] * (self.max_length - len(label))
                label += [-100] * (self.max_length - len(label))
            labels.append(label)
        tokenized['labels'] = labels
        return tokenized

def load_and_prepare_data(dataset, tokenizer):
    processor = SummarizationDataProcessor(tokenizer)
    raw_data = []
    for item in dataset:
        raw_data.append(item)
    processed_data = processor.process_summarize_from_feedback(raw_data)
    dataset = Dataset.from_list(processed_data)
    tokenized_dataset = dataset.map(
        processor.tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

###

# Load your dataset
ds_train = load_dataset("openai/summarize_from_feedback", "comparisons")
ds_train_train = ds_train["train"]
ds_train_val = ds_train["validation"]

model_name="Qwen/Qwen3-0.6B-Base"
train_dataset=ds_train_train
val_dataset=ds_train_val
output_dir="./trained_models/qwen-sft-summarization"
num_epochs=1
batch_size=4 
learning_rate = 5e-5

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = peft.LoraConfig(
    r=4,                         # Rank (typical: 4, 8, or 16)
    lora_alpha=8,               # Alpha scaling factor
    # target_modules=["q_proj", "v_proj"],  # Depends on model architecture
    lora_dropout=0.1,           # Dropout applied to LoRA layers
    bias="none",                 # Can be "none", "all", or "lora_only"
    task_type="CAUSAL_LM"  # For decoder-only models like Qwen
)

model = peft.get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)

processed_train_dataset = load_and_prepare_data(train_dataset, tokenizer)
processed_val_dataset = load_and_prepare_data(val_dataset, tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

train_dataloader = DataLoader(
    processed_train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator, 
    num_workers=8, 
    pin_memory=True
)

val_dataloader = DataLoader(
    processed_val_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator, 
    num_workers=8, 
    pin_memory=True
)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)

scaler = torch.GradScaler("cuda")

model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0.0
    model.train()

    progress_bar = tqdm(train_dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast("cuda"):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # outputs = model(**batch)
        # loss = outputs.loss
        # loss.backward()
        # optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        progress_bar.set_description(f"Train Loss: {avg_loss:.4f}")

# Validation phase
model.eval()
val_loss = 0.0
with torch.no_grad():
    val_progress = tqdm(val_dataloader, desc="Validation")
    for step, batch in enumerate(val_progress):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        val_loss += loss.item()
        avg_val_loss = val_loss / (step + 1)
        val_progress.set_description(f"Val Loss: {avg_val_loss:.4f}")
