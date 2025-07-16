
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your dataset
ds_train = load_dataset("openai/summarize_from_feedback", "comparisons")
# ds_train_train = ds_train["train"]
# ds_train_val = ds_train["validation"]
train_dataset = ds_train["train"]
val_dataset = ds_train["validation"]

class SummarizationDataProcessor:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    # ... existing code ...
    def format_prompt_no_title(self, post: str) -> str:
        return f"Post: {post}\n\nSummary:"
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
            rejected_summary = summaries[1-choice]['text']
            # if subreddit:
            #     prompt = self.format_prompt_with_subreddit(post, title, subreddit)
            # else:
            #     prompt = self.format_prompt(post, title)
            
            # make model more generic
            prompt = self.format_prompt_no_title(post)
            
            if chosen_summary.strip():
                processed.append({
                    'input_text': prompt,
                    'chosen': chosen_summary,
                    'rejected': rejected_summary,
                    # 'full_text': prompt + " " + chosen_summary, 
                    'post_id': info['id'], 
                    'worker_id': item.get('worker', 'unknown'), 
                    'choice': float(choice), 

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

    def tokenize_pair(self, examples):
        # examples is a dict of lists, not list of dicts
        input_texts = examples['input_text']
        chosen_texts = examples['chosen']
        rejected_texts = examples['rejected']

        # Tokenize prompt+chosen
        chosen_inputs = self.tokenizer(
            [i + " " + c for i, c in zip(input_texts, chosen_texts)],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        
        # Tokenize prompt+rejected
        rejected_inputs = self.tokenizer(
            [i + " " + c for i, c in zip(input_texts, rejected_texts)],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        
        return {
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"],
        }

def load_and_prepare_data(dataset, tokenizer):
    processor = SummarizationDataProcessor(tokenizer)
    raw_data = []
    for item in dataset:
        raw_data.append(item)
    processed_data = processor.process_summarize_from_feedback(raw_data)
    dataset = Dataset.from_list(processed_data)
    tokenized_dataset = dataset.map(
        processor.tokenize_pair,
        # processor.tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

class RewardModel(nn.Module):
    def __init__(self, sft_model):
        super().__init__()
        self.model = sft_model
        self.v_head = nn.Linear(self.model.config.hidden_size, 1)  # reward head

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # last_hidden_state = output.last_hidden_state  # (batch, seq_len, hidden)
        last_hidden_state = output.hidden_states[-1]  # [B, T, H]
        
        # Use last non-padding token's hidden state
        lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden_state[torch.arange(len(lengths)), lengths]
        last_token_hidden = last_token_hidden.to(self.v_head.weight.dtype) # I use mixed precision, but last_hidden_state use float

        reward = self.v_head(last_token_hidden).squeeze(-1)  # (batch,)
        return reward

def collate_fn(batch):
    batch = {k: torch.tensor([example[k] for example in batch]) for k in batch[0]}
    return batch

model_name="Qwen/Qwen3-0.6B-Base"
output_dir="./trained_models/qwen-sft-summarization"
num_epochs=1
batch_size=4
learning_rate = 5e-5
num_workers=8
is_sft_train_again = False

# Load PEFT config to get base model info
lora_repo = "hiki-t/gpt_qwen_from_scratch"
peft_config = peft.PeftConfig.from_pretrained(lora_repo)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and merge LoRA
model = peft.PeftModel.from_pretrained(model, lora_repo)
sft_model = model.merge_and_unload()  # Permanently merges LoRA into base model

if is_sft_train_again:
    # train both sft and the last reward head
    lora_config = peft.LoraConfig(
        r=4,                         # Rank (typical: 4, 8, or 16)
        lora_alpha=8,               # Alpha scaling factor
        # target_modules=["q_proj", "v_proj"],  # Depends on model architecture
        lora_dropout=0.1,           # Dropout applied to LoRA layers
        bias="none",                 # Can be "none", "all", or "lora_only"
        task_type="CAUSAL_LM"  # For decoder-only models like Qwen
    )

    sft_model = peft.get_peft_model(sft_model, lora_config)
    reward_model = RewardModel(sft_model)
else:
    # train the last reward head
    reward_model = RewardModel(sft_model)

    for param in reward_model.model.parameters():
        param.requires_grad = False
    for param in reward_model.v_head.parameters():
        param.requires_grad = True

# Ensure v_head is on the same device as inputs later, or:
reward_model.v_head.to(next(sft_model.parameters()).device)

processed_train_dataset = load_and_prepare_data(train_dataset, tokenizer)
processed_val_dataset = load_and_prepare_data(val_dataset, tokenizer)

train_dataloader = DataLoader(
    processed_train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=collate_fn, 
    num_workers=num_workers, 
    pin_memory=True
)

val_dataloader = DataLoader(
    processed_val_dataset,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=collate_fn, 
    num_workers=num_workers, 
    pin_memory=True
)

optimizer = torch.optim.AdamW(reward_model.parameters(), lr=learning_rate)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)

# scaler = torch.GradScaler("cuda")
reward_model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress_bar = tqdm(train_dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        # batch = {k: v.to(device) for k, v in batch.items()}
        # with torch.autocast("cuda"):
        #     outputs = reward_model(**batch)
        #     loss = outputs.loss
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # outputs = reward_model(**batch)        
        # loss = outputs.loss
        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)

        chosen_rewards = reward_model(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = reward_model(rejected_input_ids, rejected_attention_mask)

        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Track loss
        total_loss += loss.item()

        # Track accuracy
        correct = (chosen_rewards > rejected_rewards).sum().item()
        total_correct += correct
        total_examples += chosen_rewards.size(0)

        avg_loss = total_loss / (step + 1)
        avg_accuracy = total_correct / total_examples

        progress_bar.set_description(f"Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.4f}")

    reward_model.push_to_hub("gpt_qwen_from_scratch_reward")

# # Validation phase
# reward_model.eval()
# correct = 0
# total = 0
# val_loss = 0.0
# with torch.no_grad():
#     val_progress = tqdm(val_dataloader, desc="Validation")
#     for step, batch in enumerate(val_progress):

#         chosen_input_ids = batch["chosen_input_ids"].to(device)
#         chosen_attention_mask = batch["chosen_attention_mask"].to(device)
#         rejected_input_ids = batch["rejected_input_ids"].to(device)
#         rejected_attention_mask = batch["rejected_attention_mask"].to(device)

#         chosen_rewards = reward_model(chosen_input_ids, chosen_attention_mask)
#         rejected_rewards = reward_model(rejected_input_ids, rejected_attention_mask)

#         # Loss
#         loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
#         val_loss += loss.item()

#         # Accuracy
#         correct += (chosen_rewards > rejected_rewards).sum().item()
#         total += chosen_rewards.size(0)

#         avg_val_loss = val_loss / (step + 1)
#         accuracy = correct / total
#         val_progress.set_description(f"Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.4f}")
