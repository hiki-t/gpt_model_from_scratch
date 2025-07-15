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

###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###

class SummarizationDataProcessor:
    def __init__(self, tokenizer, max_length=2048):
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

# def train_sft_model(
#     model_name: str = "Qwen/Qwen3-0.6B-Base",
#     train_dataset=None, 
#     val_dataset=None, 
#     output_dir: str = "./trained_models/qwen-sft-summarization",
#     num_epochs: int = 1,
#     batch_size: int = 4,
#     learning_rate: float = 5e-5
# ):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     processed_train_dataset = load_and_prepare_data(train_dataset, tokenizer)
#     processed_val_dataset = load_and_prepare_data(val_dataset, tokenizer)
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=num_epochs,
#         per_device_train_batch_size=batch_size,
#         gradient_accumulation_steps=4,
#         warmup_steps=500,
#         learning_rate=learning_rate,
#         logging_steps=100,
#         save_steps=1000,
#         eval_strategy="no",
#         save_strategy="steps",
#         fp16=True,
#         dataloader_drop_last=True,
#         remove_unused_columns=False,
#         # report_to="tensorboard", 
#         push_to_hub=True, 
#         hub_model_id="hiki-t/gpt_qwen_from_scratch", 
#     )

#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False,
#         pad_to_multiple_of=8,
#         return_tensors="pt"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=processed_train_dataset, 
#         eval_dataset=processed_val_dataset, 
#         tokenizer=tokenizer,
#         data_collator=data_collator
#     )

#     trainer.train()
#     trainer.save_model(output_dir) # Save model in HuggingFace format (for later push)
#     tokenizer.save_pretrained(output_dir)
#     # Save as sft_model.pt (PyTorch state_dict)
#     torch.save(model.state_dict(), f"{output_dir}/sft_model.pt")

#     return model, tokenizer

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
    batch_size=4,
    collate_fn=data_collator
)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)


model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    progress_bar = tqdm(train_dataloader)
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        break

