
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from datasets import Dataset, load_dataset
from tqdm import tqdm
from typing import List, Dict

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is the model that generates summaries and is updated during PPO

class ActorCriticModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model  # Hugging Face model (e.g., Qwen)
        hidden_size = base_model.config.hidden_size
        self.v_head = nn.Linear(hidden_size, 1)  # Value head  # <-- THIS is the value model?

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        values = self.v_head(hidden_states).squeeze(-1)  # (B, T)
        return outputs, values  # <-- values = value model output

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model  # HuggingFace model
        hidden_size = self.model.config.hidden_size
        self.v_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (B, T, H)
        values = self.v_head(last_hidden).squeeze(-1)  # (B, T)

        if attention_mask is not None:
            # Use value of last non-padding token
            last_token_idx = attention_mask.sum(dim=1) - 1  # (B,)
            reward = values.gather(1, last_token_idx.unsqueeze(1)).squeeze(1)  # (B,)
        else:
            reward = values[:, -1]  # fallback

        return reward

def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + gamma * values[:, t + 1] * masks[:, t] - values[:, t]
        advantages[:, t] = last_advantage = delta + gamma * lam * masks[:, t] * last_advantage

    returns = advantages + values[:, :-1]
    return advantages.detach(), returns.detach()

class SummarizationDataProcessor:
    def __init__(self, tokenizer, max_length=256):
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
                    'query': prompt,
                    'response': chosen_summary,
                    'rejected_response': rejected_summary,
                    'post_id': info['id'], 
                    'worker_id': item.get('worker', 'unknown'), 
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

    def ppo_tokenize_fn(self, examples, **kwargs):
        queries = [str(q) for q in examples["query"]]
        responses = [str(r) for r in examples["response"]]
        full_texts = [q + " " + r for q, r in zip(queries, responses)]
    
        encoded = self.tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
        encoded = {k: v.tolist() for k, v in encoded.items()}  # Convert to list for compatibility with HuggingFace Dataset
    
        # Add raw strings back if needed
        encoded["query"] = queries
        encoded["response"] = responses
    
        return encoded

def load_and_prepare_data(dataset, tokenizer):
    processor = SummarizationDataProcessor(tokenizer)
    raw_data = []
    for item in dataset:
        raw_data.append(item)
    processed_data = processor.process_summarize_from_feedback(raw_data)
    dataset = Dataset.from_list(processed_data)
    tokenized_dataset = dataset.map(
        # processor.tokenize_pair,
        # processor.tokenize_function,
        processor.ppo_tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def collate_fn(batch):
    batch = {k: torch.tensor([example[k] for example in batch]) for k in batch[0]}
    return batch

def ppo_collate_fn(batch):
    batch_dict = {key: [item[key] for item in batch] for key in batch[0]}
    return {key: torch.tensor(val) for key, val in batch_dict.items() if isinstance(val[0], (int, float, list))}

### 
### 
### 

model_name="Qwen/Qwen3-0.6B-Base"
output_dir="./trained_models/qwen-sft-summarization"
num_epochs=1
batch_size=1
learning_rate = 5e-5
num_workers=8
is_sft_train_again = False
saved_model = "reward_model.pt"
hf_reward_repo = "hiki-t/gpt_qwen_from_scratch_reward"
is_reweard_trained = False
is_there_trained_reward_lora_weight = False

### 

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
base_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

tokenizer.pad_token = tokenizer.eos_token # this is because gpt2 doesn't have pad token

# Load models and tokenizer
policy_model = ActorCriticModel(base_model).to(device) # this also contains a value model
reward_model = RewardModel(base_model).to(device)
ref_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(device).eval()

# Load your dataset
ds_train = load_dataset("openai/summarize_from_feedback", "comparisons")
train_dataset = ds_train["train"]
val_dataset = ds_train["validation"]

processed_train_dataset = load_and_prepare_data(train_dataset, tokenizer)
train_dataloader = DataLoader(
    processed_train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=ppo_collate_fn, 
    num_workers=num_workers, 
    pin_memory=True
)

# processed_val_dataset = load_and_prepare_data(val_dataset, tokenizer)
# val_dataloader = DataLoader(
#     processed_val_dataset,
#     shuffle=True,
#     batch_size=batch_size,
#     collate_fn=ppo_collate_fn, 
#     num_workers=num_workers, 
#     pin_memory=True
# )

optimizer = optim.AdamW(policy_model.parameters(), lr=learning_rate)

### 
### 🔁 PPO TRAINING LOOP (BATCHED & AUTOREGRESSIVE)
### 

for epoch in range(num_epochs):

    epoch_total_loss = 0
    epoch_policy_loss = 0
    epoch_value_loss = 0
    epoch_kl = 0
    epoch_reward = 0
    num_batches = 0
    
    progress_bar = tqdm(train_dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):  # each batch contains `posts`
        # Move all tensors in batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        # === 1. Tokenize and generate summaries ===
        with torch.no_grad():
            # 1. Generate sequences with current policy (old policy at sampling time)
            gen_ids = policy_model.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=64,
                do_sample=True,
                top_k=50,
                temperature=1.0
            )
            full_input = gen_ids  # [B, T]

            # 2. Compute old log probs here, *with no grad*
            old_logits = policy_model.model(full_input).logits
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_action_log_probs = old_log_probs.gather(2, full_input.unsqueeze(-1)).squeeze(-1)
        
        summaries = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # === 2. Compute logprobs (actor & reference) ===
        actor_logits = policy_model.model(full_input).logits
        ref_logits = ref_model(full_input).logits

        log_probs = F.log_softmax(actor_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        action_log_probs = log_probs.gather(2, full_input.unsqueeze(-1)).squeeze(-1)
        ref_action_log_probs = ref_log_probs.gather(2, full_input.unsqueeze(-1)).squeeze(-1)

        # === 3. KL divergence penalty ===
        kl = action_log_probs - ref_action_log_probs
        kl_penalty = kl.mean(dim=1)

        # === 4. Get rewards ===
        with torch.no_grad():
            rewards = []
            for s in summaries:
                r = reward_model(tokenizer(s, return_tensors="pt").input_ids.to(device))[0].item()
                rewards.append(r)
            rewards = torch.tensor(rewards).unsqueeze(1).expand_as(kl).to(device)

        # === 5. Value estimation ===
        _, values = policy_model(full_input)
        # Add dummy zero at T+1 for GAE
        values = torch.cat([values, torch.zeros(values.size(0), 1).to(device)], dim=1)

        masks = (full_input != tokenizer.pad_token_id).float().to(device)

        # === 6. GAE: compute advantage + returns ===
        advantages, returns = compute_gae(rewards, values, masks)

        # === 7. PPO loss ===
        ratio = (action_log_probs - action_log_probs.detach()).exp()
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        value_loss = F.mse_loss(values[:, :-1], returns)
        total_loss = policy_loss + 0.5 * value_loss + kl_penalty.mean() * 0.1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"loss={total_loss.item():.4f} reward={rewards.mean():.4f} kl={kl_penalty.mean().item():.4f}")

        # Accumulate
        epoch_total_loss += total_loss.item()
        epoch_policy_loss += policy_loss.item()
        epoch_value_loss += value_loss.item()
        epoch_kl += kl_penalty.mean().item()
        epoch_reward += rewards.mean().item()
        num_batches += 1

    print(f"[Epoch {epoch+1}] avg_loss={epoch_total_loss / num_batches:.4f} "
          f"policy_loss={epoch_policy_loss / num_batches:.4f} "
          f"value_loss={epoch_value_loss / num_batches:.4f} "
          f"kl={epoch_kl / num_batches:.4f} "
          f"reward={epoch_reward / num_batches:.4f}")
