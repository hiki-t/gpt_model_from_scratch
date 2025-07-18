import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import peft
from huggingface_hub import hf_hub_download

from datasets import Dataset, load_dataset
from tqdm import tqdm
from typing import List, Dict

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is the model that generates summaries and is updated during PPO

### ------------------------------------------------------------------------------ ###
### custom functions
### ------------------------------------------------------------------------------ ###

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

    # def ppo_tokenize_fn(self, examples, **kwargs):
    #     queries = [str(q) for q in examples["query"]]
    #     responses = [str(r) for r in examples["response"]]
    #     full_texts = [q + " " + r for q, r in zip(queries, responses)]
    
    #     encoded = self.tokenizer(
    #         full_texts,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_length,
    #         return_tensors="pt"
    #     )
    
    #     encoded = {k: v.tolist() for k, v in encoded.items()}  # Convert to list for compatibility with HuggingFace Dataset
    
    #     # Add raw strings back if needed
    #     encoded["query"] = queries
    #     encoded["response"] = responses
    
    #     return encoded

    def ppo_tokenize_fn(self, examples, **kwargs):
        queries = [str(q) for q in examples["query"]]
        responses = [str(r) for r in examples["response"]]
        full_texts = [q + " " + r for q, r in zip(queries, responses)]

        # Tokenize full (query + response)
        encoded = self.tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tokenize only the query (prompt) part
        query_encoded = self.tokenizer(
            queries,
            padding="longest",          # or "max_length" with same truncation
            truncation=True,
            max_length=self.max_length,  # Optional: same max_length
            return_tensors="pt"
        )

        # Convert everything to lists for HF dataset compatibility
        encoded = {k: v.tolist() for k, v in encoded.items()}
        encoded["query_input_ids"] = query_encoded["input_ids"].tolist()
        encoded["query_attention_mask"] = query_encoded["attention_mask"].tolist()

        # Also store raw strings
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

def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + gamma * values[:, t + 1] * masks[:, t] - values[:, t]
        advantages[:, t] = last_advantage = delta + gamma * lam * masks[:, t] * last_advantage

    returns = advantages + values[:, :-1]
    return advantages.detach(), returns.detach()

def clean_text(text):
    # remove leading "Post: " or "Summary: " if present
    if text.startswith("Post: "):
        text = text[len("Post: "):]
    if text.startswith("Summary: "):
        text = text[len("Summary: "):]
    return text.strip()

### ------------------------------------------------------------------------------ ###
### set parameters
### ------------------------------------------------------------------------------ ###

model_name="Qwen/Qwen3-0.6B-Base"
output_dir="./trained_models/qwen-sft-summarization"
num_epochs=1
batch_size=1
learning_rate = 1e-5
num_workers=8
is_sft_train_again = False
saved_model = "reward_model.pt"
hf_reward_repo = "hiki-t/gpt_qwen_from_scratch_reward"
is_reweard_trained = False
is_there_trained_reward_lora_weight = True
lora_repo = "hiki-t/gpt_qwen_from_scratch"

### ------------------------------------------------------------------------------ ###
### load models
### ------------------------------------------------------------------------------ ###

# Load PEFT config to get base model info
peft_config = peft.PeftConfig.from_pretrained(lora_repo)

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and merge LoRA
model = peft.PeftModel.from_pretrained(base_model, lora_repo)
sft_model = model.merge_and_unload()  # Permanently merges LoRA into base model

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load models and tokenizer
policy_model = ActorCriticModel(sft_model).to(device) # this also contains a value model
ref_model = sft_model.to(device).eval()
reward_model = RewardModel(sft_model).to(device)
reward_model_path = hf_hub_download(repo_id=hf_reward_repo, filename=saved_model) # Download the reward model weights from HuggingFace Hub
reward_model.load_state_dict(torch.load(reward_model_path, map_location='cpu')) # Load the reward model weights
reward_model = reward_model.to(device).eval()

### ------------------------------------------------------------------------------ ###
### load dataset
### ------------------------------------------------------------------------------ ###

# Load your dataset
ds_train = load_dataset(
    "parquet",
    data_files={
        "train": ["https://huggingface.co/datasets/hiki-t/summarize_from_feedback/resolve/main/data/train-00000-of-00001.parquet"],
        "validation": ["https://huggingface.co/datasets/hiki-t/summarize_from_feedback/resolve/main/data/validation-00000-of-00001.parquet"]
    }
)
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

optimizer = optim.AdamW(policy_model.parameters(), lr=learning_rate)

### ------------------------------------------------------------------------------ ###
### PPO TRAINING LOOP (BATCHED & AUTOREGRESSIVE)
### ------------------------------------------------------------------------------ ###

# One-time initialization before training starts
gae_lambda = 0.85  # lower = less variance, more bias
gamma = 0.90       # discount factor
reward_mean = 0.0
reward_var = 1.0
reward_count = 1e-4  # small value to avoid division by zero
reward_decay = 0.99  # decay rate for moving average

for epoch in range(num_epochs):

    epoch_total_loss = 0
    epoch_policy_loss = 0
    epoch_value_loss = 0
    epoch_kl = 0
    epoch_reward = 0
    num_batches = 0
    
    progress_bar = tqdm(train_dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):  # each batch contains posts
        # Move all tensors in batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        posts = tokenizer.batch_decode(batch["query_input_ids"], skip_special_tokens=True)
        # === 1. Tokenize and generate summaries ===
        with torch.no_grad():
            # 1. Generate sequences with current policy (old policy at sampling time)
            gen_ids = policy_model.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=15,
                do_sample=True,
                top_p=0.95,
                # top_k=50,
                temperature=0.7, 
                no_repeat_ngram_size=5,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, 
            )
            full_input = gen_ids  # [B, T]
            
            # 2. Compute old log probs here, *with no grad*
            old_logits = policy_model.model(full_input).logits
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_action_log_probs = old_log_probs.gather(2, full_input.unsqueeze(-1)).squeeze(-1)
            
            # ✅ Add this line to get old value predictions
            _, old_values = policy_model(full_input)  # shape: [B, T]
            old_values = torch.cat([old_values, torch.zeros(old_values.size(0), 1).to(device)], dim=1)
    
        summaries = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # combined_inputs = [f"Post: {post}\n\nSummary: {summary}" for post, summary in zip(posts, summaries)]
        combined_inputs = [
            f"Post: {clean_text(post)}\n\nSummary: {clean_text(summary)}"
            for post, summary in zip(posts, summaries)
        ]

        # === 2. Compute logprobs (actor & reference) ===
        actor_logits = policy_model.model(full_input).logits
        ref_logits = ref_model(full_input).logits

        log_probs = F.log_softmax(actor_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        action_log_probs = log_probs.gather(2, full_input.unsqueeze(-1)).squeeze(-1)
        ref_action_log_probs = ref_log_probs.gather(2, full_input.unsqueeze(-1)).squeeze(-1)

        # === 3. KL divergence penalty ===
        # kl = action_log_probs - ref_action_log_probs
        # kl_penalty = kl.mean(dim=1)
        # kl = (action_log_probs - ref_action_log_probs).sum(dim=1)  # Total KL per sequence
        # kl_penalty = kl  # [batch_size]
        kl = action_log_probs - ref_action_log_probs  # ✅ shape: [B, T]
        kl_penalty = kl.mean(dim=1)  # ✅ shape: [B], used only for logging/loss weighting
        
        # === 4. Get rewards ===
        with torch.no_grad():
            rewards = []
            for s in combined_inputs:
                # print(s)
                r = reward_model(tokenizer(s, return_tensors="pt").input_ids.to(device))[0].item()
                rewards.append(r)
            # rewards = torch.tensor(rewards).unsqueeze(1).expand_as(kl).to(device)
            # rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        
            print("Reward list stats:", torch.mean(torch.tensor(rewards, dtype=torch.float)), torch.std(torch.tensor(rewards, dtype=torch.float)), torch.min(torch.tensor(rewards, dtype=torch.float)), torch.max(torch.tensor(rewards, dtype=torch.float)))
        print(combined_inputs[0])
        # rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(1)  # [B, 1]
        # rewards = rewards.expand_as(kl)  # [B, T]
        # # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # rewards = torch.clamp(rewards, min=-10, max=10)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(1)  # [B, 1]
        rewards = -1 * rewards  # <-- Invert the direction
        rewards = rewards.expand_as(kl)  # [B, T]
        
        # Get current batch stats
        batch_reward_mean = rewards.mean().item()
        
        # Update running stats (in float64 to reduce drift)
        reward_mean = reward_decay * reward_mean + (1 - reward_decay) * batch_reward_mean
        reward_var = reward_decay * reward_var + (1 - reward_decay) * ((batch_reward_mean - reward_mean) ** 2)
        
        # Normalize using updated stats
        rewards = (rewards - reward_mean) / (torch.sqrt(torch.tensor(reward_var, device=rewards.device)) + 1e-8)
        rewards = torch.clamp(rewards, min=-5.0, max=5.0)

        # # === 5. Value estimation ===
        # _, values = policy_model(full_input)
        # # Add dummy zero at T+1 for GAE
        # values = torch.cat([values, torch.zeros(values.size(0), 1).to(device)], dim=1)

        # masks = (full_input != tokenizer.pad_token_id).float().to(device)


        # # value_pred_clipped = old_values + (values - old_values).clamp(-0.2, 0.2)
        # value_pred_clipped = old_values[:, :seq_len] + (values[:, :-1] - old_values[:, :seq_len]).clamp(-0.2, 0.2)
        
        # value_loss = 0.5 * torch.max(
        #     F.mse_loss(values[:, :-1], returns, reduction="none"),
        #     F.mse_loss(value_pred_clipped, returns, reduction="none")
        # ).mean()

        # # === 6. GAE: compute advantage + returns ===
        # seq_len = min(rewards.size(1), values.size(1) - 1, masks.size(1))
        # rewards = rewards[:, :seq_len]
        # masks = masks[:, :seq_len]
        # values = values[:, :seq_len + 1]
        
        # advantages, returns = compute_gae(rewards, values, masks, gamma=gamma, lam=gae_lambda)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === 5. Value estimation ===
        # This goes AFTER computing returns and advantages

        # Compute current value predictions (used in loss)
        _, values = policy_model(full_input)  # [B, T]
        # Add dummy value at T+1 for bootstrapping (used for GAE)
        values = torch.cat([values, torch.zeros(values.size(0), 1).to(device)], dim=1)

        # === 6. GAE: compute advantage + returns ===
        masks = (full_input != tokenizer.pad_token_id).float().to(device)

        # Truncate to shortest sequence to align tensors
        seq_len = min(rewards.size(1), values.size(1) - 1, masks.size(1))
        rewards = rewards[:, :seq_len]
        masks = masks[:, :seq_len]
        values = values[:, :seq_len + 1]  # includes T+1 for GAE

        advantages, returns = compute_gae(rewards, values, masks, gamma=gamma, lam=gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === 7. Value loss with clipping ===
        # Compute new values again (same as above) so values[:, :-1] matches returns
        # values is already computed above
        value_pred_clipped = old_values[:, :seq_len] + (values[:, :-1] - old_values[:, :seq_len]).clamp(-0.2, 0.2)

        value_loss = 0.5 * torch.max(
            F.mse_loss(values[:, :-1], returns, reduction="none"),
            F.mse_loss(value_pred_clipped, returns, reduction="none")
        ).mean()

        # === 7. PPO loss ===
        ratio = (action_log_probs - old_action_log_probs.detach()).exp()
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # value_loss = F.mse_loss(values[:, :-1], returns)

        entropy = -action_log_probs.mean()
        entropy_bonus = 0.01 * entropy

        total_loss = policy_loss + 0.5 * value_loss + kl_penalty.mean() * 0.1 - entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"loss={total_loss.item():.4f} reward={rewards.mean():.4f} kl={kl_penalty.mean().item():.4f}")
        print(f"values.mean={values[:, :-1].mean().item():.4f}, returns.mean={returns.mean().item():.4f}, advantage.mean={advantages.mean().item():.4f}")

        print("Raw rewards:", rewards[0][0])
        print("Reward mean:", rewards.mean().item())
        print("Reward std:", rewards.std().item())
        
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