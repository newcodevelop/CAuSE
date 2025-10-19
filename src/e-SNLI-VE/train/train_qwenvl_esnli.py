from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from torch.optim import AdamW, Adam
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch.optim as optim
from transformers import FlavaModel, FlavaProcessor, FlavaConfig, Qwen2VLForConditionalGeneration, AutoModel
from PIL import Image
# Define a simple custom dataset
from sklearn.model_selection import train_test_split
from qwen_vl_utils import process_vision_info

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from transformers import (
    AutoProcessor,
    AutoModel,  # generic; many Qwen-VL variants can be loaded with AutoModel
    get_linear_schedule_with_warmup,
   
)
from torch.optim import AdamW
import jsonlines
from collections import OrderedDict
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


def seed_all(seed):
    if not seed:
        seed = 42

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



seed_all(42)
# -----------------------------
# Helpers: pooling and model
# -----------------------------
class QwenVLWithClassifier(nn.Module):
    """
    Wrap the base Qwen-VL model to produce a pooled multimodal embedding,
    then pass through an MLP classification head.
    """
    def __init__(self, base_model, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        # You can change pooling strategy later; currently using mean-pooling over tokens
        self.pool = lambda hs, mask=None: hs.mean(dim=1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size // 2, num_labels),
        # )
        self.lin = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(hidden_size, hidden_size//2)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(hidden_size//2,hidden_size//4)),
            ('relu2', nn.ReLU()),
            ('lo',nn.Linear(hidden_size//4,3))
        ]))

    def forward(self, **model_kwargs):
        """
        model_kwargs should be the processor outputs forwarded into base model,
        e.g., pixel_values, input_ids, attention_mask depending on model.
        """
        # We rely on the base model to return last_hidden_state
        outputs = self.base(**model_kwargs, output_hidden_states=True, return_dict=True)
        # print(outputs.keys(), len(outputs.hidden_states))
        # print(0/0)
        # print('outputs', model_kwargs)
        # last_hidden_state shape: (B, seq_len, H)
        print(outputs.hidden_states[-1].shape)
        # print(0/0)
        last_hidden = outputs.hidden_states[-1]

        mask = model_kwargs["attention_mask"]  # (B, seq_len)
        # expand mask for broadcasting
        mask = mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, seq_len, 1)
        
        # apply mask before summing
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        # pooled = self.pool(last_hidden)  # (B, H)
        logits = self.lin(pooled)
        # print(logits, logits.shape)
        return torch.nn.functional.log_softmax(logits, dim=-1), logits, pooled

# -----------------------------
# Data collator
# -----------------------------
@dataclass
class DataCollatorForQwenVL:
    processor: Any
    text_column: str
    image_column: str
    label_column: str
    device: torch.device

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Build lists
        images = [f[self.image_column] for f in features]
        texts = [f.get(self.text_column, "") for f in features]
        labels = [f[self.label_column] for f in features]

        # Processor can handle multimodal inputs (images + text)
        # The exact call depends on processor; the following is generic
        batch = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.long, device=self.device)
        return batch

# -----------------------------
# Training loop
# -----------------------------




df_train = torch.load('./train_df_enli.pt')
df_test = torch.load('./test_df_enli.pt')
def get_top_k_by_keys(d,k):
    # Sort keys in descending order and take top 1000
    sorted_keys = sorted(d.keys(), reverse=True)[:k]
    return {k: d[k] for k in sorted_keys}



# # --- reproducible shuffle ---
# random.seed(42)  # same shuffled order every run
# keys = list(df_train.keys())
# random.shuffle(keys)
# print(keys)
# df_train = {i: df_train[k] for i,k in enumerate(keys[:640])}
# # df_train = dict(sorted(df_train.items(), key=lambda x: x[0])[:640])




# random.seed(42)  # same shuffled order every run
# keys = list(df_test.keys())
# random.shuffle(keys)
# df_test = {i: df_test[k] for i,k in enumerate(keys[:64])}
# # df_test = dict(sorted(df_test.items(), key=lambda x: x[0])[:100])


print(len(df_train), len(df_test))
# print(df_train.keys())
# print(0/0)

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch.optim as optim
from transformers import FlavaModel, FlavaProcessor, FlavaConfig
from PIL import Image
# Define a simple custom dataset
class CustomDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.txt_features = torch.stack([df[i]['text_feats'] for i in range(len(df))]).squeeze().cpu()
        self.image_features = torch.stack([df[i]['img_feats'] for i in range(len(df))]).squeeze().cpu()
        self.labels = [df[i]['gold_label'] for i in range(len(df))]
        self.explanation = [df[i]['explanation'] for i in range(len(df))]
        self.img_path = [df[i]['img_path'] for i in range(len(df))]
        self.hypothesis = [df[i]['hypothesis'] for i in range(len(df))]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        self.is_train = is_train

        if self.is_train:
            self.feat_vb = torch.load('./train_vb_esnli.pt')
        else:
            self.feat_vb = torch.load('./test_vb_esnli.pt')
        #         self.tokenizer = VisualBertTokenizer.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        
    def __len__(self):
        return self.txt_features.shape[0]
    
    def __getitem__(self, idx):
        #         inputs = self.tokenizer(self.questions[idx], return_tensors="pt")
    
        
        
        gl = 'none'
        if self.labels[idx]=='contradiction':
            gl = 0
        elif self.labels[idx]=='neutral':
            gl = 1
        elif self.labels[idx]=='entailment':
            gl = 2
            
        
        
      
        # print(self.img_path[idx])
        img_path = os.path.join('/home/anonymous/unsup_nle/flickr30k_images/flickr30k_images/flickr30k_images/',self.img_path[idx])
        img  = Image.open(img_path).convert("RGB")
        txt = self.hypothesis[idx]
        
     
      

        return gl, self.explanation[idx], img_path, self.hypothesis[idx]
       
     


train_dataset_ = CustomDataset(df_train)
test_dataset_ = CustomDataset(df_test, is_train=False)

# k = []
# for i in train_dataset_:
#     k.append(i[0])

# print(k)
# print(0/0)


# -----------------------------
# Training Loop
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚡ Load processor/model ONCE
    processor = AutoProcessor.from_pretrained(args.model_name)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, trust_remote_code=True)
    hidden_size = getattr(base_model.config, "hidden_size", 4096)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(base_model, lora_config)
    for n, p in peft_model.named_parameters():
        if "lora" not in n and "adapter" not in n:
            p.requires_grad = False

    model = QwenVLWithClassifier(peft_model, hidden_size, args.num_labels).to(device)

    train_dataset = RawDataset(train_dataset_)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=10,
        pin_memory=True,
    )

    test_dataset = RawDataset(test_dataset_)
    eval_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=10,
        pin_memory=True,
    )

    print(len(train_dataset), len(test_dataset))
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # print(trainable_params)
    # print(0/0)
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = Adam(trainable_params, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps
    )

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()  # Changed
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for step, batch in tqdm(enumerate(train_loader)):
            # print('batch', batch)
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # ⚡ mixed precision
                log_probs, logits, _ = model(**batch)
                # loss = criterion(log_probs, labels)
                # logits, _, _ = model(**batch)
                print(labels)
                loss = criterion(logits, labels)
                print(loss)

            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if (step + 1) % args.log_steps == 0:
                print(f"Epoch {epoch+1} Step {step+1}/{len(train_loader)} Loss {running_loss/args.log_steps:.4f}")
                running_loss = 0.0

        val_acc = evaluate_model(model, eval_loader, device)
        print(f"Epoch {epoch+1} Validation Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save PEFT adapters and classifier head
            print("Saving best model...")
            # Save PEFT adapters (saves adapters into output_dir/adapter_model.bin + config)
            # peft save expects the underlying base model; we can call save_pretrained on the peft wrapper
            try:
                peft_model.save_pretrained(os.path.join(args.output_dir, "peft_adapters"))
            except Exception as e:
                print("Warning: could not save peft adapters directly:", e)
            # Save classifier head
            torch.save(model.lin.state_dict(), os.path.join(args.output_dir, "classifier_head.pt"))

    print("Training complete. Best val acc:", best_val_acc)

  





class RawDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset):
        self.samples = raw_dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gl, explanation, img_path, text = self.samples[idx]
        return gl, img_path, text


def collate_fn(batch, processor):
    labels, img_paths, texts = zip(*batch)

    messages = []
    for text, img_path in zip(texts, img_paths):
        messages.append([{"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": text},
        ]}])

    chat_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    enc = processor(
        text=chat_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,             # <- batch padded correctly
        truncation=True,
        return_tensors="pt"
    )
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc



# -----------------------------
# Preprocessed Dataset
# -----------------------------
class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, processor):
        self.samples = []
        for gl, explanation, img_path, text in tqdm(raw_dataset):
            message = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": text},
            ]}]
            chat_text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(message)
            enc = processor(
                text=chat_text, images=image_inputs, videos=video_inputs,
                padding="max_length", truncation=True,
                return_tensors="pt"
            )
            enc = {k: v.squeeze(0) for k, v in enc.items()}  # remove batch dim
            enc["labels"] = torch.tensor(gl, dtype=torch.long)
            self.samples.append(enc)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_model(model, dataloader, device):
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        _, logits, _ = model(**batch)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    print("F1 Macro:", f1_score(all_labels, all_preds, average="macro"))
    print("F1 Micro:", f1_score(all_labels, all_preds, average="micro"))
    return acc



def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA + Qwen-VL classifier head")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    # parser.add_argument("--dataset_name", type=str, required=True,
    #                     help="HuggingFace dataset id or local dataset path. Should contain image, text, label.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="./qwenvl_lora_classifier_esnli")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")  # tweak based on model
    parser.add_argument("--test_size", type=float, default=0.1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)