#!/usr/bin/env python3
"""
train_qwenvl_lora_classifier.py

Requirements:
  pip install transformers accelerate peft datasets evaluate torch torchvision pillow

Usage (example):
  python train_qwenvl_lora_classifier.py \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name your_dataset_name_or_path \
    --image_column image_path \
    --text_column text \
    --label_column label \
    --num_labels 5 \
    --output_dir ./qwenvl_lora_classifier \
    --batch_size 8 \
    --epochs 3
"""

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
    AdamW,
)
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
            ('lo',nn.Linear(hidden_size//4,520))
        ]))

    def forward(self, **model_kwargs):
        """
        model_kwargs should be the processor outputs forwarded into base model,
        e.g., pixel_values, input_ids, attention_mask depending on model.
        """
        # We rely on the base model to return last_hidden_state
        outputs = self.base(**model_kwargs, output_hidden_states=True, return_dict=True)
        # print('outputs', model_kwargs)
        # last_hidden_state shape: (B, seq_len, H)
        # print(outputs.hidden_states[0].shape)
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







# print(num_to_element)
# print(0/0)



import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch.optim as optim
from transformers import FlavaModel, FlavaProcessor, FlavaConfig, Qwen2VLForConditionalGeneration
from PIL import Image
# Define a simple custom dataset
from sklearn.model_selection import train_test_split
from qwen_vl_utils import process_vision_info

class CustomDataset(Dataset):
    def __init__(self, data, element_to_num, num_to_element, is_train=True):
        self.label, self.exp, self.text, self.img_id = [],[],[],[]
        self.element_to_num = element_to_num
        self.num_to_element = num_to_element
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        self.is_train = is_train
        kk = 0

        print('data sample', data[:2])
        
        for d in data:
            # print(d)
            # max_key = max(d['label'], key=d['label'].get)
            # self.label.append(self.element_to_num[max_key])
            # self.exp.append(d['explanation'][0])
            # self.text.append(d['sent'])
            # self.img_id.append(d['img_id'])
            
            try:
                max_key = max(d['label'], key=d['label'].get)
                self.label.append(self.element_to_num[max_key])
                self.exp.append(d['explanation'][0])
                self.text.append(d['sent'])
                self.img_id.append(d['img_id'])
            except:
                kk+=1
                print('here')
                continue
        self.label_tr, self.label_ts, self.exp_tr, self.exp_ts, self.text_tr, self.text_ts, self.img_id_tr, self.img_id_ts = train_test_split(
            self.label, self.exp, self.text, self.img_id,
            test_size=0.1485,
            random_state=42
        )
        if is_train:
            self.label, self.exp, self.text, self.img_id = self.label_tr, self.exp_tr, self.text_tr, self.img_id_tr
        else:
            self.label, self.exp, self.text, self.img_id = self.label_ts, self.exp_ts, self.text_ts, self.img_id_ts




        print(kk)
            
            
        
    def __len__(self):
        return len(self.exp)
    
    def __getitem__(self, idx):
        #         inputs = self.tokenizer(self.questions[idx], return_tensors="pt")
    
        



         
        gl = self.label[idx]

        explanation = self.exp[idx]

        ip = self.img_id[idx]
        text = self.text[idx]
            
        
      
        # print(self.img_path[idx])
        img_path = os.path.join('./train2014/', ip+'.jpg')




        img  = Image.open(img_path).convert("RGB")
        # explanation = self.explanation[idx]
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            )
        """

        # print(inputs)
        # enc = self.processor(
        #     text=text,
        #     images=img,
        #     padding="max_length",
        #     max_length=128,
        #     return_tensors="pt",
        # )

        # enc = {k: v.squeeze(0) for k, v in enc.items()}

        

        # target = torch.zeros(3)
        # for l, s in zip(self.annotations[idx]["labels"], self.annotations[idx]["scores"]):
        #     target[l] = s
        # enc["labels"] = target
        # return enc

        # return inputs, gl, explanation, img_path, text
        return gl, explanation, img_path, text
       
        #return multimodal_embeds[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]
        # return inputs , gl, self.explanation[idx], self.img_path[idx], self.hypothesis[idx]





















from tqdm import tqdm
from transformers import Qwen2VLModel, Qwen2VLProcessor


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading processor and base model:", args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)
    # load base model (use AutoModel to get the encoder-like model that returns last_hidden_state)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, trust_remote_code=True)
    base_model.eval()

    # Infer hidden size from base model config / embeddings
    try:
        hidden_size = base_model.config.hidden_size
    except AttributeError:
        # fallback: try reading a tensor shape
        with torch.no_grad():
            # create dummy inputs via processor may be heavy; default fallback
            hidden_size = 4096

    # Wrap with PEFT LoRA configuration
    print("Preparing LoRA config...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION  # classifier on top of base
    )

    # Apply PEFT (this will add adapter modules to the base_model)
    peft_model = get_peft_model(base_model, lora_config)
    # Freeze base model parameters EXCEPT LoRA adapters (PEFT sets .requires_grad appropriately)
    for n, p in peft_model.named_parameters():
        if "lora" not in n and "adapter" not in n:
            p.requires_grad = False

    # Wrap with classifier head module
    model = QwenVLWithClassifier(peft_model, hidden_size=hidden_size, num_labels=args.num_labels).to(device)

    # Prepare datasets
    print("Loading dataset:")
    # This expects a dataset in HF datasets format; if your dataset is local, load accordingly
    # ds = load_dataset(args.dataset_name, split=args.split)  # user must adapt if necessary
    # # Optionally, if the image column contains file paths not PIL images, map to PIL open inside a map function:
    # # ds = ds.map(lambda x: {"image": Image.open(x["image_path"]).convert("RGB")})

    # # Simple train/val split if dataset doesn't contain
    # if args.split == "train":
    #     ds = ds.train_test_split(test_size=args.test_size)
    #     train_ds = ds["train"]
    #     eval_ds = ds["test"]
    # else:
    #     train_ds = ds
    #     eval_ds = load_dataset(args.dataset_name, split="validation")

    # # data collator
    # collator = DataCollatorForQwenVL(processor, text_column=args.text_column,
    #                                  image_column=args.image_column,
    #                                  label_column=args.label_column, device=device)

    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    # eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)



    


    import json
    
    with open('./vqax/train_x.json', 'r') as file:
        data = json.load(file)
    
    data = data[:6478]

    # data = data[:100]
    
    label, exp, text, img_id = [],[],[],[]
    
    for d in data:
        try:
            max_key = max(d['label'], key=d['label'].get)
            label.append(max_key)
            exp.append(d['explanation'][0])
            text.append(d['sent'])
            img_id.append(d['img_id'])
        except:
            continue
    
    print(len(label), len(exp), len(text), len(img_id))
    
    print(label[:2], exp[:2], text[:2], img_id[:2])
    
    # elements = list(set(label))
    
    # print(len(elements))
    
    from collections import OrderedDict
    
    # ✅ Preserves order of first appearance -VVVVVVV Important
    elements = list(OrderedDict.fromkeys(label))
    print(len(elements))
    # print(0/0)
    # Create element to number mapping
    element_to_num = {element: i for i, element in enumerate(elements)}
    # Create number to element mapping
    num_to_element = {i: element for i, element in enumerate(elements)}


    train_dataset = CustomDataset(data, element_to_num, num_to_element)
    test_dataset = CustomDataset(data, element_to_num, num_to_element, is_train=False)

    # print('total in exception', kk)
    
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
    #   worker_init_fn=seed_worker)
    
    
    print(len(train_dataset), len(test_dataset))
    
    # print(0/0)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers = 4, worker_init_fn=seed_worker)

    eval_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers = 4, worker_init_fn=seed_worker)

    
    # print(0/0)



    # Optimizer -> only params with requires_grad True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Training parameter count:", sum(p.numel() for p in trainable_params))
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps),
                                                num_training_steps=total_steps)

    # Loss
    criterion = nn.NLLLoss()

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            messages = []
            
            img_path, text = batch[2], batch[3]
            # print(img_path, text)
            for ip, txt in zip(img_path, text):
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ip},
                            {"type": "text", "text": txt},
                        ],
                    }
                ]
                messages.append(message)
        
            # print(messages)
            texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in tqdm(messages)
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            # print(image_inputs, video_inputs, texts)
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # if "pixel_values" in inputs:
            #     inputs["image_inputs"] = inputs.pop("pixel_values")
            # Drop extra keys not in model.forward signature
            # print(inputs)
            model.base.config.update({'return_dict': True})
            # print(model.base.config)
            # print(0/0)
            labels, a, b, c = batch
            optimizer.zero_grad()
            # Forward: we expect processor to produce keys that base_model accepts (like pixel_values, input_ids, attention_mask)
            # inputs{'return_dict'} = False
            # inputs.update({"return_dict": True})
            log_probs, logits, pooled = model(**inputs)
            # labels = batch["labels"]
            loss = criterion(log_probs, labels.to('cuda'))
            print(loss)
            # print(0/0)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if (step + 1) % args.log_steps == 0:
                print(f"Epoch {epoch+1} Step {step+1}/{len(train_dataloader)} Loss {running_loss / args.log_steps:.4f}")
                running_loss = 0.0

        # Eval pass
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

from sklearn.metrics import f1_score
all_ops, all_tgts = [],[]
def evaluate_model(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            messages = []
            
            img_path, text = batch[2], batch[3]
            # print(img_path, text)
            for ip, txt in zip(img_path, text):
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ip},
                            {"type": "text", "text": txt},
                        ],
                    }
                ]
                messages.append(message)
        
            # print(messages)
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in tqdm(messages)
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            # print(image_inputs, video_inputs, texts)
            
            
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # if "pixel_values" in inputs:
            #     inputs["image_inputs"] = inputs.pop("pixel_values")
            # Drop extra keys not in model.forward signature
            # print(inputs)
            model.base.config.update({'return_dict': True})
            # print(model.base.config)
            # print(0/0)
            labels, a, b, c = batch
            log_probs, logits, pooled = model(**inputs)
            preds = logits.argmax(dim=-1)
            # labels = batch["labels"]
            correct += (preds == labels.to('cuda')).sum().item()
            total += labels.size(0)
            all_ops.extend(preds.cpu().numpy())
            all_tgts.extend(labels.cpu().numpy())
    print('F1 Macro', f1_score(all_tgts, all_ops, average='macro'))
    print('F1 Micro', f1_score(all_tgts, all_ops, average='micro'))
    return correct / total if total > 0 else 0.0


# -----------------------------
# Argument parser and main
# -----------------------------
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
    parser.add_argument("--output_dir", type=str, default="./qwenvl_lora_classifier_vqax")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")  # tweak based on model
    parser.add_argument("--test_size", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)