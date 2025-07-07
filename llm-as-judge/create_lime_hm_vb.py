import torch

import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import os
import jsonlines





# for i in sample_all:
#     print(df[i].split("exp:")[-1].strip().lower())

# print(0/0)


# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--ablation', type=str, required=True)
# parser.add_argument('--counterfactual_test', type=int, required=True)

# args = parser.parse_args()

# print(f"Ablation: {args.ablation.lower()}")

# #Run checks
# if args.ablation.lower() not in ['phi2', 'phi2_ts', 'cause']:
#     raise Exception("Dataset should be phi2, phi2_ts, cause")


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

df = torch.load('../kb-fb-ds/kb_fb.pt')


import sys
sys.path.append('/home/anonymous/unsup_nle/transformers-research-projects/visual_bert')


import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

device = 'cuda'

from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config


device = 'cuda'
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn_cfg.MODEL.device = device
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

frcnn.eval()
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def get_visual_embedding(img_paths):
    images, sizes, scales_yx = image_preprocess(img_paths) # img_paths -> list of image paths
    output_dict = frcnn(
      images,
      sizes,
      scales_yx=scales_yx,
      padding="max_detections",
      max_detections=frcnn_cfg.max_detections,
      return_tensors="pt",
    )
    features = output_dict.get("roi_features")
    normalized_boxes = output_dict.get("normalized_boxes")
    return features, normalized_boxes


from PIL import Image
import numpy as np

# def get_visual_embedding(images_or_paths):
#     """
#     Accepts a list of image paths or numpy arrays
#     """
#     processed_images = []
#     for img in images_or_paths:
#         if isinstance(img, str):
#             pil_img = Image.open(img).convert("RGB")
#         elif isinstance(img, np.ndarray):
#             pil_img = Image.fromarray(img.astype('uint8')).convert("RGB")
#         else:
#             raise ValueError("Unsupported image format")

#         processed_images.append(pil_img)

#     images, sizes, scales_yx = image_preprocess(processed_images)  # Now PIL images
#     output_dict = frcnn(
#         images,
#         sizes,
#         scales_yx=scales_yx,
#         padding="max_detections",
#         max_detections=frcnn_cfg.max_detections,
#         return_tensors="pt",
#     )
#     features = output_dict.get("roi_features")
#     normalized_boxes = output_dict.get("normalized_boxes")
#     return features, normalized_boxes



import tempfile
import os
from PIL import Image
import numpy as np

# def get_visual_embedding(images_or_arrays):
#     """
#     Accepts a list of image paths, PIL images, or NumPy arrays.
#     Saves temporary files for images that are not paths.
#     Deletes them after processing.
#     """
#     temp_files = []
#     img_paths = []

#     for img in images_or_arrays:
#         if isinstance(img, str):
#             img_paths.append(img)
#         else:
#             # Convert array or PIL to file
#             if isinstance(img, np.ndarray):
#                 img = Image.fromarray(img.astype('uint8')).convert("RGB")
#             elif isinstance(img, Image.Image):
#                 img = img.convert("RGB")
#             else:
#                 raise ValueError("Expected str, np.ndarray or PIL.Image")

#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
#             img.save(temp_file.name)
#             temp_file.close()
#             temp_files.append(temp_file.name)
#             img_paths.append(temp_file.name)

#     try:
#         # Your existing preprocessing and feature extraction
#         images, sizes, scales_yx = image_preprocess(img_paths)
#         output_dict = frcnn(
#             images,
#             sizes,
#             scales_yx=scales_yx,
#             padding="max_detections",
#             max_detections=frcnn_cfg.max_detections,
#             return_tensors="pt",
#         )
#         features = output_dict["roi_features"]
#         normalized_boxes = output_dict["normalized_boxes"]
#     finally:
#         # Always clean up temp files, even on failure
#         for f in temp_files:
#             os.unlink(f)

#     return features, normalized_boxes




import torch
from torch.utils.data import DataLoader, Dataset
# from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
import torch.optim as optim

# Define a simple custom dataset
class CustomDataset(Dataset):
    def __init__(self, txt_feature, image_features, labels, explanation, img_path):
        self.txt_features = txt_feature
        self.image_features = image_features
        self.labels = labels
        self.explanation = explanation
        self.img_path = img_path
        #         self.tokenizer = VisualBertTokenizer.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        
    def __len__(self):
        return self.txt_features.shape[0]
    
    def __getitem__(self, idx):
        #         inputs = self.tokenizer(self.questions[idx], return_tensors="pt")
        multimodal_embeds = torch.cat((self.txt_features, self.image_features), dim=-1)
        
       
        #return multimodal_embeds[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]
        return self.txt_features[idx,:], self.image_features[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]

all_img_tags = list(torch.load('../tensors/img_id.pt'))
all_img_tags.extend(list(torch.load('../tensors/img_id_.pt')))

print(len(all_img_tags))

import pickle


with open('../explanation/explanation-fb-0_2749.pickle', 'rb') as h:
    p1 = pickle.load(h)

with open('../explanation/explanation-fb_remaining.pickle', 'rb') as h:
    p2 = pickle.load(h)


# all_txt_feats = []
# all_img_feats  = []

# txt_path = '/kaggle/input/tensors/tx_tensor.pt'
# img_path = 


all_txt_feats = torch.load('../tensors/tx_tensor.pt')
all_txt_feats = torch.cat((all_txt_feats, torch.load('../tensors/tx_tensor_.pt')), dim=0)

all_img_feats = torch.load('../tensors/im_tensor.pt')
all_img_feats = torch.cat((all_img_feats, torch.load('../tensors/im_tensor_.pt')), dim=0)

all_gl = torch.load('../tensors/gl.pt')
print(len(all_gl))
all_gl.extend(torch.load('../tensors/gl_.pt'))
print(all_txt_feats.shape)

id_to_txt_feats = {}


for idx, i in enumerate(all_img_tags):
    id_to_txt_feats[i] = all_txt_feats[idx]
    
    
id_to_img_feats = {}
id_to_gl = {}

for idx, i in enumerate(all_img_tags):
    id_to_img_feats[i] = all_img_feats[idx]
    
    
for idx, i in enumerate(all_img_tags):
    id_to_gl[i] = all_gl[idx]
    
    
id_to_exp = {}
exp = p1 | p2

for i in exp:
    id_to_exp['img/'+str(i).zfill(5)+'.png'] = exp[i].split('.')[0]
    
print(len(id_to_exp))

text = []
explanation = []
gl = []
tf = []
if_ = []
ip = []
c=0
for img_path in all_img_tags:
    
    #     img_path = 'img/'+i['img_path'].split('.')[0]+'.png'
    #print(img_path)
    
    
    if img_path in id_to_exp:
        
        #text.append(i['text'])
        #ip.append(img_path)
        #try:
        gl.append(id_to_gl[img_path])
        explanation.append(id_to_exp[img_path])

        tf.append(id_to_txt_feats[img_path])
        if_.append(id_to_img_feats[img_path])
        ip.append(img_path)
        #         except:
            #             continue
            
    elif id_to_gl[img_path]==0:
        gl.append(id_to_gl[img_path])
        explanation.append('not offensive')

        tf.append(id_to_txt_feats[img_path])
        if_.append(id_to_img_feats[img_path])
        ip.append(img_path)
        
        
    

tf = torch.stack(tf)
if_ = torch.stack(if_)

print(tf.shape)

print(if_.shape)
print(len(text), len(explanation), len(gl))
len(ip)

dataset = CustomDataset(tf,if_,gl,explanation,ip)
torch.manual_seed(42)
train_dataset_, test_dataset_ = torch.utils.data.random_split(dataset, [0.8, 0.2])


visual_feats_train = torch.load("../train_vb.pt")
visual_feats_test = torch.load("../test_vb.pt")

print(len(visual_feats_train), len(visual_feats_test))

def get_dict(path):
    d = {}
    with jsonlines.open(path) as reader:
        for obj in reader:
            d[obj['img']] = obj['text']
    return d
dd = {}
for i in os.listdir('../facebook-hateful-memes/hateful_memes'):
    if i.endswith('.jsonl'):
        k = '../facebook-hateful-memes/hateful_memes/' + i
        d = get_dict(k)
        #print(d)
        dd.update(d)
        
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch.optim as optim

# Define a simple custom dataset
class CustomDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.is_train = is_train

        if self.is_train:
            self.feat_vb = torch.load('../train_vb.pt')
        else:
            self.feat_vb = torch.load('../test_vb.pt')
        #         self.tokenizer = VisualBertTokenizer.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        #         inputs = self.tokenizer(self.questions[idx], return_tensors="pt")

        # self.txt_features = self.df[idx][0]
        # self.image_features = self.df[idx][1]
        
        # multimodal_embeds = torch.cat((self.txt_features, self.image_features), dim=-1)
        
        
        gl = self.df[idx][2]

        explanation = self.df[idx][3]
        ip = self.df[idx][-1]
        text = dd[ip]
            
        
        
        
         

        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors='pt')


        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)

        # visual_embeds,visual_token_type_ids,visual_attention_mask = get_visual_embedding('./data/'+ds_idx['img'])
        #print(feat_vb[os.path.join('/kaggle/input/esnli-ve/imgs/imgs',self.img_path[idx])])
        visual_embeds,visual_token_type_ids,visual_attention_mask = self.feat_vb[ip][0]

        #print(ds_idx['img'])

        inputs.update({
          "visual_embeds": torch.squeeze(visual_embeds),
          "visual_token_type_ids": torch.squeeze(visual_token_type_ids),
          "visual_attention_mask": torch.squeeze(visual_attention_mask)
        })
       
        #return multimodal_embeds[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]
        return inputs , gl, explanation, ip, text

train_dataset = CustomDataset(train_dataset_)
test_dataset = CustomDataset(test_dataset_, is_train=False)




train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 4, worker_init_fn=seed_worker)

from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from transformers import VisualBertModel


class fusion(nn.Module):
    def __init__(self):
        super(fusion, self).__init__()
        self.vb = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        self.dropout = nn.Dropout(0.1)
        self.lin = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(768, 256)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(256,128)),
            ('relu2', nn.ReLU()),
            ('lo',nn.Linear(128,2))
        ]))

    def forward(self, input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        visual_embeds = None,
        visual_attention_mask = None,
        visual_token_type_ids = None):

        p = self.vb(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
        
        p = self.dropout(p.pooler_output)
        z = self.lin(p)
        return z
    

model = fusion().to('cuda')
ce = nn.CrossEntropyLoss()

from sklearn.metrics import f1_score


# Load the model and optimizer

optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
model.train()

# for p,batch in tqdm(enumerate(train_dataloader)):
#     print(batch)
#     break
    
# print(0/0)
for epoch in range(2):  # Number of epochs
    l = 0
    cnt = 0

    for batch in tqdm(train_dataloader):
        inputs, labels, exp, ip, _ = batch
        input_ids = inputs['input_ids'].to('cuda')
        token_type_ids = inputs['token_type_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        visual_embeds = inputs['visual_embeds'].to('cuda')
        visual_token_type_ids = inputs['visual_token_type_ids'].to('cuda')
        visual_attention_mask = inputs['visual_attention_mask'].to('cuda')
        

    
   
        labels = torch.tensor(labels).unsqueeze(1)  # Adjust label shape for batch size
        
        # random labels
        
        #labels = torch.randint(0, 2, (inputs_txt.shape[0],))
        
        optimizer.zero_grad()
        outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
        
       
        loss = ce(outputs.squeeze(), labels.to('cuda').squeeze())
        #         print(loss)
        
        loss.backward()
        optimizer.step()
        l+=loss.item()
        cnt+=1
        
    print(f"Epoch {epoch+1}, Loss: {l/cnt}")
    
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers = 4)
    all_ops = []
    all_tgts = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            

            inputs, labels, exp, ip, _ = batch
            input_ids = inputs['input_ids'].to('cuda')
            token_type_ids = inputs['token_type_ids'].to('cuda')
            attention_mask = inputs['attention_mask'].to('cuda')
            visual_embeds = inputs['visual_embeds'].to('cuda')
            visual_token_type_ids = inputs['visual_token_type_ids'].to('cuda')
            visual_attention_mask = inputs['visual_attention_mask'].to('cuda')


            outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
        
            op = outputs.squeeze().argmax(dim=-1).cpu().numpy()
            labels = labels.numpy()
            all_ops.extend(op)
            all_tgts.extend(labels)
    print('F1 Macro', f1_score(all_tgts, all_ops, average='macro'))
    print('F1 Micro', f1_score(all_tgts, all_ops, average='micro'))






dd_ = {0: 'not offensive', 1: 'offensive',}
def get_test_data():
  
    TXT = []
    IP = []
    
    te_loader = DataLoader(
    test_dataset,
    batch_size=1,           # or any batch size
    shuffle=False,
    num_workers=4          # increase based on CPU core       # if using GPU
    )
    

    # with torch.no_grad():
    for idx, i in enumerate(te_loader):
        
        
        _,_,_, ip, txt = i

        
        ip = ip[0]
        txt = txt[0]
    
        IP.append(ip)
        TXT.append(txt)
        if idx==1000:
            break
  
   
    return IP, TXT

IP, TXT = get_test_data()

print(IP[:10], TXT[:10])











import numpy as np
from lime.lime_text import IndexedCharacters
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic
from skimage.color import gray2rgb
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T

# ---- Step 1: Your multimodal classifier ---- #
# Define this function: takes image and text, returns probabilities
# def multimodal_predict_fn(images, texts):
#     """
#     Your model should return a numpy array of shape (batch_size, num_classes)
#     """
#     # Replace with your own inference code
#     outputs = []
#     for img, txt in zip(images, texts):
#         # dummy probability for illustration
#         outputs.append([0.1, 0.9])  # e.g., [not-cat, cat]
#     return np.array(outputs)

# def multimodal_predict_fn(images, texts):
#     outputs = []
#     for img, txt in zip(images, texts):
#         visual_embeds, _ = get_visual_embedding(img_path)
#         visual_embeds = visual_embeds.unsqueeze(0)
#         visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
#         visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
#         inputs = tokenizer(txt, padding="max_length", truncation=True, max_length=64, return_tensors='pt')


#         # inputs['input_ids'] = inputs['input_ids'].squeeze(0)
#         # inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0)
#         # inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)

#         input_ids = inputs['input_ids'].to('cuda')
#         token_type_ids = inputs['token_type_ids'].to('cuda')
#         attention_mask = inputs['attention_mask'].to('cuda')
#         visual_embeds = inputs['visual_embeds'].to('cuda')
#         visual_token_type_ids = inputs['visual_token_type_ids'].to('cuda')
#         visual_attention_mask = inputs['visual_attention_mask'].to('cuda')


#         outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
    
#         print(outputs)
#         print(0/0)

#         score = 0.5
#         if 'muslim' in txt.lower():
#             score += 0.3
#         if np.mean(img) > 100:
#             score += 0.2
#         outputs.append([1 - score, score])
#     return np.array(outputs)



import torch
from transformers import BertTokenizer
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model (once)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Assume visual_bert_model is already loaded and moved to device
# Example: visual_bert_model = VisualBertForVisualReasoning.from_pretrained(...).to(device)
# And frcnn and frcnn_cfg already initialized for get_visual_embedding



import tempfile
from PIL import Image
import os

def save_numpy_images_to_tempfiles(images_np):
    temp_paths = []
    for i, img_np in enumerate(images_np):
        img = Image.fromarray(img_np.astype('uint8')).convert("RGB")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(temp_file.name)
        temp_paths.append(temp_file.name)
    return temp_paths

def multimodal_predict_fn(images_np, texts):
    # Step 1: Save perturbed images as temp files
    image_paths = save_numpy_images_to_tempfiles(images_np)

    # Step 2: Extract features
    visual_feats, boxes = get_visual_embedding(image_paths)  # (B, R, D), (B, R, 4)

    # Step 3: Run the VisualBERT model
    probs = run_visualbert(visual_feats, boxes, texts)  # (B, num_classes)

    # Step 4: Clean up temp files
    for path in image_paths:
        os.remove(path)

    return probs.detach().cpu().numpy()


def run_visualbert(visual_feats, boxes, texts):

    # print('vf', visual_feats.shape)
    # print(len(texts))
    # print(texts)
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors='pt')

    # Required input shape alignment
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    batch_size = input_ids.size(0)
    device = 'cuda'

    # Move everything to the same device
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    visual_feats = visual_feats.to(device)
    boxes = boxes.to(device)

    # Construct visual token type ids (1 for visual tokens)
    # visual_token_type_ids = torch.ones(visual_feats.size()[:-1], dtype=torch.long).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        visual_embeds=visual_feats,
        visual_attention_mask=torch.ones(visual_feats.shape[:-1], dtype=torch.long).to(device),
        visual_token_type_ids=torch.ones(visual_feats.shape[:-1], dtype=torch.long).to(device),
    )
    # print(outputs)
    # logits = outputs.logits  # shape: [batch_size, num_classes]
    probs = torch.softmax(outputs, dim=-1)
    return probs


# def multimodal_predict_fn(img_paths, texts):
#     """
#     Predicts class probabilities using VisualBERT.
    
#     Args:
#         img_paths: list of image paths or PIL images
#         texts: list of strings

#     Returns:
#         np.ndarray of shape (batch_size, num_classes)
#     """

#     print(img_paths)
#     visual_feats, boxes = get_visual_embedding(img_paths)  # (B, num_regions, feat_dim), (B, num_regions, 4)
#     batch_size = len(texts)

#     inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors='pt')

#     # inputs = tokenizer(
#     #     texts,
#     #     padding="max_length",
#     #     truncation=True,
#     #     max_length=128,
#     #     return_tensors="pt"
#     # )

#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)

#     visual_feats = visual_feats.to(device)
#     boxes = boxes.to(device)

#     with torch.no_grad():
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             visual_embeds=visual_feats,
#             visual_attention_mask=torch.ones(visual_feats.shape[:-1], dtype=torch.long).to(device),
#             visual_token_type_ids=torch.ones(visual_feats.shape[:-1], dtype=torch.long).to(device),
#             boxes=boxes,
#         )

#     logits = outputs.logits  # (batch_size, num_classes)
#     probs = softmax(logits, dim=-1)
#     return probs.cpu().numpy()


def get_batch_visual_embedding(img_paths):
    all_feats = []
    all_boxes = []
    for img_path in img_paths:
        feats, boxes = get_visual_embedding([img_path])  # <- expects 1 image
        all_feats.append(feats)  # shape: [1, num_regions, feat_dim]
        all_boxes.append(boxes)  # shape: [1, num_regions, 4]

    visual_feats = torch.cat(all_feats, dim=0)   # shape: [B, num_regions, feat_dim]
    box_coords = torch.cat(all_boxes, dim=0)     # shape: [B, num_regions, 4]
    return visual_feats, box_coords

def multimodal_predict_fn(images_np, texts):
    # Step 1: Save LIME-perturbed images to temp files
    image_paths = save_numpy_images_to_tempfiles(images_np)

    # Step 2: Extract visual features
    visual_feats, boxes = get_batch_visual_embedding(image_paths)  # shapes: [B, R, D], [B, R, 4]

    # Step 3: Forward pass through VisualBERT
    with torch.no_grad():
        probs = run_visualbert(visual_feats, boxes, texts)  # your classifier logic
    # print(probs)
    # Step 4: Cleanup
    for path in image_paths:
        os.remove(path)

    return probs.detach().cpu().numpy()


from skimage.segmentation import mark_boundaries
import json

from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic
import matplotlib.pyplot as plt
def explain_image(image, text, multimodal_fn, class_idx=None, num_samples=500, image_path = None):
    

    explainer = LimeImageExplainer()
    segmentation_fn = lambda x: slic(x, n_segments=100, compactness=5, sigma=1)

    def predict(images):
        preds = multimodal_fn(images, [text] * len(images))
        print(f"[DEBUG] Prediction variance: {np.var(preds, axis=0)}")
        return preds

    explanation = explainer.explain_instance(
        image=image,
        classifier_fn=predict,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn
    )
    print(explanation)
    if class_idx is None:
        class_idx = explanation.top_labels[0]
        print(f"[INFO] Automatically using top predicted class: {class_idx}")

    image, mask = explanation.get_image_and_mask(
        class_idx,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    plt.imshow(mark_boundaries(image, mask))

    # plt.imshow(image)
    plt.title(f"LIME Explanation for class {class_idx}")
    plt.axis('off')
    if image_path!=None:
        plt.savefig("./hm_vb_imgs/"+image_path)
    plt.close()
    print(f"[INFO] LIME image explanation saved to ./im1.png")

# ---- Step 3: Text LIME Explainer ---- #
def explain_text(text, image, multimodal_fn, class_idx=1, num_samples=500, image_path = None):
    explainer = LimeTextExplainer()

    def predict(texts):
        # Duplicate the original image for all perturbed texts
        return multimodal_fn([image] * len(texts), texts)

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict,
        num_features=10,
        num_samples=num_samples
    )

    # explanation.show_in_notebook(text=True)
    print("\n[INFO] Top important words:")
    df = {}
    for word, weight in explanation.as_list():
        print(f"{word}: {weight:.4f}")
        df[word] = weight

    if image_path!=None:
        image_path = str(image_path.split(".")[0])

    with open(f"./hm_vb_txts/{image_path}.json", "w") as f:
        json.dump(df, f, indent=4, ensure_ascii=False)





# Save to a JSON file



# ---- Step 4: Example Usage ---- #
import random
if __name__ == "__main__":



    

    df = torch.load("../generated_samples/phi2_ts_hm_vb_gen.pt")


    neg_indices = []
    pos_indices = []
    for j, i in enumerate(df):
        # print(df[i])
        if 'not offensive' in df[j].split("exp:")[-1].strip().lower():
            neg_indices.append(j)
        else:
            pos_indices.append(j)

    # Sample 50 elements from each
    sample_pos = random.sample(pos_indices, 100)
    sample_neg = random.sample(neg_indices, 100)


    indices = sample_pos+sample_neg

    # Load image and text

    print(indices)

    for ii in list(indices):

        image_path = "/home/anonymous/TACL/facebook-hateful-memes/hateful_memes/" + IP[ii]
        text_input = TXT[ii]

        # Preprocess image
        pil_img = Image.open(image_path).convert('RGB')
        preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.ToPILImage(),  # LIME expects PIL or numpy image
        ])
        image = preprocess(pil_img)

        image_np = np.array(image)

        # Run both explanations
        explain_image(image_np, text_input, multimodal_predict_fn, image_path = str(ii)+"_"+IP[ii].split("/")[-1])
        explain_text(text_input, image_np, multimodal_predict_fn, image_path = str(ii)+"_"+IP[ii].split("/")[-1])





# image shows always the explanation for the predicted class.
# text always shows the explanation for the 1st class, aka the offensive class.
