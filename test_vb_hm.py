import torch

import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import os
import jsonlines


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ablation', type=str, required=True)
parser.add_argument('--counterfactual_test', type=int, required=True)

args = parser.parse_args()

print(f"Ablation: {args.ablation.lower()}")

#Run checks
if args.ablation.lower() not in ['phi2', 'phi2_ts', 'cause']:
    raise Exception("Dataset should be phi2, phi2_ts, cause")


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

df = torch.load('./kb-fb-ds/kb_fb.pt')

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

all_img_tags = list(torch.load('./tensors/img_id.pt'))
all_img_tags.extend(list(torch.load('./tensors/img_id_.pt')))

print(len(all_img_tags))

import pickle


with open('./explanation/explanation-fb-0_2749.pickle', 'rb') as h:
    p1 = pickle.load(h)

with open('./explanation/explanation-fb_remaining.pickle', 'rb') as h:
    p2 = pickle.load(h)


# all_txt_feats = []
# all_img_feats  = []

# txt_path = '/kaggle/input/tensors/tx_tensor.pt'
# img_path = 


all_txt_feats = torch.load('./tensors/tx_tensor.pt')
all_txt_feats = torch.cat((all_txt_feats, torch.load('./tensors/tx_tensor_.pt')), dim=0)

all_img_feats = torch.load('./tensors/im_tensor.pt')
all_img_feats = torch.cat((all_img_feats, torch.load('./tensors/im_tensor_.pt')), dim=0)

all_gl = torch.load('./tensors/gl.pt')
print(len(all_gl))
all_gl.extend(torch.load('./tensors/gl_.pt'))
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


visual_feats_train = torch.load("./train_vb.pt")
visual_feats_test = torch.load("./test_vb.pt")

print(len(visual_feats_train), len(visual_feats_test))

def get_dict(path):
    d = {}
    with jsonlines.open(path) as reader:
        for obj in reader:
            d[obj['img']] = obj['text']
    return d
dd = {}
for i in os.listdir('./facebook-hateful-memes/hateful_memes'):
    if i.endswith('.jsonl'):
        k = './facebook-hateful-memes/hateful_memes/' + i
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
            self.feat_vb = torch.load('./train_vb.pt')
        else:
            self.feat_vb = torch.load('./test_vb.pt')
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

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
#   worker_init_fn=seed_worker)

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


from transformers import BertModel, BertTokenizer

# Load the BERT model and tokenizer
model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name).to('cuda')


# Example input text

def get_cls(input_text, bert_model, bert_tokenizer):
    
    
    # Tokenize the input text
    inputs = bert_tokenizer(input_text, return_tensors="pt").to('cuda')
    
    # Perform forward pass to get the hidden states
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Extract the hidden state of the [CLS] token
    # The [CLS] token corresponds to the first token in the sequence (index 0)
    cls_representation = outputs.last_hidden_state[:, 0, :].cpu()

    return cls_representation
    
    # print("Shape of [CLS] token representation:", cls_representation.shape)
    # print("Representation:", cls_representation)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the projection model to split vector C into C1 and C2
class ProjectionModel(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super(ProjectionModel, self).__init__()
        self.proj1 = nn.Linear(input_dim, proj_dim)
        self.proj2 = nn.Linear(input_dim, proj_dim)
        
        self.reconstr1 = nn.Linear(proj_dim, input_dim)
        self.reconstr2 = nn.Linear(proj_dim, input_dim)
        
        self.aggregator = nn.Linear(50258, 768)
        
        self.clf2 = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(768, 256)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(256,128)),
            ('relu2', nn.ReLU()),
            ('lo',nn.Linear(128,2))
        ]))
        
       
        
       

    def forward(self, C):
        C1 = self.proj1(C)
        C2 = self.proj2(C)
        
        C_1 = self.reconstr1(C1)
        C_2 = self.reconstr2(C2)
        return C1, C2, C_1, C_2

# Custom Dataset

class CustomDataset(Dataset):
    def __init__(self, Cs, texts1, texts2, L, tokenizer,  inputs, labels, explanation, img_path):
        #Cs, texts1, texts2, L, tokenizer, T, I, L,explanation,IP
        self.Cs = Cs
        self.texts1 = texts1
        self.texts2 = texts2
        self.tokenizer = tokenizer
        self.L = L
        self.inputs = inputs
        self.labels = labels
        self.explanation = explanation
        self.img_path = img_path
        
    def __len__(self):
        return len(self.Cs)

    def __getitem__(self, idx):
        #print(idx)
        C = self.Cs[idx]
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        i1 = self.tokenizer(self.texts1[idx], return_tensors="pt", padding='max_length', max_length=42, truncation=True)
        i2 = self.tokenizer(self.texts2[idx], return_tensors="pt", padding='max_length', max_length=42, truncation=True)
        input_ids1 = i1.input_ids.squeeze(0)
        input_ids2 = i2.input_ids.squeeze(0)
        
        attn_mask1 = i1.attention_mask.squeeze(0)
        attn_mask2 = i2.attention_mask.squeeze(0)

        # input_ids = self.inputs[idx]['input_ids'].to('cuda')
        # token_type_ids = self.inputs[idx]['token_type_ids'].to('cuda')
        # attention_mask = self.inputs[idx]['attention_mask'].to('cuda')
        # visual_embeds = self.inputs[idx]['visual_embeds'].to('cuda')
        # visual_token_type_ids = self.inputs[idx]['visual_token_type_ids'].to('cuda')
        # visual_attention_mask = self.inputs[idx]['visual_attention_mask'].to('cuda')

        input_ids = self.inputs[idx]['input_ids']
        token_type_ids = self.inputs[idx]['token_type_ids']
        attention_mask = self.inputs[idx]['attention_mask']
        visual_embeds = self.inputs[idx]['visual_embeds']
        visual_token_type_ids = self.inputs[idx]['visual_token_type_ids']
        visual_attention_mask = self.inputs[idx]['visual_attention_mask']
        
        
        
       
        #return multimodal_embeds[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]
        #return 

        
        
        return C, input_ids1, input_ids2, self.L[idx], attn_mask1, attn_mask2,input_ids,token_type_ids,attention_mask,visual_embeds,visual_token_type_ids,visual_attention_mask, self.labels[idx], self.explanation[idx], self.img_path[idx]

"""

class CustomDataset(Dataset):
    def __init__(self, Cs, texts1, texts2, L, tokenizer, inputs, labels, explanation, img_path):
        self.Cs = Cs
        self.L = L
        self.inputs = inputs
        self.labels = labels
        self.explanation = explanation
        self.img_path = img_path

        # Pre-tokenize texts1 and texts2 here
        self.tokenized_texts1 = tokenizer(
            texts1,
            padding='max_length',
            max_length=42,
            truncation=True,
            return_tensors="pt"
        )

        self.tokenized_texts2 = tokenizer(
            texts2,
            padding='max_length',
            max_length=42,
            truncation=True,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.Cs)

    def __getitem__(self, idx):
        return (
            self.Cs[idx],
            self.tokenized_texts1['input_ids'][idx],
            self.tokenized_texts2['input_ids'][idx],
            self.L[idx],
            self.tokenized_texts1['attention_mask'][idx],
            self.tokenized_texts2['attention_mask'][idx],
            self.inputs[idx]['input_ids'],
            self.inputs[idx]['token_type_ids'],
            self.inputs[idx]['attention_mask'],
            self.inputs[idx]['visual_embeds'],
            self.inputs[idx]['visual_token_type_ids'],
            self.inputs[idx]['visual_attention_mask'],
            self.labels[idx],
            self.explanation[idx],
            self.img_path[idx]
        )

"""


def get_hidden(model, inputs):
    
    kk=False
    
    if len(inputs['input_ids'].shape)==2:
        kk = True
        
    
    if not kk:
        input_ids = inputs['input_ids'].to('cuda').unsqueeze(0)
        token_type_ids = inputs['token_type_ids'].to('cuda').unsqueeze(0)
        attention_mask = inputs['attention_mask'].to('cuda').unsqueeze(0)
        visual_embeds = inputs['visual_embeds'].to('cuda').unsqueeze(0)
        visual_token_type_ids = inputs['visual_token_type_ids'].to('cuda').unsqueeze(0)
        visual_attention_mask = inputs['visual_attention_mask'].to('cuda').unsqueeze(0)
    else:
        input_ids = inputs['input_ids'].to('cuda')
        token_type_ids = inputs['token_type_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        visual_embeds = inputs['visual_embeds'].to('cuda')
        visual_token_type_ids = inputs['visual_token_type_ids'].to('cuda')
        visual_attention_mask = inputs['visual_attention_mask'].to('cuda')
        
    # print(input_ids.shape)
    # print(token_type_ids.shape)
    # print(attention_mask.shape)
    # print(visual_embeds.shape)
    # print(visual_token_type_ids.shape)
    # print(visual_attention_mask.shape)
    p = model.vb(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
        
    z = model.dropout(p.pooler_output)
    fin = model.lin(z)
    return z, fin
    
# projection_model.clf2



def ii_h(projection_model, inputs_base, inputs_source, kk=None):
    # we are only intervening on the first hidden layer and the last layer
    
    with torch.no_grad():
        s,_ = get_hidden(model, inputs_source)
        b,_ = get_hidden(model, inputs_base)
    
    if s.shape[1]==1:
        s = s.squeeze(dim=1)
        
    if b.shape[1]==1:
        b = b.squeeze(dim=1)
    
        
    
    
    if kk==None:
        #print('kk None randomly choosing')
        kk = random.choice([0,1])
    
    if kk==0:
        s_h = projection_model.lin.l1(s)
        b_h = projection_model.lin.l1(b)
        #print('h1', b_h.shape, s_h.shape)
         #= [1,7,4,67,89,128,167,190,210] # these are random indices
        # rate of intervention = 0.2, 20% of the 256 neurons in this layer are intervened
        _ii_rate  = 0.2
        
        intervention_idx = random.choices(list(range(0,256)), k= int(256*_ii_rate))
        try:
            b_h[:,intervention_idx] = s_h[:,intervention_idx]
        except:
            print('h1', b_h.shape, s_h.shape)
        b_h = projection_model.lin.relu1(b_h)
        b_h = projection_model.lin.lo(projection_model.lin.relu2(projection_model.lin.l2(b_h)))
    else:
    
        b_h = projection_model.lin.lo(projection_model.lin.relu2(projection_model.lin.l2(projection_model.lin.relu1(projection_model.lin.l1(b)))))
        s_h = projection_model.lin.lo(projection_model.lin.relu2(projection_model.lin.l2(projection_model.lin.relu1(projection_model.lin.l1(s)))))

        intervention_idx = random.choice([0,1])
        #print('h2', b_h.shape, s_h.shape)

        try:
            b_h[:,intervention_idx] = s_h[:,intervention_idx]
        except:
            print('h1', b_h.shape, s_h.shape)
            
    return b_h, kk, intervention_idx, s, b
    
dd_= {0: 'not offensive', 1: 'offensive'}
import gc
def get_data():
    model.eval()
    explanation = []
    text = []
    Cs = []
    T, I = [],[]
    L = []
    IP = []
    LAB = []
    IPS = []

    t_loader = DataLoader(
    train_dataset,
    batch_size=1,           # or any batch size
    shuffle=False,
    num_workers=4          # increase based on CPU core       # if using GPU
    )

    
    with torch.no_grad():
        for idx, i in tqdm(enumerate(t_loader)):
           
            
            inputs, labels, exp, ip, txt = i

            inputs = {i: inputs[i].squeeze() for i in inputs}
            labels = labels[0].item()
            exp = exp[0]
            ip = ip[0]
            txt = txt[0]

        
            
            # with torch.no_grad():
            c, fin = get_hidden(model, inputs)
    
            #print(c.shape, fin.shape, fin.argmax(dim=-1))
            
            pred_lab = fin.argmax(dim=-1)[0].item()
            LAB.append(labels)
            
            if pred_lab==labels:
                L.append(pred_lab)
                explanation.append('This pair is a {} because {}'.format(dd_[pred_lab], exp.strip()+'.'))
            else:
                L.append(pred_lab)
                explanation.append('This pair is a {}'.format(dd_[pred_lab]))
                
            
           
                    
                
            
            
            text.append(txt)
            Cs.append(c)
            #T.append(inputs_txt)
            #I.append(inputs_img)
            IP.append(ip)
            IPS.append(inputs)
            # del c
            # del fin
            if idx==31:
                break
    Cs = torch.stack(Cs)
    
    Cs = Cs.squeeze()
    return explanation, text, Cs, L, IP, LAB, IPS

dd = {}
for i in os.listdir('./facebook-hateful-memes/hateful_memes'):
    if i.endswith('.jsonl'):
        k = './facebook-hateful-memes/hateful_memes/' + i
        d = get_dict(k)
        #print(d)
        dd.update(d)
        
explanation, text, Cs, L, IP,LAB,IPS = get_data() 

Cs = [Cs[i].cpu().detach().numpy() for i in range(Cs.shape[0])]

from transformers import AutoModelForCausalLM, AutoTokenizer
model_lm = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype="auto", trust_remote_code=True).to('cuda')
tokenizer = AutoTokenizer.from_pretrained("gpt2",use_fast=False)

tokenizer.add_special_tokens({'bos_token' : '<s>'})
tokenizer.add_bos_token = True
tokenizer.pad_token = tokenizer.eos_token

phi1 = model_lm
phi2 = model_lm

import torch.nn as nn

# projection_model.clf2



def ii_l(projection_model, s, b, kk, intervention_idx):
    # we are only intervening on the first hidden layer and the last layer
    
    
   
    #print('with kk and intervention idx', kk, intervention_idx)
   
    
    
    if kk==0:
        s_h = projection_model.clf2.l1(s)
        b_h = projection_model.clf2.l1(b)
      
        b_h[:,intervention_idx] = s_h[:,intervention_idx]
        b_h = projection_model.clf2.relu1(b_h)
        b_h = projection_model.clf2.lo(projection_model.clf2.relu2(projection_model.clf2.l2(b_h)))
    else:
    
        b_h = projection_model.clf2.lo(projection_model.clf2.relu2(projection_model.clf2.l2(projection_model.clf2.relu1(projection_model.clf2.l1(b)))))
        s_h = projection_model.clf2.lo(projection_model.clf2.relu2(projection_model.clf2.l2(projection_model.clf2.relu1(projection_model.clf2.l1(s)))))

        intervention_idx = random.choice([0,1])

        b_h[:,intervention_idx] = s_h[:,intervention_idx]
        
    return b_h
    

def get_frobenius(model, pm):
    
    w_l1 = model.lin.l1.weight.detach()
    b_l1 = model.lin.l1.bias.detach()
    
    w_l2 = model.lin.l2.weight.detach()
    b_l2 = model.lin.l2.bias.detach()
    
    w_lo = model.lin.lo.weight.detach()
    b_lo = model.lin.lo.bias.detach()
    
    
    w_l1_ = pm.clf2.l1.weight
    b_l1_ = pm.clf2.l1.bias
    
    w_l2_ = pm.clf2.l2.weight
    b_l2_ = pm.clf2.l2.bias
    
    w_lo_ = pm.clf2.lo.weight
    b_lo_ = pm.clf2.lo.bias
    
    
    fn_weights = torch.norm(w_l1_ - w_l1, p='fro')**2 + torch.norm(w_l2_ - w_l2, p='fro')**2 + torch.norm(w_lo_ - w_lo, p='fro')**2
    
    fn_bias = torch.norm(b_l1_ - b_l1, p='fro')**2 + torch.norm(b_l2_ - b_l2, p='fro')**2 + torch.norm(b_lo_ - b_lo, p='fro')**2
    
    fn = (fn_weights/3) + (fn_bias/3)
    
    return fn
    
    

# Initialize components
# input_dim = 512
input_dim = 768
proj_dim = 2560  # Assuming GPT-2 uses 768-dimensional embeddings
proj_dim=768
projection_model = ProjectionModel(input_dim, proj_dim).to('cuda')


import random
from tqdm import tqdm

# phi1 = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
# phi2 = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
phi1.resize_token_embeddings(len(tokenizer))
phi2.resize_token_embeddings(len(tokenizer))



# Example data (replace with actual data)
num_examples = 10  # Arbitrary number of examples
# Cs = torch.rand(num_examples, input_dim)
texts1 = text
texts2 = explanation
kl_loss = nn.KLDivLoss(reduction="batchmean")
# Dataset and DataLoader
dataset = CustomDataset(Cs, texts1, texts2, L, tokenizer, IPS, L,explanation,IP)
dataloader = DataLoader(dataset, batch_size=32,
  worker_init_fn=seed_worker,
  shuffle=False, num_workers = 10)

# Training loop
optimizer = torch.optim.Adam(list(projection_model.parameters()) + list(phi1.parameters()) + list(phi2.parameters()), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()


# for idx, batch in tqdm(enumerate(dataloader)):
#     print(idx, batch)


for idx, batch in tqdm(enumerate(dataloader)):
    C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, input_ids,token_type_ids,attention_mask,visual_embeds,visual_token_type_ids,visual_attention_mask, labels, expl, img_path = batch
    
    # print(input_ids2)

    # Project C to get C1 and C2
    C1_batch, C2_batch, C_r1, C_r2 = projection_model(C_batch.to('cuda').float())

    # Replace the first token's embedding with C1 and C2
    inputs_embeds1 = phi1.transformer.wte(input_ids1.to('cuda'))
    inputs_embeds2 = phi2.transformer.wte(input_ids2.to('cuda'))
    
    
    
    inputs_embeds1[:, 0, :] = C1_batch

    
    inputs_embeds2[:, 0, :] = C2_batch
    
    #print(attn_mask2)
    

    
    outputs1 = phi1(inputs_embeds=inputs_embeds1, attention_mask = attn_mask1.to('cuda'), labels=input_ids1.to('cuda'))
    outputs2 = phi2(inputs_embeds=inputs_embeds2, attention_mask = attn_mask2.to('cuda'),  labels=input_ids2.to('cuda'))
    
    
    
    
    #s1 = torch.softmax(outputs1.logits, dim=-1)* attn_mask1.to('cuda').unsqueeze(dim=2)
    
    s2 = torch.softmax(outputs2.logits, dim=-1)
    
    aggregated = s2.sum(dim=1)
    
    aggregated = projection_model.aggregator(aggregated)
    
    #print(aggregated)
    
    #print(C_batch.to('cuda').shape, aggregated.shape)
    
    mse_loss = F.mse_loss(aggregated, C_batch.to('cuda'))
    #print(mse_loss)
    #print(0/0)
    
    
    
    class_ = projection_model.clf2(aggregated)
    
    
    
    
    
    

    loss1 = outputs1.loss
    loss2 = outputs2.loss
    
    loss_clf = F.cross_entropy(class_, lab.to('cuda'))
    
    
    
    tot_kl = 0
    for idx_inner, batch_inner in enumerate(dataloader):
    
        #print('idx inner', idx_inner)
    
        if idx_inner < 2:
            
            C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, input_ids_s,token_type_ids_s,attention_mask_s,visual_embeds_s,visual_token_type_ids_s,visual_attention_mask_s, labels, expl, img_path = batch_inner
            
            # print(input_ids.shape, token_type_ids.shape, visual_embeds.shape)
            
            inputs_base = {'input_ids': input_ids.to('cuda'),'token_type_ids':token_type_ids.to('cuda'),'attention_mask':attention_mask.to('cuda'),
            'visual_embeds':visual_embeds.to('cuda'),'visual_token_type_ids':visual_token_type_ids.to('cuda'),'visual_attention_mask':visual_attention_mask.to('cuda')}

            inputs_src = {'input_ids': input_ids_s.to('cuda'),'token_type_ids':token_type_ids_s.to('cuda'),'attention_mask':attention_mask_s.to('cuda'),
            'visual_embeds':visual_embeds_s.to('cuda'),'visual_token_type_ids':visual_token_type_ids_s.to('cuda'),'visual_attention_mask':visual_attention_mask_s.to('cuda')}

            # print('HHHHHHHHHHHHHHHHHHHHH')
            with torch.no_grad():
                o_h, kk, intervention_idx, s, b = ii_h(model, inputs_base, inputs_src, kk=None)
            
            inputs_embeds2_s = phi2.transformer.wte(input_ids2.to('cuda'))
            inputs_embeds2_b = phi2.transformer.wte(input_ids2.to('cuda'))
            s = projection_model.proj2(s)
            b = projection_model.proj2(b)
            inputs_embeds2_s[:, 0, :] = s
            inputs_embeds2_b[:, 0, :] = b
            
            outputs2_s = phi2(inputs_embeds=inputs_embeds2_s, attention_mask = attn_mask2.to('cuda'), labels=input_ids2.to('cuda'))
            outputs2_b = phi2(inputs_embeds=inputs_embeds2_b, attention_mask = attn_mask2.to('cuda'), labels=input_ids2.to('cuda'))
            
            s = torch.softmax(outputs2_s.logits, dim=-1).sum(dim=1)
            
            b = torch.softmax(outputs2_b.logits, dim=-1).sum(dim=1)
            
            
            
            s = projection_model.aggregator(s)
            b = projection_model.aggregator(b)
            
            o_l = ii_l(projection_model, s, b, kk, intervention_idx)
            
            #print('ol', o_l)
            
            
            # input should be a distribution in the log space
            input_ = F.log_softmax(o_l, dim=1)
            # Sample a batch of distributions. Usually this would come from the dataset
            target = F.softmax(o_h, dim=1)
            kl_div = kl_loss(input_, target)

    
    
    
            
            
            
            
            
            #print('o_h', o_h)
            
            tot_kl += kl_div
        else:
            break
    
    
    
    fn_loss = get_frobenius(model, projection_model)
    
    
    print(loss2, loss_clf, (tot_kl/2), fn_loss)
    # total_loss = loss1 + loss2 + loss_clf + (tot_kl/2) + fn_loss
    
    # total_loss = loss1 + loss2 + loss_clf 
    # remove loss1 from every  loss

    if args.ablation.lower() == 'phi2':
        total_loss = loss2  
    elif args.ablation.lower() == 'phi2_ts':
        total_loss = loss2 + loss_clf 
    elif args.ablation.lower() == 'cause':
        total_loss = loss2 + loss_clf + (tot_kl/2) + fn_loss 
    
    
    # combination of losses
    # total_loss = loss2
    # total_loss = loss2 + loss_clf       # loss_clf = TS loss
    # total_loss = loss2 + loss_clf + (tot_kl/2) + fn_loss   # (tot_kl/2) = IIT loss # total_loss = loss_CAuSE


    
    

    # Combine losses and backpropagate
    #total_loss = loss1 + loss2 + loss_clf + torch.norm(C_batch.to('cuda') - C_r1.to('cuda'), p='fro')**2 + torch.norm(C_batch.to('cuda') - C_r2.to('cuda'), p='fro')**2
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

    
    print(f"Epoch {epoch+1}, Loss: {total_loss.item()}, KL Loss: {tot_kl/2}, Frobenius: {fn_loss}, mse: {mse_loss}")
    break  

       
  
print('Loading models from Folder: ./vb_hm_{}'.format(args.ablation.lower()))


    
    
phi1.load_state_dict(torch.load('./vb_hm_{}/phi1.pt'.format(args.ablation.lower())))
phi2.load_state_dict(torch.load('./vb_hm_{}/phi2.pt'.format(args.ablation.lower())))
projection_model.load_state_dict(torch.load('./vb_hm_{}/projection_model.pt'.format(args.ablation.lower())))

dd_ = {0: 'not offensive', 1: 'offensive',}
def get_test_data(model):
    model.eval()
    explanation = []
    text = []
    Cs = []
    IPS = []
    L = []
    IP = []
    
    te_loader = DataLoader(
    test_dataset,
    batch_size=1,           # or any batch size
    shuffle=False,
    num_workers=4          # increase based on CPU core       # if using GPU
    )
    

    with torch.no_grad():
        for idx, i in enumerate(te_loader):
           
            
            inputs, labels, exp, ip, txt = i

            inputs = {i: inputs[i].squeeze() for i in inputs}
            labels = labels[0].item()
            exp = exp[0]
            ip = ip[0]
            txt = txt[0]

            
            
            c, fin = get_hidden(model, inputs)
            
            
            
            
            pred_lab = fin.argmax(dim=-1)[0].item()
            LAB.append(labels)
            
            if pred_lab==labels:
                L.append(pred_lab)
                explanation.append('This pair is a {} because {}'.format(dd_[pred_lab], exp.strip()+'.'))
            else:
                L.append(pred_lab)
                explanation.append('This pair is a {}'.format(dd_[pred_lab]))
            
            
            
            
            
            
            
            text.append(txt)
            Cs.append(c)
            IPS.append(inputs)
            IP.append(ip)
            if idx==1000:
                break
    Cs = torch.stack(Cs)
   
    return explanation, text, Cs, IPS, L, IP

explanation, text, Cs,IPS, L, IP = get_test_data(model)


import torch
import torch.nn as nn
import torch.optim as optim

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: 1024 -> 512
        
            
       
        # Decoder: 512 -> 1024
        # self.decoder = nn.Linear(768, 1024)
        # self.decoder1 = nn.Linear(1024, 2048)

        self.encoder = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(2816, 1024)),
           
            ('l2', nn.Linear(1024,768)),
            
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(768, 1024)),
           
            ('l2', nn.Linear(1024,2816)),
            
        ]))

    
        
    
    def forward(self, x):
        # Encoding step
        encoded = self.encoder(x)
        # Decoding step
        decoded = self.decoder(encoded)
        return decoded


autoencoder0 = Autoencoder().to('cuda')
autoencoder1 = Autoencoder().to('cuda')

T_v1, T_v0 = [], []
I_v1, I_v0 = [], []

IPS2, IPS1, IPS0 = [], [], []
ip2, ip1, ip0 = [],[],[]
for i in tqdm(range(len(L))):
   
    if L[i]==1:
        input_text = bert_tokenizer.decode(IPS[i]['input_ids'], skip_special_tokens=True)
        cls_repr = get_cls(input_text, bert_model, bert_tokenizer)
        cls_repr = torch.cat((IPS[i]['visual_embeds'].sum(0).unsqueeze(0), cls_repr),dim=1).squeeze(dim=0)
        
        IPS1.append(cls_repr)
        ip1.append(IPS[i])
        # T_v0.append(T[i,:])
        # I_v0.append(I[i,:])
    else:
        input_text = bert_tokenizer.decode(IPS[i]['input_ids'], skip_special_tokens=True)
        cls_repr = get_cls(input_text, bert_model, bert_tokenizer)
        cls_repr = torch.cat((IPS[i]['visual_embeds'].sum(0).unsqueeze(0), cls_repr),dim=1).squeeze(dim=0)
        IPS0.append(cls_repr)
        
        # IPS0.append(IPS[i]['visual_embeds'].sum(0))
        ip0.append(IPS[i])


IPS1 = torch.stack(IPS1)
IPS0 = torch.stack(IPS0)
# I_v1 = torch.stack(I_v1)
# I_v0 = torch.stack(I_v0)


x_cfs_0 = [] # for these the counterfactual label produced by the encoder is 1/2, F should also produce 1/2
L0 = []
C0 = []
X0 = []
X_org0 = []
for i in tqdm(range(IPS0.shape[0])):

    #these are originally all 0 labels

    
    x = IPS0[i]
    X_org0.append(x)
    with torch.no_grad():
        c, fin = get_hidden(model, ip0[i])
        C0.append(c.detach())
    #print(x.shape)


    ll_d = []
    x_cfs = []
    x_cf = []
    IPS_ = IPS1
    ips_ = ip1
   
    L_ = [1]*IPS1.shape[0]
    for j in range(IPS_.shape[0]):
        x_ = IPS_[j]
        x_cfs.append(x_)
        x_cf.append(ips_[j])
        # search for the index in concat(T_v1,i_v1) corresponding to lowest dist
        dist = (x - x_).pow(2).sum().sqrt()
        ll_d.append(dist)
    x_min_idx = torch.stack(ll_d).argmin().item()

    x_ = x_cfs[x_min_idx]
    x__ = x_cf[x_min_idx]
    L0.append(L_[x_min_idx])

    #print(x_.shape)

    mu = x_ - x

    x_cfs_0.append(mu)
    X0.append(x__)

C0 = torch.stack(C0)
# X0 = torch.stack(X0)
X_org0 = torch.stack(X_org0)

X0_ii = torch.stack([i['input_ids'] for i in X0])
X0_tti = torch.stack([i['token_type_ids'] for i in X0])
X0_am = torch.stack([i['attention_mask'] for i in X0])
X0_ve = torch.stack([i['visual_embeds'] for i in X0])
X0_vtti = torch.stack([i['visual_token_type_ids'] for i in X0])
X0_vam = torch.stack([i['visual_attention_mask'] for i in X0])


x_cfs_1 = [] # for these the counterfactual label produced by the encoder is 0/2, F should also produce 0/2
L1 = []
C1 = []
X1 = []
X_org1 = []
for i in tqdm(range(IPS1.shape[0])):

    #these are originally all 0 labels

   
    x = IPS1[i]
    X_org1.append(x)
    with torch.no_grad():
        c, fin = get_hidden(model, ip1[i])
        C1.append(c.detach())
    #print(x.shape)


    ll_d = []
    x_cfs = []
    x_cf = []
    
    IPS_ = IPS0
    ips_ = ip0
    L_ = [0]*IPS0.shape[0] 
    for j in range(IPS_.shape[0]):
        x_ = IPS_[j]
        x_cfs.append(x_)
        x_cf.append(ips_[j])
        # search for the index in concat(T_v1,i_v1) corresponding to lowest dist
        dist = (x - x_).pow(2).sum().sqrt()
        ll_d.append(dist)
    x_min_idx = torch.stack(ll_d).argmin().item()

    x_ = x_cfs[x_min_idx]
    x__ = x_cf[x_min_idx]
    L1.append(L_[x_min_idx]) # L1 contains the counterfactual labels

    #print(x_.shape)

    mu = x_ - x

    x_cfs_1.append(mu)
    X1.append(x__)

# L0

C1 = torch.stack(C1)
# X1= torch.stack(X1)
X_org1 = torch.stack(X_org1)

X1_ii = torch.stack([i['input_ids'] for i in X1])
X1_tti = torch.stack([i['token_type_ids'] for i in X1])
X1_am = torch.stack([i['attention_mask'] for i in X1])
X1_ve = torch.stack([i['visual_embeds'] for i in X1])
X1_vtti = torch.stack([i['visual_token_type_ids'] for i in X1])
X1_vam = torch.stack([i['visual_attention_mask'] for i in X1])

x_cfs_0 = torch.stack(x_cfs_0)

x_cfs_1 = torch.stack(x_cfs_1)



from torch.utils.data import TensorDataset
# Create a dataset and data loader

x_cfs1 = TensorDataset(x_cfs_1.cpu(),C1.cpu(),X1_ii.cpu(), X1_tti.cpu(), X1_am.cpu(), X1_ve.cpu(), X1_vtti.cpu(), X1_vam.cpu())
x_cfs1_dataloader = DataLoader(x_cfs1, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)


x_cfs0 = TensorDataset(x_cfs_0.cpu(),C0.cpu(),X0_ii.cpu(), X0_tti.cpu(), X0_am.cpu(), X0_ve.cpu(), X0_vtti.cpu(), X0_vam.cpu())
x_cfs0_dataloader = DataLoader(x_cfs0, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)

# Loss function and optimizer
criterion = nn.MSELoss()

optimizer1 = optim.Adam(autoencoder1.parameters(), lr=0.0001)
optimizer0 = optim.Adam(autoencoder0.parameters(), lr=0.0001)

kl_loss = nn.KLDivLoss(reduction="batchmean")



if int(args.counterfactual_test)==1:

    num_epochs = 500  # You can adjust this
    for epoch in range(num_epochs):
        for batch in x_cfs1_dataloader:

            # batch[0] = mu = x'-x (x's class is 1 as the base (normal class) class of x' is 0/2)
            # batch[1] = E(x)
            # batch[2] = x' = x+mu
            E_x = batch[1].squeeze()
            T_mu = autoencoder1.encoder(batch[0].to(torch.float32).to('cuda'))

            x_ii, x_tti, x_am, x_ve, x_vtti, x_vam = batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

            inputs_base = {'input_ids': x_ii.to('cuda'),'token_type_ids':x_tti.to('cuda'),'attention_mask':x_am.to('cuda'),
                    'visual_embeds':x_ve.to('cuda'),'visual_token_type_ids':x_vtti.to('cuda'),'visual_attention_mask':x_vam.to('cuda')}
            with torch.no_grad():
                _, fin1 = get_hidden(model, inputs_base)

            # clf1(E_x+T_mu) = clf1(E(x+mu)) = clf1(E(x'))
            fin1 = fin1.squeeze(dim=1)
            # fin2 = model.lin(E_x.to('cuda')+0.4*T_mu.to('cuda'))

            fin2 = model.lin(E_x.to('cuda')+T_mu.to('cuda'))

            inp = F.log_softmax(fin2, dim=1)
            tgt = F.softmax(fin1, dim=1)

            c_loss = kl_loss(inp,tgt)






            # Get the input data (batch)
            inputs = batch[0]

            # Zero the gradients
            optimizer1.zero_grad()

            # Forward pass: get the reconstruction
            outputs = autoencoder1(inputs.to(torch.float32).to('cuda'))

            # Compute the loss

            loss = criterion(outputs.to('cuda'), inputs.to(torch.float32).to('cuda')) + c_loss

            # Backward pass: compute gradients and update weights
            loss.backward()
            optimizer1.step()

        # Print the loss every few epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Training loop
    num_epochs = 500  # You can adjust this
    for epoch in range(num_epochs):
        for batch in x_cfs0_dataloader:
            
            
            E_x = batch[1].squeeze()
            T_mu = autoencoder0.encoder(batch[0].to(torch.float32).to('cuda')) 
            
            x_ii, x_tti, x_am, x_ve, x_vtti, x_vam = batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

            inputs_base = {'input_ids': x_ii.to('cuda'),'token_type_ids':x_tti.to('cuda'),'attention_mask':x_am.to('cuda'),
                    'visual_embeds':x_ve.to('cuda'),'visual_token_type_ids':x_vtti.to('cuda'),'visual_attention_mask':x_vam.to('cuda')}
            with torch.no_grad():
                _, fin1 = get_hidden(model, inputs_base)
                
            
            fin1 = fin1.squeeze(dim=1)
            # fin2 = model.lin(E_x.to('cuda')+0.4*T_mu.to('cuda'))

            fin2 = model.lin(E_x.to('cuda')+T_mu.to('cuda'))
            
            inp = F.log_softmax(fin2, dim=1)
            tgt = F.softmax(fin1, dim=1)
            
            c_loss = kl_loss(inp,tgt)
            
            
            
            
            
            
            
            # Get the input data (batch)
            inputs = batch[0]
            
            # Zero the gradients
            optimizer0.zero_grad()
            
            # Forward pass: get the reconstruction
            outputs = autoencoder0(inputs.to(torch.float32).to('cuda'))
            
            # Compute the loss
            
            loss = criterion(outputs.to('cuda'), inputs.to(torch.float32).to('cuda')) + c_loss
            
            # Backward pass: compute gradients and update weights
            loss.backward()
            optimizer0.step()
        
        # Print the loss every few epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')




    


with torch.no_grad():
    #ae_n is for original label n
    
    reduced_vectors1 = autoencoder1.encoder(x_cfs_1.to(torch.float32).to('cuda')) + C1.squeeze() # CF1 = T(mu) + E(x) = ae_1(mu) + E(x) :::: for all x, clf1(x)=1 and clf1(x+mu) = 0/2 and clf1(CLF1) = 0/2 :::: L1 = 0/2
    reduced_vectors0 = autoencoder0.encoder(x_cfs_0.to(torch.float32).to('cuda')) + C0.squeeze()

 




if int(args.counterfactual_test)==1:
    Cs_ = torch.cat((reduced_vectors0,reduced_vectors1),dim=0)

    L = L0+L1 # comment this line if not counterfactual 
    Cs_ = [Cs_[i].detach().cpu().numpy() for i in range(Cs_.shape[0])]

else:
    Cs_ = [Cs[i].detach().cpu().numpy() for i in range(Cs.shape[0])]




from tqdm import tqdm
from transformers.utils import logging
logging.set_verbosity_error()

# Cs_ = [Cs_[i].detach().cpu().numpy() for i in range(Cs_.shape[0])] # comment this line if not counterfactual

# # Cs_ = [Cs[i].detach().cpu().numpy() for i in range(Cs.shape[0])] # un-comment this line if not counterfactual

torch.manual_seed(0)
random.seed(0)

def get_lab(generated_list):
    if 'not offensive' in generated_list or 'not offensive.' in generated_list:
        return 0
    elif 'offensive' in generated_list or 'offensive.' in generated_list:
        return 1
    
#import shutup; shutup.please()
majority_label = []
actual_label = []
projected_label = []
projected_label_ = []
FL = []
PL = []
FL_ = []
gen_vals = []
for j in tqdm(range(len(L))):
    
    
    if int(args.counterfactual_test)==1:
        C1_batch, C2_batch, _, _ = projection_model(torch.tensor(Cs_[j]).unsqueeze(dim=0).to('cuda'))
    else:
        C1_batch, C2_batch, _, _ = projection_model(torch.tensor(Cs_[j]).to('cuda'))
    # C1_batch, C2_batch, _, _ = projection_model(torch.tensor(Cs_[j]).to('cuda'))
    # print(C2_batch.shape)
    #C2_batch = torch.randn(1,768)
    outer_label = []
    outer_acc = []
    outer_ppl = []
    outer_gen = []
    outer_generation = []
    
    proj_lab = []
    
    SS = []
    
    
    class_f = model.lin(torch.tensor(Cs_[j]).to('cuda')).argmax().item()
    if class_f!=L[j]:
        
        continue
    
    
    
    #for idx,i in enumerate([0.7,0.7,0.8,0.8,0.9,0.9,1.0]):
    for idx,i in enumerate([0.2,0.4,0.6,0.8,1.0]):
    #for idx,i in enumerate([0.7]):
    #for idx,i in enumerate([0.7,0.8,0.9,1.0]):
        gen_tokens = phi2.generate(
            inputs_embeds = C2_batch.unsqueeze(dim=1).to('cuda').half(),
            do_sample=True,
            temperature=i,
            max_length=100,
        )
        
        
        
        
        inputs_embeds2 = torch.cat( ( C2_batch.unsqueeze(dim=1).to('cuda').half(), phi2.transformer.wte(gen_tokens.to('cuda'))), dim=1)
        
        
        outputs2 = phi2(inputs_embeds=inputs_embeds2)
         
        
        
     
        
        aggregated = torch.softmax(outputs2.logits, dim=-1).sum(dim=1)
        
        aggregated = projection_model.aggregator(aggregated)
        
        
        
        class_ = projection_model.clf2(aggregated)
       
        proj_lab.append(class_.argmax(dim=-1)[0].item())
        
        
        
        
        
    
        #print('*'*10)
        gen = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        gen = gen[0].strip()
        
        
        
        #print(gen)
        #print(0/0)
       
        
        if ('offensive' in gen) or ('offensive.' in gen) or ('not offensive' in gen) or ('not offensive.' in gen):
            #print('hjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj')
            SS.append(0)

        else:
            SS.append(1)
            continue
        #generated_list = gen.split(" ")
        #print(generated_list)
        generated_list = gen
        lab = get_lab(generated_list)
        
        PL.append(lab)
        
        
        outer_label.append(lab)
        outer_generation.append(gen)
        
        kk = explanation[j].split(" ")
        #print(kk)
        lab = get_lab(kk)
        
        outer_acc.append(lab)
        
                    
        #print('gen {}'.format(idx), gen)
        #print(IP[j])
        #print('exp', explanation[j])
        #print('txt', text[j])
        outer_gen.append(gen)
    
    #print('act', explanation[j])
    #print('outer', outer_label)
    #print('act', outer_acc)
    
    
    ##############################
    
    num_zeros, num_ones = 0,0
    for i in proj_lab:
        if i==0:
            num_zeros+=1
        elif i==1:
            num_ones+=1
        
    
    idx = np.array([num_zeros, num_ones]).argmax()
    projected_label.append(idx)
    FL_.append(L[j])
    
    
    
    
    
    if sum(SS)==len(SS):
        #print('nothing generated')
        continue

    #############################
    
    
    
    FL.append(L[j])
    projected_label_.append(class_f)
    
    
    num_zeros, num_ones = 0,0
    for i in outer_acc:
        if i==0:
            num_zeros+=1
        elif i==1:
            num_ones+=1
        
    idx = np.array([num_zeros, num_ones]).argmax()
    actual_label.append(idx)


    ##############################
    
    
    num_zeros, num_ones = 0,0
    for i in outer_label:
        if i==0:
            num_zeros+=1
        elif i==1:
            num_ones+=1
       
    
    idx = np.array([num_zeros, num_ones]).argmax()
    majority_label.append(idx)
    
    
    #print(majority_label,idx)
    #print(outer_label)
    
    nn = ""
    for i in range(len(outer_label)):
        
        if outer_label[i]==idx:
            nn += outer_generation[i]+'----'
    gen_vals.append(nn+'exp: '+explanation[j])
    
    print('Num generated {}'.format(len(majority_label)))
        
print('predicted class', majority_label)
print('actual class', actual_label)
print('projected class', projected_label)

from sklearn.metrics import accuracy_score, f1_score




print(np.asarray(FL), np.asarray(majority_label))

#counterfactual f1 considering LLM output
if int(args.counterfactual_test)==1:

    print('counterfactual F1 score')

    print(f1_score(FL, majority_label, average='macro'))
    print(f1_score(FL, majority_label, average='micro'))
    print(accuracy_score(FL, majority_label))

    print(len(FL), len(projected_label_), len(majority_label), len(FL_), len(projected_label))

else:
    print('counterfactual F1 score')

    print(f1_score(FL, majority_label, average='macro'))
    print(f1_score(FL, majority_label, average='micro'))
    print(accuracy_score(FL, majority_label))

    print(len(FL), len(projected_label_), len(majority_label), len(FL_), len(projected_label))

    torch.save(gen_vals, f'./generated_samples/{args.ablation}_hm_vb_gen.pt')

   
    
    
