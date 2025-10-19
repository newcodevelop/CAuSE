import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import numpy as np
import random
import os
from tqdm import tqdm
import jsonlines
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
num_workers =  torch.multiprocessing.cpu_count()
# print(num_workers)
# print(0/0)


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
torch.use_deterministic_algorithms(True)

import torch
from torch.utils.data import DataLoader, Dataset
# from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
import torch.optim as optim



import json

with open('./vqax/train_x.json', 'r') as file:
    data = json.load(file)

data = data[:6478]

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

# print(num_to_element)
# print(0/0)



import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch.optim as optim
from transformers import FlavaModel, FlavaProcessor, FlavaConfig
from PIL import Image
# Define a simple custom dataset
from sklearn.model_selection import train_test_split
class CustomDataset(Dataset):
    def __init__(self, data, is_train=True):
        self.label, self.exp, self.text, self.img_id = [],[],[],[]
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        self.is_train = is_train
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        for d in data:
            try:
                max_key = max(d['label'], key=d['label'].get)
                self.label.append(element_to_num[max_key])
                self.exp.append(d['explanation'][0])
                self.text.append(d['sent'])
                self.img_id.append(d['img_id'])
            except:
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

        if self.is_train:
            self.feat_vb = torch.load('./train_vb_vqax.pt')
        else:
            self.feat_vb = torch.load('./test_vb_vqax.pt')
    
            
        
    def __len__(self):
        return len(self.exp)
    
    def __getitem__(self, idx):
        #         inputs = self.tokenizer(self.questions[idx], return_tensors="pt")
    
        



         
        gl = self.label[idx]

        explanation = self.exp[idx]

        ip = self.img_id[idx]
        text = self.text[idx]

        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors='pt')


        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
      
        # print(self.img_path[idx])
        img_path = os.path.join('./train2014/', ip+'.jpg')




        img  = Image.open(img_path).convert("RGB")
        # explanation = self.explanation[idx]

        visual_embeds,visual_token_type_ids,visual_attention_mask = self.feat_vb[img_path][0]
        
        inputs.update({
          "visual_embeds": torch.squeeze(visual_embeds),
          "visual_token_type_ids": torch.squeeze(visual_token_type_ids),
          "visual_attention_mask": torch.squeeze(visual_attention_mask)
        })
        

        # target = torch.zeros(3)
        # for l, s in zip(self.annotations[idx]["labels"], self.annotations[idx]["scores"]):
        #     target[l] = s
        # enc["labels"] = target
        # return enc

        return inputs, gl, explanation, img_path, text, torch.squeeze(visual_embeds)
       
        #return multimodal_embeds[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]
        # return inputs , gl, self.explanation[idx], self.img_path[idx], self.hypothesis[idx]



train_dataset = CustomDataset(data)
test_dataset = CustomDataset(data, is_train=False)

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
#   worker_init_fn=seed_worker)


print(len(train_dataset), len(test_dataset))

# print(0/0)


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 4, worker_init_fn=seed_worker)




from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from transformers import VisualBertModel


# class fusion(nn.Module):
#     def __init__(self):
#         super(fusion, self).__init__()
#         self.vb = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
#         self.dropout = nn.Dropout(0.1)
#         self.lin = nn.Sequential(OrderedDict([
#             ('l1', nn.Linear(768, 256)),
#             ('relu1', nn.ReLU()),
#             ('l2', nn.Linear(256,128)),
#             ('relu2', nn.ReLU()),
#             ('lo',nn.Linear(128,3))
#         ]))

#     def forward(self, input_ids = None,
#         attention_mask = None,
#         token_type_ids = None,
#         position_ids = None,
#         head_mask = None,
#         inputs_embeds = None,
#         visual_embeds = None,
#         visual_attention_mask = None,
#         visual_token_type_ids = None):

#         p = self.vb(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
        
#         p = self.dropout(p.pooler_output)
#         z = self.lin(p)
#         return z
    





# flava_cfg = FlavaConfig.from_pretrained("facebook/flava-full")
# hidden = flava_cfg.hidden_size 

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
            ('lo',nn.Linear(128,520))
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
# class FlavaForNLI(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flava = FlavaModel.from_pretrained("facebook/flava-full")
#         self.dropout = torch.nn.Dropout(0.1)
#         self.lin = nn.Sequential(OrderedDict([
#             ('l1', nn.Linear(768, 256)),
#             ('relu1', nn.ReLU()),
#             ('l2', nn.Linear(256,128)),
#             ('relu2', nn.ReLU()),
#             ('lo',nn.Linear(128,520))
#         ]))

#         # self.lin = nn.Sequential(OrderedDict([
#         #     ('l1', nn.Linear(768, 520))
#         # ]))
#         # self.cls = torch.nn.Linear(hidden, 3)
#         # self.loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
#         # self.log_sm   = torch.nn.LogSoftmax(dim=-1)

#     def forward(self, input_ids=None, pixel_values=None, attention_mask=None,
#                 token_type_ids=None, labels=None):
#         p = self.flava(
#             input_ids=input_ids,
#             pixel_values=pixel_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=True,
#         )
#         # pooled = out.multimodal_output.pooler_output  
#         # print(pooled.shape)
#         # logits = self.cls(self.dropout(pooled))

#         p = self.dropout(p.multimodal_output.pooler_output)
#         z = self.lin(p)
#         return z

# model = fusion().to('cuda')
model = fusion().to('cuda')
ce = nn.CrossEntropyLoss()
from sklearn.metrics import f1_score


# Load the model and optimizer

optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
epochs = 10
for epoch in range(epochs):  # Number of epochs
    l = 0
    cnt = 0

    for nos, batch in tqdm(enumerate(train_dataloader)):
        inputs, labels, a, b, c, _ = batch

        # print(inputs, labels, a, b, c)        
        # print(0/0)

       
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
   
        labels = labels.unsqueeze(1)  # Adjust label shape for batch size
        
        # random labels
        
        #labels = torch.randint(0, 2, (inputs_txt.shape[0],))
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        # print('outputs')
        # print(0/0)
       
        loss = ce(outputs.squeeze(), labels.to('cuda').squeeze())
        #         print(loss)
        
        loss.backward()
        optimizer.step()
        l+=loss.item()
        cnt+=1

        # if nos==50:
        #     break
        
    print(f"Epoch {epoch+1}, Loss: {l/cnt}")
    
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers = 4)
    all_ops = []
    all_tgts = []
    with torch.no_grad():
        for batch in test_dataloader:
            

            inputs, labels, _, _, _, _ = batch

            inputs = {k: v.to('cuda') for k, v in inputs.items()}
   
            labels = torch.tensor(labels).unsqueeze(1)  # Adjust label shape for batch size
           

            outputs = model(**inputs)
            op = outputs.squeeze().argmax(dim=-1).cpu().numpy()
            labels = labels.numpy()
            all_ops.extend(op)
            all_tgts.extend(labels)
    print('F1 Macro', f1_score(all_tgts, all_ops, average='macro'))
    print('F1 Micro', f1_score(all_tgts, all_ops, average='micro'))

# print(0/0)
model.eval()

from transformers import BertModel, BertTokenizer
import torch

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
    cls_representation = outputs.last_hidden_state[:, 0, :]

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
            ('lo',nn.Linear(128,520))
        ]))
        
        # self.clf2 = nn.Sequential(OrderedDict([
        #     ('l1', nn.Linear(768, 256)),
        #     ('relu1', nn.ReLU()),
        #     ('l2', nn.Linear(256,128)),
        #     ('relu2', nn.ReLU()),
        #     ('lo',nn.Linear(128,3))
        # ]))
        
       
        
       

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

        # input_ids = self.inputs[idx]['input_ids']
        # token_type_ids = self.inputs[idx]['token_type_ids']
        # attention_mask = self.inputs[idx]['attention_mask']
        # visual_embeds = self.inputs[idx]['visual_embeds']
        # visual_token_type_ids = self.inputs[idx]['visual_token_type_ids']
        # visual_attention_mask = self.inputs[idx]['visual_attention_mask']
        
        # inputs = {k: v.to('cuda')[idx,...].unsqueeze(0) for k, v in self.inputs.items()}

        
        
        inputs = self.inputs[idx]
       
        #return multimodal_embeds[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]
        #return 

        
        
        return C, input_ids1, input_ids2, self.L[idx], attn_mask1, attn_mask2, inputs, self.labels[idx], self.explanation[idx], self.img_path[idx]


def get_hidden(model, inputs, s = None):


    # if s==None:
    #     print(inputs)

    #     print(0/0)
    
    kk=False
    
    if len(inputs['input_ids'].shape)==2:
        kk = True

    
   
    # labels = torch.tensor(labels).unsqueeze(1)  # Adjust label shape for batch si
        
    
    
    if not kk:
        # inputs = {k: v.to('cuda').unsqueeze(0) for k, v in inputs.items()}
        input_ids = inputs['input_ids'].to('cuda').unsqueeze(0)
        token_type_ids = inputs['token_type_ids'].to('cuda').unsqueeze(0)
        attention_mask = inputs['attention_mask'].to('cuda').unsqueeze(0)
        visual_embeds = inputs['visual_embeds'].to('cuda').unsqueeze(0)
        visual_token_type_ids = inputs['visual_token_type_ids'].to('cuda').unsqueeze(0)
        visual_attention_mask = inputs['visual_attention_mask'].to('cuda').unsqueeze(0)
    else:
        # inputs = {k: v.to('cuda') for k, v in inputs.items()}
        input_ids = inputs['input_ids'].to('cuda')
        token_type_ids = inputs['token_type_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        visual_embeds = inputs['visual_embeds'].to('cuda')
        visual_token_type_ids = inputs['visual_token_type_ids'].to('cuda')
        visual_attention_mask = inputs['visual_attention_mask'].to('cuda')
        
    # print(input_ids.shape)
  
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

# def ii_h(projection_model, inputs_base, inputs_source, kk=None):
#     # we are only intervening on the first hidden layer and the last layer

#     # print('is_here', inputs_source)
#     # print('ib_here', inputs_base)
    
#     # print(0/0)
#     with torch.no_grad():
#         s,_ = get_hidden(model, inputs_source)
#         b,_ = get_hidden(model, inputs_base)
    
#     if s.shape[1]==1:
#         s = s.squeeze(dim=1)
        
#     if b.shape[1]==1:
#         b = b.squeeze(dim=1)
    
        
    
    
#     if kk==None:
#         #print('kk None randomly choosing')
#         kk = random.choice([0,1])
    
#     if kk==0:
#         s_h = projection_model.lin.l1(s)
#         b_h = projection_model.lin.l1(b)
#         #print('h1', b_h.shape, s_h.shape)
#          #= [1,7,4,67,89,128,167,190,210] # these are random indices
#         # rate of intervention = 0.2, 20% of the 256 neurons in this layer are intervened
#         _ii_rate  = 0.2
        
#         intervention_idx = random.choices(list(range(0,256)), k= int(256*_ii_rate))
#         try:
#             b_h[:,intervention_idx] = s_h[:,intervention_idx]
#         except:
#             print('h1', b_h.shape, s_h.shape)
#         b_h = projection_model.lin.relu1(b_h)
#         b_h = projection_model.lin.lo(projection_model.lin.relu2(projection_model.lin.l2(b_h)))
#     else:
    
#         b_h = projection_model.lin.lo(projection_model.lin.relu2(projection_model.lin.l2(projection_model.lin.relu1(projection_model.lin.l1(b)))))
#         s_h = projection_model.lin.lo(projection_model.lin.relu2(projection_model.lin.l2(projection_model.lin.relu1(projection_model.lin.l1(s)))))

#         intervention_idx = random.choice([0,1])
#         #print('h2', b_h.shape, s_h.shape)

#         try:
#             b_h[:,intervention_idx] = s_h[:,intervention_idx]
#         except:
#             print('h1', b_h.shape, s_h.shape)
            
#     return b_h, kk, intervention_idx, s, b
    
    

    
# dd = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

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
           
            
            inputs, labels, exp, ip, txt,_ = i

            inputs = {i: inputs[i].squeeze() for i in inputs}
            labels = labels[0].item()
            exp = exp[0]
            ip = ip[0]
            txt = txt[0]
            
            c, fin = get_hidden(model, inputs)

            #print(c.shape, fin.shape, fin.argmax(dim=-1))
            
            pred_lab = fin.argmax(dim=-1)[0].item()
            LAB.append(labels)
            
            if pred_lab==labels:
                L.append(pred_lab)
                explanation.append('This pair is a {} because {}'.format(num_to_element[pred_lab], exp.strip()+'.'))
            else:
                L.append(pred_lab)
                explanation.append('This pair is a {}'.format(num_to_element[pred_lab]))
                
            
           
                    
                
            
            
            text.append(txt)
            Cs.append(c)
           
            IP.append(ip)
            IPS.append(inputs)

            # if idx==8991:
            #     break


            if idx==31:
                break

            # if idx==160:
            #     break
            
    Cs = torch.stack(Cs)
    
    Cs = Cs.squeeze()
    return explanation, text, Cs, L, IP, LAB, IPS
            
explanation, text, Cs, L, IP,LAB,IPS = get_data()

print(IPS[0])

# print(0/0)

print(Cs.shape, len(L))

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



# def ii_l(projection_model, s, b, kk, intervention_idx):
#     # we are only intervening on the first hidden layer and the last layer
    
    
   
#     #print('with kk and intervention idx', kk, intervention_idx)
   
    
    
#     if kk==0:
#         s_h = projection_model.clf2.l1(s)
#         b_h = projection_model.clf2.l1(b)
      
#         b_h[:,intervention_idx] = s_h[:,intervention_idx]
#         b_h = projection_model.clf2.relu1(b_h)
#         b_h = projection_model.clf2.lo(projection_model.clf2.relu2(projection_model.clf2.l2(b_h)))
#     else:
    
#         b_h = projection_model.clf2.lo(projection_model.clf2.relu2(projection_model.clf2.l2(projection_model.clf2.relu1(projection_model.clf2.l1(b)))))
#         s_h = projection_model.clf2.lo(projection_model.clf2.relu2(projection_model.clf2.l2(projection_model.clf2.relu1(projection_model.clf2.l1(s)))))

#         intervention_idx = random.choice([0,1])

#         b_h[:,intervention_idx] = s_h[:,intervention_idx]
        
#     return b_h
    
    


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
  shuffle=True)

# Training loop
optimizer = torch.optim.Adam(list(projection_model.parameters()) + list(phi1.parameters()) + list(phi2.parameters()), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1):  # Number of epochs
    for idx, batch in enumerate(dataloader):
        # C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, input_ids,token_type_ids,attention_mask,visual_embeds,visual_token_type_ids,visual_attention_mask, labels, expl, img_path = batch

        C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, input_, labels, expl, img_path = batch        
       
        # Project C to get C1 and C2
        C1_batch, C2_batch, C_r1, C_r2 = projection_model(C_batch.to('cuda').float())

        # Replace the first token's embedding with C1 and C2
        inputs_embeds1 = phi1.transformer.wte(input_ids1.to('cuda'))
        inputs_embeds2 = phi2.transformer.wte(input_ids2.to('cuda'))
        
        
        
        inputs_embeds1[:, 0, :] = C1_batch

        
        inputs_embeds2[:, 0, :] = C2_batch
        
        #print(attn_mask2)
        

        
        outputs1 = phi1(inputs_embeds=inputs_embeds1, labels=input_ids1.to('cuda'))
        outputs2 = phi2(inputs_embeds=inputs_embeds2, labels=input_ids2.to('cuda'))
        
        
        
      
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
                
                C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, input_s, labels, expl, img_path = batch_inner
                
                # print(input_ids.shape, token_type_ids.shape, visual_embeds.shape)
                
                # inputs_base =  {k: v.to('cuda') for k, v in input_.items()}
                # inputs_src = {k: v.to('cuda') for k, v in input_s.items()}

                inputs_base = input_
                inputs_src = input_s

                # print('is_hersdsdse', inputs_src)
                # print('ib_hersdsdse', inputs_base)

                # inputs_src = {'input_ids': input_ids_s.to('cuda'),'token_type_ids':token_type_ids_s.to('cuda'),'attention_mask':attention_mask_s.to('cuda'),
                # 'visual_embeds':visual_embeds_s.to('cuda'),'visual_token_type_ids':visual_token_type_ids_s.to('cuda'),'visual_attention_mask':visual_attention_mask_s.to('cuda')}

                # print('HHHHHHHHHHHHHHHHHHHHH')
                with torch.no_grad():
                    o_h, kk, intervention_idx, s, b = ii_h(model, inputs_base, inputs_src, kk=None)
                
                inputs_embeds2_s = phi2.transformer.wte(input_ids2.to('cuda'))
                inputs_embeds2_b = phi2.transformer.wte(input_ids2.to('cuda'))
                s = projection_model.proj2(s)
                b = projection_model.proj2(b)
                inputs_embeds2_s[:, 0, :] = s
                inputs_embeds2_b[:, 0, :] = b
                
                outputs2_s = phi2(inputs_embeds=inputs_embeds2_s, labels=input_ids2.to('cuda'))
                outputs2_b = phi2(inputs_embeds=inputs_embeds2_b, labels=input_ids2.to('cuda'))
                
                s = torch.softmax(outputs2_s.logits, dim=-1).sum(dim=1)
                
                b = torch.softmax(outputs2_b.logits, dim=-1).sum(dim=1)
                
                
                
                s = projection_model.aggregator(s)
                b = projection_model.aggregator(b)
                
                o_l = ii_l(projection_model, s, b, kk, intervention_idx)
                
                #print('ol', o_l)
                
                
                # input should be a distribution in the log space
                
                # Sample a batch of distributions. Usually this would come from the dataset
                target = F.softmax(o_h, dim=1)
                kl_div = kl_loss(F.log_softmax(o_l, dim=1), target)

        
        
        
                
                
               
                
                
                #print('o_h', o_h)
                
                tot_kl += kl_div
            else:
                break
        
        
        
        fn_loss = get_frobenius(model, projection_model)
      
        
      
        if args.ablation.lower() == 'phi2':
            total_loss = loss2  
        elif args.ablation.lower() == 'phi2_ts':
            total_loss = loss2 + loss_clf 
        elif args.ablation.lower() == 'cause':
            total_loss = loss2 + loss_clf + (tot_kl/2) + fn_loss 
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
       
    
        if idx%10==0:
            print(f"Epoch {epoch+1}, Loss: {total_loss.item()}, KL Loss: {tot_kl/2}, Frobenius: {fn_loss}, mse: {mse_loss}")
        break


    
    
    
      
print('Loading models from Folder: ./vb_vqax_{}'.format(args.ablation.lower()))


    
    
phi1.load_state_dict(torch.load('./vb_vqax_{}/phi2.pt'.format(args.ablation.lower())))
phi2.load_state_dict(torch.load('./vb_vqax_{}/phi2.pt'.format(args.ablation.lower())))
projection_model.load_state_dict(torch.load('./vb_vqax_{}/projection_model.pt'.format(args.ablation.lower())))
    

dd = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
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
           
            
            inputs, labels, exp, ip, txt, visual_embeds = i

            # print('visual embeds shape', visual_embeds.shape)

            inputs = {i: inputs[i].squeeze() for i in inputs}
            labels = labels[0].item()
            exp = exp[0]
            ip = ip[0]
            txt = txt[0]

            # inputs, labels, exp, ip, txt = i

            # inputs = {i: inputs[i].squeeze() for i in inputs}
            # labels = labels[0].item()
            # exp = exp[0]
            # ip = ip[0]
            # txt = txt[0]

            
            
            c, fin = get_hidden(model, inputs)
            
            
            
            
            pred_lab = fin.argmax(dim=-1)[0].item()
            LAB.append(labels)
            
            if pred_lab==labels:
                L.append(pred_lab)
                explanation.append('This pair is a {} because {}'.format(num_to_element[pred_lab], exp.strip()+'.'))
            else:
                L.append(pred_lab)
                explanation.append('This pair is a {}'.format(num_to_element[pred_lab]))
            
            
            
            # inputs.update({'visual_embeds':visual_embeds.squeeze()})
            
            
            
            text.append(txt)
            Cs.append(c)
            IPS.append(inputs)
            IP.append(ip)
            # if idx==1000:
            #     break

            # if idx==1000:
            #     break
    Cs = torch.stack(Cs)
   
    return explanation, text, Cs, IPS, L, IP

explanation, text, Cs,IPS, L, IP = get_test_data(model)


print(explanation[:10])


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
autoencoder2 = Autoencoder().to('cuda')

T_v1, T_v0 = [], []
I_v1, I_v0 = [], []


ip_ = []
IPS_ = []
L_= []
for i in tqdm(range(len(L))):
    
    input_text = bert_tokenizer.decode(IPS[i]['input_ids'], skip_special_tokens=True)
    cls_repr = get_cls(input_text, bert_model, bert_tokenizer)
    # print(IPS[i]['visual_embeds'].shape)
    # print(0/0)
    cls_repr = torch.cat((IPS[i]['visual_embeds'].sum(0).unsqueeze(0), cls_repr.cpu()),dim=1).squeeze(dim=0)
    IPS_.append(cls_repr)
    
    # IPS0.append(IPS[i]['visual_embeds'].sum(0))
    ip_.append(IPS[i])
    L_.append(L[i])

IPS = torch.stack(IPS_)
ip = ip_




# I_v1 = torch.stack(I_v1)
# I_v0 = torch.stack(I_v0)

# print(IPS2)


x_cfs_0 = [] # for these the counterfactual label produced by the encoder is 1/2, F should also produce 1/2
L0 = []
C0 = []
X0 = []
X_org0 = []
for i in tqdm(range(IPS.shape[0])):

    #these are originally all 0 labels

    
    x = IPS[i]
    X_org0.append(x)
    with torch.no_grad():
        # del ip[i]['visual_embeds']
        c, fin = get_hidden(model, ip[i])
        C0.append(c.detach())
    #print(x.shape)


    ll_d = []
    x_cfs = []
    x_cf = []
    # IPS_ = torch.cat((IPS1,IPS2),dim=0)
    # ips_ = ip1+ip2
   
    # L_ = [1]*IPS1.shape[0] + [2]*IPS2.shape[0]
    # print(L[i], L)
    for j in range(len(L)):
        x_ = IPS[j]
        x_cfs.append(x_)
        x_cf.append(ip[j])
        # search for the index in concat(T_v1,i_v1) corresponding to lowest dist
        if L[j]!=L[i]:
            # print('heyyyyyyyyyyyy', L[i], L[j])
            
            dist = (x - x_).float().pow(2).sum().sqrt()
            ll_d.append(dist)
        else:
            ll_d.append(torch.tensor(1e10)) # append very high value of distance for label matches to ignore them. this stage is necessary to maintain proper indexing
    assert len(ll_d)==len(L)
    # print(ll_d)
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

# print(X0)

# print(0/0)
        
X0_ii = torch.stack([i['input_ids'] for i in X0])
X0_tti = torch.stack([i['token_type_ids'] for i in X0])
X0_am = torch.stack([i['attention_mask'] for i in X0])
# X0_pv = torch.stack([i['pixel_values'] for i in X0])

X0_ve = torch.stack([i['visual_embeds'] for i in X0])
X0_vtti = torch.stack([i['visual_token_type_ids'] for i in X0])
X0_vam = torch.stack([i['visual_attention_mask'] for i in X0])

x_cfs_0 = torch.stack(x_cfs_0)

# x_cfs_1 = torch.stack(x_cfs_1)

# x_cfs_2 = torch.stack(x_cfs_2)

from torch.utils.data import TensorDataset
# Create a dataset and data loader

# x_cfs2 = TensorDataset(x_cfs_2.cpu(),C2.cpu(),X2_ii.cpu(), X2_tti.cpu(), X2_am.cpu(), X2_pv.cpu())
# x_cfs2_dataloader = DataLoader(x_cfs2, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)


# x_cfs1 = TensorDataset(x_cfs_1.cpu(),C1.cpu(),X1_ii.cpu(), X1_tti.cpu(), X1_am.cpu(), X1_pv.cpu())
# x_cfs1_dataloader = DataLoader(x_cfs1, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)


x_cfs0 = TensorDataset(x_cfs_0.cpu(),C0.cpu(),X0_ii.cpu(), X0_tti.cpu(), X0_am.cpu(), X0_ve.cpu(), X0_vtti.cpu(), X0_vam.cpu())
x_cfs0_dataloader = DataLoader(x_cfs0, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)


# Loss function and optimizer
criterion = nn.MSELoss()
# optimizer2 = optim.Adam(autoencoder2.parameters(), lr=0.0001)
# optimizer1 = optim.Adam(autoencoder1.parameters(), lr=0.0001)
optimizer0 = optim.Adam(autoencoder0.parameters(), lr=0.0001)

kl_loss = nn.KLDivLoss(reduction="batchmean")

if int(args.counterfactual_test)==1:

    # Training loop
    num_epochs = 150  # You can adjust this
    for epoch in range(num_epochs):
        for batch in x_cfs0_dataloader:
            
            
            E_x = batch[1].squeeze()
            T_mu = autoencoder0.encoder(batch[0].to(torch.float32).to('cuda')) 
            
            x_ii, x_tti, x_am, x_ve, x_vtti, x_vam = batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

            inputs_base = {'input_ids': x_ii.to('cuda'),'token_type_ids':x_tti.to('cuda'),'attention_mask':x_am.to('cuda'),
                    'visual_embeds':x_ve.to('cuda'),'visual_token_type_ids':x_vtti.to('cuda'),'visual_attention_mask':x_vam.to('cuda')}
            

            # x_ii, x_tti, x_am, x_pv = batch[2], batch[3], batch[4], batch[5]

            # inputs_base = {'input_ids': x_ii.to('cuda'),'token_type_ids':x_tti.to('cuda'),'attention_mask':x_am.to('cuda'),
            #         'pixel_values':x_pv.to('cuda')}
            

            with torch.no_grad():
                _, fin1 = get_hidden(model, inputs_base)
                
            
            fin1 = fin1.squeeze()
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
        # if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


torch.save(autoencoder0.state_dict(), './vb_finetuned_vqax_autoencoder.pt')
with torch.no_grad():
  
    reduced_vectors0 = autoencoder0.encoder(x_cfs_0.to(torch.float32).to('cuda')) + C0.squeeze()






# if int(args.counterfactual_test)==1:
#     Cs_ = torch.cat((reduced_vectors0,reduced_vectors1,reduced_vectors2),dim=0)

#     L = L0+L1+L2
#     Cs_ = [Cs_[i].detach().cpu().numpy() for i in range(Cs_.shape[0])]

# else:
#     Cs_ = [Cs[i].detach().cpu().numpy() for i in range(Cs.shape[0])]


if int(args.counterfactual_test)==1:
    Cs_ = reduced_vectors0

    L = L0
    Cs_ = [Cs_[i].detach().cpu().numpy() for i in range(Cs_.shape[0])]

else:
    Cs_ = [Cs[i].detach().cpu().numpy() for i in range(Cs.shape[0])]

torch.manual_seed(0)
random.seed(0)

# def get_lab(gen):
#     for elem in list(element_to_num.keys()):
#             if (elem.lower() in gen) or (elem.lower()+'.' in gen):
#                 return element_to_num[elem]
    


def get_lab(gen):
    for elem in list(element_to_num.keys()):
            if (elem.lower() in gen) or (elem.lower()+'.' in gen):
                return element_to_num[elem]
    

majority_label = []
actual_label = []
projected_label = []
projected_label_ = []
FL = []
PL = []
FL_ = []
gen_vals = []
for j in tqdm(range(len(L))):

    C1_batch, C2_batch, _, _ = projection_model(torch.tensor(Cs_[j]).unsqueeze(dim=0).to('cuda'))

    # C1_batch, C2_batch, _, _ = projection_model(torch.tensor(Cs_[j]).to('cuda'))
    C1_batch = C1_batch.squeeze(dim=1)
    C2_batch = C2_batch.squeeze(dim=1)
    # print(C1_batch.shape, C2_batch.shape)
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




    for idx,i in enumerate([0.2,0.4,0.6,0.8,1.0]):

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

        # print(gen)


        flag = 0
        for elem in list(element_to_num.keys()):
            if (elem.lower() in gen.lower()) or (elem.lower()+'.' in gen.lower()):
                SS.append(0)
                flag = 1
                # print('found')
                # break
        if not flag:
            SS.append(1)
            continue
                
            



        # if ('entailment' in gen) or ('entailment.' in gen) or ('neutral' in gen) or ('neutral.' in gen) or ('contradiction' in gen) or ('contradiction.' in gen):

        #     SS.append(0)

        # else:
        #     SS.append(1)
        #     continue
        
        generated_list = gen.split(" ")

        lab = get_lab(generated_list)

        PL.append(lab)


        outer_label.append(lab)
        outer_generation.append(gen)

        kk = explanation[j].split(" ")
        #print(kk)
        lab = get_lab(kk)

        outer_acc.append(lab)



        outer_gen.append(gen)



   





    if sum(SS)==len(SS):
        #print('nothing generated')
        continue



    FL.append(L[j])
   



    num_labels = {i:0 for i in num_to_element}
    for i in outer_label:
        try:
            num_labels[i]+=1
        except:
            continue
    highest_key, highest_value = max(num_labels.items(), key=lambda x: x[1])
        
    # print(num_labels)
    majority_label.append(highest_key)

    # print(majority_label)
    # print(FL)
    # print(0/0)




    nn = ""
    for i in range(len(outer_label)):

        # if outer_label[i]==idx:
        nn += outer_generation[i]+'----'
    gen_vals.append(nn+'exp: '+explanation[j])

    # print(gen_vals)
    # print(0/0)

print('predicted class', majority_label)
print('actual class', FL)




from sklearn.metrics import accuracy_score, f1_score




# print(np.asarray(FL), np.asarray(majority_label))

#counterfactual f1 considering LLM output
if int(args.counterfactual_test)==1:

    print('counterfactual F1 score')

    print(f1_score(FL, majority_label, average='macro'))
    print(f1_score(FL, majority_label, average='micro'))
    print(accuracy_score(FL, majority_label))

    print(len(FL), len(majority_label))

else:
    print('F1 score')

    print(f1_score(FL, majority_label, average='macro'))
    print(f1_score(FL, majority_label, average='micro'))
    print(accuracy_score(FL, majority_label))

    print(len(FL), len(majority_label))

    torch.save(gen_vals, f'./generated_samples/{args.ablation}_esnli_vb_gen.pt')

   
    
    




    
    
