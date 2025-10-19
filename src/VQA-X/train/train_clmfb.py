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

        self.text_feats = torch.load("text_embeddings_clip_vqax.pt")
        self.image_feats = torch.load("image_embeddings_clip_vqax.pt")
    
            
        
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

        text_feats = self.text_feats[img_path]
        image_feats = self.image_feats[img_path]



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

        return inputs, gl, explanation, img_path, text, text_feats, image_feats
       
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


import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F



class fusion(nn.Module):
    def __init__(self,img_feat_size, txt_feat_size, is_first, K, O, DROPOUT_R):
        super(fusion, self).__init__()
        #self.__C = __C
        self.K = K
        self.O = O
        self.DROPOUT_R = DROPOUT_R

        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, K * O)
        self.proj_t = nn.Linear(txt_feat_size, K * O)

        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(K, stride = K)



        self.lin = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(512, 256)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(256,128)),
            ('relu2', nn.ReLU()),
            ('lo',nn.Linear(128,520))
        ]))

    def forward(self, img_feat, txt_feat, exp_in=1):

        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)
        txt_feat = self.proj_t(txt_feat)

        exp_out = img_feat * txt_feat
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)
        z = self.pool(exp_out) * self.K
        z = F.normalize(z.view(batch_size, -1))
        z = z.view(batch_size, -1, self.O)
        z = self.lin(z)
        return z

model = fusion(512,512,True,256,512,0.1).to('cuda')




# model = fusion().to('cuda')
ce = nn.CrossEntropyLoss()
from sklearn.metrics import f1_score


# Load the model and optimizer

optimizer = optim.AdamW(model.parameters(), lr=5e-5)






# Training loop
model.train()
for epoch in range(15):  # Number of epochs
    l = 0
    cnt = 0
    for batch in tqdm(train_dataloader):
        _, labels, _,_,_, inputs_txt, inputs_img = batch
        # print(inputs_txt.shape, inputs_img.shape)
        # print(0/0)

        labels = torch.tensor(labels).unsqueeze(1)  # Adjust label shape for batch size

       

        optimizer.zero_grad()

        outputs = model(inputs_img.to(torch.float32).to('cuda'), inputs_txt.to(torch.float32).to('cuda'))

        loss = ce(outputs.squeeze(), labels.to('cuda').squeeze())
        #         print(loss)

        loss.backward()
        optimizer.step()
        l+=loss.item()
        cnt+=1

    print(f"Epoch {epoch+1}, Loss: {l/cnt}")

    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    all_ops = []
    all_tgts = []
    with torch.no_grad():
        for batch in test_dataloader:
            _, labels, _,_,_, inputs_txt, inputs_img = batch




            outputs = model(inputs_img.to(torch.float32).to('cuda'), inputs_txt.to(torch.float32).to('cuda'))
            op = outputs.squeeze().argmax(dim=-1).cpu().numpy()
            labels = labels.numpy()
            all_ops.extend(op)
            all_tgts.extend(labels)
    print('F1 Macro', f1_score(all_tgts, all_ops, average='macro'))
    print('F1 Micro', f1_score(all_tgts, all_ops, average='micro'))

model.eval()

# print(0/0)









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

        self.aggregator = nn.Linear(50258, 512)

        self.clf2 = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(512, 256)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(256,128)),
            ('relu2', nn.ReLU()),
            ('lo',nn.Linear(128,520))
        ]))





    def forward(self, C):
        C1 = self.proj1(C)
        C2 = self.proj2(C)

        C_1 = self.reconstr1(C1)
        C_2 = self.reconstr2(C2)
        return C1, C2, C_1, C_2





# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, Cs, texts1, texts2, L, tokenizer,  txt_feature, image_features, labels, explanation, img_path):
        #Cs, texts1, texts2, L, tokenizer, T, I, L,explanation,IP
        self.Cs = Cs
        self.texts1 = texts1
        self.texts2 = texts2
        self.tokenizer = tokenizer
        self.L = L
        self.txt_features = txt_feature
        self.image_features = image_features
        self.labels = labels
        self.explanation = explanation
        self.img_path = img_path
        #self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #         self.tokenizer.add_special_tokens({'bos_token' : '<s>'})
        #         self.tokenizer.add_bos_token = True

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

       
        #return multimodal_embeds[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]
        #return



        return C, input_ids1, input_ids2, self.L[idx], attn_mask1, attn_mask2, self.txt_features[idx,:], self.image_features[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]

# tokenizer.eos_token





# len(dd)

def get_hidden(model, img_feat, txt_feat):
    batch_size = img_feat.shape[0]
    img_feat = model.proj_i(img_feat)
    txt_feat = model.proj_t(txt_feat)

    exp_out = img_feat * txt_feat
    exp_out = model.dropout(exp_out) if model.is_first else model.dropout(exp_out * exp_in)
    z = model.pool(exp_out) * model.K
    z = F.normalize(z.view(batch_size, -1))
    z = z.view(batch_size, -1, model.O)
    fin = model.lin(z)
    return z, fin

# projection_model.clf2



def ii_h(projection_model, img_feat_base, txt_feat_base, img_feat_source, txt_feat_source, kk=None):
    # we are only intervening on the first hidden layer and the last layer

    with torch.no_grad():
        s,_ = get_hidden(model, img_feat_source, txt_feat_source)
        b,_ = get_hidden(model, img_feat_base, txt_feat_base)

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

def get_data():
    model.eval()
    explanation = []
    text = []
    Cs = []
    T, I = [],[]
    L = []
    IP = []
    LAB = []
    t_loader = DataLoader(
    train_dataset,
    batch_size=1,           # or any batch size
    shuffle=False,
    num_workers=4          # increase based on CPU core       # if using GPU
    )
    with torch.no_grad():
        for idx, i in tqdm(enumerate(t_loader)):

            _, labels, exp, ip, txt, inputs_txt, inputs_img = i 
            # inputs_txt, inputs_img, labels, exp, ip, txt = i
            inputs_txt = inputs_txt.squeeze()
            inputs_img = inputs_img.squeeze()
            labels = labels[0].item()
            exp = exp[0]
            ip = ip[0]
            txt = txt[0]

            # print(inputs_txt, inputs_img, labels, exp, ip, txt)
            # print(0/0)

            c, fin = get_hidden(model, inputs_img.unsqueeze(dim=0).to(torch.float32).to('cuda'), inputs_txt.unsqueeze(dim=0).to(torch.float32).to('cuda'))


            pred_lab = fin.argmax(dim=-1)[0][0].item()
            LAB.append(labels)


            if pred_lab==labels:
                L.append(pred_lab)
                explanation.append('The answer is {} because {}'.format(num_to_element[pred_lab], exp.strip()+'.'))
            else:
                L.append(pred_lab)
                explanation.append('The answer is {}'.format(num_to_element[pred_lab]))





            text.append(txt)
            Cs.append(c)
            T.append(inputs_txt)
            I.append(inputs_img)
            IP.append(ip)

            if idx==5503:
                break
            # if idx==31:
            #     break

    Cs = torch.stack(Cs)
    T = torch.stack(T)
    I = torch.stack(I)
    return explanation, text, Cs, T, I, L, IP, LAB

explanation, text, Cs, T, I, L, IP, LAB = get_data()




Cs = Cs.squeeze()

print(Cs.shape)

Cs = [Cs[i].cpu().detach().numpy() for i in range(Cs.shape[0])]

from transformers import AutoModelForCausalLM, AutoTokenizer
model_lm = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype="auto", trust_remote_code=True).to('cuda')
tokenizer = AutoTokenizer.from_pretrained("gpt2",use_fast=False)

tokenizer.add_special_tokens({'bos_token' : '<s>'})
tokenizer.add_bos_token = True
tokenizer.pad_token = tokenizer.eos_token

phi1 = model_lm
phi2 = model_lm

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
input_dim = 512
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
dataset = CustomDataset(Cs, texts1, texts2, L, tokenizer, T, I, L,explanation,IP)
dataloader = DataLoader(dataset, batch_size=32,
  worker_init_fn=seed_worker,
  shuffle=True, num_workers = 10)

# Training loop
optimizer = torch.optim.Adam(list(projection_model.parameters()) + list(phi1.parameters()) + list(phi2.parameters()), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):  # Number of epochs
    for idx, batch in tqdm(enumerate(dataloader)):
        C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, txt_features, image_features, labels, expl, img_path = batch



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



        s2 = torch.softmax(outputs2.logits, dim=-1)

        aggregated = s2.sum(dim=1)

        aggregated = projection_model.aggregator(aggregated)



        mse_loss = F.mse_loss(aggregated, C_batch.to('cuda'))



        class_ = projection_model.clf2(aggregated)







        loss1 = outputs1.loss
        loss2 = outputs2.loss

        loss_clf = F.cross_entropy(class_, lab.to('cuda'))



        tot_kl = 0
        for idx_inner, batch_inner in enumerate(dataloader):

            #print('idx inner', idx_inner)

            if idx_inner < 2:

                C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, txt_features_s, image_features_s, labels, expl, img_path = batch_inner
                with torch.no_grad():
                    o_h, kk, intervention_idx, s, b = ii_h(model, image_features.to('cuda').float(), txt_features.to('cuda').float(), image_features_s.to('cuda').float(), txt_features_s.to('cuda').float(), kk=None)

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
                input_ = F.log_softmax(o_l, dim=1)
                # Sample a batch of distributions. Usually this would come from the dataset
                target = F.softmax(o_h, dim=1)
                kl_div = kl_loss(input_, target)



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


print('Saving models to Folder: ./clmfb_vqax_F_{}'.format(args.ablation.lower()))

torch.save(phi1.state_dict(), './clmfb_vqax_{}_F/phi1.pt'.format(args.ablation.lower()))
torch.save(phi2.state_dict(), './clmfb_vqax_{}_F/phi2.pt'.format(args.ablation.lower()))
torch.save(projection_model.state_dict(), './clmfb_vqax_{}_F/projection_model.pt'.format(args.ablation.lower()))





# print(IPS[0])

# # print(0/0)

# print(Cs.shape, len(L))

# Cs = [Cs[i].cpu().detach().numpy() for i in range(Cs.shape[0])]







# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_lm = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype="auto", trust_remote_code=True).to('cuda')
# tokenizer = AutoTokenizer.from_pretrained("gpt2",use_fast=False)

# tokenizer.add_special_tokens({'bos_token' : '<s>'})
# tokenizer.add_bos_token = True
# tokenizer.pad_token = tokenizer.eos_token

# phi1 = model_lm
# phi2 = model_lm

# import torch.nn as nn

# # projection_model.clf2



# # def ii_l(projection_model, s, b, kk, intervention_idx):
# #     # we are only intervening on the first hidden layer and the last layer
    
    
   
# #     #print('with kk and intervention idx', kk, intervention_idx)
   
    
    
# #     if kk==0:
# #         s_h = projection_model.clf2.l1(s)
# #         b_h = projection_model.clf2.l1(b)
      
# #         b_h[:,intervention_idx] = s_h[:,intervention_idx]
# #         b_h = projection_model.clf2.relu1(b_h)
# #         b_h = projection_model.clf2.lo(projection_model.clf2.relu2(projection_model.clf2.l2(b_h)))
# #     else:
    
# #         b_h = projection_model.clf2.lo(projection_model.clf2.relu2(projection_model.clf2.l2(projection_model.clf2.relu1(projection_model.clf2.l1(b)))))
# #         s_h = projection_model.clf2.lo(projection_model.clf2.relu2(projection_model.clf2.l2(projection_model.clf2.relu1(projection_model.clf2.l1(s)))))

# #         intervention_idx = random.choice([0,1])

# #         b_h[:,intervention_idx] = s_h[:,intervention_idx]
        
# #     return b_h
    
    


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
    




# def get_frobenius(model, pm):
    
#     w_l1 = model.lin.l1.weight.detach()
#     b_l1 = model.lin.l1.bias.detach()
    
#     w_l2 = model.lin.l2.weight.detach()
#     b_l2 = model.lin.l2.bias.detach()
    
#     w_lo = model.lin.lo.weight.detach()
#     b_lo = model.lin.lo.bias.detach()
    
    
#     w_l1_ = pm.clf2.l1.weight
#     b_l1_ = pm.clf2.l1.bias
    
#     w_l2_ = pm.clf2.l2.weight
#     b_l2_ = pm.clf2.l2.bias
    
#     w_lo_ = pm.clf2.lo.weight
#     b_lo_ = pm.clf2.lo.bias
    
    
#     fn_weights = torch.norm(w_l1_ - w_l1, p='fro')**2 + torch.norm(w_l2_ - w_l2, p='fro')**2 + torch.norm(w_lo_ - w_lo, p='fro')**2
    
#     fn_bias = torch.norm(b_l1_ - b_l1, p='fro')**2 + torch.norm(b_l2_ - b_l2, p='fro')**2 + torch.norm(b_lo_ - b_lo, p='fro')**2
    
#     fn = (fn_weights/3) + (fn_bias/3)
    
#     return fn
    

# # Initialize components
# # input_dim = 512
# input_dim = 768
# proj_dim = 2560  # Assuming GPT-2 uses 768-dimensional embeddings
# proj_dim=768
# projection_model = ProjectionModel(input_dim, proj_dim).to('cuda')
# import random

# # phi1 = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
# # phi2 = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
# phi1.resize_token_embeddings(len(tokenizer))
# phi2.resize_token_embeddings(len(tokenizer))



# # Example data (replace with actual data)
# num_examples = 10  # Arbitrary number of examples
# # Cs = torch.rand(num_examples, input_dim)
# texts1 = text
# texts2 = explanation
# kl_loss = nn.KLDivLoss(reduction="batchmean")
# # Dataset and DataLoader
# dataset = CustomDataset(Cs, texts1, texts2, L, tokenizer, IPS, L,explanation,IP)
# dataloader = DataLoader(dataset, batch_size=32,
#   worker_init_fn=seed_worker,
#   shuffle=True)

# # Training loop
# optimizer = torch.optim.Adam(list(projection_model.parameters()) + list(phi1.parameters()) + list(phi2.parameters()), lr=1e-4)
# loss_fn = nn.CrossEntropyLoss()

# for epoch in range(5):  # Number of epochs
#     for idx, batch in enumerate(dataloader):
#         # C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, input_ids,token_type_ids,attention_mask,visual_embeds,visual_token_type_ids,visual_attention_mask, labels, expl, img_path = batch

#         C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, input_, labels, expl, img_path = batch        
       
#         # Project C to get C1 and C2
#         C1_batch, C2_batch, C_r1, C_r2 = projection_model(C_batch.to('cuda').float())

#         # Replace the first token's embedding with C1 and C2
#         inputs_embeds1 = phi1.transformer.wte(input_ids1.to('cuda'))
#         inputs_embeds2 = phi2.transformer.wte(input_ids2.to('cuda'))
        
        
        
#         inputs_embeds1[:, 0, :] = C1_batch

        
#         inputs_embeds2[:, 0, :] = C2_batch
        
#         #print(attn_mask2)
        

        
#         outputs1 = phi1(inputs_embeds=inputs_embeds1, labels=input_ids1.to('cuda'))
#         outputs2 = phi2(inputs_embeds=inputs_embeds2, labels=input_ids2.to('cuda'))
        
        
        
      
#         #s1 = torch.softmax(outputs1.logits, dim=-1)* attn_mask1.to('cuda').unsqueeze(dim=2)
        
#         s2 = torch.softmax(outputs2.logits, dim=-1)
        
#         aggregated = s2.sum(dim=1)
        
#         aggregated = projection_model.aggregator(aggregated)
        
#         #print(aggregated)
        
#         #print(C_batch.to('cuda').shape, aggregated.shape)
        
#         mse_loss = F.mse_loss(aggregated, C_batch.to('cuda'))
#         #print(mse_loss)
#         #print(0/0)
        
        
        
#         class_ = projection_model.clf2(aggregated)
        
        
        
        
        
        

#         loss1 = outputs1.loss
#         loss2 = outputs2.loss
        
#         loss_clf = F.cross_entropy(class_, lab.to('cuda'))
        
        
        
#         tot_kl = 0
#         for idx_inner, batch_inner in enumerate(dataloader):
        
#             #print('idx inner', idx_inner)
        
#             if idx_inner < 2:
                
#                 C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, input_s, labels, expl, img_path = batch_inner
                
#                 # print(input_ids.shape, token_type_ids.shape, visual_embeds.shape)
                
#                 # inputs_base =  {k: v.to('cuda') for k, v in input_.items()}
#                 # inputs_src = {k: v.to('cuda') for k, v in input_s.items()}

#                 inputs_base = input_
#                 inputs_src = input_s

#                 # print('is_hersdsdse', inputs_src)
#                 # print('ib_hersdsdse', inputs_base)

#                 # inputs_src = {'input_ids': input_ids_s.to('cuda'),'token_type_ids':token_type_ids_s.to('cuda'),'attention_mask':attention_mask_s.to('cuda'),
#                 # 'visual_embeds':visual_embeds_s.to('cuda'),'visual_token_type_ids':visual_token_type_ids_s.to('cuda'),'visual_attention_mask':visual_attention_mask_s.to('cuda')}

#                 # print('HHHHHHHHHHHHHHHHHHHHH')
#                 with torch.no_grad():
#                     o_h, kk, intervention_idx, s, b = ii_h(model, inputs_base, inputs_src, kk=None)
                
#                 inputs_embeds2_s = phi2.transformer.wte(input_ids2.to('cuda'))
#                 inputs_embeds2_b = phi2.transformer.wte(input_ids2.to('cuda'))
#                 s = projection_model.proj2(s)
#                 b = projection_model.proj2(b)
#                 inputs_embeds2_s[:, 0, :] = s
#                 inputs_embeds2_b[:, 0, :] = b
                
#                 outputs2_s = phi2(inputs_embeds=inputs_embeds2_s, labels=input_ids2.to('cuda'))
#                 outputs2_b = phi2(inputs_embeds=inputs_embeds2_b, labels=input_ids2.to('cuda'))
                
#                 s = torch.softmax(outputs2_s.logits, dim=-1).sum(dim=1)
                
#                 b = torch.softmax(outputs2_b.logits, dim=-1).sum(dim=1)
                
                
                
#                 s = projection_model.aggregator(s)
#                 b = projection_model.aggregator(b)
                
#                 o_l = ii_l(projection_model, s, b, kk, intervention_idx)
                
#                 #print('ol', o_l)
                
                
#                 # input should be a distribution in the log space
                
#                 # Sample a batch of distributions. Usually this would come from the dataset
#                 target = F.softmax(o_h, dim=1)
#                 kl_div = kl_loss(F.log_softmax(o_l, dim=1), target)

        
        
        
                
                
               
                
                
#                 #print('o_h', o_h)
                
#                 tot_kl += kl_div
#             else:
#                 break
        
        
        
#         fn_loss = get_frobenius(model, projection_model)
      
        
      
#         if args.ablation.lower() == 'phi2':
#             total_loss = loss2  
#         elif args.ablation.lower() == 'phi2_ts':
#             total_loss = loss2 + loss_clf 
#         elif args.ablation.lower() == 'cause':
#             total_loss = loss2 + loss_clf + (tot_kl/2) + fn_loss 
        
#         total_loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
       
    
#         if idx%10==0:
#             print(f"Epoch {epoch+1}, Loss: {total_loss.item()}, KL Loss: {tot_kl/2}, Frobenius: {fn_loss}, mse: {mse_loss}")

        
# print('Saving models to Folder: ./clmfb_vqax_{}'.format(args.ablation.lower()))

# # torch.save(phi1.state_dict(), './flava_esnli_{}/phi1.pt'.format(args.ablation.lower()))
# torch.save(phi2.state_dict(), './clmfb_vqax_{}/phi2.pt'.format(args.ablation.lower()))
# torch.save(projection_model.state_dict(), './clmfb_vqax_{}/projection_model.pt'.format(args.ablation.lower()))
    
    
    


    
    
