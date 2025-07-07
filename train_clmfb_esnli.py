import torch
import numpy as np
import random
import os
import jsonlines
import os


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



from tqdm import tqdm



import pandas as pd

df_train = torch.load('./train_df_enli.pt')
df_test = torch.load('./test_df_enli.pt')



import torch
from torch.utils.data import DataLoader, Dataset
# from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
import torch.optim as optim

# Define a simple custom dataset
class CustomDataset(Dataset):
    def __init__(self, df, txt_feats, img_feats):
        self.txt_features = txt_feats
        self.image_features = img_feats
        self.labels = [df[i]['gold_label'] for i in range(len(df))]
        self.explanation = [df[i]['explanation'] for i in range(len(df))]
        self.img_path = [df[i]['img_path'] for i in range(len(df))]
        self.hypothesis = [df[i]['hypothesis'] for i in range(len(df))]
        #         self.tokenizer = VisualBertTokenizer.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

    def __len__(self):
        return self.txt_features.shape[0]

    def __getitem__(self, idx):
        #         inputs = self.tokenizer(self.questions[idx], return_tensors="pt")
        # multimodal_embeds = torch.cat((self.txt_features, self.image_features), dim=-1)


        gl = 'none'
        if self.labels[idx]=='contradiction':
            gl = 0
        elif self.labels[idx]=='neutral':
            gl = 1
        elif self.labels[idx]=='entailment':
            gl = 2




        
        # return  gl, self.explanation[idx], self.img_path[idx], self.hypothesis[idx]
        return self.txt_features[idx,:], self.image_features[idx,:], gl, self.explanation[idx], self.img_path[idx], self.hypothesis[idx]


txt_feats_train = torch.stack([df_train[i]['text_feats'] for i in range(len(df_train))]).squeeze().cpu()
img_feats_train = torch.stack([df_train[i]['img_feats'] for i in range(len(df_train))]).squeeze().cpu()

txt_feats_test = torch.stack([df_test[i]['text_feats'] for i in range(len(df_test))]).squeeze().cpu()
img_feats_test = torch.stack([df_test[i]['img_feats'] for i in range(len(df_test))]).squeeze().cpu()




train_dataset = CustomDataset(df_train, txt_feats_train, img_feats_train)

test_dataset = CustomDataset(df_test, txt_feats_test, img_feats_test)

len(train_dataset), len(test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
  worker_init_fn=seed_worker, num_workers = 4)




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
            ('lo',nn.Linear(128,3))
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

ce = nn.CrossEntropyLoss()

from tqdm import tqdm

from sklearn.metrics import f1_score

# labels

frac = 0.2
# Load the model and optimizer

optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Training loop
model.train()
for epoch in range(4):  # Number of epochs
    l = 0
    cnt = 0
    for batch in tqdm(train_dataloader):
        inputs_txt, inputs_img, labels, exp, ip, _ = batch

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
            inputs_txt, inputs_img, labels, exp, ip, _ = batch




            outputs = model(inputs_img.to(torch.float32).to('cuda'), inputs_txt.to(torch.float32).to('cuda'))
            op = outputs.squeeze().argmax(dim=-1).cpu().numpy()
            labels = labels.numpy()
            all_ops.extend(op)
            all_tgts.extend(labels)
    print('F1 Macro', f1_score(all_tgts, all_ops, average='macro'))
    print('F1 Micro', f1_score(all_tgts, all_ops, average='micro'))

model.eval()

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
            ('lo',nn.Linear(128,3))
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

        multimodal_embeds = torch.cat((self.txt_features, self.image_features), dim=-1)


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

dd = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}



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


            inputs_txt, inputs_img, labels, exp, ip, txt = i
            inputs_txt = inputs_txt.squeeze()
            inputs_img = inputs_img.squeeze()
            labels = labels[0].item()
            exp = exp[0]
            ip = ip[0]
            txt = txt[0]

            c, fin = get_hidden(model, inputs_img.unsqueeze(dim=0).to(torch.float32).to('cuda'), inputs_txt.unsqueeze(dim=0).to(torch.float32).to('cuda'))


            pred_lab = fin.argmax(dim=-1)[0][0].item()
            LAB.append(labels)

            if pred_lab==labels:
                L.append(pred_lab)
                explanation.append('This pair is a {} because {}'.format(dd[pred_lab], exp.strip()+'.'))
            else:
                L.append(pred_lab)
                explanation.append('This pair is a {}'.format(dd[pred_lab]))







            text.append(txt)
            Cs.append(c)
            T.append(inputs_txt)
            I.append(inputs_img)
            IP.append(ip)

            if idx==8991:
                break

    Cs = torch.stack(Cs)
    T = torch.stack(T)
    I = torch.stack(I)
    return explanation, text, Cs, T, I, L, IP, LAB

explanation, text, Cs, T, I, L, IP,LAB = get_data()

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


print('Saving models to Folder: ./clmfb_esnli_{}'.format(args.ablation.lower()))

torch.save(phi1.state_dict(), './clmfb_esnli_{}/phi1.pt'.format(args.ablation.lower()))
torch.save(phi2.state_dict(), './clmfb_esnli_{}/phi2.pt'.format(args.ablation.lower()))
torch.save(projection_model.state_dict(), './clmfb_esnli_{}/projection_model.pt'.format(args.ablation.lower()))