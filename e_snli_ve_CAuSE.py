# -*- coding: utf-8 -*-


import torch
import numpy as np
import random

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

df_train = torch.load(os.path.join('/kaggle/working', 'train_df_enli.pt'))
df_test = torch.load(os.path.join('/kaggle/working', 'test_df_enli.pt'))



import torch
from torch.utils.data import DataLoader, Dataset
# from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
import torch.optim as optim

# Define a simple custom dataset
class CustomDataset(Dataset):
    def __init__(self, df):
        self.txt_features = torch.stack([df[i]['text_feats'] for i in range(len(df))]).squeeze()
        self.image_features = torch.stack([df[i]['img_feats'] for i in range(len(df))]).squeeze()
        self.labels = [df[i]['gold_label'] for i in range(len(df))]
        self.explanation = [df[i]['explanation'] for i in range(len(df))]
        self.img_path = [df[i]['img_path'] for i in range(len(df))]
        self.hypothesis = [df[i]['hypothesis'] for i in range(len(df))]
        #         self.tokenizer = VisualBertTokenizer.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

    def __len__(self):
        return self.txt_features.shape[0]

    def __getitem__(self, idx):
        #         inputs = self.tokenizer(self.questions[idx], return_tensors="pt")
        multimodal_embeds = torch.cat((self.txt_features, self.image_features), dim=-1)


        gl = 'none'
        if self.labels[idx]=='contradiction':
            gl = 0
        elif self.labels[idx]=='neutral':
            gl = 1
        elif self.labels[idx]=='entailment':
            gl = 2




        #return multimodal_embeds[idx,:], self.labels[idx], self.explanation[idx], self.img_path[idx]
        return self.txt_features[idx,:], self.image_features[idx,:], gl, self.explanation[idx], self.img_path[idx], self.hypothesis[idx]

train_dataset = CustomDataset(df_train)

test_dataset = CustomDataset(df_test)

len(train_dataset), len(test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
  worker_init_fn=seed_worker)

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

        """
        idx = random.sample(range(len(labels)), int(len(labels) * frac))

        for i in idx:
            if labels[i] ==0:
                labels[i] = random.choice([1, 2])  # Change 1 to 0 or 2
            elif labels[i] == 1:
                labels[i] = random.choice([0, 2])  # Change 1 to 0 or 2
            elif labels[i] == 2:
                labels[i] = random.choice([0, 1])  # Change 2 to 0 or 1
        """

        optimizer.zero_grad()

        outputs = model(inputs_img.to(torch.float32).to('cuda'), inputs_txt.to(torch.float32).to('cuda'))

        loss = ce(outputs.squeeze(), labels.to('cuda').squeeze())
        #         print(loss)

        loss.backward()
        optimizer.step()
        l+=loss.item()
        cnt+=1

    print(f"Epoch {epoch+1}, Loss: {l/cnt}")

    test_dataloader = DataLoader(test_dataset, batch_size=32)
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

GT = []
P = []
IP = []
EXP = []
HYP = []
for batch in train_dataloader:
    inputs_txt, inputs_img, labels, exp, ip, hypothesis = batch




    outputs = model(inputs_img.to(torch.float32).to('cuda'), inputs_txt.to(torch.float32).to('cuda'))
    op = outputs.squeeze().argmax(dim=-1).cpu().numpy()
    labels = labels.numpy()
    P.extend(op)
    GT.extend(labels)
    IP.extend(ip)
    EXP.extend(exp)
    HYP.extend(hypothesis)

len(P), len(GT), len(IP), len(EXP), len(HYP)



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

!pip install jsonlines

import jsonlines
import os

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
    with torch.no_grad():
        for idx, i in enumerate(train_dataset):


            inputs_txt, inputs_img, labels, exp, ip, txt = i

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

    Cs = torch.stack(Cs)
    T = torch.stack(T)
    I = torch.stack(I)
    return explanation, text, Cs, T, I, L, IP, LAB

explanation, text, Cs, T, I, L, IP,LAB = get_data()

Cs = Cs.squeeze()

# multimodal medium
M = torch.cat((I,T), dim=-1)

import torch
import torch.nn as nn
import torch.optim as optim

# Define the simple neural network for vector reconstruction
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, x):
        # Pass input through encoder
        encoded = self.encoder(x)

        return encoded

# Hyperparameters
input_dim = 512  # Input vector size
hidden_dim = 1024  # Size of hidden layer (bottleneck)

# Initialize the model, loss function, and optimizer
model_ = SimpleAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to('cuda')
criterion = nn.MSELoss()  # Reconstruction loss
optimizer = optim.SGD(model_.parameters(), lr=0.01)

# Example training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Example input vector
    input_vector = Cs

    # Forward pass: reconstruct the input
    output_vector = model_(input_vector.float().to('cuda'))

    # Calculate the loss
    loss = criterion(output_vector,M.float().to('cuda'))

    # Backward pass and SGD update
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update weights

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

import torch
import torch.nn.functional as F
# Function F(c)
def Fu(c,M):
    # Example function: sum of squares of c (L2 norm squared)
    img_feat, txt_feat = M[0:512], M[512:1024]
    c, fin = get_hidden(model, img_feat.unsqueeze(0).to('cuda').float(), txt_feat.unsqueeze(0).to('cuda').float())
    orig = fin.argmax(dim=-1)[0][0].item()


    changed = model.lin(c).argmax(dim=-1)[0][0].item()

    return changed, orig

# Loss function with respect to output class y
def loss_function(output, target):
    #return torch.nn.CrossEntropyLoss()(output, target)
    return nn.MSELoss()(output, target)

# SGD update with constraint F(c) = k
def constrained_sgd_step(model, data, target, k, lr=0.01):
    # Forward pass
    data.requires_grad_()
    output = model_(data.float())
    loss = loss_function(output.float(), target.to('cuda').float())




    # Manually compute the gradient of the loss with respect to c
    c_grad = torch.autograd.grad(loss, data, create_graph=True, allow_unused=True)

    c_grad = c_grad[0]

    # Direct SGD update: c = c - lr * c_grad
    with torch.no_grad():
        data -= lr * c_grad

        # Projection step: Ensure F(c) = k
        current_value,k = Fu(data,target)
        scaling_factor = torch.sqrt(torch.tensor((k+0.01) / (current_value+0.01)))
        data *= scaling_factor
    return data




# Desired value of F(c)
k = 1.0

# Learning rate
lr = 0.01

kk = []
for i in range(M.shape[0]):

    data = Cs[i,:]
    target = M[i,:]
    # Perform the constrained SGD step
    p = constrained_sgd_step(model, data, target, k, lr)
    kk.append(p)

kk = torch.stack(kk)



Cs = kk

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
dataloader = DataLoader(dataset, batch_size=4,
  worker_init_fn=seed_worker,
  shuffle=True)

# Training loop
optimizer = torch.optim.Adam(list(projection_model.parameters()) + list(phi1.parameters()) + list(phi2.parameters()), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):  # Number of epochs
    for idx, batch in enumerate(dataloader):
        C_batch, input_ids1, input_ids2, lab, attn_mask1, attn_mask2, txt_features, image_features, labels, expl, img_path = batch



        # Project C to get C1 and C2
        C1_batch, C2_batch, C_r1, C_r2 = projection_model(C_batch.to('cuda').float())

        # Replace the first token's embedding with C1 and C2
        inputs_embeds1 = phi1.transformer.wte(input_ids1.to('cuda'))
        inputs_embeds2 = phi2.transformer.wte(input_ids2.to('cuda'))

        #inputs_embeds1 = phi1.base_model.model.model.embed_tokens(input_ids1.to('cuda'))
        #inputs_embeds2 = phi2.base_model.model.model.embed_tokens(input_ids2.to('cuda'))

        #         C1_batch = C1_batch.unsqueeze(1).repeat(1,42,1)
        #         C2_batch = C2_batch.unsqueeze(1).repeat(1,42,1)

        #         inputs_embeds1 = inputs_embeds1 + 0.01*C1_batch
        #         inputs_embeds2 = inputs_embeds2 + 0.01*C2_batch



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









                #print('o_h', o_h)

                tot_kl += kl_div
            else:
                break



        fn_loss = get_frobenius(model, projection_model)



        total_loss = loss1 + loss2 + loss_clf + (tot_kl/2) + fn_loss









        # Combine losses and backpropagate

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if idx%10==0:
            print(f"Epoch {epoch+1}, Loss: {total_loss.item()}, KL Loss: {tot_kl/2}, Frobenius: {fn_loss}, mse: {mse_loss}")

# torch.save(phi1.state_dict(), '/kaggle/working/frobenius_wograd/phi1.pt')
# torch.save(phi2.state_dict(), '/kaggle/working/frobenius_wograd/phi2.pt')
# torch.save(projection_model.state_dict(), '/kaggle/working/frobenius_wograd/projection_model.pt')

!ls /kaggle/working/

phi1.load_state_dict(torch.load('/kaggle/working/only_mle/phi1.pt'))
phi2.load_state_dict(torch.load('/kaggle/working/only_mle/phi2.pt'))
projection_model.load_state_dict(torch.load('/kaggle/working/only_mle/projection_model.pt'))

model

dd = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
def get_test_data(model):
    model.eval()
    explanation = []
    text = []
    Cs = []
    T, I = [],[]
    L = []
    IP = []
    with torch.no_grad():
        for idx, i in enumerate(test_dataset):


            inputs_txt, inputs_img, labels, exp, ip, txt = i

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
            if idx==1000:
                break
    Cs = torch.stack(Cs)
    T = torch.stack(T)
    I = torch.stack(I)
    return explanation, text, Cs, T, I, L, IP

explanation, text, Cs, T, I, L, IP = get_test_data(model)

import torch
import torch.nn as nn
import torch.optim as optim

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: 1024 -> 512
        self.encoder = nn.Linear(1024, 512)


        # Decoder: 512 -> 1024
        self.decoder = nn.Linear(512, 1024)



    def forward(self, x):
        # Encoding step
        encoded = self.encoder(x)
        # Decoding step
        decoded = self.decoder(encoded)
        return decoded

# Instantiate the model
autoencoder0 = Autoencoder().to('cuda')
autoencoder1 = Autoencoder().to('cuda')
autoencoder2 = Autoencoder().to('cuda')

















T_v2, T_v1, T_v0 = [], [], []
I_v2, I_v1, I_v0 = [], [], []


for i in range(len(L)):
    if L[i]==2:
        T_v2.append(T[i,:])
        I_v2.append(I[i,:])
    elif L[i]==1:
        T_v1.append(T[i,:])
        I_v1.append(I[i,:])
    elif L[i]==0:
        T_v0.append(T[i,:])
        I_v0.append(I[i,:])


T_v2 = torch.stack(T_v2)
T_v1 = torch.stack(T_v1)
T_v0 = torch.stack(T_v0)
I_v2 = torch.stack(I_v2)
I_v1 = torch.stack(I_v1)
I_v0 = torch.stack(I_v0)

T_v2.shape, T_v1.shape, T_v0.shape, I_v2.shape, I_v1.shape, I_v0.shape

from sklearn.decomposition import PCA

pca = PCA(n_components=512)

from tqdm import tqdm

x_cfs_0 = [] # for these the counterfactual label produced by the encoder is 1/2, F should also produce 1/2
L0 = []
C0 = []
X0 = []
X_org0 = []
for i in tqdm(range(T_v0.shape[0])):

    #these are originally all 0 labels

    x = torch.cat((T_v0[i,:], I_v0[i,:]))
    X_org0.append(x)
    with torch.no_grad():
        c, fin = get_hidden(model, I_v0[i,:].unsqueeze(dim=0).to(torch.float32).to('cuda'), T_v0[i,:].unsqueeze(dim=0).to(torch.float32).to('cuda'))
        C0.append(c.detach())
    #print(x.shape)


    ll_d = []
    x_cfs = []
    T_V_ = torch.cat((T_v1,T_v2),dim=0)
    I_V_ = torch.cat((I_v1,I_v2),dim=0)
    L_ = [1]*T_v1.shape[0] + [2]*T_v2.shape[0]
    for j in range(T_V_.shape[0]):
        x_ = torch.cat((T_V_[j,:], I_V_[j,:]))
        x_cfs.append(x_)
        # search for the index in concat(T_v1,i_v1) corresponding to lowest dist
        dist = (x - x_).pow(2).sum().sqrt()
        ll_d.append(dist)
    x_min_idx = torch.stack(ll_d).argmin().item()

    x_ = x_cfs[x_min_idx]
    L0.append(L_[x_min_idx])

    #print(x_.shape)

    mu = x_ - x

    x_cfs_0.append(mu)
    X0.append(x_)

C0 = torch.stack(C0)
X0 = torch.stack(X0)
X_org0 = torch.stack(X_org0)

C0.shape, X0.shape

x_cfs_1 = [] # for these the counterfactual label produced by the encoder is 0/2, F should also produce 0/2
L1 = []
C1 = []
X1 = []
X_org1 = []
for i in tqdm(range(T_v1.shape[0])):

    #these are originally all 0 labels

    x = torch.cat((T_v1[i,:], I_v1[i,:]))
    X_org1.append(x)
    with torch.no_grad():
        c, fin = get_hidden(model, I_v1[i,:].unsqueeze(dim=0).to(torch.float32).to('cuda'), T_v1[i,:].unsqueeze(dim=0).to(torch.float32).to('cuda'))
        C1.append(c.detach())
    #print(x.shape)


    ll_d = []
    x_cfs = []
    T_V_ = torch.cat((T_v0,T_v2),dim=0)
    I_V_ = torch.cat((I_v0,I_v2),dim=0)
    L_ = [0]*T_v0.shape[0] + [2]*T_v2.shape[0]
    for j in range(T_V_.shape[0]):
        x_ = torch.cat((T_V_[j,:], I_V_[j,:]))
        x_cfs.append(x_)
        # search for the index in concat(T_v1,i_v1) corresponding to lowest dist
        dist = (x - x_).pow(2).sum().sqrt()
        ll_d.append(dist)
    x_min_idx = torch.stack(ll_d).argmin().item()

    x_ = x_cfs[x_min_idx]
    L1.append(L_[x_min_idx]) # L1 contains the counterfactual labels

    #print(x_.shape)

    mu = x_ - x

    x_cfs_1.append(mu)
    X1.append(x_)

# L0

C1 = torch.stack(C1)
X1= torch.stack(X1)
X_org1 = torch.stack(X_org1)

C1.shape, X1.shape, X_org1.shape



x_cfs_2 = [] # for these the counterfactual label produced by the encoder is 0/1, F should also produce 0/1
L2 = []
C2 = []
X2 = []
X_org2 = []
for i in tqdm(range(T_v2.shape[0])):

    #these are originally all 0 labels

    x = torch.cat((T_v2[i,:], I_v2[i,:]))
    with torch.no_grad():
        c, fin = get_hidden(model, I_v2[i,:].unsqueeze(dim=0).to(torch.float32).to('cuda'), T_v2[i,:].unsqueeze(dim=0).to(torch.float32).to('cuda'))
        C2.append(c.detach())
    X_org2.append(x)
    #print(x.shape)


    ll_d = []
    x_cfs = []
    T_V_ = torch.cat((T_v0,T_v1),dim=0)
    I_V_ = torch.cat((I_v0,I_v1),dim=0)
    L_ = [0]*T_v0.shape[0] + [1]*T_v2.shape[0]
    for j in range(T_V_.shape[0]):
        x_ = torch.cat((T_V_[j,:], I_V_[j,:]))
        x_cfs.append(x_)
        # search for the index in concat(T_v1,i_v1) corresponding to lowest dist
        dist = (x - x_).pow(2).sum().sqrt()
        ll_d.append(dist)
    x_min_idx = torch.stack(ll_d).argmin().item()

    x_ = x_cfs[x_min_idx]
    L2.append(L_[x_min_idx])

    #print(x_.shape)

    mu = x_ - x

    x_cfs_2.append(mu)
    X2.append(x_)

C2 = torch.stack(C2)
X2 = torch.stack(X2)
X_org2 = torch.stack(X_org2)





x_cfs_0 = torch.stack(x_cfs_0)

x_cfs_1 = torch.stack(x_cfs_1)

x_cfs_2 = torch.stack(x_cfs_2)

from torch.utils.data import TensorDataset
# Create a dataset and data loader

x_cfs2 = TensorDataset(x_cfs_2,C2,X2)
x_cfs2_dataloader = DataLoader(x_cfs2, batch_size=32, shuffle=True)


x_cfs1 = TensorDataset(x_cfs_1,C1,X1)
x_cfs1_dataloader = DataLoader(x_cfs1, batch_size=32, shuffle=True)


x_cfs0 = TensorDataset(x_cfs_0,C0,X0)
x_cfs0_dataloader = DataLoader(x_cfs0, batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer2 = optim.Adam(autoencoder2.parameters(), lr=0.001)
optimizer1 = optim.Adam(autoencoder1.parameters(), lr=0.001)
optimizer0 = optim.Adam(autoencoder0.parameters(), lr=0.001)

kl_loss = nn.KLDivLoss(reduction="batchmean")



# Training loop
num_epochs = 500  # You can adjust this
for epoch in range(num_epochs):
    for batch in x_cfs2_dataloader:

        # batch[0] = mu = x'-x (x's class is 1 as the base (normal class) class of x' is 0/2)
        # batch[1] = E(x)
        # batch[2] = x' = x+mu
        #print(batch)
        E_x = batch[1].squeeze()
        T_mu = autoencoder2.encoder(batch[0].to(torch.float32))

        x_t,x_i = batch[2][:,:512], batch[2][:,512:]
        with torch.no_grad():
            _, fin1 = get_hidden(model, x_i.to(torch.float32).to('cuda'), x_t.to(torch.float32).to('cuda'))

        # clf1(E_x+T_mu) = clf1(E(x+mu)) = clf1(E(x'))
        fin1 = fin1.squeeze()
        fin2 = model.lin(E_x.to('cuda')+0.4*T_mu.to('cuda'))




        inp = F.log_softmax(fin2, dim=1)
        tgt = F.softmax(fin1, dim=1)

        c_loss = kl_loss(inp,tgt)






        # Get the input data (batch)
        inputs = batch[0]

        # Zero the gradients
        optimizer2.zero_grad()

        # Forward pass: get the reconstruction
        outputs = autoencoder2(inputs.to(torch.float32))

        # Compute the loss

        loss = criterion(outputs, inputs.to(torch.float32)) + c_loss

        # Backward pass: compute gradients and update weights
        loss.backward()
        optimizer2.step()

    # Print the loss every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Training loop
num_epochs = 500  # You can adjust this
for epoch in range(num_epochs):
    for batch in x_cfs1_dataloader:

        # batch[0] = mu = x'-x (x's class is 1 as the base (normal class) class of x' is 0/2)
        # batch[1] = E(x)
        # batch[2] = x' = x+mu
        E_x = batch[1].squeeze()
        T_mu = autoencoder1.encoder(batch[0].to(torch.float32))

        x_t,x_i = batch[2][:,:512], batch[2][:,512:]
        with torch.no_grad():
            _, fin1 = get_hidden(model, x_i.to(torch.float32).to('cuda'), x_t.to(torch.float32).to('cuda'))

        # clf1(E_x+T_mu) = clf1(E(x+mu)) = clf1(E(x'))
        fin1 = fin1.squeeze()
        fin2 = model.lin(E_x.to('cuda')+0.4*T_mu.to('cuda'))

        inp = F.log_softmax(fin2, dim=1)
        tgt = F.softmax(fin1, dim=1)

        c_loss = kl_loss(inp,tgt)






        # Get the input data (batch)
        inputs = batch[0]

        # Zero the gradients
        optimizer1.zero_grad()

        # Forward pass: get the reconstruction
        outputs = autoencoder1(inputs.to(torch.float32))

        # Compute the loss

        loss = criterion(outputs, inputs.to(torch.float32)) + c_loss

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
        T_mu = autoencoder0.encoder(batch[0].to(torch.float32))

        x_t,x_i = batch[2][:,:512], batch[2][:,512:]
        with torch.no_grad():
            _, fin1 = get_hidden(model, x_i.to(torch.float32).to('cuda'), x_t.to(torch.float32).to('cuda'))


        fin1 = fin1.squeeze()
        fin2 = model.lin(E_x.to('cuda')+0.4*T_mu.to('cuda'))

        inp = F.log_softmax(fin2, dim=1)
        tgt = F.softmax(fin1, dim=1)

        c_loss = kl_loss(inp,tgt)







        # Get the input data (batch)
        inputs = batch[0]

        # Zero the gradients
        optimizer0.zero_grad()

        # Forward pass: get the reconstruction
        outputs = autoencoder0(inputs.to(torch.float32))

        # Compute the loss

        loss = criterion(outputs, inputs.to(torch.float32)) + c_loss

        # Backward pass: compute gradients and update weights
        loss.backward()
        optimizer0.step()

    # Print the loss every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    #ae_n is for original label n
    reduced_vectors2 = 0.4*autoencoder2.encoder(x_cfs_2.to(torch.float32)) + C2.squeeze()
    reduced_vectors1 = 0.4*autoencoder1.encoder(x_cfs_1.to(torch.float32)) + C1.squeeze() # CF1 = T(mu) + E(x) = ae_1(mu) + E(x) :::: for all x, clf1(x)=1 and clf1(x+mu) = 0/2 and clf1(CLF1) = 0/2 :::: L1 = 0/2
    reduced_vectors0 = 0.4*autoencoder0.encoder(x_cfs_0.to(torch.float32)) + C0.squeeze()

reduced_vectors2.shape, reduced_vectors1.shape, reduced_vectors0.shape

len(L0), len(L1), len(L2)

Cs_ = torch.cat((reduced_vectors0,reduced_vectors1,reduced_vectors2),dim=0)

Cs_.shape

L = L0+L1+L2

len(L)

# if doing counterfactual use the following line
M = torch.cat((X_org0,X_org1,X_org2),dim=0)
# else use the following
# multimodal medium
# M = torch.cat((I,T), dim=-1)

#M = torch.cat((I,T), dim=-1)

#Cs = Cs.squeeze()



# if use counterfactual use
Cs = Cs_
# else
# Cs = Cs.squeeze()



Cs.shape

import torch
from torch import nn
import torch.nn.functional as F
# Function F(c)
def Fu(c,M, counterfactual=False):
    # Example function: sum of squares of c (L2 norm squared)
    if counterfactual:
        txt_feat, img_feat = M[0:512], M[512:1024]
    else:
        img_feat, txt_feat = M[0:512], M[512:1024]
    c, fin = get_hidden(model, img_feat.unsqueeze(0).to('cuda').float(), txt_feat.unsqueeze(0).to('cuda').float())
    orig = fin.argmax(dim=-1)[0][0].item()


    changed = model.lin(c).argmax(dim=-1)[0][0].item()

    return changed, orig

# Loss function with respect to output class y
def loss_function(output, target):
    #return torch.nn.CrossEntropyLoss()(output, target)
    return nn.MSELoss()(output, target)

# SGD update with constraint F(c) = k
def constrained_sgd_step(model, data, target, k, lr=0.01):
    # Forward pass
    data.requires_grad_()
    output = model_(data.float())
    loss = loss_function(output.float(), target.to('cuda').float())




    # Manually compute the gradient of the loss with respect to c
    c_grad = torch.autograd.grad(loss, data, create_graph=True, allow_unused=True)

    c_grad = c_grad[0]

    # Direct SGD update: c = c - lr * c_grad
    with torch.no_grad():
        data -= lr * c_grad

        # Projection step: Ensure F(c) = k
        current_value,k = Fu(data,target,counterfactual=True)
        scaling_factor = torch.sqrt(torch.tensor((k+0.01) / (current_value+0.01)))
        data *= scaling_factor
    return data




# Desired value of F(c)
k = 1.0

# Learning rate
lr = 0.01

kk = []
for i in range(999):

    data = Cs[i,:]
    target = M[i,:]
    # Perform the constrained SGD step
    p = constrained_sgd_step(model, data, target, k, lr)
    kk.append(p)

kk = torch.stack(kk)

kk.shape

Cs = kk

Cs.shape



explanation[0:5]



len(Cs_)

# torch.save(Cs_, '/kaggle/working/mu.pt')

# Cs__ = torch.load('/kaggle/working/mu.pt')

# Cs_[0]

# Cs__[0]

!pip install shutup

# import shutup; shutup.please()

from transformers.utils import logging
logging.set_verbosity_error()

Cs_ = [Cs[i].detach().cpu().numpy() for i in range(Cs.shape[0])]

torch.manual_seed(0)
random.seed(0)

def get_lab(generated_list):
    if 'entailment' in generated_list:
        return 2
    elif 'neutral' in generated_list:
        return 1
    else:
        return 0

from tqdm import tqdm

len(L)

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

    C1_batch, C2_batch, _, _ = projection_model(torch.tensor(Cs_[j]).unsqueeze(dim=0).to('cuda'))
    #print(C1_batch.shape)
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




        if ('entailment' in gen) or ('entailment.' in gen) or ('neutral' in gen) or ('neutral.' in gen) or ('contradiction' in gen) or ('contradiction.' in gen):

            SS.append(0)

        else:
            SS.append(1)
            continue
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


    num_zeros, num_ones, num_twos = 0,0,0
    for i in proj_lab:
        if i==0:
            num_zeros+=1
        elif i==1:
            num_ones+=1
        else:
            num_twos+=1

    idx = np.array([num_zeros, num_ones, num_twos]).argmax()
    projected_label.append(idx)
    FL_.append(L[j])





    if sum(SS)==len(SS):
        #print('nothing generated')
        continue



    FL.append(L[j])
    projected_label_.append(class_f)


    num_zeros, num_ones, num_twos = 0,0,0
    for i in outer_acc:
        if i==0:
            num_zeros+=1
        elif i==1:
            num_ones+=1
        else:
            num_twos+=1

    idx = np.array([num_zeros, num_ones, num_twos]).argmax()
    actual_label.append(idx)



    num_zeros, num_ones, num_twos = 0,0,0
    for i in outer_label:
        if i==0:
            num_zeros+=1
        elif i==1:
            num_ones+=1
        else:
            num_twos+=1

    idx = np.array([num_zeros, num_ones, num_twos]).argmax()
    majority_label.append(idx)




    nn = ""
    for i in range(len(outer_label)):

        if outer_label[i]==idx:
            nn += outer_generation[i]+'----'
    gen_vals.append(nn+'exp: '+explanation[j])


print('predicted class', majority_label)
print('actual class', actual_label)
print('projected class', projected_label)





