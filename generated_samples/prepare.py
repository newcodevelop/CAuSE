import torch


df_test = torch.load('../test_df_enli.pt')

print(df_test[0])

print(0/0)


#ESNLI + CLMFB + CAuSE
k_cause = torch.load("cause_esnli_clmfb_gen.pt")
k_ts = torch.load("phi2_ts_esnli_clmfb_gen.pt")
k_phi = torch.load("phi2_esnli_clmfb_gen.pt")

same_strs = {}

for j, i in enumerate(k_cause):
    l = i.split("----")

    labs = []
    for pred_str in l:
        
        lab_list = ["entailment", "contradiction", "neutral"]
        for lab in lab_list:
            if lab in pred_str:
                labs.append(lab.lower())
                continue

    if len(list(set(labs)))==1:
        # same_strs.append((j,i))
        same_strs[j] = i
         
print(len(same_strs))

enumerator = list(same_strs.keys())

df = {"GT" : [], "CauSE": [], "TS": [], "phi2": []}

for s,(p,q) in enumerate(zip(k_ts, k_phi)):
    if s in enumerator:
        df["GT"].append(same_strs[s].split("exp:")[-1])
        df["CauSE"].append(same_strs[s].split("exp:")[0])
        df["TS"].append(p)
        df["phi2"].append(q)

import pandas as pd

pd.DataFrame(df).to_csv("./df_clmfb_esnli.csv")






    
    


