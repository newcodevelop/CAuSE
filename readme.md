# CAuSE: Post-hoc Natural Language Explanation of Multimodal Classifiers through Causal Abstraction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

**CAuSE** (**C**ausal **A**bstraction **u**nder **S**imulated **E**xplanation) is a framework designed to generate faithful natural language explanations for multimodal classifiers [web:7]. It leverages **Interchange Intervention Training (IIT)** to align the explanation generation process with the causal mechanisms of the underlying classifier.

---

## 📂 Repository Structure

The code is organized by dataset (e.g., `src/HM` for Hateful Memes) [attached_file:1]. Within each dataset directory, the code is split into `train` and `test` folders.

### Directory Layout
src/
└── <Dataset>/ # e.g., HM (Hateful Memes)
├── train/ # Training scripts
│ ├── train_qwenvl_hm.py # (1) Train the Classifier (Qwen-VL)
│ ├── qwenvl_hm_train.py # (2) Train CAuSE Explainer (Qwen-VL backbone)
│ ├── train_vb_hm.py # (2) Train CAuSE Explainer (VisualBERT backbone)
│ ├── train_flava_hm.py # (2) Train CAuSE Explainer (FLAVA backbone)
│ └── train_clmfb_hm.py # (2) Train CAuSE Explainer (CLIP-Multimodal backbone)
└── test/ # Testing scripts
├── test_qwenvl_hm.py # Testing (structure mirrors train/)
└── ...


---

## 🚀 Usage

### 1. Train the Multimodal Classifier
Before training the CAuSE explainer, you must train the target "black-box" classifier (e.g., Qwen-VL, FLAVA, VisualBERT). For Qwen-VL, a separate file is used to train it

**Example (Hateful Memes - Qwen-VL):**
```python 
src/HM/train/train_qwenvl_hm.py
```


### 2. Train CAuSE
Train the explanation generator using the CAuSE framework [attached_file:1]. You must specify the `--ablation` argument to define the training mode.

**Required Arguments:**
- `--ablation` (str, required): The specific configuration to train.
    - `cause`: The full CAuSE framework (with IIT)
    - `phi2`: Baseline (standard language model training)
    - `phi2_ts`: Teacher-Student baseline configuration

**Example (Hateful Memes with Qwen-VL backbone):**


**Example (Hateful Memes with Qwen-VL backbone):**
### Train the CAuSE model (with Causal Abstraction)
```python src/HM/train/qwenvl_hm_train.py --ablation cause```

### Train the phi2 baseline
```python src/HM/train/qwenvl_hm_train.py --ablation phi2```

### Train the phi2_ts baseline
```python src/HM/train/qwenvl_hm_train.py --ablation phi2_ts```


