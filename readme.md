# CAuSE: Post-hoc Natural Language Explanation of Multimodal Classifiers through Causal Abstraction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

**CAuSE** (**C**ausal **A**bstraction **u**nder **S**imulated **E**xplanation) is a framework designed to generate faithful natural language explanations for multimodal classifiers. It leverages **Interchange Intervention Training (IIT)** to align the explanation generation process with the causal mechanisms of the underlying classifier.

---

## 📂 Repository Structure

The code is organized by dataset (e.g., `src/HM` for Hateful Memes). Within each dataset directory, the code is split into `train` and `test` folders.

### Directory Layout


```text
src/
└── <Dataset>/                      # e.g., HM (Hateful Memes)
    ├── train/                      # Training scripts
    │   ├── train_qwenvl_hm.py      # (1) Train the Classifier (Qwen-VL)
    │   ├── qwenvl_hm_train.py      # (2) Train CAuSE Explainer (Qwen-VL backbone)
    │   ├── train_vb_hm.py          # (2) Train CAuSE Explainer (VisualBERT backbone)
    │   ├── train_flava_hm.py       # (2) Train CAuSE Explainer (FLAVA backbone)
    │   └── train_clmfb_hm.py       # (2) Train CAuSE Explainer (CLIP-Multimodal backbone)
    └── test/                       # Testing scripts
        ├── test_qwenvl_hm.py       # Testing (mirrors train/)
        └── ...
```

---

## 🚀 Usage

### 1. Train the Multimodal Classifier
Before training the CAuSE explainer, you must train the target "black-box" classifier (e.g., Qwen-VL, FLAVA, VisualBERT). For Qwen-VL, a separate file is used to train it

**Example (Hateful Memes - Qwen-VL):**
```python 
python src/HM/train/train_qwenvl_hm.py
```


### 2. Train CAuSE
Train the explanation generator using the CAuSE framework. You must specify the `--ablation` argument to define the training mode.

**Required Arguments:**
- `--ablation` (str, required): The specific configuration to train.
    - `cause`: The full CAuSE framework (with IIT)
    - `phi2`: Baseline (standard language model training)
    - `phi2_ts`: Teacher-Student baseline configuration

**Example (Hateful Memes with Qwen-VL backbone):**

### a) Train the CAuSE model (with Causal Abstraction)
```python src/HM/train/qwenvl_hm_train.py --ablation cause```

### b) Train the phi2 baseline
```python src/HM/train/qwenvl_hm_train.py --ablation phi2```

### c) Train the phi2_ts baseline
```python src/HM/train/qwenvl_hm_train.py --ablation phi2_ts```


**Supported Backbones:**
We support various multimodal backbones. Ensure you run the corresponding script for your desired architecture:
- **Qwen-VL**: `src/HM/train/qwenvl_hm_train.py`
- **VisualBERT**: `src/HM/train/train_vb_hm.py`
- **FLAVA**: `src/HM/train/train_flava_hm.py`
- **CLIP-Multimodal**: `src/HM/train/train_clmfb_hm.py`

---

## 📊 Evaluation & Testing

Testing follows the same directory structure (replace `train` with `test` in the path). The testing scripts calculate performance metrics (F1) or faithfulness metrics (CCMR) based on the flags provided.

**Required Arguments:**
- `--ablation` (str, required): The model configuration to load (must match the training phase: `cause`, `phi2`, or `phi2_ts`)
- `--counterfactual_test` (int, required): Flag for the metric type
    - `0`: Calculate standard **F1 Score** (Performance)
    - `1`: Calculate **CCMR** (Counterfactual Consistency Metric / Faithfulness)

**Example: Calculate Standard F1**
```python
python src/HM/test/test_qwenvl_hm.py
--ablation cause
--counterfactual_test 0
```


**Example: Calculate Faithfulness (CCMR)**
```python
python src/HM/test/test_qwenvl_hm.py
--ablation cause
--counterfactual_test 1
```

### The above codes are shown corresponding to a single dataset (HM). To run on datasets, just replace HM with the corresponding dataset name.


---

## 📧 Contact

For questions or inquiries, please contact:
- **Dibyanayan Bandyopadhyay**: [dibyanayan@gmail.com](mailto:dibyanayan@gmail.com)

---


## Citation

If you find this code useful, please cite:

```bibtex
@article{10.1162/TACL.a.686,
    author = {Bandyopadhyay, Dibyanayan and Bhattacharjee, Soham and Hasanuzzaman, Mohammed and Ekbal, Asif},
    title = {CAuSE: Decoding Multimodal Classifiers using Faithful Natural Language Explanation},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {14},
    pages = {829-851},
    year = {2026},
    month = {06},
    abstract = {Multimodal classifiers function as opaque black box models. While several techniques exist to interpret their predictions, very few of them are as intuitive and accessible as natural language explanations (NLEs). To build trust, such explanations must faithfully capture the classifier’s internal decision making behavior, a property known as faithfulness. In this paper, we propose CAuSE (Causal Abstraction under Simulated Explanations), a novel framework to generate faithful NLEs for any pretrained multimodal classifier. We demonstrate that CAuSE generalizes across datasets and models through extensive empirical evaluation. Theoretically, we show that CAuSE, trained via interchange intervention, forms a causal abstraction of the underlying classifier. We further validate this through a redesigned metric for measuring causal faithfulness in multimodal settings. CAuSE surpasses other methods on this metric, with qualitative analysis reinforcing its advantages. We also perform detailed error analysis to pinpoint the failure cases of CAuSE1.},
    issn = {2307-387X},
    doi = {10.1162/TACL.a.686},
    url = {https://doi.org/10.1162/TACL.a.686},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/TACL.a.686/2607018/tacl.a.686.pdf},
}
```


## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.



