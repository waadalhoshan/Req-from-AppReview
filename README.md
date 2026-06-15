# REV4RE: Eliciting Explicit and Implicit Requirements from App Reviews

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-REV4RE-blue.svg)](#dataset)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

This repository contains the dataset, code, and supplementary materials for the paper:

> **Eliciting Explicit and Implicit Requirements from App Reviews via Grounded Theory and BERT-Based Classification**

## Overview

App reviews contain valuable user requirements—some stated explicitly as feature requests or bug reports, others implied within narratives of use, complaints, or comparisons to competing apps. Existing approaches predominantly focus on explicit feedback, overlooking implicit requirements that carry actionable information for app improvement.

This work presents:
1. A **grounded theory-derived classification scheme** with 14 requirement characteristics (ER1–ER4, MR1–MR6, IR1–IR4) distinguishing explicit, implicit, and irrelevant reviews
2. **REV4RE**, a gold-standard dataset of manually annotated app reviews from the Apple App Store and Google Play Store (κ = 0.74)
3. A **comparative evaluation** of four fine-tuned BERT-based models (BERT, DistilBERT, ALBERT, RoBERTa) with balancing strategies for automated classification

## Repository Structure

```
REV4RE/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── dataset/
│   ├── README.md              # Dataset description and format
│   └── rev4re.csv             # REV4RE dataset
├── code/
│   ├── preprocessing/
│   │   └── preprocess.py      # Data cleaning, splitting, and balancing
│   ├── fine-tuning/
│   │   └── train.py           # BERT-based model fine-tuning
│   └── evaluation/
│       └── evaluate.py        # Metrics computation and ROC curves
├── models/
│   └── README.md              # Pre-trained model links and config
├── figures/
├── tables/
└── docs/
    └── annotation_guidelines.md  # Labeling manual and coding scheme
```

## Dataset

The **REV4RE** dataset contains app reviews collected from the Apple App Store and Google Play Store (February–May 2022), annotated into three classes:

| Class | Label | Description |
|-------|-------|-------------|
| Explicit | `explicit` | Directly stated requirements (feature requests, bug reports) |
| Implicit | `implicit` | Implied requirements (experiential narratives, comparisons) |
| Irrelevant | `irrelevant` | Non-requirements content (service feedback, pricing) |

See [`dataset/README.md`](dataset/README.md) for format details.

## Requirements Classification Scheme

The classification scheme was derived through grounded theory and comprises 14 requirement characteristics:

### Explicit Requirements (ER1–ER4)
- **ER1**: Specific app features not working as intended
- **ER2**: Complaints about bugs, glitches, or functional issues
- **ER3**: Requests for help with app features
- **ER4**: Missing features users wish to be added

### Implicit Requirements (MR1–MR6)
- **MR1**: Opinions on overall app performance
- **MR2**: Non-functional complaints (battery, heating)
- **MR3**: Storytelling-style feature suggestions
- **MR4**: Competitor comparisons implying missing capabilities
- **MR5**: UI/UX navigation difficulties
- **MR6**: Device/platform incompatibility

### Irrelevant Reviews (IR1–IR4)
- **IR1**: Customer support feedback
- **IR2**: Pricing or value-for-money comments
- **IR3**: Community or social feature discussions
- **IR4**: Content or media quality reviews

## Quick Start

### Installation

```bash
git clone https://github.com/[username]/REV4RE.git
cd REV4RE
pip install -r requirements.txt
```

### Preprocessing

```bash
# Split into train/test (80/20) with random under-sampling
python code/preprocessing/preprocess.py \
    --data dataset/rev4re.csv \
    --output_dir dataset/ \
    --balance undersample
```

### Fine-Tuning a Model

```bash
python code/fine-tuning/train.py \
    --model roberta \
    --train_data dataset/rev4re_train.csv \
    --test_data dataset/rev4re_test.csv \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --balance_strategy undersample
```

### Evaluating a Model

```bash
python code/evaluation/evaluate.py \
    --model_path outputs/roberta/best_model \
    --model_type roberta \
    --test_data dataset/rev4re_test.csv
```

## Results Summary

| Model | Params | Imbalanced F1 | Weighted F1 | Under-sampled F1 |
|-------|--------|---------------|-------------|-------------------|
| BERT | 110M | 0.88 | 0.87 | 0.79 |
| DistilBERT | 60M | 0.88 | 0.87 | 0.79 |
| ALBERT | 11M | 0.88 | 0.86 | 0.78 |
| **RoBERTa** | **125M** | **0.89** | **0.87** | **0.79** |

- Best AUC scores: **0.91–0.92** (under-sampled configuration)
- Random under-sampling improved implicit requirements detection by **~40%**
- RoBERTa F1-score (87%) closely matches human annotator performance (87%)

## Citation

If you use REV4RE or the classification scheme in your research, please cite:

```bibtex
@article{[key],
  title={Eliciting Explicit and Implicit Requirements from App Reviews via Grounded Theory and BERT-Based Classification},
  author={[Authors]},
  journal={[Journal]},
  year={[Year]},
  doi={[DOI]}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, please open an issue or contact [corresponding author email].
