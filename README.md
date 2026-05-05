# Membership Inference Attack — Reproducing the Best Leaderboard Result

## Overview

This codebase implements an RMIA-based membership inference attack against a pretrained ResNet-18 image classifier, as described in [Zarifzadeh et al. (2024)](https://arxiv.org/pdf/2312.03262).

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU supported but slow)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Download Assets

Run the following to download the model, datasets, and official template into your working directory:

```bash
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt" \
     "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt" \
     "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt" \
     "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/task_template.py"
```

---

## Reproduce the Best Result

From the repository root, run:

```bash
python src/main.py \
  --directory . \
  --output submission.csv \
  --num_ref_models 4 \
  --epochs 5 \
  --batch_size 128 \
  --gamma 1.0 \
  --a 0.3 \
  --Z_size 1000 \
  --val_size 0.2 \
  --seed 42
```

This will:
1. Train 4 reference models on random 50% subsets of `pub.pt`
2. Compute RMIA scores for all 14k samples in `priv.pt`
3. Write `submission.csv` to the working directory
4. Validate the submission format automatically

---

## Submit to Leaderboard

```bash
export MIA_API_KEY="your_api_key_here"
python submit.py --csv submission.csv
```

---

## Project Structure

```
.
├── requirements.txt
├── submit.py               # Leaderboard submission script
└── src/
    ├── main.py             # Entry point — orchestrates the full pipeline
    ├── dataset.py          # Dataset classes for pub.pt / priv.pt
    ├── model.py            # ResNet-18 architecture definition
    ├── rmia.py             # RMIA scoring algorithm
    ├── train.py            # Reference model training loop
    ├── utils.py            # Reference dataset creation + probability computation
    └── metrics.py          # TPR@FPR evaluation metric
```
