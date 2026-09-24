# Detect Hallucinations in RAG from Retrieval and Generation Signals

A lightweight, black-box hallucination detection framework for Retrieval-Augmented Generation (RAG) systems. The framework extracts 17 externally observable signals from the retrieval, generation, and alignment stages of any RAG pipeline and trains supervised classifiers to predict whether a generated answer is hallucinated without requiring access to internal model states.

## Results

| Model | AUROC | vs TPA |
|---|---|---|
| **Random Forest (best)** | **0.904** | **+0.080** |
| Logistic Regression | 0.892 | +0.068 |
| MLP | 0.890 | +0.066 |
| XGBoost | 0.890 | +0.066 |
| TPA (ACL 2026) [white-box] | 0.824 | — |
| Best single-signal baseline | 0.548 | — |

Ablation results(LR):

| Signal group | AUROC |
|---|---|
| Retrieval only | 0.801 |
| Generation only | 0.782 |
| Alignment only | 0.682 |
| All 17 combined | 0.892 |

## Dataset

[RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) (Niu et al., ACL 2024) — QA subset, good quality examples, query-level deduplicated (839 unique queries). 200 training examples, 100 held-out test examples.

## Pipeline

```
RAGTruth QA queries
        ↓
Qwen2.5-7B-Instruct (4-bit NF4) — answer generation
        ↓
Signal extraction (17 signals)
    ├── Retrieval signals (6)  — cosine similarity via all-MiniLM-L6-v2
    ├── Generation signals (8) — token log-probabilities from model.generate()
    └── Alignment signals (3)  — ROUGE-L, BERTScore, lexical overlap
        ↓
3-Judge LLM ensemble — majority vote faithfulness label
    ├── GPT-oss-120b (Groq Account 1)
    ├── GPT-oss-20b  (Groq Account 2)
    └── Qwen3.8-27b  (Groq Account 3)
        ↓
Calibration classifier training (LR / MLP / Random Forest / XGBoost)
        ↓
AUROC evaluation on held-out test set
```

## Requirements

```bash
pip install transformers datasets sentence-transformers torch xgboost scikit-learn numpy pandas tqdm rouge-score bitsandbytes accelerate groq
```

## Setup

1. Clone the repository
2. Add the following secrets to Google Colab or Kaggle:
   - `GROQ_API_KEY_1` — Groq account 1 (GPT-oss-120b)
   - `GROQ_API_KEY_2` — Groq account 2 (GPT-oss-20b)
   - `GROQ_API_KEY_3` — Groq account 3 (Qwen3.8-27b)
3. Open `RAG_Reliability_RAGTruth_Final.ipynb` in Google Colab or Kaggle
4. Run all cells in order

## Hardware

- GPU: NVIDIA T4 (16 GB VRAM) — Google Colab or Kaggle
- Runtime: ~90 minutes (200 train + 100 test examples)
- GPU memory: ~5 GB (4-bit NF4 quantisation)

## Cached Results

The pipeline saves intermediate results to CSV files:
- `ragtruth_train_signals.csv` — training signals and labels
- `ragtruth_test_signals.csv` — test signals and labels

If both CSVs exist, model loading and generation are skipped automatically and the pipeline resumes from the calibration training step.
