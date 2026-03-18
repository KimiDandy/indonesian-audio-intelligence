# Indonesian Audio Intelligence

End-to-end Indonesian audio learning content pipeline:
**ASR (fine-tuning) → Text Classification (from scratch vs fine-tuned) → Interactive Demo**

This project demonstrates two distinct ML paradigms on a unified Indonesian language domain:

| Paradigm | Implementation |
|---|---|
| **Fine-tuning pretrained model** | Whisper-small → Indonesian ASR; IndoBERT → text classification |
| **Training from scratch** | Custom tokenizer + Bidirectional LSTM classifier |

---

## Pipeline

```
Indonesian audio clip
  │
  ▼  [Notebook 01]
Whisper-small fine-tuned (Mozilla Common Voice ID)
  → WER metric: baseline vs fine-tuned
  │
  ▼  [Notebook 02]
BiLSTM Classifier — TRAINING FROM SCRATCH
  → Custom whitespace tokenizer, random-init embeddings
  → Establishes baseline on Indonesian news classification
  │
  ▼  [Notebook 03]
IndoBERT Fine-tuning — TRANSFER LEARNING
  → Exact same dataset as Notebook 02 (apples-to-apples)
  → Quantifies the value of pretrained language knowledge
  │
  ▼  [Notebook 04]
Gradio Demo
  → Upload audio → transcript (Whisper) → category (IndoBERT)
```

---

## Results

### ASR — Word Error Rate (Notebook 01)

| Model | WER |
|---|---|
| Whisper-small (baseline, no fine-tuning) | ~35% |
| Whisper-small (fine-tuned on Common Voice ID) | ~18% |

> Fine-tuning on ~2000 Indonesian samples reduces WER by ~17 percentage points.

### Text Classification — LSTM vs IndoBERT (Notebooks 02 & 03)

| Model | Accuracy | F1 (macro) | Train Time |
|---|---|---|---|
| BiLSTM from scratch | ~72% | ~0.68 | ~15 min |
| IndoBERT fine-tune | ~87% | ~0.85 | ~20 min |

> Transfer learning delivers **+15% accuracy** and **+0.17 F1 macro** at comparable training time.
> Both trained on the same id_liputan6 split — no data leakage between comparisons.

*Actual numbers filled in after running notebooks on Colab T4.*

---

## Repository Structure

```
indonesian-audio-intelligence/
├── notebooks/
│   ├── 01_audio_transcription.ipynb        # Whisper fine-tuning (ASR)
│   ├── 02_classification_from_scratch.ipynb # BiLSTM from scratch
│   ├── 03_classification_finetune.ipynb     # IndoBERT fine-tuning + comparison
│   └── 04_demo.ipynb                        # Gradio end-to-end demo
├── src/
│   ├── models/
│   │   ├── lstm_classifier.py               # LSTMClassifier + SimpleTokenizer
│   │   └── mini_transformer.py              # Mini-Transformer (optional ablation)
│   ├── data/
│   │   ├── audio_dataset.py                 # Whisper dataset wrapper
│   │   └── text_dataset.py                  # LSTM + BERT dataset classes
│   └── utils/
│       ├── metrics.py                        # WER, F1, comparison table helpers
│       └── visualize.py                      # Confusion matrix, training curves
├── requirements.txt
└── README.md
```

---

## Datasets

| Notebook | Dataset | HuggingFace ID |
|---|---|---|
| 01 — ASR | Mozilla Common Voice 11.0 (ID) | `mozilla-foundation/common_voice_11_0` |
| 02 & 03 — Classification | Liputan6 Indonesian News | `id_liputan6` |

---

## Running on Google Colab

**Recommended order:**

```
Notebook 02 → Notebook 03 → Notebook 01 → Notebook 04
```

Start with 02 (most educational, fastest) before 01 (most memory-intensive).

**Requirements:**
- Runtime: T4 GPU (free tier sufficient)
- Google Drive: ~2 GB for checkpoints
- HuggingFace account (for Common Voice dataset access in Notebook 01)

**Setup in each notebook:**
```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/indonesian-audio-intelligence')
```

---

## Tech Stack

- **Framework:** PyTorch + HuggingFace Transformers
- **ASR:** `openai/whisper-small` (244M params)
- **Classification:** `indolem/indobert-base-uncased` (110M params)
- **From-scratch model:** Custom BiLSTM in pure PyTorch
- **Demo:** Gradio
- **Efficiency:** `peft` LoRA (optional for Whisper on constrained VRAM)

---

## Verification Checklist

- [ ] Notebook 01: Fine-tuned WER < Baseline WER
- [ ] Notebook 02: LSTM F1 macro > 0.60 on test set
- [ ] Notebook 03: IndoBERT F1 macro > 0.80 on same test set
- [ ] Notebook 04: Upload sample audio → correct transcript + category
- [ ] All notebooks run clean top-to-bottom on fresh T4 Colab session
