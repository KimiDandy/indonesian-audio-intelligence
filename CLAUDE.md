# CLAUDE.md — indonesian-audio-intelligence

> **Context:** This project was fully planned and designed before implementation. All architecture
> decisions, dataset choices, model selections, and expected benchmarks were deliberate. Do NOT
> redesign from scratch — extend and refine what exists. Read `.claude/` files for full context.

---

## What This Project Is

**Portfolio ML project** demonstrating two ML paradigms on a single Indonesian language domain:

| Paradigm | Implementation |
|---|---|
| Training from scratch | Custom BiLSTM + custom whitespace tokenizer (Notebook 02) |
| Fine-tuning pretrained | Whisper-small for ASR (Notebook 01), IndoBERT for classification (Notebook 03) |

**The central portfolio story:**
> "Transfer learning delivers +15% accuracy and -17pp WER over scratch-trained models at comparable
> training time — on the same datasets, controlled comparison."

This is an edtech/audio domain project inspired by Indonesian podcast learning platforms.
The Indonesian NLP domain is intentionally chosen: underrepresented in public benchmarks,
which makes the portfolio piece more differentiated.

**Target runtime:** Google Colab T4 GPU (free tier). All training must fit in ~15GB VRAM with fp16.

---

## Notebook Execution Order

**Run in this order — dependencies flow forward:**

```
02_classification_from_scratch   ← START HERE (fastest, most educational, ~15 min)
  ↓ produces: checkpoints/lstm_best.pt

03_classification_finetune       ← loads lstm_best.pt for comparison table (~20 min)
  ↓ produces: checkpoints/indobert-best/, classification_results.json

01_audio_transcription           ← most memory-intensive, run last (~45-60 min)
  ↓ produces: whisper-id-finetuned/, wer_results.json

04_demo                          ← loads all above, launches Gradio
```

**Why this order (not 01→02→03→04):**
- Notebook 02 is fastest and builds intuition — good to start with
- Notebook 01 needs ~45–60 min and is most likely to OOM — do last
- Notebooks 02+03 share dataset loading code — run them back-to-back while data is warm in memory

---

## Checkpoint Contract

Every notebook saves specific files to Google Drive. Other notebooks depend on these exact paths.

```
/content/drive/MyDrive/indonesian-audio-intelligence/
├── checkpoints/
│   ├── lstm_best.pt                   # Notebook 02 saves, Notebook 03+04 reads
│   │   Keys: model_state_dict, tokenizer_word2idx, vocab_size,
│   │         num_classes, categories, test_accuracy, test_f1_macro, train_time_min
│   │
│   ├── indobert-best/                 # Notebook 03 saves, Notebook 04 reads
│   │   Files: pytorch_model.bin (or model.safetensors), config.json,
│   │          tokenizer_config.json, vocab.txt, tokenizer.json
│   │
│   └── indobert-classification/       # Trainer epoch checkpoints (all epochs)
│
├── whisper-id-finetuned/              # Notebook 01 saves, Notebook 04 reads
│   Files: model.safetensors, config.json, preprocessor_config.json,
│          tokenizer.json, vocab.json, merges.txt, normalizer.json
│
├── classification_results.json        # Notebook 03 saves, Notebook 04 reads
│   Schema: {
│     "lstm":     {"accuracy": float, "f1_macro": float, "train_time_min": float},
│     "indobert": {"accuracy": float, "f1_macro": float, "train_time_min": float},
│     "categories": {"bisnis": 0, "teknologi": 1, ...}
│   }
│
├── wer_results.json                   # Notebook 01 saves
│   Schema: {"baseline_wer": float, "finetuned_wer": float}
│
├── lstm_training_curves.png           # Notebook 02
├── lstm_confusion_matrix.png          # Notebook 02
├── model_comparison.png               # Notebook 03 (LSTM vs IndoBERT bar chart)
├── indobert_confusion_matrix.png      # Notebook 03
└── wer_comparison.png                 # Notebook 01
```

---

## Google Drive Setup (required in every notebook)

Every notebook must start with this block:

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
REPO_DIR = '/content/drive/MyDrive/indonesian-audio-intelligence'
sys.path.append(REPO_DIR)

import os
os.makedirs(f'{REPO_DIR}/checkpoints', exist_ok=True)
```

`REPO_DIR` is the canonical path used everywhere. If the user has a different Drive folder name,
update only this variable — all other paths derive from it.

---

## Dataset Details

### id_liputan6 — Notebooks 02 & 03

```python
ds = load_dataset('id_liputan6', 'complete', trust_remote_code=True)
# Splits: train / validation / test
# Relevant fields: 'clean_article' (article body), 'url' (used to extract category)
```

**Category extraction — from URL path segment:**
```python
CATEGORIES = {
    'bisnis': 0, 'teknologi': 1, 'otomotif': 2, 'bola': 3,
    'health': 4, 'lifestyle': 5, 'showbiz': 6, 'regional': 7,
}
# URL example: https://www.liputan6.com/bisnis/read/5234567/...
# Articles that don't match any known category are skipped (label = -1)
```

Sample sizes (modest for Colab, sufficient for portfolio):
- Train: 8,000 | Val: 1,500 | Test: 1,500

**Critical:** Notebooks 02 and 03 must use **identical** split sizes so the comparison is valid.
`process_split()` iterates deterministically — same result every run assuming no upstream shuffle.

### Mozilla Common Voice 11.0 Indonesian — Notebook 01

```python
ds = load_dataset('mozilla-foundation/common_voice_11_0', 'id',
                  split='train[:2000]', trust_remote_code=True)
```

**Requires HuggingFace account + dataset acceptance:**
1. Visit https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
2. Accept terms of use
3. In Colab: `huggingface-cli login` → paste token

Audio is originally 48kHz, resampled to 16kHz in-place:
```python
ds = ds.cast_column('audio', Audio(sampling_rate=16_000))
```

---

## Model Architecture

### BiLSTM Classifier (Notebook 02) — from scratch

```
Embedding(vocab_size=15_000, dim=128, padding_idx=0)
  → Dropout(0.3)
  → BiLSTM(128 → 256 hidden, 2 layers, dropout=0.3)
  → Concat last fwd+bwd hidden state → shape (batch, 512)
  → Dropout(0.5)
  → Linear(512, 8)
```

**SimpleTokenizer rules:**
- `<PAD>` at index 0, `<UNK>` at index 1
- Vocabulary: top 15,000 tokens by frequency from training corpus
- max_len = 128 (truncate long texts, right-pad short ones with PAD)
- **Must call `tokenizer.build_vocab(train_texts)` before any `encode()` call**

**Training hyperparameters:**
- Optimizer: Adam, lr=1e-3
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Loss: CrossEntropyLoss | Gradient clip: max_norm=1.0
- Batch: 64 | Epochs: 10 | Best model saved by F1 macro on val set

### IndoBERT Fine-tuning (Notebook 03) — transfer learning

```
indolem/indobert-base-uncased (110M params, BERT-base architecture)
  → AutoModelForSequenceClassification(num_labels=8)
  → HuggingFace Trainer API
```

**Training hyperparameters:**
- lr: 2e-5 | warmup_ratio: 0.1 | weight_decay: 0.01
- Batch: 16 | fp16=True | Epochs: 5
- EarlyStoppingCallback(patience=2), best model by F1 macro
- max_len: 128 (matches LSTM for fairness)

### Whisper Fine-tuning (Notebook 01) — ASR

```
openai/whisper-small (244M params)
  → Seq2SeqTrainer
  → Forced decoder: language=indonesian, task=transcribe
  → Labels padded with -100 (cross-entropy ignores padding)
```

**Training hyperparameters:**
- lr: 1e-5 | warmup_steps: 100
- Batch: 8 | gradient_accumulation_steps: 2 (effective batch = 16)
- fp16=True | Epochs: 3 | Best model by WER on val set

**If T4 runs OOM — use LoRA (reduces trainable params from 244M to ~2M):**
```python
from peft import LoraConfig, get_peft_model, TaskType
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=32, lora_alpha=64,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none',
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: trainable params: ~2M || all params: 244M || trainable%: 0.9%
```

---

## Expected Benchmarks

| Model | Task | Metric | Target | Minimum acceptable |
|---|---|---|---|---|
| Whisper baseline | ASR | WER | ~35% | — (reference point) |
| Whisper fine-tuned | ASR | WER | ~18% | Must be < baseline |
| BiLSTM (scratch) | Classification | F1 macro | ~0.68 | > 0.60 |
| IndoBERT (fine-tuned) | Classification | F1 macro | ~0.85 | > 0.80, must beat LSTM |

**The portfolio story only works if IndoBERT clearly beats LSTM.**
If LSTM unexpectedly gets >0.80 F1, investigate class imbalance or data leakage before drawing conclusions.

---

## `src/` Module Reference

```
src/
├── models/
│   ├── lstm_classifier.py    LSTMClassifier, SimpleTokenizer
│   └── mini_transformer.py   MiniTransformerClassifier (optional ablation study)
├── data/
│   ├── audio_dataset.py      CommonVoiceIDDataset, WhisperDataCollator
│   └── text_dataset.py       LSTMTextDataset, BERTTextDataset, load_liputan6_splits()
└── utils/
    ├── metrics.py            compute_wer(), compute_classification_metrics(),
    │                         format_comparison_table()
    └── visualize.py          plot_confusion_matrix(), plot_training_curves(),
                              plot_model_comparison(), plot_wer_comparison()
```

Modules are importable after `sys.path.append(REPO_DIR)` where REPO_DIR = repo root on Drive.

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `AuthorizationError` on Common Voice | HuggingFace token missing | `huggingface-cli login` before `load_dataset` |
| CUDA OOM on Whisper training | T4 ~15GB VRAM tight | Add LoRA (see above), reduce batch to 4 |
| `RuntimeError: Call build_vocab() before encode()` | SimpleTokenizer used before fit | Call `tokenizer.build_vocab(train_texts)` |
| Notebook 03 can't load `lstm_best.pt` | Notebook 02 not run | Run 02 first, verify file on Drive |
| Notebook 04 missing checkpoint | 01 or 03 incomplete | Check both WHISPER_CKPT and INDOBERT_CKPT paths |
| All labels = -1 in id_liputan6 | URL patterns changed upstream | Print sample URLs, adjust `CATEGORIES` keys |
| Class imbalance in categories | URL pattern match misses some | Log Counter(train_labels) after process_split() |
| Gradio share link dead | Colab session expired | Re-run Notebook 04 — links are session-scoped |

---

## Invariants — Do NOT Change Without Understanding Consequences

1. **`CATEGORIES` dict** — must be identical across Notebooks 02, 03, and 04
2. **`process_split()` max_samples** — changing breaks the apples-to-apples comparison
3. **`MAX_LEN = 128`** — shared between LSTM tokenizer and BERT tokenizer; models saved with this assumption
4. **`REPO_DIR` path** — only change if Drive folder structure is different
5. **Label extraction logic** — URL-based, not a field in the dataset; any change here changes what "categories" mean

---

## Verification Checklist

After running all notebooks, verify:

- [ ] `wer_results.json` exists and `finetuned_wer < baseline_wer`
- [ ] `classification_results.json` exists and `indobert.f1_macro > lstm.f1_macro`
- [ ] `lstm_best.pt` loads cleanly
- [ ] `checkpoints/indobert-best/` contains `config.json` + tokenizer files
- [ ] `whisper-id-finetuned/` contains `config.json` + `preprocessor_config.json`
- [ ] Notebook 04 produces a live Gradio `share.gradio.live` URL
- [ ] 4 PNG comparison plots saved to Drive root

---

## Deep Context Files

- `.claude/decisions.md` — why each model/dataset was chosen, alternatives considered
- `.claude/colab-playbook.md` — step-by-step first-time Colab setup + troubleshooting
- `.claude/notebook-internals.md` — detailed logic walkthrough per notebook
