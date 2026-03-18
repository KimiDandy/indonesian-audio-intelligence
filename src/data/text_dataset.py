"""
Dataset utilities for text classification.
Supports both custom LSTM tokenizer and HuggingFace AutoTokenizer (for IndoBERT).
"""

import torch
from torch.utils.data import Dataset

from src.models.lstm_classifier import SimpleTokenizer


class LSTMTextDataset(Dataset):
    """
    Token-ID dataset for LSTM training.
    Expects texts and integer labels.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: SimpleTokenizer,
        max_len: int = 128,
    ):
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.encodings = [tokenizer.encode(t, max_len=max_len) for t in texts]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTTextDataset(Dataset):
    """
    HuggingFace tokenizer dataset for IndoBERT fine-tuning.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int = 128,
    ):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_liputan6_splits(
    max_train: int = 10_000,
    max_val: int = 2_000,
    max_test: int = 2_000,
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int], dict]:
    """
    Load id_liputan6 dataset and map category strings to integer labels.

    Returns (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, label_map)
    """
    from datasets import load_dataset

    ds = load_dataset("id_liputan6", "complete")

    # id_liputan6 has 'title', 'url', 'clean_article', 'extractive_highlights'
    # We derive category from URL path segment (e.g. /bisnis/, /teknologi/)
    import re

    CATEGORY_PATTERNS = {
        "bisnis": 0,
        "teknologi": 1,
        "otomotif": 2,
        "bola": 3,
        "health": 4,
        "lifestyle": 5,
        "showbiz": 6,
        "regional": 7,
    }
    label_map = CATEGORY_PATTERNS

    def extract_label(url: str) -> int:
        for cat, idx in CATEGORY_PATTERNS.items():
            if f"/{cat}/" in url.lower():
                return idx
        return -1  # unknown

    def process_split(split_name: str, max_samples: int):
        split = ds[split_name]
        texts, labels = [], []
        for item in split:
            label = extract_label(item.get("url", ""))
            if label == -1:
                continue
            texts.append(item["clean_article"][:512])
            labels.append(label)
            if len(texts) >= max_samples:
                break
        return texts, labels

    train_texts, train_labels = process_split("train", max_train)
    val_texts, val_labels = process_split("validation", max_val)
    test_texts, test_labels = process_split("test", max_test)

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, label_map
