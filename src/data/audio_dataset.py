"""
Dataset utilities for audio data — wraps HuggingFace datasets for Whisper fine-tuning.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer


@dataclass
class WhisperDataCollator:
    """
    Collates audio samples into batches for Whisper seq2seq training.
    Pads/truncates mel features and decoder input IDs consistently.
    """

    processor_feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor_feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip decoder start token if prepended
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class CommonVoiceIDDataset(Dataset):
    """
    Wraps Mozilla Common Voice (Indonesian) HuggingFace dataset.

    Each item:
        input_features: mel spectrogram (80, 3000) as float32
        labels: tokenized transcript IDs
    """

    def __init__(
        self,
        hf_dataset,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        sample_rate: int = 16_000,
    ):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        audio = item["audio"]

        # Resample if necessary (Common Voice is typically 48kHz)
        array = audio["array"]
        sr = audio["sampling_rate"]
        if sr != self.sample_rate:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=self.sample_rate)

        input_features = self.feature_extractor(
            array,
            sampling_rate=self.sample_rate,
            return_tensors="np",
        ).input_features[0]

        labels = self.tokenizer(item["sentence"]).input_ids

        return {"input_features": input_features, "labels": labels}
