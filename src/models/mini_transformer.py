"""
Mini Transformer text classifier — trained from scratch.

Serves as an optional alternative / ablation study vs. the BiLSTM.

Architecture:
    Embedding + PositionalEncoding → TransformerEncoder (2 layers) → mean pool → Linear
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MiniTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self._padding_idx = padding_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        padding_mask = x == self._padding_idx                       # (batch, seq_len) True = ignore

        embedded = self.pos_enc(self.embedding(x))                  # (batch, seq_len, d_model)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)  # (batch, seq_len, d_model)

        # Mean pool over non-padding tokens
        mask = (~padding_mask).unsqueeze(-1).float()                # (batch, seq_len, 1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (batch, d_model)

        return self.fc(self.dropout(pooled))                        # (batch, num_classes)
