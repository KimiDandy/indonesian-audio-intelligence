"""
Bidirectional LSTM text classifier — trained from scratch.

Architecture:
    Embedding (random init) → BiLSTM (2 layers) → Dropout → Linear
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout + 0.2)  # slightly higher at output
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))                  # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)                        # hidden: (num_layers*2, batch, hidden_dim)
        # Concat last forward and last backward hidden state
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)         # (batch, hidden_dim*2)
        return self.fc(self.dropout(hidden))                        # (batch, num_classes)


class SimpleTokenizer:
    """Character-agnostic whitespace tokenizer with vocabulary built from training corpus."""

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, max_vocab: int = 10_000):
        self.max_vocab = max_vocab
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._built = False

    def build_vocab(self, texts: list[str]) -> None:
        from collections import Counter

        counter: Counter = Counter()
        for text in texts:
            counter.update(text.lower().split())

        vocab = [self.PAD, self.UNK] + [
            word for word, _ in counter.most_common(self.max_vocab - 2)
        ]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._built = True

    def encode(self, text: str, max_len: int = 128) -> list[int]:
        if not self._built:
            raise RuntimeError("Call build_vocab() before encode()")
        tokens = text.lower().split()[:max_len]
        ids = [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]
        # Pad to max_len
        ids += [self.word2idx[self.PAD]] * (max_len - len(ids))
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
