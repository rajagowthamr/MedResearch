"""Corpus loading, the character vocabulary, and batching.

Extracted from train.py so the vocabulary is built by ONE function. The rule
that matters: a checkpoint's embeddings are only meaningful against the exact
stoi that produced them, so `build_char_vocab` must be deterministic given
(text, min_freq). It is -- np.unique returns sorted codepoints -- which is what
lets train.py refuse a RESUME whose stoi differs instead of silently scrambling
every embedding.

Note that finetune and benchmark do NOT call build_char_vocab: they read stoi
back out of the checkpoint they are loading, which is always the correct source.
Only pretraining, which has no checkpoint yet, derives a vocabulary from text.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

UNKNOWN = "�"  # the "�" replacement character; last slot in every vocab


def read_text(path: str | Path, limit: int | None = None) -> str:
    """Read a corpus. errors='ignore' because the scrape contains stray bytes."""
    with open(path, errors="ignore") as f:
        return f.read() if limit is None else f.read(limit)


def build_char_vocab(text: str, min_freq: int = 10) -> tuple[list[str], dict, dict]:
    """Character vocabulary, pruned by frequency.

    Codepoints are counted in one vectorised pass over a utf-32 view rather than
    a 60-million-iteration Python loop.

    min_freq drops the long tail: v1 kept all 1103 characters including ones
    seen exactly once, spending embedding capacity on scrape artefacts. At
    min_freq=10 the medical corpus yields 249 characters.
    """
    codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    uniq, counts = np.unique(codepoints, return_counts=True)
    chars = [chr(c) for c in uniq[counts >= min_freq]] + [UNKNOWN]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = dict(enumerate(chars))
    return chars, stoi, itos


def encode_corpus(text: str, stoi: dict) -> torch.Tensor:
    """Whole corpus -> int16 tensor, via a codepoint lookup table.

    int16, not int64: 60M tokens is 120MB instead of 480MB. Callers cast to
    .long() per batch, because nn.Embedding requires int64 but only for the
    few thousand tokens actually in flight.
    """
    codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    unk_id = stoi[UNKNOWN]
    lut = np.full(int(codepoints.max()) + 1, unk_id, dtype=np.int16)
    for c, i in stoi.items():
        if ord(c) < len(lut):
            lut[ord(c)] = i
    return torch.from_numpy(lut[codepoints])


def encode_text(text: str, stoi: dict) -> torch.Tensor:
    """Short string -> int64 tensor. For prompts and small files, where the
    lookup-table setup in encode_corpus would cost more than it saves."""
    unk = stoi.get(UNKNOWN, 0)
    return torch.tensor([stoi.get(c, unk) for c in text], dtype=torch.long)


def split_train_val(data: torch.Tensor, val_fraction: float = 0.1):
    """Chronological split, not random: a random split would leak, because
    adjacent windows of the same article would land on both sides."""
    n = int((1.0 - val_fraction) * len(data))
    return data[:n], data[n:]


class BatchSampler:
    """Random fixed-length windows from a token tensor.

    The corpus is parked on the GPU once (60.7M int16 tokens = 121MB, nothing
    against 12GB). Otherwise every step ships batch_size slices across PCIe and
    on a rented GPU that copy, not the arithmetic, becomes the bottleneck.
    """

    def __init__(self, train_data, val_data, block_size: int, batch_size: int, device: str):
        # Only worth relocating for cuda; mps and cpu share memory with the host
        # so there is no transfer to avoid.
        self.data_device = device if device == "cuda" else "cpu"
        self.train = train_data.to(self.data_device)
        self.val = val_data.to(self.data_device)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self._offsets = torch.arange(block_size + 1, device=self.data_device)

    def __call__(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.train if split == "train" else self.val
        ix = torch.randint(
            len(d) - self.block_size - 1, (self.batch_size,), device=self.data_device
        )
        # One vectorised gather beats batch_size Python slices plus a stack.
        chunk = d[ix[:, None] + self._offsets].to(self.device).long()
        return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
