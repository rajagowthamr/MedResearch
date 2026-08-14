"""The character vocabulary and batching.

The property that actually matters here is DETERMINISM: a checkpoint's
embeddings are only meaningful against the exact stoi that produced them, so
build_char_vocab must return the same mapping for the same input. train.py
relies on this to refuse a mismatched RESUME rather than scramble every
embedding row.
"""
import torch

from medresearch.gpt.data import (
    UNKNOWN,
    BatchSampler,
    build_char_vocab,
    encode_corpus,
    encode_text,
    split_train_val,
)

SAMPLE = "the patient presented with acute pneumonia. " * 50


def test_vocab_is_deterministic():
    a_chars, a_stoi, _ = build_char_vocab(SAMPLE, min_freq=1)
    b_chars, b_stoi, _ = build_char_vocab(SAMPLE, min_freq=1)
    assert a_chars == b_chars
    assert a_stoi == b_stoi


def test_unknown_token_is_last():
    chars, stoi, itos = build_char_vocab(SAMPLE, min_freq=1)
    assert chars[-1] == UNKNOWN
    assert stoi[UNKNOWN] == len(chars) - 1
    assert itos[len(chars) - 1] == UNKNOWN


def test_min_freq_prunes_the_long_tail():
    text = ("a" * 100) + "z"          # 'z' appears once
    kept_all, _, _ = build_char_vocab(text, min_freq=1)
    pruned, _, _ = build_char_vocab(text, min_freq=10)
    assert "z" in kept_all
    assert "z" not in pruned
    assert "a" in pruned


def test_encode_decode_round_trip():
    _, stoi, itos = build_char_vocab(SAMPLE, min_freq=1)
    ids = encode_corpus(SAMPLE, stoi)
    assert "".join(itos[int(i)] for i in ids) == SAMPLE


def test_encode_corpus_is_int16_to_save_memory():
    """int64 would quadruple a 60M-token corpus from 121MB to 485MB."""
    _, stoi, _ = build_char_vocab(SAMPLE, min_freq=1)
    assert encode_corpus(SAMPLE, stoi).dtype is torch.int16


def test_unseen_characters_map_to_unknown():
    _, stoi, _ = build_char_vocab("abc", min_freq=1)
    ids = encode_text("abcZ", stoi)
    assert int(ids[-1]) == stoi[UNKNOWN]


def test_split_is_chronological_not_random():
    """A random split leaks: adjacent windows of one article would land on both
    sides, and val loss would measure memorisation."""
    data = torch.arange(1000, dtype=torch.int16)
    train, val = split_train_val(data, val_fraction=0.1)
    assert len(train) == 900 and len(val) == 100
    assert int(train[-1]) + 1 == int(val[0])


def test_batch_shapes_and_offset_by_one():
    _, stoi, _ = build_char_vocab(SAMPLE, min_freq=1)
    data = encode_corpus(SAMPLE, stoi)
    train, val = split_train_val(data)
    sampler = BatchSampler(train, val, block_size=8, batch_size=4, device="cpu")
    x, y = sampler("train")
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype is torch.int64, "nn.Embedding requires int64"
    # y is x shifted one position: next-character prediction
    assert torch.equal(x[:, 1:], y[:, :-1])
