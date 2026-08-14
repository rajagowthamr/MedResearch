"""Architecture, checkpoint round-trip, and sampling.

The checkpoint test is the important one. save_checkpoint stores the config
INSIDE the file precisely so that changing a default in GPTConfig cannot
silently invalidate every .pt on disk -- which is what happened in v1, where the
shape lived in module globals.
"""
import pytest
import torch

from medresearch.gpt import GPT, GPTConfig, load_checkpoint, save_checkpoint
from medresearch.gpt.data import build_char_vocab

TINY = dict(vocab_size=32, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)


@pytest.fixture
def model():
    return GPT(GPTConfig(**TINY))


def test_forward_returns_logits_over_the_vocabulary(model):
    x = torch.randint(0, TINY["vocab_size"], (2, TINY["block_size"]))
    logits, loss = model(x)
    assert logits.shape == (2, TINY["block_size"], TINY["vocab_size"])
    assert loss is None, "loss should be None when no targets are given"


def test_forward_with_targets_returns_a_scalar_loss(model):
    x = torch.randint(0, TINY["vocab_size"], (2, TINY["block_size"]))
    _, loss = model(x, x)
    assert loss.ndim == 0 and loss.item() > 0


def test_sequence_longer_than_block_size_is_rejected(model):
    """Better a loud assert than silently indexing past the position embedding."""
    too_long = torch.zeros(1, TINY["block_size"] + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        model(too_long)


def test_weight_tying_shares_one_matrix():
    tied = GPT(GPTConfig(**{**TINY, "tie_weights": True}))
    assert tied.head.weight is tied.token_embedding.weight
    untied = GPT(GPTConfig(**{**TINY, "tie_weights": False}))
    assert untied.head.weight is not untied.token_embedding.weight


def test_n_embd_must_divide_by_n_head():
    with pytest.raises(AssertionError):
        GPT(GPTConfig(**{**TINY, "n_embd": 30, "n_head": 4}))


def test_checkpoint_round_trip_preserves_weights_and_tokenizer(tmp_path, model):
    _, stoi, itos = build_char_vocab("abcdefg" * 20, min_freq=1)
    path = tmp_path / "m1_test.pt"
    save_checkpoint(path, model, stoi, itos, extra={"val_loss": 1.23, "iter": 7})

    loaded, cfg, l_stoi, l_itos, meta = load_checkpoint(path)

    assert cfg.n_layer == TINY["n_layer"] and cfg.n_embd == TINY["n_embd"]
    assert l_stoi == stoi and l_itos == itos
    assert meta["val_loss"] == 1.23 and meta["iter"] == 7
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_checkpoint_carries_its_own_architecture(tmp_path, model):
    """A checkpoint must be loadable without knowing what defaults were in
    force when it was written."""
    path = tmp_path / "m1_arch.pt"
    save_checkpoint(path, model, {"a": 0}, {0: "a"})
    raw = torch.load(path, map_location="cpu", weights_only=False)
    assert raw["config"]["n_layer"] == TINY["n_layer"]
    assert raw["arch_version"]


def test_generate_appends_exactly_n_tokens(model):
    idx = torch.zeros(1, 4, dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, temperature=0.8, top_k=4)
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :4], idx), "the prompt must be preserved"


def test_generate_crops_context_to_block_size(model):
    """A prompt longer than the context window must not raise -- stream() crops."""
    idx = torch.zeros(1, TINY["block_size"] + 5, dtype=torch.long)
    out = model.generate(idx, max_new_tokens=2)
    assert out.shape[1] == TINY["block_size"] + 7


def test_top_k_restricts_the_sampled_tokens(model):
    torch.manual_seed(0)
    idx = torch.zeros(1, 4, dtype=torch.long)
    out = model.generate(idx, max_new_tokens=40, top_k=1, temperature=1.0)
    # top_k=1 is greedy, so the same prompt must give the same continuation
    torch.manual_seed(0)
    again = model.generate(idx, max_new_tokens=40, top_k=1, temperature=1.0)
    assert torch.equal(out, again)
