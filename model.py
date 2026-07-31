"""
The from-scratch GPT.  Architecture version: v2.

What changed from v1, and WHY:

 1. block_size 12 -> 256.  This is the single biggest quality win.  v1 could
    only ever see 12 characters, so it could not learn a word longer than a
    word, let alone a clinical sentence.  Loss stalling near 1.87 was the
    model hitting that ceiling, not the training being wrong.
 2. Fused multi-head attention through F.scaled_dot_product_attention (Flash
    Attention).  v1 ran n_layer * n_head = 36 separate attention modules in a
    Python loop per forward pass.  This does all heads in one batched kernel:
    identical math, far less overhead — which is what makes block_size=256
    affordable at all.
 3. GELU instead of ReLU, and weight tying (the token embedding shares its
    matrix with the output head).  Both are standard in real GPTs; tying also
    removes vocab_size * n_embd parameters for free.
 4. Scaled residual init — keeps activations from blowing up as n_layer grows.
 5. The architecture now lives in a GPTConfig that is SAVED INSIDE the
    checkpoint.  v1 hardcoded its shape here as module globals, so changing a
    number in this file silently broke every checkpoint on disk.  Now each
    checkpoint describes itself and old versions stay loadable forever.
 6. generate() takes temperature / top_k, and can stream token by token.
"""
from dataclasses import dataclass, asdict, field
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

ARCH_VERSION = "v2"


@dataclass
class GPTConfig:
    """Everything needed to rebuild this model. Gets saved into the checkpoint."""
    vocab_size: int
    block_size: int = 256     # context window, in characters
    n_embd: int = 256         # embedding width  (must be divisible by n_head)
    n_head: int = 8           # attention heads  -> head_size = 256/8 = 32
    n_layer: int = 6          # transformer blocks
    dropout: float = 0.2
    activation: str = "gelu"  # "relu" reproduces v1
    tie_weights: bool = True  # False reproduces v1

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head


# The shape v1 hardcoded as module globals. Kept so v1 checkpoints still load.
LEGACY_V1_CONFIG = dict(block_size=12, n_embd=192, n_head=6, n_layer=6,
                        dropout=0.1, activation="relu", tie_weights=False)


class CausalSelfAttention(nn.Module):
    """All heads at once: one qkv projection, one Flash-Attention call."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide evenly by n_head"
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        # one matrix producing query, key and value together
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size) so heads become a batch dim
        view = lambda t: t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k, v = view(q), view(k), view(v)
        # is_causal=True applies the "can't see the future" mask internally,
        # so we no longer need a tril buffer.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # re-merge the heads
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    """Position-wise 2-layer MLP."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        act = nn.GELU() if cfg.activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            act,
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention + feed-forward, both with residuals."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ff = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """embeddings -> transformer blocks -> vocabulary logits."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

        self.apply(self._init_weights)
        # Residual layers add into the stream n_layer times over, so shrink
        # their output init by 1/sqrt(2*n_layer) to keep the variance steady.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        if cfg.tie_weights:
            # same matrix used to look a character up and to score it
            self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.position_embedding.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"sequence of {T} exceeds block_size {self.cfg.block_size}"
        tok = self.token_embedding(idx)                                    # (B,T,C)
        pos = self.position_embedding(torch.arange(T, device=idx.device))   # (T,C)
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                              # (B,T,vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, logits.size(-1)), targets.reshape(B * T)
            )
        return logits, loss

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def stream(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Yield one new token id at a time (lets a UI print as it generates).

        temperature <1 = safer/more repetitive, >1 = more inventive.
        top_k = only ever sample from the k most likely next characters.
        """
        self.eval()
        idx = idx.to(next(self.parameters()).device)   # caller may hand us a CPU tensor
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]     # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                cutoff = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            yield next_id

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Same thing, but collected into one finished sequence."""
        for next_id in self.stream(idx, max_new_tokens, temperature, top_k):
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    @torch.no_grad()
    def evaluate_text(self, ids):
        """Quality of this model on real text, as a next-character exam.

        Returns (perplexity, top1_accuracy, top5_accuracy).  Perplexity is
        "on average, how many characters was it torn between?" — 1.0 is
        perfect, vocab_size is a model that learned nothing.
        """
        self.eval()
        T = min(len(ids) - 1, self.cfg.block_size)
        x = ids[:T].unsqueeze(0).to(next(self.parameters()).device)
        y = ids[1:T + 1].unsqueeze(0).to(x.device)
        logits, loss = self(x, y)
        ranked = logits[0].topk(5, dim=-1).indices          # (T, 5)
        target = y[0].unsqueeze(-1)                          # (T, 1)
        top1 = (ranked[:, :1] == target).any(-1).float().mean().item()
        top5 = (ranked == target).any(-1).float().mean().item()
        return math.exp(loss.item()), top1, top5


# ------------------------------------------------------------------
# checkpoint I/O  — the part that makes versioning actually work
# ------------------------------------------------------------------
def save_checkpoint(path, model, stoi, itos, extra=None):
    torch.save({
        "arch_version": ARCH_VERSION,
        "config": asdict(model.cfg),
        "model": model.state_dict(),
        "stoi": stoi, "itos": itos,
        "vocab_size": model.cfg.vocab_size,
        **(extra or {}),
    }, path)


def _convert_v1_state_dict(sd, cfg):
    """v1 stored every attention head as its own key/query/value Linear
    (blocks.0.sa.heads.3.key.weight).  Concatenate them into the single fused
    qkv matrix this version uses — the math is unchanged, only the layout."""
    new = {k: v for k, v in sd.items() if ".sa." not in k}
    for i in range(cfg.n_layer):
        parts = [
            torch.cat([sd[f"blocks.{i}.sa.heads.{h}.{n}.weight"]
                       for h in range(cfg.n_head)], dim=0)
            for n in ("query", "key", "value")           # fused order is q,k,v
        ]
        new[f"blocks.{i}.attn.qkv.weight"] = torch.cat(parts, dim=0)
        new[f"blocks.{i}.attn.out_proj.weight"] = sd[f"blocks.{i}.sa.proj.weight"]
        new[f"blocks.{i}.attn.out_proj.bias"] = sd[f"blocks.{i}.sa.proj.bias"]
    return new


def load_checkpoint(path, device="cpu"):
    """Load ANY version of this model. Returns (model, cfg, stoi, itos, meta)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" in ckpt:                       # v2 and later: self-describing
        cfg = GPTConfig(**ckpt["config"])
        sd = ckpt["model"]
    else:                                      # v1: shape lived in model.py
        cfg = GPTConfig(vocab_size=ckpt["vocab_size"], **LEGACY_V1_CONFIG)
        sd = _convert_v1_state_dict(ckpt["model"], cfg)
    model = GPT(cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    meta = {k: v for k, v in ckpt.items()
            if k not in ("model", "stoi", "itos", "config")}
    meta.setdefault("arch_version", "v1")
    return model, cfg, ckpt["stoi"], ckpt["itos"], meta
