"""
Measure a checkpoint properly, and log the numbers to MLflow.

    python benchmark.py                          # every checkpoint
    python benchmark.py checkpoints/foo.pt       # just one

WHAT TO MEASURE FOR A GENERATIVE MODEL, AND WHY
-----------------------------------------------
"Accuracy" in the Random-Forest sense does not exist here — there is no single
correct continuation to compare against. These are the metrics that do apply:

  bits/char (BPC)  THE headline number for a character model. It is the loss
                   converted from nats to bits: loss / ln(2). "How many bits of
                   information does the model still need to identify the next
                   character." Well-modelled English is ~1.1-1.5 bpc; uniform
                   guessing over 249 characters is 7.96. Unlike perplexity this
                   stays interpretable across vocabularies.

  perplexity       e^loss. "Effectively torn between how many characters."
                   ONLY comparable between models sharing a tokenizer — a
                   smaller vocabulary lowers it for free.

  top-1 / top-5    How often the true character was the model's best guess, or
                   in its top five. Measured against the SAME raw text for
                   every model, so it is comparable across tokenizers. This is
                   the fair number for v1-vs-v2.

  accuracy vs      Does a longer context actually help? Accuracy at position 0
  context length   (no context) versus position 255 (255 chars of context).
                   If the curve is flat, the context window is being wasted and
                   block_size is not earning its compute.

  throughput       Generation speed in chars/sec, and one forward pass in ms.
                   A quality number without a cost number cannot support a
                   deployment decision.

Every metric goes to MLflow under its own run so versions stay comparable.
"""
import math
import os
import sys
import time

import mlflow
import torch

from medresearch import config
from medresearch.gpt import load_checkpoint

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HELDOUT_CHARS = 400_000          # from the tail of the corpus = the val split
N_WINDOWS = 256
LN2 = math.log(2)


def heldout_text():
    """Tail of the corpus — inside train.py's 10% validation split, so no
    checkpoint has ever trained on it."""
    size = os.path.getsize(config.MEDICAL_TEXT)
    with open(config.MEDICAL_TEXT, errors="ignore") as f:
        f.seek(max(0, size - HELDOUT_CHARS * 2))
        tail = f.read()
    return tail[len(tail) // 2:]


def encode(text, stoi):
    # each checkpoint carries its own tokenizer; always use the model's own
    unk = stoi.get("�", 0)
    return torch.tensor([stoi.get(c, unk) for c in text],
                        dtype=torch.long, device=DEVICE)


@torch.no_grad()
def quality(model, ids, n_windows=N_WINDOWS, batch=16):
    B = model.cfg.block_size
    if len(ids) <= B + 1:
        return None
    g = torch.Generator().manual_seed(0)          # same windows for every model
    starts = torch.randint(0, len(ids) - B - 1, (n_windows,), generator=g)

    tot_loss = hit1 = hit5 = last_hits = seen = 0
    by_pos = torch.zeros(B, device=DEVICE)        # accuracy at each position
    for i in range(0, n_windows, batch):
        s = starts[i:i + batch]
        x = torch.stack([ids[j:j + B] for j in s])
        y = torch.stack([ids[j + 1:j + B + 1] for j in s])
        logits, loss = model(x, y)
        top5 = logits.topk(5, dim=-1).indices
        tgt = y.unsqueeze(-1)
        c1 = (top5[..., :1] == tgt).any(-1)       # (b, B) bool
        c5 = (top5 == tgt).any(-1)

        tot_loss += loss.item() * x.size(0)
        hit1 += c1.sum().item()
        hit5 += c5.sum().item()
        last_hits += c1[:, -1].sum().item()
        by_pos += c1.float().sum(0)
        seen += x.size(0)

    mean_loss = tot_loss / seen
    by_pos = (by_pos / seen).tolist()
    return {
        "heldout_loss": mean_loss,
        "bits_per_char": mean_loss / LN2,
        "perplexity": math.exp(mean_loss),
        "top1_last": last_hits / seen,
        "top1_all": hit1 / (seen * B),
        "top5_all": hit5 / (seen * B),
        # does more context help? first 16 positions vs last 16
        "top1_first16": sum(by_pos[:16]) / 16,
        "top1_last16": sum(by_pos[-16:]) / 16,
        "windows": seen,
    }


@torch.no_grad()
def speed(model, ids, gen_chars=200):
    """Generation throughput and single-forward latency."""
    B = model.cfg.block_size
    prompt = ids[:B].unsqueeze(0)

    for _ in range(3):                            # warm up the MPS kernels
        model(prompt)
    if DEVICE == "mps":
        torch.mps.synchronize()

    t = time.perf_counter()
    for _ in range(10):
        model(prompt)
    if DEVICE == "mps":
        torch.mps.synchronize()
    fwd_ms = (time.perf_counter() - t) / 10 * 1000

    t = time.perf_counter()
    n = sum(1 for _ in model.stream(prompt, gen_chars, 0.8, 40))
    if DEVICE == "mps":
        torch.mps.synchronize()
    gen_s = time.perf_counter() - t

    return {
        "forward_ms": fwd_ms,
        "gen_chars_per_sec": n / gen_s,
        # no KV cache: each new char re-runs the whole context, so this is the
        # cost of that design choice, in one number
        "ms_per_generated_char": gen_s / n * 1000,
    }


def run(path, text):
    model, cfg, stoi, itos, meta = load_checkpoint(path, DEVICE)
    ids = encode(text, stoi)
    q = quality(model, ids)
    s = speed(model, ids)
    return {
        "checkpoint": os.path.basename(path),
        "version": meta.get("version", "v1"),
        "params_millions": model.n_params() / 1e6,
        "disk_mb": os.path.getsize(path) / 1e6,
        "block_size": cfg.block_size,
        "vocab_size": cfg.vocab_size,
        **(q or {}), **s,
    }


if __name__ == "__main__":
    paths = sys.argv[1:] or [str(p) for p in config.find_checkpoints()]
    text = heldout_text()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"{config.EXPERIMENT_NAME}-benchmark")

    rows = []
    for p in paths:
        print(f"benchmarking {p} …")
        r = run(p, text)
        rows.append(r)
        with mlflow.start_run(run_name=f"bench-{r['checkpoint']}"):
            mlflow.log_params({"checkpoint": r["checkpoint"], "version": r["version"],
                               "block_size": r["block_size"], "vocab_size": r["vocab_size"]})
            mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, (int, float))})

    hdr = ["checkpoint", "params_millions", "bits_per_char", "perplexity",
           "top1_last", "top5_all", "top1_first16", "top1_last16",
           "gen_chars_per_sec"]
    w = [26, 8, 8, 8, 9, 8, 12, 11, 9]
    print("\n" + "  ".join(h[:c].ljust(c) for h, c in zip(hdr, w)))
    print("-" * (sum(w) + 2 * len(w)))
    for r in sorted(rows, key=lambda x: -x.get("top1_last", 0)):
        cells = [r["checkpoint"][:26].ljust(26)]
        for k, c in zip(hdr[1:], w[1:]):
            v = r.get(k, 0)
            cells.append((f"{v*100:.1f}%" if k.startswith("top") else f"{v:.2f}").rjust(c))
        print("  ".join(cells))

    print("\nbits_per_char: lower is better. ~1.1-1.5 = well-modelled English,"
          " 7.96 = random guessing over 249 chars.")
    print("top1_first16 vs top1_last16: if these are close, the long context is"
          " not paying for itself.")
