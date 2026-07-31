"""
Train the from-scratch GPT and publish it as a numbered model version.

HOW VERSIONING WORKS HERE
-------------------------
v1 of this script overwrote gpt_medical.pt every run and called
mlflow.log_artifact().  That gave you loss curves but no versions: nothing
recorded which file came from which run, and yesterday's model was gone.

Now every run does three things:
  1. writes checkpoints/gpt_medical_<VERSION>.pt (old versions are never
     overwritten, and the checkpoint carries its own architecture config)
  2. calls mlflow.pytorch.log_model(registered_model_name=...), which enters
     the model in the MLflow **Model Registry** -> Version 1, 2, 3, ...
     (that's the "Model registry" tab in the sidebar of your MLflow UI)
  3. moves the "champion" alias onto this version ONLY if it beat the best
     val_loss ever registered.  So "champion" always means "our best model",
     independent of which run happened to be last.

To make the NEXT version: change something in CONFIG (or model.py), bump
VERSION, re-run.  MLflow keeps both, and the UI lets you overlay their curves.

Registry needs a database backend, which you already have — MLflow is
auto-detecting ./mlflow.db, so it works locally with no server.
"""
import math
import os
import time

import numpy as np
import torch
import mlflow
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

from model import GPT, GPTConfig, ARCH_VERSION, save_checkpoint

# ------------------------------------------------------------------
# 1. CONFIG  -  every knob in ONE place.
#    Change these, bump VERSION, re-run = a new registered model version.
# ------------------------------------------------------------------
MODEL_NAME = "medresearch-gpt"     # the name in the MLflow Model Registry
VERSION = "v2"                     # your label for this attempt

CONFIG = {
    # ---- training ----
    "batch_size": 32,
    "max_iters": 5000,
    "eval_interval": 250,
    "eval_iters": 50,
    "learning_rate": 1e-3,         # v1: 3e-4 flat. Higher peak + decay = faster.
    "min_lr": 1e-4,                # cosine decay floor
    "warmup_iters": 200,           # ramp up slowly so early steps don't diverge
    "weight_decay": 0.1,           # regularisation, on matrices only
    "grad_clip": 1.0,              # stops a single bad batch wrecking the run
    "seed": 1337,
    # ---- architecture (v1 values in comments) ----
    "block_size": 256,             # v1: 12  <- the big one
    "n_embd": 256,                 # v1: 192
    "n_head": 8,                   # v1: 6
    "n_layer": 6,                  # v1: 6
    "dropout": 0.2,                # v1: 0.1
    "activation": "gelu",          # v1: relu
    "tie_weights": True,           # v1: False
    # ---- data ----
    "dataset": "gamino/wiki_medical_terms",
    "min_char_freq": 10,           # v1 kept all 1103 chars, incl. ones seen once
}

torch.manual_seed(CONFIG["seed"])
device = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
ckpt_path = os.path.join(CKPT_DIR, f"gpt_medical_{VERSION}.pt")

# ------------------------------------------------------------------
# 2. DATA + TOKENIZER
# ------------------------------------------------------------------
with open("medical_text.txt", "r") as f:
    text = f.read()

# Codepoints in one vectorised shot instead of a 60M-iteration Python loop.
codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
uniq, counts = np.unique(codepoints, return_counts=True)

# Prune the long tail: characters seen fewer than min_char_freq times are
# stray unicode from the scrape.  v1 spent embedding capacity on ~700 of them.
keep = uniq[counts >= CONFIG["min_char_freq"]]
chars = [chr(c) for c in keep] + ["�"]          # last slot = "unknown"
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
unk_id = stoi["�"]

lut = np.full(int(uniq.max()) + 1, unk_id, dtype=np.int16)   # codepoint -> token id
for c, i in stoi.items():
    if ord(c) < len(lut):
        lut[ord(c)] = i

# int16 not int64: 60M tokens is 120MB instead of 480MB of RAM.
data = torch.from_numpy(lut[codepoints])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"vocab: {vocab_size} chars (v1 had 1103) | tokens: {len(data)/1e6:.1f}M")

cfg = GPTConfig(
    vocab_size=vocab_size,
    **{k: CONFIG[k] for k in
       ("block_size", "n_embd", "n_head", "n_layer", "dropout",
        "activation", "tie_weights")},
)


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (CONFIG["batch_size"],))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])
    # .long() here, because embedding lookups need int64
    return x.to(device).long(), y.to(device).long()


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(CONFIG["eval_iters"])
        for k in range(CONFIG["eval_iters"]):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(it):
    """Linear warmup, then cosine decay to min_lr.

    v1 held the LR flat the whole way, which both learns too slowly at the
    start and bounces around the optimum at the end."""
    if it < CONFIG["warmup_iters"]:
        return CONFIG["learning_rate"] * (it + 1) / CONFIG["warmup_iters"]
    progress = (it - CONFIG["warmup_iters"]) / max(
        1, CONFIG["max_iters"] - CONFIG["warmup_iters"])
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return CONFIG["min_lr"] + cosine * (CONFIG["learning_rate"] - CONFIG["min_lr"])


# ------------------------------------------------------------------
# 3. BUILD MODEL
# ------------------------------------------------------------------
model = GPT(cfg).to(device)
n_params = model.n_params()

# Decay matrices, not biases/LayerNorms — decaying a bias just biases it to 0.
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": CONFIG["weight_decay"]},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=CONFIG["learning_rate"], betas=(0.9, 0.99), eps=1e-8,
)
print(f"Device: {device} | Parameters: {n_params/1e6:.2f}M | arch {ARCH_VERSION} {VERSION}")

# ------------------------------------------------------------------
# 4. TRAIN INSIDE AN MLFLOW RUN
# ------------------------------------------------------------------
mlflow.set_experiment("medresearch-gpt")

with mlflow.start_run(run_name=f"{VERSION}-b{cfg.block_size}-e{cfg.n_embd}-l{cfg.n_layer}"):
    mlflow.set_tags({
        "version": VERSION,
        "arch_version": ARCH_VERSION,
        "attention": "fused-sdpa",
    })
    mlflow.log_params(CONFIG)
    mlflow.log_params({
        "device": device,
        "vocab_size": vocab_size,
        "total_chars": len(data),
        "n_params_millions": round(n_params / 1e6, 3),
        "tokens_per_step": CONFIG["batch_size"] * cfg.block_size,
    })

    best_val = float("inf")
    best_iter = -1
    t0 = time.time()

    for it in range(CONFIG["max_iters"] + 1):
        lr = lr_at(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # eval at step 0, every eval_interval, AND at the very last step
        # (v1's loop ended at 2700 and never measured the finished model)
        if it % CONFIG["eval_interval"] == 0 or it == CONFIG["max_iters"]:
            losses = estimate_loss()
            elapsed = time.time() - t0
            print(f"step {it}: train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {elapsed/60:.1f}m")
            mlflow.log_metrics({
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "overfit_gap": losses["val"] - losses["train"],
                "val_perplexity": math.exp(losses["val"]),   # readable: "torn between N chars"
                "learning_rate": lr,
            }, step=it)

            # keep the BEST model, not the last one. Late steps can regress.
            if losses["val"] < best_val:
                best_val, best_iter = losses["val"], it
                save_checkpoint(ckpt_path, model, stoi, itos,
                                extra={"val_loss": best_val, "iter": it,
                                       "version": VERSION, "config_train": CONFIG})

        if it == CONFIG["max_iters"]:
            break

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

    mins = (time.time() - t0) / 60
    mlflow.log_metrics({
        "best_val_loss": best_val,
        "best_val_perplexity": math.exp(best_val),
        "best_iter": best_iter,
        "train_minutes": mins,
    })
    print(f"\nBest val {best_val:.4f} (ppl {math.exp(best_val):.2f}) at step {best_iter}")
    print(f"Saved {ckpt_path}  ({mins:.1f} min)")

    # ------------------------------------------------------------------
    # 5. REGISTER  -  this is what creates Version 1 / 2 / 3 in the registry
    # ------------------------------------------------------------------
    # Reload the best weights so we register the checkpoint we actually kept,
    # not whatever the final step left in memory.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    mlflow.log_artifact(ckpt_path)

    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.int64), (-1, cfg.block_size), "idx")]),
        outputs=Schema([TensorSpec(np.dtype(np.float32),
                                   (-1, cfg.block_size, vocab_size), "logits")]),
    )
    try:
        mlflow.pytorch.log_model(
            model, name="model",
            registered_model_name=MODEL_NAME,
            code_paths=["model.py"],          # so the class can be rebuilt later
            signature=signature,
            serialization_format="pickle",    # pt2 can't trace a (logits, loss) return
        )
        client = MlflowClient()
        mine = max(client.search_model_versions(f"name='{MODEL_NAME}'"),
                   key=lambda v: int(v.version))
        client.set_model_version_tag(MODEL_NAME, mine.version, "val_loss", f"{best_val:.4f}")
        client.set_model_version_tag(MODEL_NAME, mine.version, "version", VERSION)

        # promote to "champion" only if this is the best model we've ever had
        others = [float(v.tags["val_loss"])
                  for v in client.search_model_versions(f"name='{MODEL_NAME}'")
                  if v.version != mine.version and "val_loss" in v.tags]
        if not others or best_val < min(others):
            client.set_registered_model_alias(MODEL_NAME, "champion", mine.version)
            print(f"Registered {MODEL_NAME} v{mine.version} -> promoted to @champion")
        else:
            print(f"Registered {MODEL_NAME} v{mine.version} "
                  f"(champion kept: best existing val_loss {min(others):.4f})")
    except Exception as e:
        # never let a registry hiccup lose a finished training run
        print(f"WARNING: registry step failed ({e}). Checkpoint is safe at {ckpt_path}")
