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
import contextlib
import math
import os
import time

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

from medresearch import config
from medresearch.gpt import ARCH_VERSION, GPT, GPTConfig, save_checkpoint
from medresearch.gpt.data import (
    BatchSampler,
    build_char_vocab,
    encode_corpus,
    read_text,
    split_train_val,
)

# ------------------------------------------------------------------
# 1. CONFIG  -  every knob in ONE place.
#    Change these, bump VERSION, re-run = a new registered model version.
# ------------------------------------------------------------------
MODEL_NAME = config.REGISTERED_MODEL_NAME   # the name in the MLflow Model Registry
VERSION = "v2"                              # your label for this attempt

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

# Environment overrides, so a Colab/Kaggle cell can retune without editing this
# file (a file edit is lost every time the notebook's disk is recycled):
#   MAX_ITERS=20000 BATCH_SIZE=64 VERSION=v4 python train.py
# Every override is logged to MLflow via log_params(CONFIG), so a tuned run is
# still fully reproducible from its run page.
VERSION = os.environ.get("VERSION", VERSION)
for _key in CONFIG:
    _env = os.environ.get(_key.upper())
    if _env is None:
        continue
    _old = CONFIG[_key]
    # bool() of any non-empty string is True, so "FALSE" would read as True.
    CONFIG[_key] = (_env.lower() in ("1", "true", "yes") if isinstance(_old, bool)
                    else type(_old)(_env))
    print(f"override: {_key} {_old} -> {CONFIG[_key]}")

torch.manual_seed(CONFIG["seed"])

# cuda FIRST, so this file runs unchanged on a free Colab/Kaggle T4. Without
# the cuda branch a GPU notebook silently falls through to "cpu" and trains
# ~40x slower than the Mac it was written on — the worst possible outcome,
# because nothing errors.
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

# Mixed precision, cuda only. MPS autocast exists but is slower than fp32 for
# a model this small, and CPU fp16 is pointless.
#   bf16 (Ampere+: A100, L4, RTX 30xx) needs no loss scaling.
#   fp16 (T4, P100 — what the free tiers actually give you) does.
AMP = device == "cuda"
amp_dtype = (torch.bfloat16 if AMP and torch.cuda.is_bf16_supported()
             else torch.float16)
scaler = torch.amp.GradScaler(device, enabled=AMP and amp_dtype is torch.float16)
autocast = (lambda: torch.autocast(device, dtype=amp_dtype)) if AMP \
    else contextlib.nullcontext
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True     # free speed on Ampere+
    torch.backends.cudnn.allow_tf32 = True

# Continue training an existing checkpoint instead of starting from random.
#   RESUME=checkpoints/gpt_medical_v2-best.pt python train.py
# NOTE: save_checkpoint() stores weights but NOT optimizer state, so AdamW's
# moments restart from zero. That is what warmup_iters is for — leave it at a
# few hundred steps or the first resumed step will kick the model backwards.
RESUME = os.environ.get("RESUME", "")

config.ensure_dirs()
ckpt_path = str(config.checkpoint_path(VERSION))

# ------------------------------------------------------------------
# 2. DATA + TOKENIZER
# ------------------------------------------------------------------
text = read_text(config.require(
    config.MEDICAL_TEXT, "Build it with: python tools/fetch_wiki_medical.py"))

chars, stoi, itos = build_char_vocab(text, CONFIG["min_char_freq"])
vocab_size = len(chars)
data = encode_corpus(text, stoi)
train_data, val_data = split_train_val(data)
print(f"vocab: {vocab_size} chars (v1 had 1103) | tokens: {len(data)/1e6:.1f}M")

cfg = GPTConfig(
    vocab_size=vocab_size,
    **{k: CONFIG[k] for k in
       ("block_size", "n_embd", "n_head", "n_layer", "dropout",
        "activation", "tie_weights")},
)


# BatchSampler parks the corpus on the GPU once and gathers windows there --
# see medresearch/gpt/data.py for why that matters on a rented box.
get_batch = BatchSampler(train_data, val_data, cfg.block_size,
                         CONFIG["batch_size"], device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(CONFIG["eval_iters"])
        for k in range(CONFIG["eval_iters"]):
            with autocast():
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

# Warm-start from an earlier checkpoint when RESUME is set. The tokenizer must
# match exactly: stoi is rebuilt from medical_text.txt above, and if the old
# checkpoint used a different character->id map then every pretrained embedding
# now points at the wrong character. Cheaper to refuse than to debug.
if RESUME:
    prev_ckpt = torch.load(RESUME, map_location=device, weights_only=False)
    if prev_ckpt.get("stoi") != stoi:
        raise SystemExit(
            f"{RESUME} has a different tokenizer ({prev_ckpt['vocab_size']} vs "
            f"{vocab_size} tokens). Resuming would scramble the embeddings — "
            "keep min_char_freq and medical_text.txt identical, or train fresh.")
    model.load_state_dict(prev_ckpt["model"])
    model.train()
    print(f"Resumed {RESUME} (step {prev_ckpt.get('iter')}, "
          f"val {prev_ckpt.get('val_loss', float('nan')):.4f})")

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
# Set the URI explicitly. Relying on MLflow to auto-detect ./mlflow.db means a
# run launched from a different working directory silently starts a NEW file
# store: training "succeeds", the registry step fails, and the run never appears
# beside the others.
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.EXPERIMENT_NAME)

with mlflow.start_run(run_name=f"{VERSION}-b{cfg.block_size}-e{cfg.n_embd}-l{cfg.n_layer}"):
    mlflow.set_tags({
        "version": VERSION,
        "arch_version": ARCH_VERSION,
        "attention": "fused-sdpa",
    })
    mlflow.log_params(CONFIG)
    mlflow.log_params({
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "precision": str(amp_dtype).replace("torch.", "") if AMP else "float32",
        "resumed_from": RESUME or "scratch",
        "vocab_size": vocab_size,
        "total_chars": len(data),
        "n_params_millions": round(n_params / 1e6, 3),
        "tokens_per_step": CONFIG["batch_size"] * cfg.block_size,
    })

    # Seed best_val from any checkpoint ALREADY at this path, so a re-run can
    # only replace it by genuinely beating it.
    #
    # Why this matters: best_val used to start at inf every run, and the very
    # first eval is at step 0 — where the model is random. So simply re-running
    # train.py without bumping VERSION overwrote a finished 51-minute model
    # with random weights before completing a single step. Bump VERSION (or
    # delete the file) when you want a clean slate.
    best_val = float("inf")
    best_iter = -1
    if os.path.exists(ckpt_path):
        prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best_val = prev.get("val_loss", float("inf"))
        best_iter = prev.get("iter", -1)
        print(f"NOTE: {ckpt_path} already holds val {best_val:.4f} "
              f"(step {best_iter}). This run will only overwrite it on a better "
              f"val_loss. Bump VERSION for a fresh file.")
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
        with autocast():
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # Unscale BEFORE clipping: clipping fp16-scaled grads would compare them
        # against the wrong threshold and effectively disable grad_clip.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

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
            # Ship the whole package, not just model.py: the pickle references
            # medresearch.gpt.model, so a bare model.py would not re-import.
            code_paths=[str(config.PROJECT_ROOT / "src" / "medresearch")],
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
