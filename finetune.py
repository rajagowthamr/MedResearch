"""
Fine-tune v2 into a model that answers instead of only continuing.  -> v3-chat

THIS IS THE ANSWER TO "where do I add hi means hello".
Not in model.py. You add it to the DATA (chat_data.py), then run this, which
continues training the existing v2 weights on that data.

WHY FINE-TUNE INSTEAD OF TRAINING FROM SCRATCH
----------------------------------------------
chat_data.txt is 72KB. Trained from scratch on that alone you would get a model
that can say "Hello." and nothing else — it would never have learned English.
Fine-tuning starts from v2, which already knows medical spelling and grammar
from 60MB of text, and only teaches it the dialogue SHAPE on top.

THE TWO THINGS THAT MAKE FINE-TUNING WORK
-----------------------------------------
1. LOW LEARNING RATE (1e-4, not 1e-3). A high rate would blast away the
   language knowledge you spent an hour training.
2. DATA MIXING. Half of every batch is still original medical text. Train on
   chat data alone and the model suffers "catastrophic forgetting" — it learns
   to say Hello and forgets how to write a sentence. This is the single most
   common mistake when fine-tuning.

Run:  python chat_data.py && python finetune.py
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

from model import load_checkpoint, save_checkpoint

MODEL_NAME = "medresearch-gpt"
BASE = "checkpoints/gpt_medical_v2.pt"     # the model we start from
VERSION = "v3-chat"

CONFIG = {
    "base_checkpoint": BASE,
    # 1200 steps put 68 EPOCHS over a 72KB corpus, which memorises it. At
    # batch 32 / block 256 / chat_fraction 0.5 the epoch count is
    #     iters * (batch*frac) * block / len(chat_data.txt)
    # so 300 steps is ~17 epochs — enough to learn the dialogue format,
    # few enough to leave some generality.
    "max_iters": 300,
    "batch_size": 32,
    "eval_interval": 25,
    "eval_iters": 20,
    "learning_rate": 1e-4,      # 10x lower than pretraining, on purpose
    "min_lr": 1e-5,
    "warmup_iters": 30,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "chat_fraction": 0.35,      # was 0.5 — less chat per batch = less rote
    "seed": 1337,
}

torch.manual_seed(CONFIG["seed"])
device = "mps" if torch.backends.mps.is_available() else "cpu"
ckpt_path = f"checkpoints/gpt_medical_{VERSION}.pt"

# ------------------------------------------------------------------
# 1. LOAD v2  -  weights, architecture AND tokenizer all come from the file
# ------------------------------------------------------------------
if not os.path.exists(BASE):
    raise SystemExit(f"{BASE} not found — let train.py finish first.")
model, cfg, stoi, itos, base_meta = load_checkpoint(BASE, device)
model.train()
print(f"Base: {BASE} (step {base_meta.get('iter')}, "
      f"val {base_meta.get('val_loss', float('nan')):.4f})")

# Reuse v2's tokenizer exactly. Rebuilding it from the new text would shuffle
# every character id and make the pretrained embeddings meaningless.
unk = stoi.get("�", 0)
encode = lambda s: torch.tensor([stoi.get(c, unk) for c in s], dtype=torch.long)

# ------------------------------------------------------------------
# 2. DATA  -  chat + a slice of the original corpus to mix back in
# ------------------------------------------------------------------
if not os.path.exists("chat_data.txt"):
    raise SystemExit("chat_data.txt not found — run `python chat_data.py` first.")
chat_text = open("chat_data.txt").read()

# A slice of medical text big enough to keep the language knowledge alive.
with open("medical_text.txt", errors="ignore") as f:
    med_text = f.read(8_000_000)

chat_train = encode(chat_text)
med_ids = encode(med_text)

# The honest validation set: dialogues built from user phrasings that appear
# NOWHERE in chat_data.txt. Slicing chat_data.txt itself would not work —
# every dialogue in it draws from the same small pool of replies, so a random
# split shares target strings with the training data and reports memorisation
# as if it were generalisation.
if not os.path.exists("chat_holdout.txt"):
    raise SystemExit("chat_holdout.txt missing — re-run `python chat_data.py`.")
chat_val = encode(open("chat_holdout.txt").read())

epochs = (CONFIG["max_iters"] * int(CONFIG["batch_size"] * CONFIG["chat_fraction"])
          * cfg.block_size / max(len(chat_train), 1))
print(f"chat: {len(chat_train):,} chars | holdout: {len(chat_val):,} chars "
      f"| medical mix-in: {len(med_ids):,} chars")
print(f"epochs over chat data: {epochs:.0f}  (68 memorised it; aim for <20)")

n_chat = int(CONFIG["batch_size"] * CONFIG["chat_fraction"])
n_med = CONFIG["batch_size"] - n_chat


def sample(source, count):
    B = cfg.block_size
    ix = torch.randint(len(source) - B - 1, (count,))
    x = torch.stack([source[i:i + B] for i in ix])
    y = torch.stack([source[i + 1:i + B + 1] for i in ix])
    return x, y


def get_batch():
    """Part chat, part medical — the anti-forgetting mix."""
    xc, yc = sample(chat_train, n_chat)
    xm, ym = sample(med_ids, n_med)
    return (torch.cat([xc, xm]).to(device), torch.cat([yc, ym]).to(device))


@torch.no_grad()
def estimate_loss():
    """Three numbers, because each catches a different failure:

      chat_train  - is it learning the dialogue format at all?
      chat_val    - can it handle UNSEEN phrasings? If chat_train keeps
                    falling while this flattens or rises, it is memorising.
      medical     - is it forgetting the language it took 51 min to learn?
                    Rising here is catastrophic forgetting.

    A single blended loss hides all three.
    """
    model.eval()
    out = {}
    for name, src in (("chat_train", chat_train), ("chat_val", chat_val),
                      ("medical", med_ids)):
        losses = torch.zeros(CONFIG["eval_iters"])
        for k in range(CONFIG["eval_iters"]):
            x, y = sample(src, CONFIG["batch_size"])
            _, loss = model(x.to(device), y.to(device))
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    # the overfitting signal, in one number
    out["memorisation_gap"] = out["chat_val"] - out["chat_train"]
    return out


def lr_at(it):
    if it < CONFIG["warmup_iters"]:
        return CONFIG["learning_rate"] * (it + 1) / CONFIG["warmup_iters"]
    p = (it - CONFIG["warmup_iters"]) / max(1, CONFIG["max_iters"] - CONFIG["warmup_iters"])
    cos = 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return CONFIG["min_lr"] + cos * (CONFIG["learning_rate"] - CONFIG["min_lr"])


# ------------------------------------------------------------------
# 3. SAMPLING WITH A STOP SEQUENCE
# ------------------------------------------------------------------
def reply(user_text, max_new=200, temperature=0.7, top_k=30):
    """Format the prompt the way training formatted it, and stop the moment the
    model starts a new 'User:' turn — otherwise it happily role-plays both
    sides of the conversation forever."""
    model.eval()
    prompt = f"User: {user_text}\nDoctor:"
    idx = encode(prompt).unsqueeze(0).to(device)
    out = ""
    for nxt in model.stream(idx, max_new, temperature, top_k):
        out += itos[nxt.item()]
        if "\nUser:" in out:
            out = out.split("\nUser:")[0]
            break
    model.train()
    return out.strip()


# ------------------------------------------------------------------
# 4. FINE-TUNE
# ------------------------------------------------------------------
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": CONFIG["weight_decay"]},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=CONFIG["learning_rate"], betas=(0.9, 0.99),
)

mlflow.set_experiment("medresearch-gpt")
with mlflow.start_run(run_name=f"{VERSION}-finetune"):
    mlflow.set_tags({"version": VERSION, "stage": "finetune",
                     "base_run": base_meta.get("version", "v2")})
    mlflow.log_params(CONFIG)
    mlflow.log_params({"device": device, "vocab_size": cfg.vocab_size,
                       "chat_chars": len(chat_ids)})

    best = float("inf")
    t0 = time.time()
    for it in range(CONFIG["max_iters"] + 1):
        lr = lr_at(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % CONFIG["eval_interval"] == 0 or it == CONFIG["max_iters"]:
            L = estimate_loss()
            print(f"step {it}: chat_train {L['chat_train']:.4f} | "
                  f"chat_val {L['chat_val']:.4f} | gap {L['memorisation_gap']:+.4f} "
                  f"| medical {L['medical']:.4f} | {(time.time()-t0)/60:.1f}m")
            mlflow.log_metrics({"chat_train_loss": L["chat_train"],
                                "chat_val_loss": L["chat_val"],
                                "memorisation_gap": L["memorisation_gap"],
                                "medical_loss": L["medical"],
                                "learning_rate": lr}, step=it)
            # Early stopping on the HOLDOUT, not on training loss. Training
            # loss will keep falling all the way to memorisation, so selecting
            # on it would guarantee we ship the most overfit checkpoint.
            if L["chat_val"] < best:
                best = L["chat_val"]
                save_checkpoint(ckpt_path, model, stoi, itos,
                                extra={"val_loss": L["chat_val"],
                                       "chat_train_loss": L["chat_train"],
                                       "memorisation_gap": L["memorisation_gap"],
                                       "medical_loss": L["medical"],
                                       "iter": it, "version": VERSION,
                                       "config_train": CONFIG,
                                       "chat_format": "User: {q}\nDoctor:"})

        if it == CONFIG["max_iters"]:
            break

        x, y = get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

    mlflow.log_metric("best_chat_val_loss", best)
    print(f"\nBest chat loss {best:.4f} -> {ckpt_path}")

    # ------------------------------------------------------------------
    # 5. DOES IT ACTUALLY SAY HELLO?
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    print("\n--- sanity check ---")
    for q in ["hi", "hello", "who are you", "i have a headache", "what is asthma"]:
        r = reply(q)
        print(f"User: {q}\nDoctor: {r}\n")
        mlflow.log_text(f"User: {q}\nDoctor: {r}", f"samples/{q.replace(' ','_')}.txt")

    mlflow.log_artifact(ckpt_path)
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.int64), (-1, cfg.block_size), "idx")]),
        outputs=Schema([TensorSpec(np.dtype(np.float32),
                                   (-1, cfg.block_size, cfg.vocab_size), "logits")]),
    )
    try:
        mlflow.pytorch.log_model(model, name="model",
                                 registered_model_name=MODEL_NAME,
                                 code_paths=["model.py"], signature=signature,
                                 serialization_format="pickle")
        client = MlflowClient()
        mine = max(client.search_model_versions(f"name='{MODEL_NAME}'"),
                   key=lambda v: int(v.version))
        client.set_model_version_tag(MODEL_NAME, mine.version, "version", VERSION)
        client.set_model_version_tag(MODEL_NAME, mine.version, "chat_val_loss", f"{best:.4f}")
        # a chat model is a different job from a text model, so give it its own
        # alias rather than competing for @champion on val_loss
        client.set_registered_model_alias(MODEL_NAME, "chat", mine.version)
        print(f"Registered {MODEL_NAME} v{mine.version} as @chat")
    except Exception as e:
        print(f"WARNING: registry step failed ({e}). Checkpoint safe at {ckpt_path}")
