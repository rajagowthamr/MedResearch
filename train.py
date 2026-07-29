"""
Train the from-scratch GPT with enterprise-grade experiment tracking (MLflow).

Why MLflow (vs cloud tools): it's open-source and SELF-HOSTED — all run data,
metrics, and model files stay on YOUR machine (./mlruns). That's the standard,
HIPAA-friendly choice for clinical/medical work where data can't leave premises.

Best practices demonstrated here:
  - A single CONFIG dict = the one source of truth for every hyperparameter.
  - Fixed random seed for reproducibility.
  - Params logged once; metrics logged per eval step (become loss curves).
  - The trained model + tokenizer logged as a run Artifact (model lineage).
"""
import torch
import mlflow
import model as M               # import the module so we can read/log its hyperparams
from model import GPT, block_size

# ------------------------------------------------------------------
# 1. CONFIG  -  every knob in ONE place (best practice).
# ------------------------------------------------------------------
CONFIG = {
    # training
    "batch_size": 32,
    "max_iters": 3000,
    "eval_interval": 300,
    "eval_iters": 50,
    "learning_rate": 3e-4,
    "seed": 1337,
    # model architecture (mirrored from model.py so it's logged too)
    "block_size": M.block_size,
    "n_embd": M.n_embd,
    "n_head": M.n_head,
    "n_layer": M.n_layer,
    "dropout": M.dropout,
    # data
    "dataset": "gamino/wiki_medical_terms",
}

torch.manual_seed(CONFIG["seed"])   # reproducibility
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ------------------------------------------------------------------
# 2. DATA
# ------------------------------------------------------------------
with open("medical_text.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (CONFIG["batch_size"],))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


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


# ------------------------------------------------------------------
# 3. BUILD MODEL
# ------------------------------------------------------------------
model = GPT(vocab_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
print(f"Device: {device} | Parameters: {n_params/1e6:.2f}M")

# ------------------------------------------------------------------
# 4. TRAIN INSIDE AN MLFLOW RUN
#    "experiment" = the project folder; each run() = one training attempt.
# ------------------------------------------------------------------
mlflow.set_experiment("medresearch-gpt")

with mlflow.start_run():
    # log all hyperparameters + derived facts ONCE
    mlflow.log_params(CONFIG)
    mlflow.log_params({
        "device": device,
        "vocab_size": vocab_size,
        "total_chars": len(data),
        "n_params_millions": round(n_params / 1e6, 3),
    })

    for it in range(CONFIG["max_iters"]):
        if it % CONFIG["eval_interval"] == 0:
            losses = estimate_loss()
            print(f"step {it}: train {losses['train']:.4f} | val {losses['val']:.4f}")
            # log metrics for this step -> MLflow turns these into curves.
            # step=it lines all metrics up on the same x-axis.
            mlflow.log_metrics({
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "overfit_gap": losses["val"] - losses["train"],  # grows = overfitting
            }, step=it)

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ------------------------------------------------------------------
    # 5. SAVE  -  locally AND as an MLflow artifact (tied to this run)
    # ------------------------------------------------------------------
    ckpt = {"model": model.state_dict(), "stoi": stoi, "itos": itos,
            "vocab_size": vocab_size}
    torch.save(ckpt, "gpt_medical.pt")
    mlflow.log_artifact("gpt_medical.pt")   # versioned with this run

    print("Saved gpt_medical.pt and logged it to MLflow.")
