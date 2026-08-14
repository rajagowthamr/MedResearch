"""
Test bench for the from-scratch medical GPT.

    streamlit run gpt_app.py

Three tabs:
  Generate  - type a prompt, watch the model continue it
  Score     - measure the model on held-out text it never trained on
  Compare   - run every checkpoint through the same exam, side by side

A NOTE ON "ACCURACY".  This is a generative character-level model, so there
is no single right answer to score against — "accuracy" the way the Random
Forest in app.py has it doesn't apply.  What IS measurable, and what this
page reports, is the next-character exam: hide a character, ask the model to
predict it, check the answer.

  top-1 accuracy   how often its single best guess was correct
  top-5 accuracy   how often the right character was in its top five
  perplexity       how many characters it was effectively torn between.
                   1.0 = certain and right; vocab_size = learned nothing.
                   This is exp(val_loss) — the same loss curve in the MLflow
                   UI, just in units you can reason about.
"""
import glob
import hashlib
import math
import os
import time

import pandas as pd
import streamlit as st
import torch

from model import load_checkpoint

# Prompts that suit this model. It is a CONTINUATION model trained on
# encyclopedic medical text, not a chatbot — so each one is the opening of a
# sentence it can carry on, written in the register of the corpus.
EXAMPLE_PROMPTS = [
    "Type 2 diabetes mellitus is a chronic metabolic disorder characterised by",
    "The most common symptoms of bacterial pneumonia include fever,",
    "Diagnosis is confirmed by",
    "Treatment usually consists of",
    "The differential diagnosis includes",
]

st.set_page_config(page_title="MedResearch GPT", page_icon="🧠", layout="wide")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VAL_TAIL_CHARS = 400_000        # sample the exam from the end of the corpus,
                                # which is inside training's 10% val split


# ------------------------------------------------------------------
# loading
# ------------------------------------------------------------------
def find_checkpoints():
    """Newest first. Includes the v1 file that lives in the project root."""
    paths = sorted(glob.glob("checkpoints/*.pt"))
    if os.path.exists("gpt_medical.pt"):
        paths.append("gpt_medical.pt")          # the original v1 model
    return sorted(paths, key=os.path.getmtime, reverse=True)


@st.cache_data(show_spinner=False)
def file_digest(path, mtime=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


@st.cache_resource(show_spinner=False)
def _load(path, mtime):
    # mtime is part of the cache key on purpose: train.py rewrites the SAME
    # path every time it finds a better checkpoint, so keying on path alone
    # would pin this page to whatever it loaded first (e.g. the random-init
    # save from step 0) and never pick up the improved weights.
    return load_checkpoint(path, DEVICE)


def load(path):
    return _load(path, os.path.getmtime(path))


@st.cache_data(show_spinner=False)
def load_heldout():
    """Tail of the corpus = data the model was validated on, never trained on."""
    size = os.path.getsize("medical_text.txt")
    with open("medical_text.txt", "r", errors="ignore") as f:
        f.seek(max(0, size - VAL_TAIL_CHARS * 2))
        tail = f.read()
    return tail[len(tail) // 2:]                # skip any partial char at the seek


def encode(text, stoi):
    """Each checkpoint carries its OWN tokenizer, so always encode with the
    model's own stoi — v1 has 1103 characters, v2 prunes the rare tail.

    Returns a tensor already on DEVICE, so no caller can hand the model a CPU
    tensor and hit "Placeholder storage has not been allocated on MPS device".
    """
    unk = stoi.get("�", 0)
    return torch.tensor([stoi.get(c, unk) for c in text],
                        dtype=torch.long, device=DEVICE)


# ------------------------------------------------------------------
# the exam
# ------------------------------------------------------------------
@torch.no_grad()
def exam(model, ids, n_windows=256, batch=16, seed=0):
    """Next-character exam over random held-out windows.

    Reports two accuracies because they answer different questions:

      last_pos  - accuracy predicting ONE character with the model's full
                  context available.  The fair way to compare models with
                  different block_size, since each gets its own best shot.
      all_pos   - accuracy averaged over every position in the window,
                  including position 0 which has almost no context to go on.
    """
    B = model.cfg.block_size
    if len(ids) <= B + 1:
        return None
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - B - 1, (n_windows,), generator=g)

    tot_loss = hit1 = hit5 = seen = last_hits = 0
    for i in range(0, n_windows, batch):
        s = starts[i:i + batch]
        x = torch.stack([ids[j:j + B] for j in s]).to(DEVICE)
        y = torch.stack([ids[j + 1:j + B + 1] for j in s]).to(DEVICE)
        logits, loss = model(x, y)
        top5 = logits.topk(5, dim=-1).indices             # (b, B, 5)
        tgt = y.unsqueeze(-1)                             # (b, B, 1)
        correct1 = (top5[..., :1] == tgt).any(-1)         # (b, B)
        correct5 = (top5 == tgt).any(-1)

        tot_loss += loss.item() * x.size(0)
        hit1 += correct1.sum().item()
        hit5 += correct5.sum().item()
        last_hits += correct1[:, -1].sum().item()
        seen += x.size(0)

    return {
        "loss": tot_loss / seen,
        "perplexity": math.exp(tot_loss / seen),
        "top1_all": hit1 / (seen * B),
        "top5_all": hit5 / (seen * B),
        "top1_last": last_hits / seen,
        "windows": seen,
        "block_size": B,
    }


@torch.no_grad()
def next_char_table(model, ids, itos, k=8):
    """What the model thinks comes next after the prompt, with confidences."""
    x = ids[-model.cfg.block_size:].unsqueeze(0).to(DEVICE)
    logits, _ = model(x)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top = probs.topk(k)
    label = lambda c: {" ": "␣ (space)", "\n": "⏎ (newline)"}.get(c, repr(c))
    return pd.DataFrame({
        "character": [label(itos[i]) for i in top.indices.tolist()],
        "confidence": top.values.tolist(),
    })


# ------------------------------------------------------------------
# sidebar: pick a version
# ------------------------------------------------------------------
paths = find_checkpoints()
if not paths:
    st.error("No checkpoints found. Run `python train.py` first.")
    st.stop()

st.sidebar.title("🧠 MedResearch GPT")
path = st.sidebar.selectbox("Model version", paths,
                            format_func=lambda p: os.path.basename(p))
with st.spinner("Loading model…"):
    model, cfg, stoi, itos, meta = load(path)

st.sidebar.caption(f"device `{DEVICE}`")

# train.py rewrites this file mid-run, so the dropdown can hand you a model
# that is only a few hundred steps in. Say so loudly — an early checkpoint is
# near-random, and a near-random model emits mostly rare unicode (62% of this
# vocabulary is non-ASCII), which reads like garbled Greek rather than a bug.
done, target = meta.get("iter"), (meta.get("config_train") or {}).get("max_iters")
if done is not None and target and done < target:
    st.sidebar.warning(
        f"Still training: step {done:,} of {target:,} ({done/target*100:.0f}%). "
        "Early checkpoints produce gibberish. Re-run to pick up newer weights."
    )
st.sidebar.metric("Parameters", f"{model.n_params()/1e6:.2f}M")
st.sidebar.metric("Context window", f"{cfg.block_size} chars")
if "val_loss" in meta:
    st.sidebar.metric("Val loss (from training)", f"{meta['val_loss']:.4f}",
                      help=f"best at step {meta.get('iter', '?')}")
st.sidebar.write({
    "arch": meta.get("arch_version"), "version": meta.get("version", "v1"),
    "n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_layer": cfg.n_layer,
    "vocab": cfg.vocab_size, "activation": cfg.activation,
})

# Provenance: ties the text you see to a specific file of bytes on your disk.
# If you ever doubt the output is local, the airtight test is to disable Wi-Fi
# and generate again — no remote API can answer with the network off.
with st.sidebar.expander("Provenance — is this really my model?"):
    st.write(f"Weights read from `{os.path.abspath(path)}`")
    st.write(f"Size {os.path.getsize(path)/1e6:.1f} MB · "
             f"modified {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(path)))}")
    st.code(f"sha256  {file_digest(path)}", language=None)
    st.caption(
        "Every number above comes out of that file. Change the dropdown to an "
        "older checkpoint and the output quality visibly drops — a hosted API "
        "would not do that. For proof: turn off Wi-Fi and press Generate."
    )

st.sidebar.divider()
st.sidebar.subheader("Sampling")
max_new = st.sidebar.slider("Characters to generate", 50, 1000, 400, 50)
temperature = st.sidebar.slider(
    "Temperature", 0.1, 2.0, 0.8, 0.05,
    help="Low = safe and repetitive. High = inventive and more likely to babble.")
top_k = st.sidebar.slider(
    "Top-k", 1, min(100, cfg.vocab_size), min(40, cfg.vocab_size), 1,
    help="Only ever sample from the k most likely next characters.")

gen_tab, chat_tab, score_tab, cmp_tab = st.tabs(
    ["Generate", "Chat", "Score", "Compare versions"])

# ------------------------------------------------------------------
# Generate
# ------------------------------------------------------------------
with gen_tab:
    with st.expander("How to prompt this model", expanded=False):
        st.markdown(
            """
This **continues text**, it does not answer questions. It was trained to
predict the next character of encyclopedic medical writing, so:

- **Write the start of a sentence, not a question.** `What is diabetes?` gets
  you a continuation of that *question*. `Diabetes mellitus is` gets you a
  definition.
- **Longer prompts work much better.** The model has a 256-character window
  and uses every character of it. Two words give it almost nothing to condition
  on, so it falls back on generic corpus phrasing.
- **Match the corpus register** — the training text is wiki-style medical
  reference prose, so write like the opening line of an article.
- **Temperature 0.6–0.8** reads best. Below 0.4 it loops; above 1.2 it invents
  words.
            """
        )
        st.caption("Click one to load it:")
        for i, ex in enumerate(EXAMPLE_PROMPTS):
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state.prompt_box = ex

    # A dialogue-tuned checkpoint will role-play BOTH sides forever here,
    # because this tab has no stop sequence. Point people at the Chat tab.
    if meta.get("chat_format"):
        st.warning(
            "This is a dialogue-tuned checkpoint. On this tab it will keep "
            "writing `User:`/`Doctor:` turns until it runs out of characters, "
            "because there is no stop sequence. Use the **Chat** tab for "
            "one reply at a time."
        )

    prompt = st.text_area("Prompt", key="prompt_box",
                          value=st.session_state.get("prompt_box",
                                                     EXAMPLE_PROMPTS[0]),
                          height=100)

    # How much of the context window the prompt actually fills. A 17-character
    # prompt uses 7% of what the model can attend to — that is the single
    # biggest reason short prompts produce vague output.
    used = len(prompt)
    st.caption(f"Context used: **{used} / {cfg.block_size}** characters "
               f"({used/cfg.block_size*100:.0f}% of the window)"
               + ("  ·  short prompts give the model little to work with"
                  if used < cfg.block_size * 0.25 else ""))

    if st.button("Generate", type="primary", use_container_width=True):
        if not prompt:
            st.warning("Type a prompt first.")
        else:
            ids = encode(prompt, stoi)
            unknown = [c for c in set(prompt) if c not in stoi]
            if unknown:
                st.info(f"Not in this model's vocabulary, mapped to unknown: {unknown}")

            box = st.empty()
            out = prompt
            idx = ids.unsqueeze(0).to(DEVICE)
            # st.code, NOT st.markdown: the model emits raw characters, and
            # markdown would interpret **, $, #, > as formatting — silently
            # italicising and LaTeX-rendering chunks of the model's output.
            for i, nxt in enumerate(model.stream(idx, max_new, temperature, top_k)):
                out += itos[nxt.item()]
                if i % 8 == 0:                       # repaint every 8 chars
                    box.code(out + "▌", language=None, wrap_lines=True)
            box.code(out, language=None, wrap_lines=True)

            st.divider()
            st.caption("Most likely next character right after your prompt:")
            st.dataframe(
                next_char_table(model, ids, itos),
                hide_index=True, use_container_width=True,
                column_config={"confidence": st.column_config.ProgressColumn(
                    "confidence", format="percent", min_value=0.0, max_value=1.0)},
            )

# ------------------------------------------------------------------
# Chat  -  only meaningful for a checkpoint fine-tuned by finetune.py
# ------------------------------------------------------------------
@torch.no_grad()
def chat_reply(transcript, temperature, top_k, max_new=200):
    """Generate one Doctor turn and STOP at the next 'User:'.

    Without this stop the model role-plays both sides of the conversation
    forever — it was trained on transcripts, so continuing the transcript is
    exactly what it has learned to do. The stop is what turns a text
    continuer into something that takes turns.
    """
    idx = encode(transcript, stoi).unsqueeze(0)
    out = ""
    for nxt in model.stream(idx, max_new, temperature, top_k):
        out += itos[nxt.item()]
        if "\nUser:" in out:
            out = out.split("\nUser:")[0]
            break
    return out.strip()


with chat_tab:
    fmt = meta.get("chat_format")
    if not fmt:
        st.info(
            "This checkpoint was not fine-tuned for dialogue, so it has never "
            "seen a greeting answered. Run `python chat_data.py && python "
            "finetune.py`, then pick **gpt_medical_v3-chat.pt** above."
        )
    st.caption(f"Prompts are wrapped as `{fmt or 'User: …\\nDoctor:'}` — the "
               "exact format used during fine-tuning — and generation stops at "
               "the next turn boundary.")

    if st.button("Reset conversation"):
        st.session_state.history = []
    history = st.session_state.setdefault("history", [])

    for role, msg in history:
        st.chat_message("user" if role == "User" else "assistant").write(msg)

    if user_msg := st.chat_input("Say hi"):
        st.chat_message("user").write(user_msg)
        # Rebuild the whole transcript so multi-turn context carries over —
        # finetune.py trained on two-turn dialogues, not just single replies.
        transcript = "".join(f"User: {u}\nDoctor: {d}\n" for u, d in history)
        transcript += f"User: {user_msg}\nDoctor:"
        transcript = transcript[-cfg.block_size:]      # keep within the window
        with st.chat_message("assistant"):
            with st.spinner(""):
                answer = chat_reply(transcript, temperature, top_k)
            st.write(answer or "_(empty reply)_")
        history.append(("User", user_msg))
        history.append(("Doctor", answer))

    st.caption(
        "⚠️ A 4.86M-parameter character model. The safety refusals were trained "
        "in deliberately, but nothing it says is medical advice."
    )


# ------------------------------------------------------------------
# Score
# ------------------------------------------------------------------
with score_tab:
    st.write("Measures this version on held-out text from the end of "
             "`medical_text.txt` — inside the 10% validation split, so the "
             "model never trained on it.")
    n_windows = st.slider("Exam size (windows)", 32, 512, 128, 32)
    if st.button("Run exam", type="primary", use_container_width=True):
        with st.spinner("Scoring…"):
            ids = encode(load_heldout(), stoi)
            r = exam(model, ids, n_windows=n_windows)
        if r is None:
            st.error("Held-out sample is shorter than the model's context window.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Top-1 accuracy", f"{r['top1_last']*100:.1f}%",
                     help="Predicting one character with the full context window. "
                          "The fair number to compare across versions.")
            b.metric("Top-5 accuracy", f"{r['top5_all']*100:.1f}%",
                     help="Right character somewhere in the model's top five.")
            c.metric("Perplexity", f"{r['perplexity']:.2f}",
                     help=f"Effectively torn between this many of "
                          f"{cfg.vocab_size} characters. Lower is better.")
            d.metric("Held-out loss", f"{r['loss']:.4f}")
            st.caption(
                f"{r['windows']} windows x {r['block_size']} chars of context. "
                f"All-position top-1 was {r['top1_all']*100:.1f}% — lower than the "
                "headline because it includes the first characters of each window, "
                "where the model has almost nothing to go on."
            )

# ------------------------------------------------------------------
# Compare versions
# ------------------------------------------------------------------
with cmp_tab:
    st.write("Same exam, every checkpoint. This is the payoff of versioning: "
             "you can prove a new version is actually better.")
    st.caption("Perplexity is **not** comparable across versions with different "
               "vocabulary sizes — a smaller vocab is an easier guess. Compare on "
               "top-1 accuracy, which is measured against the same raw text.")
    if st.button("Compare all", type="primary", use_container_width=True):
        heldout = load_heldout()
        rows = []
        bar = st.progress(0.0)
        for i, p in enumerate(paths):
            m, c, s, _, mt = load(p)
            r = exam(m, encode(heldout, s), n_windows=128)
            rows.append({
                "checkpoint": os.path.basename(p),
                "version": mt.get("version", "v1"),
                "params (M)": round(m.n_params() / 1e6, 2),
                "context": c.block_size,
                "vocab": c.vocab_size,
                "top-1 acc": r["top1_last"] if r else None,
                "top-5 acc": r["top5_all"] if r else None,
                "perplexity": round(r["perplexity"], 2) if r else None,
            })
            bar.progress((i + 1) / len(paths))
        bar.empty()
        df = pd.DataFrame(rows).sort_values("top-1 acc", ascending=False)
        st.dataframe(
            df, hide_index=True, use_container_width=True,
            column_config={
                "top-1 acc": st.column_config.ProgressColumn(
                    "top-1 acc", format="percent", min_value=0.0, max_value=1.0),
                "top-5 acc": st.column_config.ProgressColumn(
                    "top-5 acc", format="percent", min_value=0.0, max_value=1.0),
            },
        )
        best = df.iloc[0]
        st.success(f"Best: **{best['checkpoint']}** — "
                   f"{best['top-1 acc']*100:.1f}% top-1 accuracy")
