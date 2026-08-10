# MedResearch — roadmap to on-device clinical field extraction

**Goal:** extract structured fields (vitals, medications, labs) from clinical
notes, running locally on an 8GB-RAM edge device.
**Training hardware:** RTX 3060.
**Data:** `note_train.txt` (4,737 notes, ~3,225 chars each) + `note_val.txt`.

---

## Read this first: the constraint is not what you think

8GB of RAM is **roomy** for this task, not tight. Sizes at 4-bit quantization:

| model | RAM | share of 8GB |
|---|---|---|
| your v2 (4.86M) | ~3 MB | 0.04% |
| DeBERTa-v3-small (140M) | ~90 MB | 1% |
| Qwen2.5-0.5B | ~0.4 GB | 5% |
| Llama-3.2-1B | ~0.8 GB | 10% |
| Llama-3.2-3B | ~2.0 GB | 25% |

Your current model uses **0.04%** of the budget. You are not limited by memory —
you are limited by what fits in 4.86M parameters, which is spelling and grammar
with no capacity left for facts or task behaviour.

**So the plan is not "make the from-scratch model bigger."** It is: fine-tune a
small *pretrained* model, which already knows English and clinical vocabulary,
and spend your data teaching it the extraction task only. A 0.5B model at 4-bit
is 5% of your RAM budget and will beat a from-scratch 30M model at this task by
a margin that is not close.

Extraction is also the *right* task to have picked. The answer is present in the
input text, so the model transforms rather than recalls — which means
hallucination is measurable and largely avoidable, unlike open-ended Q&A.

### What happens to the from-scratch GPT

Keep it. It is a finished, working artifact and the reason you now understand
attention, tokenizers, LR schedules, mixed precision and overfitting from the
inside. That understanding is the prerequisite for everything below, not a
detour from it. But it is not the path to this goal, and continuing to scale it
would be sunk-cost reasoning.

---

## The actual bottleneck: you have no labels

Extraction is **supervised** learning. It needs pairs:

```
input:  "...Vital Signs: BP 128/76, Pulse: 80, Temperature: 97.7..."
output: {"bp_systolic":128, "bp_diastolic":76, "pulse":80, "temp_f":97.7}
```

You have 4,737 notes and **zero** of those pairs. Producing them is the project.
Everything else is comparatively routine.

---

## M0 — Define the schema and build a gold test set  *(~1 day, no GPU)*

Do not skip this. Without a gold set you cannot tell whether any later step
helped, and every subsequent milestone is unmeasurable.

- [ ] **Write the JSON schema.** Start with 8–12 fields, not 50. Suggested:
      `bp_systolic`, `bp_diastolic`, `pulse`, `temp_f`, `respiratory_rate`,
      `weight_lbs`, `medications[]` (name, dose, unit, frequency),
      `allergies[]`, `labs[]` (name, value, unit).
- [ ] **Decide what "absent" means.** Most notes lack most fields — only 757 of
      4,737 have a `Vital Signs:` header. `null` vs omitted vs `"not stated"`
      must be one consistent choice, or your metric is meaningless.
- [ ] **Hand-label 200 notes.** Yes, by hand. This is your test set and it is
      the only thing standing between you and confidently shipping garbage.
      Sample across specialties, not the first 200.
- [ ] **Lock it.** Never train on these. Never tune on them more than you must.

**Done when:** 200 verified note→JSON pairs exist and a scoring script reports
per-field precision/recall/F1 against them.

---

## M1 — Rule-based baseline  *(~1 day, no GPU)*

Before any ML, write the regex extractor. Two reasons: some fields are genuinely
solved by rules, and **every model you train later must beat this number** or it
is not worth deploying.

- [ ] Parse by section header first (`Vital Signs:`, `MEDICATIONS:`,
      `ALLERGIES:`, `LABORATORY DATA:`), then apply patterns within the section.
      Sectioning first is what stops the date-vs-blood-pressure failure.
- [ ] **Be suspicious of every pattern.** `\d{2,3}/\d{2,3}` matches `06/30`.
      Require a `BP`/`blood pressure` anchor nearby and a plausible range
      (systolic 70–250, diastolic 40–150).
- [ ] Score on the M0 gold set. Record per-field F1 here.

**Expected:** high precision, mediocre recall. Vitals with explicit labels may
reach 0.85+ F1; medications with free-text dosing ("one tablet twice daily")
will be poor. That gap is exactly what the model is for.

---

## M2 — Bootstrap the training labels  *(~2 days, uses the 3060)*

You need thousands of labeled notes and you will not hand-label thousands. Two
sources, combined:

- [ ] **Rules from M1** over all 4,737 notes → weak labels, high precision on the
      easy fields.
- [ ] **Distillation from a larger model.** An RTX 3060 12GB runs Qwen2.5-7B at
      4-bit (~4.7GB) comfortably. Prompt it with the schema and each note, in
      JSON mode, to produce labels. This is knowledge distillation: a big model
      you cannot deploy teaches a small one you can.
      - On a 6GB 3060, use Qwen2.5-3B instead — still a strong teacher.
- [ ] **Reconcile the two.** Where rules and teacher agree, trust it. Where they
      disagree, either drop the example or review it. Disagreement rate is a
      free quality signal — a field where they disagree 40% of the time is a
      field whose schema definition is ambiguous.
- [ ] Validate a 100-note sample of the result by hand before training on it.
      Distilled labels inherit the teacher's mistakes, silently.

**Done when:** ~4,000 note→JSON training pairs exist, spot-checked, with a
recorded estimate of label noise.

---

## M3 — Fine-tune the deployable model  *(~4h on the 3060)*

- [ ] **Start with Qwen2.5-0.5B.** Small enough that full fine-tuning fits on the
      3060, deploys at ~0.4GB, and is strong at structured output. Move up to
      Llama-3.2-1B only if 0.5B underperforms the M1 baseline.
- [ ] Format as instruction pairs: system prompt with the schema, user turn with
      the note, assistant turn with the JSON.
- [ ] **Mask the loss to the JSON output only.** Training on the input note as
      well wastes capacity teaching it to reproduce notes — the same
      catastrophic-forgetting reasoning behind the data mixing in
      `finetune.py`, applied in reverse.
- [ ] Use LoRA if VRAM is tight; full fine-tuning if it fits. At 0.5B on 12GB
      with bf16 (the 3060 is Ampere, so bf16 needs no GradScaler), it fits.
- [ ] Stack: `transformers` + `peft` + `trl`. Keep logging to MLflow — that
      setup already works and carries over unchanged.
- [ ] **Constrain the output.** Grammar-constrained decoding (llama.cpp GBNF, or
      `outlines`) makes invalid JSON structurally impossible rather than merely
      unlikely. For an extraction system this is worth more than a point of F1.

**Gate:** must beat M1's rule baseline on the M0 gold set, per field. If it does
not, the problem is label quality (back to M2), not model size.

---

## M4 — Quantize and deploy  *(~1 day)*

- [ ] Convert to GGUF, quantize to Q4_K_M. `llama.cpp` is the right runtime for
      an 8GB device.
- [ ] **Re-score the quantized model on the gold set.** Quantization costs
      accuracy, and how much is task-specific — you must measure it, not assume
      it. If Q4 hurts, try Q5_K_M or Q8_0; you have RAM to spare.
- [ ] Measure real latency and peak RSS on the actual device, with a full
      3,225-char note as input. Context length drives the KV cache, which is
      often what actually blows the memory budget — not the weights.
- [ ] Ship the rule extractor alongside as a fallback and a cross-check.
      Disagreement between rules and model is a useful confidence signal.

**Done when:** F1 on the gold set, tokens/sec, and peak RAM are all measured on
the target hardware and written down.

---

## M5 — Close the loop  *(ongoing)*

- [ ] Log every low-confidence extraction for review.
- [ ] Hand-correct those, add to training data, retrain. Corrections on cases the
      model actually got wrong are worth far more per example than fresh random
      labels.
- [ ] Track per-field F1 over versions in MLflow, exactly as you already track
      loss curves.

---

## Summary

| # | milestone | effort | hardware | gate |
|---|---|---|---|---|
| M0 | schema + 200 gold notes | 1 day | none | scoring script runs |
| M1 | rule baseline | 1 day | none | per-field F1 recorded |
| M2 | bootstrap ~4k labels | 2 days | 3060 (teacher) | 100 spot-checked |
| M3 | fine-tune 0.5B | 4h | 3060 | beats M1 |
| M4 | quantize + deploy | 1 day | 3060 + device | measured on device |
| M5 | correction loop | ongoing | 3060 | F1 trends up |

**The order is deliberate.** M0 and M1 involve no machine learning at all and
take two days, and skipping them is the most common way this kind of project
fails — you end up with a model you cannot evaluate, solving a task you never
pinned down.

---

## Open question: your RTX 3060's VRAM

The desktop 3060 is 12GB; laptop and some variants are 6GB. It changes M2 (7B vs
3B teacher) and M3 (full fine-tune vs LoRA). Check with `nvidia-smi` and adjust.
