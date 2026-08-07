"""
Pull tchebonenko/MedicalTranscriptions and prepare it for THIS project's model.

WHY THIS DATASET IS DIFFERENT FROM wiki_medical_terms
-----------------------------------------------------
wiki_medical_terms is encyclopedia prose ABOUT diseases. This is 4,999 real
dictated clinical notes — SOAP notes, operative reports, discharge summaries.
It is the register your model actually needs to speak, and 60MB of Wikipedia
will never teach it, because Wikipedia never writes "SUBJECTIVE:," or
"POSTOPERATIVE DIAGNOSIS:".

Columns: description (1-line summary, ~132 chars), medical_specialty (40
classes), sample_name, transcription (the note, ~3,052 chars), keywords.

TWO FACTS ABOUT v2's TOKENIZER THAT BREAK IF YOU IGNORE THEM
------------------------------------------------------------
v2's vocab was frozen from wiki text, which used typographic quotes. So the
plain ASCII apostrophe (U+0027) is NOT in the 249-char vocab — and these
transcriptions use it 4,320 times. Unmapped, every one becomes the unknown
token and the model learns that contractions are noise. Same for "@".
NORMALIZE below folds them onto characters v2 already knows. After that,
out-of-vocab coverage is 5 characters totalling 5 occurrences in 15M.

Run:  python fetch_transcriptions.py
"""
import random

from datasets import load_dataset

random.seed(1337)

# Fold characters absent from v2's vocab onto ones it knows. Verified against
# checkpoints/gpt_medical_v2.pt — do not add a mapping without checking stoi.
NORMALIZE = {
    "'": "’",   # ASCII apostrophe -> typographic, which v2 HAS
    "@": " at ",     # 12 occurrences, all in addresses
}

VAL_FRACTION = 0.05


def clean(text):
    for bad, good in NORMALIZE.items():
        text = text.replace(bad, good)
    return text.strip()


def build():
    print("Downloading tchebonenko/MedicalTranscriptions...")
    ds = load_dataset("tchebonenko/MedicalTranscriptions", split="train")

    # 33 of 4,999 rows have a null transcription. Keep only complete pairs so
    # the two outputs below are built from exactly the same set of documents.
    rows = [
        {
            "desc": clean(r["description"]),
            "note": clean(r["transcription"]),
            # medical_specialty ships with a LEADING SPACE (" Surgery").
            # Skip the strip and you get 40 classes that all look wrong.
            "spec": r["medical_specialty"].strip(),
        }
        for r in ds
        if r["transcription"] and r["description"]
    ]
    print(f"{len(rows)} usable notes out of {len(ds)} rows.")

    # ------------------------------------------------------------------
    # OUTPUT 1: raw corpus, for mixing into further pretraining.
    # Same shape as medical_text.txt, so train.py / finetune.py can read it
    # with no code change.
    # ------------------------------------------------------------------
    corpus = "\n\n".join(r["note"] for r in rows)
    with open("transcriptions.txt", "w") as f:
        f.write(corpus)
    print(f"transcriptions.txt : {len(corpus):,} chars "
          f"({len(corpus)/1_000_000:.1f} MB) of real clinical notes")

    # ------------------------------------------------------------------
    # OUTPUT 2: conditioned-generation pairs,  DESC -> opening of the NOTE.
    #
    # WHY THIS DIRECTION AND NOT SUMMARISATION.  The obvious task is
    # note -> description (summarise the note). It is impossible with this
    # model: block_size is 256 characters and the median note is 2,667. The
    # model would have to summarise text it can never see all of. Reversed,
    # the 132-char description fits comfortably and the model learns to
    # continue in note register — which is a task that FITS the context it has.
    #
    # The split holds out whole DOCUMENTS, stratified by specialty. Unlike
    # chat_data.py this is an honest random split: every note is a distinct
    # document, so there is no shared-target leakage to engineer around.
    # ------------------------------------------------------------------
    by_spec = {}
    for r in rows:
        by_spec.setdefault(r["spec"], []).append(r)

    train, val = [], []
    for spec, group in sorted(by_spec.items()):
        group = group[:]
        random.shuffle(group)
        # max(1, ...) would steal the only example from 6-note specialties,
        # so tiny classes stay entirely in train.
        n_val = int(len(group) * VAL_FRACTION)
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    random.shuffle(train)

    def fmt(r):
        # "SPECIALTY:" is included so the model can be steered at generation
        # time. Plain ASCII markers again — every character is already in the
        # 249-char vocab, so the tokenizer and the v2 weights stay loadable.
        return (f"SPECIALTY: {r['spec']}\n"
                f"DESC: {r['desc']}\n"
                f"NOTE: {r['note']}\n")

    for name, split in (("note_train.txt", train), ("note_val.txt", val)):
        with open(name, "w") as f:
            f.write("\n".join(fmt(r) for r in split))
        print(f"{name:16s} : {len(split)} notes")

    print(f"\nSpecialties: {len(by_spec)}. Heavily imbalanced — "
          f"{max(len(g) for g in by_spec.values())} notes in the largest, "
          f"{min(len(g) for g in by_spec.values())} in the smallest.")
    print("  ^ If you train a specialty classifier on this, accuracy is a lie: "
          "always-predict-Surgery already scores ~22%. Use macro-F1.")


if __name__ == "__main__":
    build()
