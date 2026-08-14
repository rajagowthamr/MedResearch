from datasets import load_dataset

from medresearch import config

# ------------------------------------------------------------------
# Download a medical-text dataset from Hugging Face and write it all
# into medical_text.txt (which train.py reads).
#
# Dataset: gamino/wiki_medical_terms  -> 6,861 medical wiki articles.
# The text lives in the "page_text" column.
# ------------------------------------------------------------------

print("Downloading dataset from Hugging Face...")
ds = load_dataset("gamino/wiki_medical_terms", split="train")
print(f"Got {len(ds)} articles.")

# join every article's text into one big string
texts = [row["page_text"] for row in ds if row["page_text"]]
all_text = "\n\n".join(texts)

config.ensure_dirs()
with open(config.MEDICAL_TEXT, "w") as f:
    f.write(all_text)

print(f"Wrote {len(all_text):,} characters "
      f"({len(all_text)/1_000_000:.1f} MB) to medical_text.txt")
