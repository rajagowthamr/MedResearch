"""Quick CLI sample from a checkpoint.  For the interactive version:

    streamlit run apps/gpt_app.py

    python scripts/generate.py                 # newest checkpoint
    python scripts/generate.py path/to.pt      # a specific one
    python scripts/generate.py path/to.pt "The patient presented with"
"""
import sys

import torch

from medresearch import config
from medresearch.gpt import load_checkpoint
from medresearch.gpt.data import encode_text

DEFAULT_PROMPT = "The patient presented with"

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    # find_checkpoints() covers both the m1_* naming and the older
    # gpt_medical_* files, so nothing already trained becomes unreachable.
    found = config.find_checkpoints()
    if not found:
        sys.exit(f"No checkpoint in {config.CHECKPOINTS_DIR} — "
                 "run `python scripts/train_gpt.py` first.")
    path = str(found[0])

prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT

# load_checkpoint rebuilds the right architecture from the config inside the
# file, so this works for v1 and every version after it.
model, cfg, stoi, itos, meta = load_checkpoint(path, device)
print(f"{path}  |  {meta.get('version', 'v1')}  |  context {cfg.block_size} chars")

idx = encode_text(prompt, stoi).unsqueeze(0).to(device)
out = model.generate(idx, max_new_tokens=300, temperature=0.8, top_k=40)
print("".join(itos[i] for i in out[0].tolist()))
