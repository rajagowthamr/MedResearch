"""Quick CLI sample from a checkpoint.  For the interactive version:

    streamlit run gpt_app.py
"""
import glob
import os
import sys

import torch
from model import load_checkpoint

device = "mps" if torch.backends.mps.is_available() else "cpu"

# default to the newest checkpoint; override with: python generate.py <path>
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    found = sorted(glob.glob("checkpoints/*.pt") + glob.glob("gpt_medical.pt"),
                   key=os.path.getmtime, reverse=True)
    if not found:
        sys.exit("No checkpoint found — run `python train.py` first.")
    path = found[0]

# load_checkpoint rebuilds the right architecture from the config inside the
# file, so this works for v1 and every version after it.
model, cfg, stoi, itos, meta = load_checkpoint(path, device)
print(f"{path}  |  {meta.get('version', 'v1')}  |  context {cfg.block_size} chars")

prompt = "The patient presented with"
idx = torch.tensor([[stoi.get(c, stoi.get("�", 0)) for c in prompt]],
                   dtype=torch.long, device=device)

out = model.generate(idx, max_new_tokens=300, temperature=0.8, top_k=40)
print("".join(itos[i] for i in out[0].tolist()))
