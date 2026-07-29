import torch
from model import GPT

device = "mps" if torch.backends.mps.is_available() else "cpu"

# load what we saved during training
ckpt = torch.load("gpt_medical.pt", map_location=device)
stoi, itos = ckpt["stoi"], ckpt["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

model = GPT(ckpt["vocab_size"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# give it a starting prompt
prompt = "The patient presented with"
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

out = model.generate(idx, max_new_tokens=300)
print(decode(out[0].tolist()))
