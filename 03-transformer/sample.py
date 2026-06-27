import argparse

import torch

from gpt import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="out/ckpt.pt")
p.add_argument("--tokens", type=int, default=500)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-k", type=int, default=None)
p.add_argument("--prompt", default="\n")
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
model = GPT(GPTConfig(**ckpt["cfg"]))
model.load_state_dict(ckpt["model"])
model.eval()
stoi = {c: i for i, c in enumerate(ckpt["chars"])}
idx = torch.tensor([[stoi[c] for c in args.prompt]])
out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
print("".join(ckpt["chars"][i] for i in out[0].tolist()))
