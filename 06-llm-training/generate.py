import argparse

import torch
import torch.nn.functional as F

from model import ModelConfig, Transformer

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="out/ckpt.pt")
p.add_argument("--prompt", default="ROMEO:")
p.add_argument("--tokens", type=int, default=300)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-p", type=float, default=0.9)
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
model = Transformer(ModelConfig(**ckpt["cfg"]))
model.load_state_dict(ckpt["model"])
model.eval()
idx = torch.tensor([list(args.prompt.encode("utf-8"))])
with torch.no_grad():
    for _ in range(args.tokens):
        logits, _ = model(idx[:, -model.cfg.seq_len:])
        probs = F.softmax(logits[:, -1] / args.temperature, dim=-1)
        sorted_p, sorted_ix = torch.sort(probs, descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < args.top_p
        sorted_p = sorted_p * keep
        nxt = sorted_ix.gather(-1, torch.multinomial(sorted_p / sorted_p.sum(), 1))
        idx = torch.cat([idx, nxt], dim=1)
print(bytes(idx[0].tolist()).decode("utf-8", errors="replace"))
