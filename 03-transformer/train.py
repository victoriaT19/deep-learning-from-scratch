import argparse
import time
from pathlib import Path

import torch

from data import CharData, load_text
from gpt import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--max-iters", type=int, default=5000)
p.add_argument("--eval-interval", type=int, default=500)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--block-size", type=int, default=256)
p.add_argument("--n-layer", type=int, default=6)
p.add_argument("--n-head", type=int, default=6)
p.add_argument("--n-embd", type=int, default=384)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--out", default="out/ckpt.pt")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
data = CharData(load_text())
cfg = GPTConfig(vocab_size=data.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
model = GPT(cfg).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {device}")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)


@torch.no_grad()
def estimate_loss(iters=50):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.stack([model(*data.batch(split, args.block_size, args.batch_size, device))[1] for _ in range(iters)])
        out[split] = losses.mean().item()
    model.train()
    return out


t0 = time.time()
for it in range(args.max_iters + 1):
    if it % args.eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {it:5d} train {losses['train']:.4f} val {losses['val']:.4f} ({time.time() - t0:.0f}s)")
    if it == args.max_iters:
        break
    _, loss = model(*data.batch("train", args.block_size, args.batch_size, device))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "chars": data.chars}, args.out)
