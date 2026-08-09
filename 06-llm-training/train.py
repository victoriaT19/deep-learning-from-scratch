import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig, Transformer

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--out", default="out/ckpt.pt")
p.add_argument("--dim", type=int, default=384)
p.add_argument("--n-layers", type=int, default=6)
p.add_argument("--n-heads", type=int, default=6)
p.add_argument("--n-kv-heads", type=int, default=2)
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--grad-accum", type=int, default=4)
p.add_argument("--max-steps", type=int, default=5000)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--min-lr", type=float, default=6e-5)
p.add_argument("--weight-decay", type=float, default=0.1)
p.add_argument("--grad-clip", type=float, default=1.0)
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--eval-iters", type=int, default=50)
p.add_argument("--resume", action="store_true")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
autocast = torch.autocast(device_type=device, dtype=dtype) if dtype != torch.float32 else nullcontext()
torch.manual_seed(0)

data = {s: np.memmap(Path(args.data_dir) / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}


def get_batch(split):
    d = data[split]
    ix = np.random.randint(0, len(d) - args.seq_len - 1, args.batch_size)
    x = torch.from_numpy(np.stack([d[i:i + args.seq_len] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([d[i + 1:i + 1 + args.seq_len] for i in ix]).astype(np.int64))
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


cfg = ModelConfig(vocab_size=256, seq_len=args.seq_len, dim=args.dim, n_layers=args.n_layers,
                  n_heads=args.n_heads, n_kv_heads=args.n_kv_heads)
model = Transformer(cfg).to(device)
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda")
start_step, best_val = 0, float("inf")
if args.resume and Path(args.out).exists():
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    start_step, best_val = ckpt["step"], ckpt["best_val"]
print(f"{model.num_params() / 1e6:.2f}M parameters, {device}, {dtype}")


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(args.eval_iters):
            with autocast:
                losses.append(model(*get_batch(split))[1].item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
t0, last_log = time.time(), start_step
for step in range(start_step, args.max_steps + 1):
    if step % args.eval_every == 0 or step == args.max_steps:
        losses = evaluate()
        bpb = losses["val"] / math.log(2)
        print(f"step {step:5d} train {losses['train']:.4f} val {losses['val']:.4f} ({bpb:.3f} bits/byte)")
        if losses["val"] < best_val:
            best_val = losses["val"]
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg.__dict__,
                        "step": step, "best_val": best_val}, args.out)
    if step == args.max_steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    for _ in range(args.grad_accum):
        with autocast:
            _, loss = model(*get_batch("train"))
        (loss / args.grad_accum).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 50 == 0:
        dt, n_steps = time.time() - t0, step + 1 - last_log
        t0, last_log = time.time(), step + 1
        print(f"step {step:5d} loss {loss.item():.4f} lr {lr_at(step):.2e} grad norm {norm:.2f} "
              f"{tokens_per_step * n_steps / max(dt, 1e-9):,.0f} tok/s")
