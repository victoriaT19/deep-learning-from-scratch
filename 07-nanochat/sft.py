import argparse
from pathlib import Path

import torch

from chat_format import VOCAB_SIZE, decode, render_conversation, render_prompt
from chat_model import ChatModel, Config
from engine import generate
from tasks import dataset

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--out", default="out/chat.pt")
args = p.parse_args()

torch.manual_seed(0)
cfg = Config(vocab_size=VOCAB_SIZE)
model = ChatModel(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
train, test = dataset(20000, seed=0), dataset(200, seed=1)


def batch(items):
    rows = [render_conversation(conv) for conv, _ in items]
    T = max(len(ids) for ids, _ in rows)
    x = torch.zeros(len(rows), T - 1, dtype=torch.long)
    y = torch.zeros_like(x)
    m = torch.zeros_like(x, dtype=torch.float)
    for i, (ids, mask) in enumerate(rows):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1])
        y[i, :n] = torch.tensor(ids[1:])
        m[i, :n] = torch.tensor(mask[1:], dtype=torch.float)
    return x, y, m


def accuracy(items):
    hits = 0
    for conv, answer in items:
        out = generate(model, render_prompt(conv[:1]), max_new_tokens=8)
        hits += decode(out).strip() == answer
    model.train()
    return hits / len(items)


for step in range(args.steps + 1):
    idx = torch.randint(0, len(train), (args.batch_size,))
    x, y, m = batch([train[i] for i in idx])
    _, loss = model(x, y, m)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 500 == 0:
        print(f"step {step:5d} masked loss {loss.item():.4f} test accuracy {accuracy(test[:100]):.0%}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.out)
for conv, answer in test[:5]:
    print(conv[0]["content"], "->", decode(generate(model, render_prompt(conv[:1]))).strip(), f"(expected {answer})")
