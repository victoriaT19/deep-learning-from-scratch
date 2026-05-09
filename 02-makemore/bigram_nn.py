import torch
import torch.nn.functional as F

from common import load_words, steps_arg, vocab

words = load_words()
stoi, itos = vocab(words)
V = len(stoi)
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for a, b in zip(chs, chs[1:]):
        xs.append(stoi[a])
        ys.append(stoi[b])
xs, ys = torch.tensor(xs), torch.tensor(ys)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((V, V), generator=g, requires_grad=True)
xenc = F.one_hot(xs, num_classes=V).float()

for step in range(steps_arg(200)):
    logits = xenc @ W
    loss = F.cross_entropy(logits, ys) + 0.01 * (W**2).mean()
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad
    if step % 50 == 0:
        print(f"step {step:4d} loss {loss.item():.4f}")
print(f"final loss {loss.item():.4f} (the count-based bigram reaches about 2.45)")
