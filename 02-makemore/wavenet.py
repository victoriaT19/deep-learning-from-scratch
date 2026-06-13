import torch
import torch.nn.functional as F

from common import splits, steps_arg
from layers import BatchNorm1d, Embedding, FlattenConsecutive, Linear, Sequential, Tanh

block_size, n_embd, n_hidden = 8, 24, 128
(Xtr, Ytr), (Xdev, Ydev), _, stoi, itos = splits(block_size)
V = len(stoi)
g = torch.Generator().manual_seed(42)

model = Sequential([
    Embedding(V, n_embd, g),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, V, generator=g),
])
with torch.no_grad():
    model.layers[-1].weight *= 0.1
params = model.parameters()
for p in params:
    p.requires_grad = True
print(f"parameters: {sum(p.nelement() for p in params)}")

steps = steps_arg(200000)
for i in range(steps):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
    for p in params:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < steps * 3 // 4 else 0.01
    for p in params:
        p.data += -lr * p.grad
    if i % 10000 == 0:
        print(f"{i:7d}/{steps}: {loss.item():.4f}")

model.train(False)
with torch.no_grad():
    print(f"train {F.cross_entropy(model(Xtr), Ytr).item():.4f}  dev {F.cross_entropy(model(Xdev), Ydev).item():.4f}")
    for _ in range(8):
        out, context = [], [0] * block_size
        while True:
            ix = torch.multinomial(F.softmax(model(torch.tensor([context])), 1), 1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        print("".join(out))
