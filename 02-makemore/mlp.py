import torch
import torch.nn.functional as F

from common import splits, steps_arg

block_size, n_embd, n_hidden = 3, 10, 200
(Xtr, Ytr), (Xdev, Ydev), (Xte, Yte), stoi, itos = splits(block_size)
V = len(stoi)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((V, n_embd), generator=g)
W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) * 0.2
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, V), generator=g) * 0.01
b2 = torch.zeros(V)
params = [C, W1, b1, W2, b2]
for p in params:
    p.requires_grad = True


def forward(X):
    emb = C[X].view(X.shape[0], -1)
    h = torch.tanh(emb @ W1 + b1)
    return h @ W2 + b2


steps = steps_arg(200000)
for i in range(steps):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    loss = F.cross_entropy(forward(Xtr[ix]), Ytr[ix])
    for p in params:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < steps // 2 else 0.01
    for p in params:
        p.data += -lr * p.grad
    if i % 10000 == 0:
        print(f"{i:7d}/{steps}: {loss.item():.4f}")


@torch.no_grad()
def split_loss(X, Y):
    return F.cross_entropy(forward(X), Y).item()


print(f"train {split_loss(Xtr, Ytr):.4f}  dev {split_loss(Xdev, Ydev):.4f}")

for _ in range(10):
    out, context = [], [0] * block_size
    while True:
        probs = F.softmax(forward(torch.tensor([context])), dim=1)
        ix = torch.multinomial(probs, 1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    print("".join(out))
