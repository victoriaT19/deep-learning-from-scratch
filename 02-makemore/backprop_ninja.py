import torch
import torch.nn.functional as F

from common import splits

block_size, n_embd, n_hidden, batch = 3, 10, 64, 32
(Xtr, Ytr), _, _, stoi, _ = splits(block_size)
V = len(stoi)
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((V, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / (n_embd * block_size) ** 0.5
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn((n_hidden, V), generator=g) * 0.1
b2 = torch.randn(V, generator=g) * 0.1
bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1
params = [C, W1, b1, W2, b2, bngain, bnbias]
for p in params:
    p.requires_grad = True

ix = torch.randint(0, Xtr.shape[0], (batch,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]
n = batch
emb = C[Xb]
embcat = emb.view(n, -1)
hprebn = embcat @ W1 + b1
bnmean = hprebn.mean(0, keepdim=True)
bndiff = hprebn - bnmean
bnvar = (bndiff**2).sum(0, keepdim=True) / (n - 1)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yb)
for p in params:
    p.grad = None
loss.backward()

with torch.no_grad():
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    dhpreact = (1 - h**2) * dh
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    dhprebn = bngain * bnvar_inv / n * (n * dhpreact - dhpreact.sum(0) - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0))
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    dC.index_add_(0, Xb.view(-1), demb.view(-1, n_embd))

for name, manual, auto in [("C", dC, C), ("W1", dW1, W1), ("b1", db1, b1), ("W2", dW2, W2), ("b2", db2, b2),
                           ("bngain", dbngain, bngain), ("bnbias", dbnbias, bnbias)]:
    print(f"{name:7s} exact {torch.all(manual == auto.grad).item()!s:5s} close {torch.allclose(manual, auto.grad, atol=1e-6)!s:5s} "
          f"max diff {(manual - auto.grad).abs().max().item():.2e}")
