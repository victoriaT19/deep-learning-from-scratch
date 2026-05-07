import torch

from common import load_words, vocab

words = load_words()
stoi, itos = vocab(words)
V = len(stoi)
N = torch.zeros((V, V), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]
    for a, b in zip(chs, chs[1:]):
        N[stoi[a], stoi[b]] += 1

P = (N + 1).float()
P /= P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    out, ix = [], 0
    while True:
        ix = torch.multinomial(P[ix], 1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print("".join(out))

log_likelihood, n = 0.0, 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for a, b in zip(chs, chs[1:]):
        log_likelihood += torch.log(P[stoi[a], stoi[b]]).item()
        n += 1
print(f"average negative log likelihood: {-log_likelihood / n:.4f}")
