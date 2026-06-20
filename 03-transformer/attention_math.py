import torch
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

bow_loop = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        bow_loop[b, t] = x[b, : t + 1].mean(0)

tril = torch.tril(torch.ones(T, T))
weights = tril / tril.sum(1, keepdim=True)
bow_matmul = weights @ x

scores = torch.zeros(T, T).masked_fill(tril == 0, float("-inf"))
bow_softmax = F.softmax(scores, dim=-1) @ x

print("loop == matmul:", torch.allclose(bow_loop, bow_matmul, atol=1e-6))
print("loop == masked softmax:", torch.allclose(bow_loop, bow_softmax, atol=1e-6))

head_size = 16
key, query, value = (torch.nn.Linear(C, head_size, bias=False) for _ in range(3))
k, q, v = key(x), query(x), value(x)
wei = q @ k.transpose(-2, -1) * head_size**-0.5
wei = F.softmax(wei.masked_fill(tril == 0, float("-inf")), dim=-1)
out = wei @ v
print("attention weights of the first sequence, last position:", wei[0, -1].detach().numpy().round(3))
print("variance of q·k with and without 1/sqrt(d) scaling:",
      round((q @ k.transpose(-2, -1)).var().item(), 2), round((q @ k.transpose(-2, -1) * head_size**-0.5).var().item(), 2))
