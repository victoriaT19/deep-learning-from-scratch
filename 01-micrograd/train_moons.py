import random

from sklearn.datasets import make_moons

from engine import Value
from nn import MLP

random.seed(1337)
X, y = make_moons(n_samples=100, noise=0.1, random_state=0)
y = y * 2 - 1
model = MLP(2, [16, 16, 1])


def loss_fn():
    scores = [model([Value(a), Value(b)]) for a, b in X]
    hinge = [(1 + -yi * s).relu() for yi, s in zip(y, scores)]
    data_loss = sum(hinge) * (1.0 / len(hinge))
    reg = 1e-4 * sum(p * p for p in model.parameters())
    acc = sum((yi > 0) == (s.data > 0) for yi, s in zip(y, scores)) / len(y)
    return data_loss + reg, acc


for step in range(100):
    total, acc = loss_fn()
    model.zero_grad()
    total.backward()
    lr = 1.0 - 0.9 * step / 100
    for p in model.parameters():
        p.data -= lr * p.grad
    if step % 10 == 0 or step == 99:
        print(f"step {step:3d} loss {total.data:.4f} accuracy {acc:.0%}")
