import numpy as np


def relu(z):
    return np.maximum(z, 0)


def init(hidden, seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.choice([-1.0, 1.0], hidden) * rng.uniform(0.5, 1.5, hidden)
    kinks = rng.uniform(-3, 3, hidden)
    return {
        "w1": w1,
        "b1": -w1 * kinks,
        "w2": rng.normal(0, 1 / np.sqrt(hidden), hidden),
        "b2": 0.0,
    }


def forward(p, x):
    z = np.outer(x, p["w1"]) + p["b1"]
    h = relu(z)
    return h @ p["w2"] + p["b2"], (z, h)


def grads(p, x, y):
    out, (z, h) = forward(p, x)
    d = 2 * (out - y) / len(x)
    dh = np.outer(d, p["w2"]) * (z > 0)
    return {
        "w1": (dh * x[:, None]).sum(0),
        "b1": dh.sum(0),
        "w2": h.T @ d,
        "b2": d.sum(),
    }, np.mean((out - y) ** 2)


def train(x, y, hidden=20, lr=0.02, steps=20000):
    p = init(hidden)
    for step in range(steps):
        g, loss = grads(p, x, y)
        for k in p:
            p[k] = p[k] - lr * g[k]
    return p, loss


def joints(p, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        kinks = -p["b1"] / p["w1"]
    return int(np.sum((kinks > lo) & (kinks < hi)))


if __name__ == "__main__":
    x = np.linspace(-3, 3, 200)
    y = np.sin(2 * x)
    p, loss = train(x, y)
    print(f"final mse {loss:.4f}, joints inside the data range: {joints(p, -3, 3)}")
