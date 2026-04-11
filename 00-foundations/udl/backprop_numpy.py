import numpy as np


def init(d_in, d_hidden, d_out, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / d_in), (d_hidden, d_in)),
        "b1": np.zeros(d_hidden),
        "W2": rng.normal(0, np.sqrt(2 / d_hidden), (d_out, d_hidden)),
        "b2": np.zeros(d_out),
    }


def forward(p, x):
    f1 = p["W1"] @ x + p["b1"][:, None]
    h1 = np.maximum(f1, 0)
    f2 = p["W2"] @ h1 + p["b2"][:, None]
    return f2, (f1, h1)


def loss_and_grads(p, x, y):
    out, (f1, h1) = forward(p, x)
    n = x.shape[1]
    loss = np.sum((out - y) ** 2) / n
    d2 = 2 * (out - y) / n
    g = {"W2": d2 @ h1.T, "b2": d2.sum(1)}
    d1 = (p["W2"].T @ d2) * (f1 > 0)
    g["W1"] = d1 @ x.T
    g["b1"] = d1.sum(1)
    return loss, g


def numerical_grad(p, x, y, key, eps=1e-6):
    g = np.zeros_like(p[key])
    for idx in np.ndindex(p[key].shape):
        old = p[key][idx]
        p[key][idx] = old + eps
        up = loss_and_grads(p, x, y)[0]
        p[key][idx] = old - eps
        down = loss_and_grads(p, x, y)[0]
        p[key][idx] = old
        g[idx] = (up - down) / (2 * eps)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(3, 8)), rng.normal(size=(2, 8))
    p = init(3, 5, 2)
    _, g = loss_and_grads(p, x, y)
    for key in p:
        err = np.max(np.abs(g[key] - numerical_grad(p, x, y, key)))
        print(f"{key}: max abs difference {err:.2e}")
