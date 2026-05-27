import numpy as np


def build(f, a, b, n):
    xs = np.linspace(a, b, n + 1)
    ys = f(xs)
    slopes = np.diff(ys) / np.diff(xs)
    weights = np.concatenate([[slopes[0]], np.diff(slopes)])
    return xs[:-1], weights, ys[0]


def realize(net, x):
    breakpoints, weights, bias = net
    return bias + np.maximum(x[:, None] - breakpoints[None, :], 0) @ weights


if __name__ == "__main__":
    f = np.cos
    lipschitz = 1.0
    x = np.linspace(0, 2 * np.pi, 10001)
    print("neurons  max error  bound L*h")
    for n in [4, 8, 16, 32, 64, 128]:
        net = build(f, 0, 2 * np.pi, n)
        err = np.max(np.abs(realize(net, x) - f(x)))
        print(f"{n:7d}  {err:9.5f}  {lipschitz * 2 * np.pi / n:9.5f}")
