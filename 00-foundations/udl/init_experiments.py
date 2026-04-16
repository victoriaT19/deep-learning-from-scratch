import numpy as np


def activation_variances(scale, depth=50, width=100, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(width, n))
    out = []
    for _ in range(depth):
        W = rng.normal(0, np.sqrt(scale / width), (width, width))
        f = W @ h
        h = np.maximum(f, 0)
        out.append(float(f.var()))
    return out


if __name__ == "__main__":
    for scale, name in [(1.0, "too small (1/D)"), (2.0, "He (2/D)"), (3.0, "too large (3/D)")]:
        v = activation_variances(scale)
        print(f"{name:18s} variance at layers 1, 10, 50: {v[0]:.3g}, {v[9]:.3g}, {v[-1]:.3g}")
