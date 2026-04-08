import numpy as np


def make_problem(n=2000, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true + rng.normal(0, 0.5, n)
    w_star = np.linalg.lstsq(X, y, rcond=None)[0]
    return X, y, w_star


def sgd(X, y, steps, step_size, batch=1, seed=1):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    for n in range(1, steps + 1):
        idx = rng.integers(0, len(y), batch)
        g = 2 * X[idx].T @ (X[idx] @ w - y[idx]) / batch
        w = w - step_size(n) * g
    return w


if __name__ == "__main__":
    X, y, w_star = make_problem()
    schedules = {
        "constant 0.01": lambda n: 0.01,
        "constant 0.002": lambda n: 0.002,
        "decreasing 1/(10+n)": lambda n: 1.0 / (10 + n),
    }
    print("steps   " + "  ".join(f"{k:>20s}" for k in schedules))
    for steps in [100, 1000, 10000, 50000]:
        errs = [np.sum((sgd(X, y, steps, s) - w_star) ** 2) for s in schedules.values()]
        print(f"{steps:6d}  " + "  ".join(f"{e:20.2e}" for e in errs))
