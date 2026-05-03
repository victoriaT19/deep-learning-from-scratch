import numpy as np


def features(x, degree):
    return np.vander(x, degree + 1, increasing=True)


def ridge(x, y, degree, lam):
    X = features(x, degree)
    return np.linalg.solve(X.T @ X + lam * np.eye(degree + 1), X.T @ y)


def mse(w, x, y, degree):
    return np.mean((features(x, degree) @ w - y) ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(2 * np.pi * x)
    x_tr = rng.uniform(0, 1, 15)
    y_tr = f(x_tr) + rng.normal(0, 0.2, 15)
    x_va = rng.uniform(0, 1, 200)
    y_va = f(x_va) + rng.normal(0, 0.2, 200)
    degree = 12
    print("lambda     train mse  valid mse  ||w||")
    for lam in [0.0, 1e-6, 1e-4, 1e-2, 1.0]:
        w = ridge(x_tr, y_tr, degree, lam)
        print(f"{lam:8.0e}  {mse(w, x_tr, y_tr, degree):9.4f}  {mse(w, x_va, y_va, degree):9.4f}  {np.linalg.norm(w):9.1f}")
