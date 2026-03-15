import numpy as np


def make_data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2, n)
    y = 0.5 + 1.8 * x + rng.normal(0, 0.2, n)
    return x, y


def closed_form(x, y):
    X = np.column_stack([np.ones_like(x), x])
    return np.linalg.solve(X.T @ X, X.T @ y)


def loss(phi, x, y):
    return np.mean((phi[0] + phi[1] * x - y) ** 2)


def gradient(phi, x, y):
    r = phi[0] + phi[1] * x - y
    return np.array([2 * r.mean(), 2 * (r * x).mean()])


def gradient_descent(x, y, lr=0.1, steps=500):
    phi = np.zeros(2)
    history = []
    for _ in range(steps):
        phi = phi - lr * gradient(phi, x, y)
        history.append(loss(phi, x, y))
    return phi, history


if __name__ == "__main__":
    x, y = make_data()
    exact = closed_form(x, y)
    approx, history = gradient_descent(x, y)
    print("closed form:", exact.round(4), "loss", round(loss(exact, x, y), 5))
    print("gradient descent:", approx.round(4), "loss", round(history[-1], 5))
