import numpy as np


def gaussian_nll(y, mu, sigma=1.0):
    return np.mean(0.5 * np.log(2 * np.pi * sigma**2) + (y - mu) ** 2 / (2 * sigma**2))


def mse(y, mu):
    return np.mean((y - mu) ** 2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, logits):
    p = sigmoid(logits)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def cross_entropy(labels, logits):
    return -np.mean(log_softmax(logits)[np.arange(len(labels)), labels])


def categorical_nll(labels, logits):
    p = np.exp(log_softmax(logits))
    return -np.mean(np.log(p[np.arange(len(labels)), labels]))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y, mu = rng.normal(size=100), rng.normal(size=100)
    print("NLL - (MSE/2 + const):", gaussian_nll(y, mu) - (mse(y, mu) / 2 + 0.5 * np.log(2 * np.pi)))
    labels, logits = rng.integers(0, 5, 64), rng.normal(size=(64, 5))
    print("CE vs categorical NLL:", cross_entropy(labels, logits), categorical_nll(labels, logits))
    yb, zb = rng.integers(0, 2, 64), rng.normal(size=64)
    two_class = np.column_stack([np.zeros(64), zb])
    print("BCE vs 2-class CE:", binary_cross_entropy(yb, zb), cross_entropy(yb, two_class))
