import numpy as np


def quadratic(kappa=50.0):
    A = np.diag([1.0, kappa])
    return (lambda x: 0.5 * x @ A @ x), (lambda x: A @ x), np.zeros(2)


def rosenbrock():
    f = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    g = lambda x: np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])
    return f, g, np.ones(2)


def gd(grad, x, lr, steps, **_):
    for _ in range(steps):
        x = x - lr * grad(x)
    return x


def momentum(grad, x, lr, steps, beta=0.9, **_):
    v = np.zeros_like(x)
    for _ in range(steps):
        v = beta * v + grad(x)
        x = x - lr * v
    return x


def nesterov(grad, x, lr, steps, beta=0.9, **_):
    v = np.zeros_like(x)
    for _ in range(steps):
        v = beta * v + grad(x - lr * beta * v)
        x = x - lr * v
    return x


def adagrad(grad, x, lr, steps, eps=1e-8, **_):
    s = np.zeros_like(x)
    for _ in range(steps):
        g = grad(x)
        s += g * g
        x = x - lr * g / (np.sqrt(s) + eps)
    return x


def rmsprop(grad, x, lr, steps, beta=0.99, eps=1e-8, **_):
    s = np.zeros_like(x)
    for _ in range(steps):
        g = grad(x)
        s = beta * s + (1 - beta) * g * g
        x = x - lr * g / (np.sqrt(s) + eps)
    return x


def adam(grad, x, lr, steps, b1=0.9, b2=0.999, eps=1e-8, **_):
    m, v = np.zeros_like(x), np.zeros_like(x)
    for t in range(1, steps + 1):
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return x


METHODS = {"gd": gd, "momentum": momentum, "nesterov": nesterov, "adagrad": adagrad, "rmsprop": rmsprop, "adam": adam}

if __name__ == "__main__":
    for name, (f, g, xstar), x0, lrs, steps in [
        ("quadratic (kappa=50)", quadratic(), np.array([1.0, 1.0]), {"gd": 0.02, "momentum": 0.02, "nesterov": 0.02, "adagrad": 0.5, "rmsprop": 0.01, "adam": 0.05}, 300),
        ("rosenbrock", rosenbrock(), np.array([-1.2, 1.0]), {"gd": 1e-3, "momentum": 1e-4, "nesterov": 1e-4, "adagrad": 0.1, "rmsprop": 1e-3, "adam": 0.02}, 5000),
    ]:
        print(name)
        for m, fn in METHODS.items():
            x = fn(g, x0.copy(), lrs[m], steps)
            print(f"  {m:9s} distance to minimum {np.linalg.norm(x - xstar):.2e}  f = {f(x):.2e}")
