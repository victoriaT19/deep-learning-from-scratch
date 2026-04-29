import numpy as np


def truth(x):
    return np.sin(2 * np.pi * x)


def fit_predict(x, y, degree, x_test):
    coefs = np.polyfit(x, y, degree)
    return np.polyval(coefs, x_test)


def decompose(degree, n=20, trials=300, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x_test = np.linspace(0.05, 0.95, 50)
    preds = []
    for _ in range(trials):
        x = rng.uniform(0, 1, n)
        y = truth(x) + rng.normal(0, noise, n)
        preds.append(fit_predict(x, y, degree, x_test))
    preds = np.array(preds)
    bias2 = np.mean((preds.mean(0) - truth(x_test)) ** 2)
    variance = np.mean(preds.var(0))
    return bias2, variance, noise**2


if __name__ == "__main__":
    print("degree  bias^2   variance  expected test mse")
    for degree in [1, 2, 3, 5, 7]:
        b, v, s = decompose(degree)
        print(f"{degree:6d}  {b:7.4f}  {v:8.4f}  {b + v + s:8.4f}")
