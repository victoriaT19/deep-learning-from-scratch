# UDL ch. 2 — Supervised learning

- A model is a family of functions `y = f[x, φ]`; learning means choosing the parameters `φ` from pairs `{x_i, y_i}`.
- The loss `L[φ]` scores how badly the model fits the training data; training is `φ̂ = argmin L[φ]`.
- Linear regression in 1D: `f[x, φ] = φ0 + φ1 x`, least-squares loss `Σ (φ0 + φ1 x_i − y_i)²`.
- The loss surface of linear regression is convex: one minimum, reachable in closed form or by gradient descent.
- Testing on held-out data is the only honest measure; a low training loss says little about new inputs.

Questions I wrote down:
- Why squared error? (answered in ch. 5: it is the negative log-likelihood of a Gaussian.)
- What changes when the model is not linear in the parameters? (ch. 3: the surface stops being convex.)

Code: `linear_regression.py` (closed form vs. gradient descent on the same data).
