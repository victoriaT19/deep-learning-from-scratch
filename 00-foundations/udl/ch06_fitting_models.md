# UDL ch. 6 — Fitting models

- Gradient descent: `φ ← φ − α ∂L/∂φ`. On non-convex surfaces it finds a local minimum or a saddle point, depending on where it starts.
- Stochastic gradient descent uses a random minibatch per step: cheaper steps, noisy gradients, and the noise helps escape poor regions. An epoch is one pass over the data.
- Momentum keeps a running average of past gradients: it smooths the zig-zag in narrow valleys. Nesterov evaluates the gradient at the look-ahead point.
- Adam normalises each parameter's step by a running estimate of the gradient's second moment, with bias correction for the first steps. It is robust to the scale of each parameter, which is why it became the default.
- Learning-rate schedules (step decay, cosine, warm-up) matter as much as the optimiser.

Code: the optimisers are implemented in `../math-dl/optimizers.py`, compared on the same badly conditioned quadratic and on the Rosenbrock function.
