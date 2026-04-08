# Part III — Optimization (GF, GD, SGD)

- Gradient flow (ch. 5): the ODE `Θ'(t) = −∇L(Θ(t))`. Along it, `L(Θ(t))` is non-increasing, since `d/dt L = −‖∇L‖²`. For strongly convex `L` it converges exponentially; Łojasiewicz-type inequalities extend this beyond convexity.
- Gradient descent (ch. 6) is the explicit Euler discretization of gradient flow. For `L`-smooth, `μ`-strongly convex objectives with step `γ ≤ 1/L`, the error contracts by `(1 − γμ)` per step. Too large a step diverges; on a quadratic the threshold is `2/λ_max`.
- Momentum and Nesterov accelerate the rate on ill-conditioned problems from `κ` to roughly `√κ` iterations (κ = condition number).
- Stochastic gradient descent (ch. 7): with constant step, SGD converges to a neighbourhood of the minimum whose size scales with the step and the gradient noise; decreasing steps with `Σγ_n = ∞, Σγ_n² < ∞` give convergence, at rate `O(1/n)` in mean square for strongly convex problems.
- Adaptive methods (Adagrad, RMSprop, Adam) rescale each coordinate; the book gives their exact update rules and notes that their theory is much thinner than for plain SGD.

Code:
- `optimizers.py`: GD, momentum, Nesterov, Adagrad, RMSprop and Adam on an ill-conditioned quadratic and on Rosenbrock.
- `sgd_convergence.py`: SGD on least squares with constant vs. decreasing step sizes, error as a function of the number of steps.
