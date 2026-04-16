# UDL ch. 7 — Gradients and initialization

- Backpropagation = the chain rule applied in the reverse order of the forward pass. The forward pass stores pre-activations; the backward pass multiplies by local derivatives.
- For `h = a[β + Ω h_prev]`: `∂L/∂Ω = (∂L/∂f) h_prevᵀ` and `∂L/∂h_prev = Ωᵀ (a'(f) ⊙ ∂L/∂h)`.
- Algorithmic differentiation (reverse mode) makes the gradient cost about the same as one forward pass, whatever the number of parameters.
- Initialization controls how variance flows: if weights are too small, activations and gradients shrink layer after layer; too large, they explode.
- He initialization for ReLU: `Var[Ω] = 2 / D_in` keeps the variance of pre-activations constant through depth (the factor 2 compensates ReLU zeroing half the inputs). Xavier/Glorot `2 / (D_in + D_out)` is the compromise for symmetric activations.

Code:
- `backprop_numpy.py`: two-layer network with a hand-written backward pass, checked against finite differences.
- `init_experiments.py`: activation variance through 50 ReLU layers for different weight scales.
