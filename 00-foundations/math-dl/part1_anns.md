# Mathematical Introduction to Deep Learning — Part I (ANNs)

Jentzen, Kuckuck, von Wurstemberger. Read alongside UDL: same objects, written with full mathematical precision.

- A fully-connected feedforward ANN is described by its architecture `(l_0, l_1, ..., l_L)` and a parameter vector `θ ∈ ℝ^d` with `d = Σ_k l_k (l_{k−1} + 1)`. The realization function maps `ℝ^{l_0} → ℝ^{l_L}` by alternating affine maps and componentwise activations.
- Two views of the same network: the *vectorized* description (one long parameter vector, convenient for optimization) and the *structured* one (a tuple of weight matrices and bias vectors, convenient for composition).
- Activations covered: ReLU, clipping, softplus, GELU, sigmoid, tanh, swish. Many proofs only use that the activation is piecewise linear or smooth.
- Chapter 1 also defines CNNs, ResNets and RNNs in the same formalism; a ResNet adds the identity through skip connections, `x_{k+1} = x_k + A_k(x_k)`.
- Chapter 2, ANN calculus: compositions, parallelizations and sums of ANNs are again ANNs, with explicit bounds on depth and number of parameters. This is the toolkit later used to *construct* networks that approximate a function.

What I found useful: writing `d = Σ l_k (l_{k−1} + 1)` by hand for a few architectures and checking it against `sum(p.numel())` in PyTorch.
