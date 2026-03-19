# UDL ch. 3 — Shallow neural networks

- A shallow network computes hidden units `h_d = a[θ_d0 + θ_d1 x]` and outputs `y = φ0 + Σ φ_d h_d`.
- With ReLU, each hidden unit adds one "joint": the output is piecewise linear with at most `D` joints for `D` hidden units.
- Universal approximation: with enough hidden units a shallow network can approximate any continuous function on a compact set as closely as wanted. It says nothing about how many units, or whether training finds them.
- Multivariate inputs: each hidden unit splits the input space with a hyperplane; the regions are convex polytopes.
- The number of linear regions grows with width, but slowly compared with what depth gives (ch. 4).

Code: `shallow_network.py` fits `sin(2x)` with a ReLU network and hand-written gradients, then counts the joints of the learned function.
