# UDL ch. 4 — Deep neural networks

- Composing two shallow networks "folds" the input space: the second network sees the first one's output, so each of its linear regions is replicated in every region of the first.
- A deep network is a sequence of affine maps and nonlinearities: `h_k = a[β_k + Ω_k h_{k−1}]`.
- With the same number of parameters, depth creates far more linear regions than width (exponential in depth vs. polynomial in width).
- Hyperparameters: depth `K`, width `D_k`. They are chosen with held-out data, not learned.
- Depth is not free: deeper networks are harder to train (gradients, initialization — ch. 7) and need tricks like residual connections (ch. 11).

Takeaway: depth buys expressiveness per parameter; the price is optimization difficulty.
