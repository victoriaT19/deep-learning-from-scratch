# Part II — Approximation

- One dimension first (ch. 3): any continuous piecewise linear function with `n` breakpoints is exactly the realization of a ReLU network with one hidden layer of about `n` neurons. Since piecewise linear interpolation of a Lipschitz function on a grid of spacing `h` has error `≤ L h`, a ReLU network with `O(1/ε)` neurons achieves uniform error `ε`.
- The construction is explicit: the slope change at each breakpoint becomes the output weight of one ReLU placed at that breakpoint.
- Multivariate (ch. 4): approximation rates depend on the dimension; for general Lipschitz functions the number of parameters grows like `ε^{−d}` (curse of dimensionality). Better rates need more structure.
- These results are existence statements: they say a good network exists, not that gradient descent finds it.

Code: `relu_interpolation.py` builds the interpolating ReLU network directly (no training) and checks the error bound as the grid gets finer.
