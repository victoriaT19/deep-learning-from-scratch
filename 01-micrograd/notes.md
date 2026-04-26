# micrograd — notes (nn-zero-to-hero, lecture 1)

Reimplemented from scratch after watching "The spelled-out intro to neural networks and backpropagation" and reading karpathy/micrograd.

- A `Value` wraps a scalar and remembers the operation and the operands that produced it, so every expression builds a DAG.
- `backward()` topologically sorts the DAG and calls each node's local `_backward` in reverse order. Each local rule is one line of calculus: `+` routes the gradient unchanged, `*` swaps the operands, `tanh` multiplies by `1 − t²`.
- Gradients must accumulate (`+=`), not be assigned: a node used twice (e.g. `a + a`) receives gradient from both paths. This was the bug in my first version.
- `zero_grad` before each backward pass, for the same reason.
- An MLP is just neurons built from `Value`s; PyTorch does exactly the same thing with tensors instead of scalars, which is why the test in `test_engine.py` can compare the two directly.

Results: `train_moons.py` trains a 2-16-16-1 MLP with a max-margin loss and L2 penalty on `make_moons`, reaching 100% training accuracy in 100 steps (slow: every scalar is a Python object).

Connection to UDL ch. 7: this is reverse-mode algorithmic differentiation, the same algorithm as `00-foundations/udl/backprop_numpy.py`, only automatic.
