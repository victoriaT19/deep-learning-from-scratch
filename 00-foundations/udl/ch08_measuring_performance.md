# UDL ch. 8 — Measuring performance

- Test error decomposes into noise (irreducible), bias (the model family cannot represent the truth) and variance (the fit depends on which training sample we drew).
- Classical picture: more capacity lowers bias and raises variance, so test error is U-shaped in capacity.
- Double descent: past the interpolation threshold (enough parameters to fit the training set exactly), test error can fall again. Among the many interpolating solutions, training tends to pick smooth ones.
- In high dimensions, intuition breaks: most of the volume of a hypercube is near its corners, and random points are almost equidistant.
- Choosing hyperparameters on the test set leaks information; use a validation split or cross-validation.

Code: `bias_variance.py` refits polynomials of increasing degree on many resampled training sets and estimates bias² and variance at fixed test inputs.
