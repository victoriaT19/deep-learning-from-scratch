# UDL ch. 9 — Regularization

- Explicit regularization adds a term to the loss: L2 (weight decay) is a Gaussian prior on the weights; MAP estimation instead of maximum likelihood.
- Implicit regularization: gradient descent with a finite step behaves like gradient flow plus a penalty on the squared gradient norm; SGD adds a further term that favours flat regions.
- Early stopping limits how far the weights travel from the initialization, which acts like L2.
- Ensembles average models trained from different seeds or data resamples; dropout approximates an ensemble of subnetworks by randomly zeroing hidden units during training.
- Data augmentation encodes known invariances; label smoothing stops the network from becoming overconfident.
- Transfer learning and self-supervised pretraining are regularization too: they start from parameters that already encode useful structure.

Code: `regularization.py` fits a degree-12 polynomial on 15 points with and without an L2 penalty and compares validation error over λ.
