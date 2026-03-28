# UDL ch. 5 — Loss functions

The recipe that made losses click for me: let the network predict the parameters of a distribution over the output, then minimise the negative log-likelihood of the training targets.

| task | distribution | network predicts | loss |
|---|---|---|---|
| regression | normal | mean (fixed variance) | squared error |
| regression, heteroscedastic | normal | mean and variance | Gaussian NLL |
| binary classification | Bernoulli | logit → sigmoid | binary cross-entropy |
| multiclass | categorical | logits → softmax | cross-entropy |

- Squared error is the Gaussian NLL up to constants, so least squares is maximum likelihood.
- Cross-entropy and KL divergence differ only by the entropy of the data, which does not depend on the model.
- At inference you can return the mode, mean or full distribution.

Code: `loss_functions.py` checks numerically that MSE equals the Gaussian NLL up to an affine transform and that cross-entropy equals the categorical NLL.
