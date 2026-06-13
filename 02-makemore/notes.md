# makemore — notes (nn-zero-to-hero, lectures 2–6)

Character-level language models on 32k names, rebuilt step by step. `common.py` downloads `names.txt` into `data/` (not committed) and builds the train/dev/test splits.

## Bigram (lecture 2)
- `bigram_counts.py`: count table, add-one smoothing, sampling, average NLL ≈ 2.45 on the training set.
- `bigram_nn.py`: the same model as a single linear layer on one-hot inputs trained with gradient descent; it converges to the same loss, because both estimate the same conditional distribution. The L2 term plays the role of the count smoothing.

## MLP (lecture 3, Bengio et al. 2003)
- `mlp.py`: embedding table `C`, context of 3 characters, tanh hidden layer. The lecture reaches a dev loss of about 2.17 with the full 200k steps; much better than the bigram's 2.45.
- Learning-rate search: sweep `10^linspace(-3, 0)` for a few hundred steps, pick where the loss falls fastest.
- Splits matter: overfitting shows up as a growing gap between train and dev loss.

## Activations, gradients, BatchNorm (lecture 4)
- `layers.py`: Linear, BatchNorm1d, Tanh, Embedding written by hand to mirror `torch.nn`.
- At initialization the expected loss is `−log(1/27) ≈ 3.30`; a much larger value means overconfident logits → scale down the last layer.
- Saturated tanh units (|t| > 0.97) pass no gradient. Kaiming-style scaling (gain 5/3 for tanh) and BatchNorm keep activations in range: in `activations_batchnorm.py` saturation drops from 11–22% to ~2–3% per layer.
- BatchNorm couples the examples of a batch (a regularizer) and needs running statistics for inference.

## Backprop ninja (lecture 5)
- `backprop_ninja.py`: manual backward pass through cross-entropy, linear, tanh, BatchNorm (in its compact closed form) and the embedding lookup (`index_add_`), checked against autograd to ~1e-8.
- Writing the BatchNorm backward by hand is what made the "sum over the batch dimension" pattern click.

## WaveNet (lecture 6)
- `wavenet.py`: context of 8, characters fused hierarchically two at a time (`FlattenConsecutive`) instead of all at once. BatchNorm needs the mean/var over both batch and time dimensions for 3D inputs.
