# Research program

Goal: the lowest validation bits per byte on Tiny Shakespeare within a fixed training-time budget.

Rules for each experiment:
- Change one thing at a time in the configuration (`experiment.py --config`), so the effect can be attributed.
- Every run gets the same time budget; a change only counts if it beats the best run so far on validation, not training, loss.
- Log every run in `results.tsv`, including the ones that got worse or crashed.
- Keep a change only if it improves the best result; otherwise revert to the best configuration.
- Prefer simpler configurations when results are within noise (re-run the best one to estimate the noise).

Knobs: depth, width, heads, learning rate, warm-up, batch size, sequence length, weight decay.
