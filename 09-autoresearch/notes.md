# Automated experiments — notes (karpathy/autoresearch)

autoresearch hands a small but real LLM training setup to an AI coding agent: the agent edits the training script, runs a short experiment with a fixed time budget, checks whether validation bits per byte improved, keeps or reverts the change, logs it, and repeats overnight. The human writes the research program (`program.md`), not the code changes.

What I rebuilt to understand the mechanics:
- `program.md`: the rules of the game — one change at a time, same budget, judge on validation, log everything including failures, keep only improvements.
- `experiment.py`: trains the Llama-style model from `06-llm-training` for a fixed number of *seconds* (not steps) and prints validation bits per byte as JSON. A time budget makes runs comparable across configurations of different cost: a bigger model gets fewer steps, so the search optimises what matters in practice.
- `loop.py`: greedy local search. Each round mutates one knob of the best configuration, runs it as a separate process (a crash does not kill the loop), keeps it if it beats the best, and appends a row to `results.tsv`. Configurations already tried are skipped. Here the "researcher" is a random proposal; in autoresearch it is an LLM agent that reads the code and the log and proposes edits, including structural ones a config search cannot express.

Observations from short local runs:
- With budgets of a few seconds the noise between runs is as large as most effects; re-running the baseline is necessary to know what an improvement is.
- The fixed-time setup penalises larger models early: within a short budget a smaller, faster model often wins. The ranking can flip with longer budgets.

This connects back to the foundations: it is hyperparameter search with a validation set (UDL ch. 8), automated, and it only works if the evaluation is cheap and trustworthy.
