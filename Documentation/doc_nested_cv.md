# Nested Cross-Validation

## What it does

`nested_cv.py` performs time-ordered nested walk-forward search. Inner folds
select XGBoost parameters; untouched outer folds estimate performance, reducing
the chance of tuning directly to the test period.

## How to use it

Import `nested_walk_forward_search` and pass features, labels, a parameter grid,
and optional fixed parameters. The result is one `FoldResult` per usable outer
fold with its chosen parameters and AUC score. No files are written directly.

## Key terms

- **Nested CV:** separate tuning folds inside separate evaluation folds.
- **Embargo:** a gap that prevents overlapping future-return leakage.
- **Hyperparameter:** a model setting chosen before fitting.
