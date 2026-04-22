# tune_xgb_best_tickers.py — What It Does and How to Run It

## What This Script Does (Plain English)

`tune_xgb_best_tickers.py` is the **hyperparameter tuner** for XGBoost. Instead of using the default model settings in `settings.py`, it systematically tests different combinations of parameters to find what works best for each specific ticker.

**Important:** The tuner uses a 4-way temporal split so the test set is never touched during the search. This prevents "hyperparameter overfitting" — where params look great on the test set but fail in live trading because they were tuned to that specific period.

---

## How to Run It

```bash
# Find best params for AAPL
python tune_xgb_best_tickers.py --ticker AAPL

# Find best params AND immediately retrain with them
python tune_xgb_best_tickers.py --ticker AAPL --apply

# Tune multiple tickers (slower — can take 30–60 min per ticker)
python tune_xgb_best_tickers.py --ticker MSFT TSLA GOOGL
```

**Output:**
- `models/tuning/<TICKER>_best_xgb_params.json` — the winning parameter set
- If `--apply` is used: the model in `models/` is retrained with the best params

---

## What Parameters Are Tuned?

| Parameter | What It Controls | Default |
|---|---|---|
| `n_estimators` | Number of trees in the forest | 500 |
| `max_depth` | How deep each tree can grow. Deeper = more complex = more likely to overfit | 6 |
| `learning_rate` | How much each tree adjusts the prediction. Smaller = slower but more robust | 0.05 |
| `subsample` | Fraction of training rows each tree sees. Prevents overfitting | 0.8 |
| `colsample_bytree` | Fraction of features each tree can use | 0.7 |
| `min_child_weight` | Minimum data points required to create a leaf node | 3 |
| `gamma` | Minimum gain required to make a split. Higher = more conservative trees | 0.1 |

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Hyperparameter** | A setting that controls how the model trains, not what it learns from data. Must be chosen before training begins. |
| **Grid search** | Try every combination in a predefined grid and pick the best one. Gets slow with many parameters. |
| **4-way split** | train / calibration (early stopping) / validation (param selection) / test (final report only). Test is read-only during search. |
| **Hyperparameter overfitting** | When params are "tuned" on the same data used to evaluate them. The numbers look good but won't generalize. This script prevents it. |
| **--apply flag** | Without it, tuning is read-only. The model in `models/` doesn't change. Add `--apply` only when you've reviewed the output and are ready to commit. |
