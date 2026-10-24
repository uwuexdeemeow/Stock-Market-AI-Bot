> **Legacy (moved 2026-10-25):** this script now lives in `legacy/` and needs
> `pip install -r requirements-legacy.txt`. Live trading does not use it.
> See `Documentation/LEGACY_ML_CLEANUP_PLAN.md`.

# model_self_check.py — What It Does and How to Run It

## What This Script Does (Plain English)

`model_self_check.py` is a **health check** for trained models. Before predict.py runs, it verifies that:
- The model artifact files exist and load cleanly
- The features the model expects are present in the data
- Critical features aren't missing (which would silently break predictions)
- Optional features (like sentiment) are flagged but don't block the run

Think of it as asking: "Is everything ready for this ticker before we generate signals?"

---

## How to Run It

```bash
# Check all tickers that have model files in models/
python legacy/model_self_check.py

# Check a specific ticker
python legacy/model_self_check.py --ticker AAPL

# Quiet mode (only print errors)
python legacy/model_self_check.py --quiet
```

**Example output:**
```
=== SELF CHECK: AAPL ===
model_version - xgb_complete_v2
feature_count - 147
selected_mode - xgboost_only
parquet       - OK (all training feature columns present)
xgb_dir       - OK (model loads, can predict)
xgb_ret       - OK (model loads, can predict)
scaler        - OK
calibrator    - OK
✓ AAPL passed all checks
```

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Critical features** | Features the model absolutely needs. If missing, predictions will be garbage or crash. Examples: RSI, MACD, returns. |
| **Optional features** | Features that improve predictions but aren't required. Examples: sentiment scores. If unavailable, they get zero-filled. |
| **Scaler** | The feature normalizer saved at training time. Must match the features in the parquet file. |
| **Calibrator** | The confidence adjuster saved at training time. If missing, falls back to a fixed threshold. |
| **model_version** | A string tag written during training (e.g., `xgb_complete_v2`) so you know which training run produced the model. |
| **xgb_dir** | Direction model artifact — predicts UP or DOWN. |
| **xgb_ret** | Return-bucket model artifact — predicts magnitude category. |
