# xgb_feature_engineering.py — What It Does and How to Use It

## What This Script Does (Plain English)

`xgb_feature_engineering.py` is a **feature expander** specifically for XGBoost. Neural networks naturally see a 90-day sequence window. XGBoost sees a single row — just one day's worth of numbers. Without enrichment, XGBoost would miss all the temporal pattern information.

This module fixes that by taking each raw feature and adding:
- Rolling **mean** over 3, 5, and 10 days
- Rolling **standard deviation** over 3, 5, and 10 days
- **Delta** vs rolling mean (is today above or below its recent average?)

So one input feature like `rsi_14` becomes 10 features: `rsi_14`, `rsi_14__mean_3`, `rsi_14__mean_5`, `rsi_14__mean_10`, `rsi_14__std_3`, etc.

---

## How to Use It (in Code)

```python
from xgb_feature_engineering import build_xgb_matrix

# Build XGBoost input matrix from a feature DataFrame
X, feature_cols = build_xgb_matrix(
    df=feature_dataframe,               # output of pipeline_shared
    feature_cols_raw=base_feature_list, # the raw column names to expand
    xgb_feature_cols=saved_cols,        # exact column order from training (ensures consistency)
)
# X is a numpy array ready for xgboost.XGBClassifier().predict(X)
```

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Rolling mean** | Average of the last N values. `rsi__mean_5` = average RSI over the last 5 days. |
| **Rolling std** | Standard deviation over last N days. High std = the feature has been noisy recently. |
| **Delta vs mean** | `rsi__delta_mean_5` = today's RSI minus its 5-day average. Tells XGBoost if RSI is currently above or below its recent trend. |
| **Feature expansion** | Creating many derived features from one raw feature to give the model more signal. |
| **Consistent column order** | `xgb_feature_cols` ensures the training-time column order matches inference time. If they differ, the model predicts on the wrong features. |
| **nan_to_num** | Replaces NaN/Inf with 0.0. XGBoost can handle some NaN internally, but explicit cleanup prevents warnings and edge cases. |

---

## Why This Matters

XGBoost is a tree-based model. Decision trees split on single feature values like "is RSI > 65?" Rolling summaries let it ask richer questions like "is RSI elevated AND has been trending up for 5 days?" — a much stronger signal than either alone.
