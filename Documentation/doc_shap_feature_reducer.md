# shap_feature_reducer.py — SHAP Feature Recommendation

## What It Does

This research utility measures how much each input changed XGBoost predictions
across actual training rows. It recommends the strongest features—15 by
default—and writes a complete ranking to `models/selected_features.json`.

It does not automatically replace production features. A smaller model must
first pass the same out-of-sample walk-forward checks as the current model.

## How To Run It

Prepare an NPZ file containing the training matrix `X` and its
`feature_names`, then run:

```bash
python3 shap_feature_reducer.py \
  --model models/pooled_xgb_dir.json \
  --matrix models/pooled_training_matrix.npz \
  --top-k 15
```

Expected output: `models/selected_features.json`. The report says
`research_only_requires_oos_validation` until a later comparison confirms the
reduced model preserves out-of-sample performance.

## Key Concepts

- **SHAP value:** How much one feature moved one prediction away from the base.
- **Mean absolute SHAP:** Average contribution size, ignoring direction.
- **Feature reduction:** Keeping strong inputs and dropping weak/redundant ones.
- **OOS validation:** Testing on later data the reducer did not use.
