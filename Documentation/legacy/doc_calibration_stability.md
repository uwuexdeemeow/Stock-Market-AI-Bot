> **Legacy (moved 2026-10-25):** this script now lives in `legacy/` and needs
> `pip install -r requirements-legacy.txt`. Live trading does not use it.
> See `Documentation/LEGACY_ML_CLEANUP_PLAN.md`.

# Calibration Stability

## What it does

`calibration_stability.py` splits time-ordered model probabilities into folds
and measures whether their distributions change too much between periods.

## How to use it

Import `calibration_stability(probabilities, labels)`. Inputs are equally sized
pandas Series. The returned dictionary contains fold statistics, KS distances,
`max_ks`, and a `stable` result. It returns a clear not-enough-samples result
instead of inventing evidence from tiny folds.

## Key terms

- **Calibration:** a 70% forecast should succeed about 70% of the time.
- **Fold:** one time block used for evaluation.
- **KS distance:** distribution difference from 0 (same) to 1 (fully apart).
