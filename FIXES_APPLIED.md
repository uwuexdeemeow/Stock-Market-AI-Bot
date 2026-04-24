# Fixes Applied

This package fixes the issues identified before feature ablation:

1. **Paper trading triple commission bug**
   - Entry commission is deducted at fill time.
   - Exit now subtracts only the exit leg from cash and realized P&L.
   - Trade logs still record total round-trip commission.

2. **Slippage mismatch**
   - `paper_trading.py` now uses the same base slippage constant as `backtest.py`
   - Paper and backtest are directly comparable.

3. **Calibrator leakage**
   - `train.py` now uses a true three-way split:
     - train
     - calibration
     - final test
   - The isotonic calibrator is fit on the calibration split only.

4. **XGBoost single-day weakness**
   - Added `xgb_feature_engineering.py`
   - XGBoost now sees rolling tabular summaries (means/stds/deltas over 3, 5, 10 days)

5. **Strict v4 model validation**
   - `model_self_check.py` now validates `v4_hybrid` with the same strict dual-head path.

6. **XGBoost multiclass metric bug**
   - Return-bin XGBoost now uses `eval_metric="mlogloss"` instead of binary `logloss`.
