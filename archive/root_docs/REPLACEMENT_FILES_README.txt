Replacement files included:
- train_complete.py
- predict_complete.py
- walk_forward_backtest_complete.py
- moomoo_paper_trading_complete.py

What they fix:
1) leakage / split discipline
   - train_complete.py fits scaler on train only
   - calibrator on calibration only
   - test metrics only on unseen test split

2) in-sample backtest problem
   - walk_forward_backtest_complete.py retrains in expanding windows and tests only on future windows

3) confidence threshold doing too much work
   - train_complete.py saves confidence buckets + threshold sweep
   - predict_complete.py uses confidence-bucket precision for signal quality

4) circular signal-quality logic
   - signal quality is based on empirical bucket precision, not confidence + expected return restatement

5) dangerous SHORT side in live trading
   - predict_complete.py marks live shorts non-actionable
   - moomoo_paper_trading_complete.py filters to LONG-only by default

6) sentiment features largely zero
   - train_complete.py removes sentiment features by default
   - predict_complete.py handles degraded sentiment explicitly

7) recent-accuracy gate
   - predict_complete.py only activates it if enough samples exist

Suggested usage:
- Replace train.py with train_complete.py
- Replace predict.py with predict_complete.py
- Use walk_forward_backtest_complete.py as your real validator
- Use moomoo_paper_trading_complete.py to enforce your approved live universe
