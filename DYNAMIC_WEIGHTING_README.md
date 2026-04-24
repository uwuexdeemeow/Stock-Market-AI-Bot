# Dynamic Weighting Upgrade

This package includes a new regime-aware ensemble weighting layer.

What changed:
- Added `dynamic_weighting.py`
- Added `portfolio_manager.py`
- `predict.py` now adjusts model weights by regime at inference time
- `backtest.py` applies the same regime-aware weights historically
- `train.py` still learns base walk-forward weights, then the regime layer tilts them

How it works:
- Stable / calm regime -> XGBoost gets a boost
- Defensive / bear / high-VIX regime -> sequence models get more weight
- Crisis regime -> strongest tilt toward LSTM+Attention and Transformer

Use:
- `python predict.py --ticker AAPL`
- `python backtest.py --ticker AAPL`
