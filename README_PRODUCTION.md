# Quant Pipeline Production Package

This package upgrades the renamed stock-ML pipeline into a more production-ready,
quant-style workflow.

Included files:
- settings.py
- scanner.py
- research.py
- train.py
- pipeline_shared.py
- sentiment_engine.py
- social_sentiment.py  (copied unchanged)
- setup.py             (copied unchanged)
- model_self_check.py
- predict.py
- backtest.py
- clean_ticker_models.py

## What was implemented from ANALYSIS.md
- scanner market-regime filter
- scanner model-status tagging (needs_research vs refresh_only)
- research target horizon aligned to RETURN_HORIZON_DAYS
- historical options leakage mitigated by neutral historical options features
- pandas bfill deprecation fixed
- hardcoded Finnhub key removed from settings
- weighted return loss to reduce "flat" overprediction
- walk-forward epochs increased to 35
- Transformer recency weighting added
- simple cross-source headline deduplication in sentiment_engine
- live social sentiment integrated into research and prediction
- signal_quality field added to prediction output
- fixed-confidence default for backtesting instead of in-sample threshold sweep
- Sharpe ratio annualised by trades/year instead of sqrt(252)
- stronger position sizing logic in backtest

## Known limitations
- Historical social sentiment remains sparse unless you have paid historical access.
- Historical options features are neutralized to avoid leakage; live prediction still uses a current snapshot.
- The confidence score is improved, but not a full probability-calibration model.

## Recommended order
1. python setup.py
2. python scanner.py
3. python research.py --ticker AAPL
4. python train.py --ticker AAPL
5. python model_self_check.py --ticker AAPL
6. python predict.py --ticker AAPL
7. python backtest.py --ticker AAPL
