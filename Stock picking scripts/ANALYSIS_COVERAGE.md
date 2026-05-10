Analysis coverage checklist

Implemented:
[x] scanner regime filter
[x] scanner model status tagging
[x] target horizon mismatch fix
[x] historical options leakage mitigation
[x] pandas .bfill() compatibility
[x] secure Finnhub env-var settings
[x] return-loss class weighting
[x] longer walk-forward estimate
[x] transformer recency weighting
[x] cross-source exact-title dedup
[x] social sentiment integration
[x] signal_quality
[x] Sharpe annualisation fix
[x] improved position sizing

Partially addressed:
[~] confidence calibration (improved margin-based, not full baseline-calibration)
[~] threshold overfitting (replaced by fixed default; no fancy threshold optimiser)
[~] look-ahead bias verification (not implemented as a standalone audit script)
[~] fundamental fraud/SEC filter (not implemented)

Optional future work:
[ ] XGBoost scanner ranker
[ ] CPI / Fed calendar features
[ ] bid-ask spread microstructure features
[ ] explicit feature look-ahead audit utility
