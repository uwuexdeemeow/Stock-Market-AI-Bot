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
[x] confidence calibration with saved calibrators and stability diagnostics
[~] threshold overfitting (replaced by fixed default; no fancy threshold optimiser)
[x] standalone multi-layer look-ahead audit in `leakage_audit.py`
[~] fundamental fraud/SEC filter (not implemented)

Optional future work:
[x] XGBoost cross-sectional ranker utilities and daily IC evaluation
[ ] CPI / Fed calendar features — research-gated until a versioned,
    point-in-time event calendar is supplied; do not infer old calendars from
    today's page.
[ ] bid-ask spread microstructure features — live quotes are available for
    execution guards, but no historical quote archive exists for honest model
    training yet.
[x] explicit feature look-ahead audit utility
