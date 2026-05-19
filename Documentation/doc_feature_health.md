# feature_health.py

## What It Does
`feature_health.py` checks whether the strategy is relying on enough independent signals. A "feature" is one input used to rank stocks, such as momentum, liquidity, volatility, or benchmark-relative strength. Some features are really the same idea in different forms, so this script groups related features into clusters.

The script also checks whether recent research says a feature is decaying. Weak or decaying features can be quarantined so they do not help the live score.

The live safety gate currently requires:

- At least 6 active feature clusters
- No single cluster above 25% of total factor weight

If the gate fails, real-capital overlay trading should stay blocked until the factor set is healthier.

## How To Run It
Usually this script is called by `alpha_factor_backtest.py`, `core_satellite_alpha.py`, or diagnostics. To refresh the health profile from the default factor specs, run the normal research/backtest command that builds scores:

```bash
python3 alpha_factor_backtest.py
```

Expected outputs:

- `signals/feature_health_profile.json`
- `signals/feature_health_profile.csv`

## Key Concepts
- Feature: A numeric signal used to rank stocks.
- Cluster: A group of features that measure the same basic idea.
- Active cluster: A cluster that still contributes to the score after quarantine checks.
- Quarantine: A feature is excluded because recent evidence says it has weakened too much.
- Gate: A pass/fail safety check before allowing the overlay to trade with real capital.
