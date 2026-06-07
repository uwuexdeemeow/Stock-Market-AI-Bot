# feature_health.py

## What It Does
`feature_health.py` checks whether the strategy is relying on enough independent signals. A "feature" is one input used to rank stocks, such as momentum, liquidity, volatility, or benchmark-relative strength. Some features are really the same idea in different forms, so this script groups related features into clusters.

The script also checks whether recent research says a feature is decaying. Weak or decaying features can be quarantined so they do not help the live score.

The live safety gate currently requires:

- At least 6 active feature clusters
- No single cluster above 25% of total factor weight

If the gate fails, real-capital overlay trading should stay blocked until the factor set is healthier.

## How To Run It
Usually this script is called by `alpha_factor_backtest.py`, `core_satellite_alpha.py`, or diagnostics.

To refresh the health profile directly and print a short summary:

```bash
python3 feature_health.py
```

Useful options:

```bash
# Use fewer shortlisted features
python3 feature_health.py --max-specs 24

# Print more clusters/features in the console summary
python3 feature_health.py --limit 15

# Preview without rewriting the JSON/CSV profile files
python3 feature_health.py --no-write
```

To refresh the health profile as part of the normal research/backtest command that builds scores:

```bash
python3 alpha_factor_backtest.py
```

Expected outputs:

- `signals/feature_health_profile.json`
- `signals/feature_health_profile.csv`

Both files are written atomically, meaning the script writes a complete
temporary file first and then swaps it into place so live trading never reads a
half-written health profile.

The direct command now prints:

- Whether the feature-health gate passed
- Number of raw features and feature clusters
- Number of active clusters
- Maximum cluster weight
- Quarantined and watchlist features
- Top active clusters and their weights

## Key Concepts
- Feature: A numeric signal used to rank stocks.
- Cluster: A group of features that measure the same basic idea.
- Active cluster: A cluster that still contributes to the score after quarantine checks.
- Quarantine: A feature is excluded because recent evidence says it has weakened too much.
- Gate: A pass/fail safety check before allowing the overlay to trade with real capital.
