# alpha_factor_backtest.py

## What It Does
`alpha_factor_backtest.py` tests the raw factor ranking system before the model gets credit. It builds factor scores, simulates factor portfolios, compares them with benchmarks, and reports whether the overlay has enough recent edge to be trusted.

It also carries the feature-health gate into the factor research path. That means the backtest and live gate use the same diversification rules:

- At least 6 active feature clusters
- No single cluster above 25% of total factor weight

## How To Run It
Run the main factor backtest:

```bash
python3 alpha_factor_backtest.py
```

Common output files are written under `signals/`. Exact filenames can vary depending on the command options and current research setup.

## Key Concepts
- Factor score: A combined ranking score built from multiple market features.
- Cross-sectional ranking: Comparing stocks against each other on the same date.
- Overlay: The satellite stock-picking sleeve that sits on top of the core ETF exposure.
- Gate: A risk check that can block live capital even when the script still produces research output.
- Feature-health summary: Metadata explaining whether the current feature set is diverse enough for live use.
