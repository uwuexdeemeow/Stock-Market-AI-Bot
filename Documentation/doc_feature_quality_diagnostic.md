# feature_quality_diagnostic.py — Feature Quality Report

## What It Does

The JSON and CSV outputs are written atomically, so live gates never see a
half-written feature-quality report if the script crashes mid-refresh.

`feature_quality_diagnostic.py` checks whether the factor features used by the
core-satellite overlay are useful enough for live paper trading. It measures
predictive power, stability, regime behavior, signal decay, turnover, and
feature correlation.

The daily signal generator reads this report before trading. If the report is
missing, stale, or says too many features are weak, the bot blocks trading.

Regime warnings are labeled by the weak side. For example, a feature with
`bull_ic=0.0047` and `bear_ic=0.0086` is reported as "weaker in bull markets",
not "bull-only".

## How To Run It

```bash
python3 feature_quality_diagnostic.py
python3 feature_quality_diagnostic.py --top 48
```

Inputs:

- `logs/feature_ic_shortlist.csv` — ranked feature shortlist from research.
- `data/*.parquet` — per-ticker factor data with raw feature columns.

Outputs:

- `signals/feature_quality_report.json` — detailed feature diagnostics.
- `signals/feature_quality_summary.csv` — compact table of grades and metrics.

Note: `--top` limits the diagnostic feature set only. It no longer rewrites the
live `feature_health_profile` with a partial feature set.

## Key Terms

- **Feature** — one input signal, such as recent return, liquidity, or sector
  relative strength.
- **IC** — information coefficient; rank correlation between feature score and
  future return.
- **Regime** — market environment, such as bull or bear.
- **Turnover** — how often the top-ranked stocks change.
- **Correlation cluster** — a group of features that mostly say the same thing.
