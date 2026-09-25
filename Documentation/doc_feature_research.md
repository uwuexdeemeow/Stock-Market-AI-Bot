# feature_research.py - Where, When and How Each Feature Works

## What It Does

`feature_quality_diagnostic.py` answers **whether** a feature predicts future
returns. `feature_research.py` answers **where, when and under what
conditions** it does. For each of the top features it runs six checks:

1. **Sector IC**: does the feature work in tech but fail in energy?
2. **IC time trend**: is the feature's edge shrinking over the years?
3. **Feature pairs**: which two features work better together than alone?
4. **Best holding period**: does the feature predict 5, 10 or 20 days ahead best?
5. **Recent health**: has the feature weakened in the last ~6 months?
6. **Conditional IC**: does it change with market fear (VIX), the yield curve,
   or earnings dates?

It then groups the features into plain categories (sector-specific,
decaying, horizon mismatch, conditional, strengthening, good pairs) so you
know what to look at next.

The script only **reads** data and writes a report. It never trades and never
changes the live strategy.

## How To Run It

```bash
python feature_research.py                # analyse the top 24 features
python feature_research.py --top 10       # only the top 10
python feature_research.py --pairs 10     # limit the feature-pair check to 10 features
python feature_research.py --skip-pairs   # skip the slow feature-pair check (~3 min faster)
```

Inputs:

- Price and factor files in `data/` for the broad (~147 ticker) universe.
- The ranked feature list from `alpha_factor_backtest.load_feature_specs`
  (run `research.py` first if no features are found).

Expected outputs:

- `signals/feature_research_report.json`: the full report, one entry per feature.
- `signals/feature_research_summary.csv`: one flat row per feature.
- A console summary grouped by recommendation.

`feature_health.py` and `factor_data_health.py` read
`feature_research_summary.csv`, and the daily workflow saves both files as
evidence. Run `feature_quality_diagnostic.py --top 48` afterwards, as
`core_satellite_alpha.py` suggests.

## Key Concepts

- **Feature**: a number computed from market data for each stock and day,
  such as 5-day return or RSI. The strategy ranks stocks with features.
- **IC (information coefficient)**: how well a feature's ranking of stocks on
  one day matches the ranking of their later returns. +1 is perfect, 0 is
  random, negative means the ranking points the wrong way. It is measured
  with a Spearman (rank) correlation.
- **t-stat**: IC average divided by its noise. Above about 2 suggests the
  edge is not just luck.
- **Forward return**: a stock's gain or loss over the next N trading days.
- **Sector-excess return**: the stock's return minus its sector's average,
  so the feature is judged on picking winners *within* a sector.
- **Regime**: the market's current state, for example calm vs fearful or
  above vs below its 200-day average.
- **Decay**: a feature's edge getting weaker over time, often because other
  traders found it too.
