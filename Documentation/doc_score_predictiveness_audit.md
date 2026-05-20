# score_predictiveness_audit.py - Audit Selector Quality

## What It Does

`score_predictiveness_audit.py` checks whether the inner walkforward metrics
that selected a config actually predicted the later out-of-sample result.

It reads a walkforward CSV, builds the matching OOS objective score, and ranks
inner/config fields by correlation with the OOS result. Positive correlation
means the field helped point toward better OOS performance. Negative
correlation means the field pointed the selector the wrong way.

The report includes both raw correlations and a fold-year-detrended
correlation. The detrended value removes a simple time trend, which helps avoid
mistaking "this happened later in history" for real predictive power.

## How To Run It

Default alpha objective audit:

```bash
python3 score_predictiveness_audit.py
```

Audit a specific walkforward:

```bash
python3 score_predictiveness_audit.py --csv signals/core_satellite_nested_walkforward_alphaqqq.csv --objective alpha_vs_qqq
```

Print more rows:

```bash
python3 score_predictiveness_audit.py --limit 20
```

Expected outputs:

- `signals/score_predictiveness_audit.csv`
- `signals/score_predictiveness_audit.json`

## Important Limitation

This is a selected-config audit. It studies the configs that the walkforward
actually selected. It does not replay every rejected candidate config unless
the walkforward JSON contains full candidate-level history.

That means it is best for identifying suspicious score components, not for
proving a replacement selector by itself. After changing a component, rerun the
nested walkforward and analyzer.

## Key Concepts

- Inner score: The validation score used to choose a config before OOS testing.
- OOS: Out-of-sample, the held-out year used as the real test.
- Objective: The formula used to judge OOS quality, such as `alpha_vs_qqq`.
- Correlation: A value from -1 to +1. Positive is good here; negative means
  the field moved against future OOS quality.
- Selected-config audit: An audit of chosen configs only, not every possible
  candidate in the grid.
