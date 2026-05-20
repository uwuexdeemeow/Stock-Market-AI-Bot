# walkforward_analyzer.py - Validate Walkforward Results

## What it does in plain language

After you run the nested walkforward, this script reads the results CSV and
checks whether the strategy validation is trustworthy.

It does not only ask, "Did the strategy beat QQQ?" It also checks whether the
inner score that selected each config actually predicted the later
out-of-sample result.

The score-predictiveness check is now target-aware. If the selector is using
the hybrid objective, the analyzer compares the inner score against the OOS
hybrid objective. If the selector is using `alpha_vs_qqq`, the analyzer checks
against OOS QQQ-alpha quality. This prevents a noisy scoring model from looking
healthy just because a different metric happened to improve.

## How to run it

Default results file:

```bash
python walkforward_analyzer.py
```

Custom results file:

```bash
python walkforward_analyzer.py --csv signals/core_satellite_nested_walkforward_alphaqqq.csv
```

Save a JSON report beside the CSV:

```bash
python walkforward_analyzer.py --json
```

Choose which objective the predictiveness check should use:

```bash
python walkforward_analyzer.py --csv signals/core_satellite_nested_walkforward_alphaqqq.csv --objective alpha_vs_qqq --json
python walkforward_analyzer.py --csv signals/core_satellite_nested_walkforward_hybrid.csv --objective hybrid --json
python walkforward_analyzer.py --csv signals/core_satellite_nested_walkforward.csv --objective sharpe --json
```

## Main checks

1. Score predictiveness
   - Compares the inner score against the matching OOS objective score.
   - Also prints extra correlations versus OOS Sharpe and OOS QQQ alpha.
   - PASS means high inner scores tend to become strong OOS results.
   - FAIL means the selector is backwards: it likes configs that do worse OOS.

2. Model calibration
   - Compares how often inner folds expected positive QQQ alpha against how
     often OOS folds actually delivered positive QQQ alpha.
   - A large optimism gap means the model is overconfident.

3. Concentration vulnerability
   - Checks whether the strategy loses alpha when QQQ strongly beats SPY.
   - This catches strategies that look fine in broad markets but struggle when
     mega-cap tech dominates.

4. Config stability
   - Checks whether the walkforward keeps selecting similar configs.
   - Too much config hopping suggests the selector is chasing noise.

## Pass / Warn / Fail thresholds

| Check | PASS | WARN | FAIL |
|-------|------|------|------|
| Score predictiveness | corr > 0.3 | 0 to 0.3 | corr < 0 |
| Calibration | direction >= 60% and gap < 30pp | direction >= 40% and gap < 50pp | otherwise |
| Concentration | corr > -0.3 and delta > -5% | corr > -0.5 and delta > -10% | otherwise |
| Config stability | top config freq >= 30% and uniqueness < 70% | top config freq >= 20% | otherwise |

## Key terms

- OOS: Out-of-sample. The test period the model did not use for selection.
- Inner score: The validation score used to choose a config before OOS testing.
- Objective: The formula used for selection, such as `sharpe`,
  `alpha_vs_qqq`, or `hybrid`.
- Correlation: A number from -1 to +1 that shows whether two values move
  together. Positive is good here; negative means the selector is backwards.
- QQQ alpha: Strategy return minus QQQ return. Positive means the strategy beat
  QQQ.
- Concentration proxy: QQQ return minus SPY return for a year. High values mean
  mega-cap tech led the market.

## When to use it

Run this after every nested walkforward and before publishing a live config.
Also run it after changing `robustness_scoring.py`, because that file controls
the objective used by config selection.
