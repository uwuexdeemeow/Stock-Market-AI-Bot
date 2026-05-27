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

The check also treats cross-year score correlation as a smoke test, not a
complete verdict by itself. Inner scores from different market years are not
always directly comparable. If the raw score correlation is negative but the
inner folds still correctly predict the sign of QQQ alpha at least 60% of the
time, the analyzer reports WARN instead of FAIL and points you to selector
replay diagnostics for the deeper ranking test.

For config stability, newer walkforward CSVs include a
`stable_family_signature` column. The analyzer uses that family column when it
exists, because small knob changes inside the same behavior family should not
count as totally unrelated strategy hopping.

## How to run it

Default results file:

```bash
python walkforward_analyzer.py
```

Custom results file:

```bash
python walkforward_analyzer.py --csv signals/core_satellite_nested_walkforward_alphaqqq.csv
```

Research baseline files are supported too.  If a CSV was produced by
`walkforward_selector_diagnostics.py fixed`, the analyzer maps
`oos_total_return_pct` to `oos_return_pct` and prints the checks that can be
computed.  Score predictiveness and calibration are skipped when the file has
no inner-score columns, because a fixed baseline has no selector.

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

1. Fold completeness
   - Fails the report when any requested walkforward fold has no OOS result.
   - Prints missing fold years and their saved failure reasons.
   - This prevents a partial run from looking deployable just because hard
     years dropped out.

2. Score predictiveness
   - Compares the inner score against the matching OOS objective score.
   - Also prints extra correlations versus OOS Sharpe and OOS QQQ alpha.
   - PASS means high inner scores tend to become strong OOS results.
   - WARN can mean the raw cross-year correlation is weak or negative but
     calibration is still healthy.
   - FAIL means the selector is backwards: it likes configs that do worse OOS.

3. Model calibration
   - Compares how often inner folds expected positive QQQ alpha against how
     often OOS folds actually delivered positive QQQ alpha.
   - A large optimism gap means the model is overconfident.

4. Concentration vulnerability
   - Checks whether the strategy loses alpha when QQQ strongly beats SPY.
   - This catches strategies that look fine in broad markets but struggle when
     mega-cap tech dominates.

5. Config stability
   - Checks whether the walkforward keeps selecting similar configs.
   - Uses `stable_family_signature` when present, otherwise exact
     `selected_config`.
   - Too much config hopping suggests the selector is chasing noise.

## Pass / Warn / Fail thresholds

| Check | PASS | WARN | FAIL |
|-------|------|------|------|
| Fold completeness | no missing OOS folds | - | any missing OOS fold |
| Score predictiveness | corr > 0.3 | 0 to 0.3, or negative corr with healthy sign calibration | negative corr without healthy sign calibration |
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
- Stable family: A grouped config label that keeps major behavior choices
  together while ignoring small tuning differences such as exact volatility
  mode.
- QQQ alpha: Strategy return minus QQQ return. Positive means the strategy beat
  QQQ.
- Concentration proxy: QQQ return minus SPY return for a year. High values mean
  mega-cap tech led the market.

## When to use it

Run this after every nested walkforward and before publishing a live config.
Also run it after changing `robustness_scoring.py`, because that file controls
the objective used by config selection.
