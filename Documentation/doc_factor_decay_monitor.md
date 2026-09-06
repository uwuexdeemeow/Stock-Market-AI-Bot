# factor_decay_monitor.py - Factor Decay Monitor

The report identity includes the deployment gross-exposure ceiling so the
validation bundle cannot combine decay evidence with a different portfolio
configuration.

A non-positive top bucket based on fewer than three independent holding-period
cohorts is classified as advisory rather than warning. Once three or more
cohorts exist, the same deterioration remains a paper-trading warning.

## What It Does

`factor_decay_monitor.py` checks whether the overlay score still has recent
predictive power and whether recent overlay trades are still adding alpha.

The report names top-decile minus rest return with both the legacy
`top_bucket_excess_return_pct` field and the clearer
`top_vs_rest_return_pct` field. They are the same non-overlapping-cohort
statistic.

Plain language: it watches for the strategy edge getting weaker before the
paper/live system relies on it too much.

## How To Run It

```bash
python factor_decay_monitor.py
```

Inputs:

- Current approved core-satellite config
- Factor data in `data/`
- `signals/core_satellite_alpha_trades.csv`

Expected outputs:

- `signals/factor_decay_monitor.csv`
- `logs/factor_decay_monitor.json`

Both outputs are written atomically, so daily gates never read a half-written
decay report.

## Key Concepts

- IC: information coefficient; rank correlation between score and future return.
- Non-overlapping cohort: the actual rebalance dates used by the strategy. A
  20-day future return is sampled every 20 trading dates, not every day, so 19
  overlapping observations cannot pretend to be independent evidence.
- Decile monotonicity: whether average returns generally rise from the lowest
  score group to the highest score group.
- Newey-West t-stat: a conservative significance estimate that adjusts for
  remaining serial dependence.
- Overlay alpha: return from the stock overlay above the benchmark.
- Lookback window: recent period, such as 60 or 120 trading days.
- Real-capital block: a severe status that should stop real-money promotion.

Saved JSON records the strategy and dataset fingerprints measured. Old decay
evidence cannot be reused to approve a newer configuration.
The fingerprint also records selection shape, weighting, overlay size,
volatility mode, and risk mode even when a field does not alter the IC formula.
Schema version 2 records `higher_is_better` score direction, exact cohort count,
decile returns, and the non-overlapping sampling method.


## Remaining audit repair, September 2026

The old overlay_alpha_sum_pct field is retired. Historical overlay totals are explicitly legacy_overlay_raw_return_sum_pct and are not benchmark-relative alpha. Only current versioned paired-ledger evidence with at least 20 mature non-overlapping cohorts and positive lower 95% confidence bounds for excess return and rank IC can support pass. Otherwise edge health remains advisory and real-capital evidence remains blocked; advisory is not an automatic paper-strategy cutover or halt. JSON, CSV, notification text and status consumers share the migrated fields. Legacy return sums remain diagnostic only. The corrected source is signals/corrected_audit/edge_monitor.json. Do not run this monitor merely to test it: its normal CLI can send configured status notifications. Test offline with python -m pytest tests/test_medium_gap_fixes.py tests/test_corrected_audit.py -q.

Historical results affected by these changes must be regenerated. Original audit evidence is preserved; no corrected historical claim is made when source checks are blocked.
