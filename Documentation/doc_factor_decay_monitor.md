# factor_decay_monitor.py - Factor Decay Monitor

## What It Does

`factor_decay_monitor.py` checks whether the overlay score still has recent
predictive power and whether recent overlay trades are still adding alpha.

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
