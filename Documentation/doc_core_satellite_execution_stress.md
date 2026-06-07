# core_satellite_execution_stress.py - Execution Stress Check

## What It Does

`core_satellite_execution_stress.py` reruns the selected core-satellite config
under harder execution assumptions, such as delayed entries and extra turnover
costs.

Plain language: it checks whether the strategy still works if real fills are
worse than the ideal backtest.

## How To Run It

```bash
python core_satellite_execution_stress.py
```

Inputs:

- Current approved core-satellite config
- Factor data in `data/`
- ETF benchmark data

Expected outputs:

- `signals/core_satellite_execution_stress.csv`
- `logs/core_satellite_execution_stress.json`

Both outputs are written atomically, so medium-risk review gates never read a
half-written stress report.

## Key Concepts

- Entry delay: buying one trading day later than the signal.
- Turnover cost: extra simulated trading cost from rebalancing.
- Stress scenario: a harder version of the same backtest.
- Gate: a pass/fail safety check before promoting a config.
