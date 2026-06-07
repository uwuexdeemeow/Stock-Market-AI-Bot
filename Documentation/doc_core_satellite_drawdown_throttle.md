# core_satellite_drawdown_throttle.py - Drawdown Throttle Research

## What It Does

`core_satellite_drawdown_throttle.py` tests simple rules that reduce overlay
exposure after losses. It does not change the live config by itself.

Plain language: it asks whether cutting risk during drawdowns would improve
the selected strategy's drawdown or Sharpe without losing too much alpha.

## How To Run It

```bash
python core_satellite_drawdown_throttle.py
```

Inputs:

- Current approved core-satellite config
- Factor data in `data/`
- ETF benchmark data

Expected outputs:

- `signals/core_satellite_drawdown_throttle.csv`
- `logs/core_satellite_drawdown_throttle.json`

Both outputs are written atomically, so dashboards and research gates never read
a half-written throttle report.

## Key Concepts

- Drawdown: decline from a previous equity high.
- Throttle: rule that temporarily reduces exposure.
- Overlay: individual stock sleeve around the ETF core.
- Promotion candidate: a throttle rule that improves risk without breaking
  benchmark alpha.
