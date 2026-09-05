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

## September 2026 submission and historical-data repair

This evaluator now loads the complete core/satellite candidate panel with `require_forward_returns=False`. Stocks with missing future returns stay eligible for ranking; the shared core/satellite engine validates the selected holdings after selection and stops with a ticker/date error if a required outcome cannot be measured. It excludes incomplete evaluation periods as whole periods. This prevents future data availability from choosing today's holdings.

Use the run command and inputs described above as before. Expected output is the usual evaluation report, or a clear missing-price error to resolve before reporting performance. A candidate is a stock considered for selection; a forward return is its later gain or loss. Historical reports made before this repair must be regenerated before comparison with corrected results. Run `python -m pytest tests/test_submission_history_guards.py -q` for offline regressions.
