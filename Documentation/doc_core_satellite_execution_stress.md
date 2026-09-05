# core_satellite_execution_stress.py - Execution Stress Check

The report identity includes the deployment gross-exposure ceiling. This
prevents evidence produced for an unscaled research portfolio from being
mistaken for evidence produced with the paper broker's exposure ceiling.

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

The JSON report includes the exact strategy and dataset fingerprints. The
validation bundle rejects a report from an older config or data snapshot.
Volatility mode, leverage choice, and risk-control mode are included in that
identity so a report cannot be mislabeled as a different strategy.

## September 2026 submission and historical-data repair

This evaluator now loads the complete core/satellite candidate panel with `require_forward_returns=False`. Stocks with missing future returns stay eligible for ranking; the shared core/satellite engine validates the selected holdings after selection and stops with a ticker/date error if a required outcome cannot be measured. It excludes incomplete evaluation periods as whole periods. This prevents future data availability from choosing today's holdings.

Use the run command and inputs described above as before. Expected output is the usual evaluation report, or a clear missing-price error to resolve before reporting performance. A candidate is a stock considered for selection; a forward return is its later gain or loss. Historical reports made before this repair must be regenerated before comparison with corrected results. Run `python -m pytest tests/test_submission_history_guards.py -q` for offline regressions.
