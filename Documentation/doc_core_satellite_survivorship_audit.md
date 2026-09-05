# core_satellite_survivorship_audit.py - Survivorship Stress Check

The report identity includes the deployment gross-exposure ceiling so the
validation bundle can prove the audit tested the exact paper configuration.

## What It Does

`core_satellite_survivorship_audit.py` tests whether the selected core-satellite
strategy still behaves well when locally available failed or delisted tickers are
added back into the research universe.

Plain language: it checks whether the strategy only looked good because it
mostly tested companies that survived.

## How To Run It

```bash
python core_satellite_survivorship_audit.py
```

Inputs:

- Current approved core-satellite config
- Factor data in `data/`
- Failed-name audit profiles from `survivorship_audit.py`

Expected outputs:

- `signals/core_satellite_survivorship_audit.csv`
- `logs/core_satellite_survivorship_audit.json`

Both outputs are written atomically, so approval gates never read a half-written
stress report.

## Key Concepts

- Survivorship bias: backtests can look better if failed companies are missing.
- Audit ticker: a failed or delisted ticker available for stress testing.
- Failed-name coverage: available failed-name histories divided by all named
  audit failures. Capital evidence requires 100% coverage.
- Point-in-time universe: a dated constituent list showing which companies
  were actually eligible on each historical date. Capital evidence cannot pass
  while this table is incomplete.

The JSON report records both completeness checks explicitly. A partial stress
test remains useful provisional paper evidence, but it cannot clear the
capital-approval gate regardless of its backtest return.
- Stressed universe: normal watchlist plus available audit tickers.
- Alpha: return above a benchmark such as QQQ or a blended benchmark.

The report carries strategy and dataset fingerprints. Testing remains partial
until date-effective membership and delisted-name data are complete, so it
cannot authorize real capital.
The fingerprint includes volatility and risk-control modes, preventing results
for one strategy variant from being attached to another.

## September 2026 submission and historical-data repair

This evaluator now loads the complete core/satellite candidate panel with `require_forward_returns=False`. Stocks with missing future returns stay eligible for ranking; the shared core/satellite engine validates the selected holdings after selection and stops with a ticker/date error if a required outcome cannot be measured. It excludes incomplete evaluation periods as whole periods. This prevents future data availability from choosing today's holdings.

Use the run command and inputs described above as before. Expected output is the usual evaluation report, or a clear missing-price error to resolve before reporting performance. A candidate is a stock considered for selection; a forward return is its later gain or loss. Historical reports made before this repair must be regenerated before comparison with corrected results. Run `python -m pytest tests/test_submission_history_guards.py -q` for offline regressions.
