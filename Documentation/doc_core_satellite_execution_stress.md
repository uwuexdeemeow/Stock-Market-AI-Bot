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
The runner now copies every setting from the approved validation bundle and
checks that today's metrics agree. This includes each market regime's QQQ and
stock weights. Previously, omitting that preset made the stress test use a
different built-in allocation.
Each scenario also lists `failed_gates`. For example, a delayed entry may
trail QQQ during the 2023–2026 holdout even when its full-history return is
positive. This is a strategy result, not a missing-price error.

## September 2026 submission and historical-data repair

This evaluator now loads the complete core/satellite candidate panel with `require_forward_returns=False`. Stocks with missing future returns stay eligible for ranking; the shared core/satellite engine validates the selected holdings after selection and stops with a ticker/date error if a required outcome cannot be measured. It excludes incomplete evaluation periods as whole periods. This prevents future data availability from choosing today's holdings.

Use the run command and inputs described above as before. Expected output is the usual evaluation report, or a clear missing-price error to resolve before reporting performance. A candidate is a stock considered for selection; a forward return is its later gain or loss. Historical reports made before this repair must be regenerated before comparison with corrected results. Run `python -m pytest tests/test_submission_history_guards.py -q` for offline regressions.

## Testing a research candidate without publishing it (`--candidate-json`)

Normally this script can only test the **approved live config**. That is a
problem for research: a new walk-forward winner (for example from Colab)
would have to be published before it could be stress-tested, and publishing
is forbidden until it passes. The `--candidate-json` option fixes that.

```bash
# A walk-forward result (settings are read from approved_live_config.config)
python core_satellite_execution_stress.py --candidate-json logs/wf_delay_robust_lowturnover_20260926.json

# A plain JSON file that holds only the settings, with a custom short name
python core_satellite_execution_stress.py --candidate-json my_config.json --candidate-name lowturn_v1
```

Inputs:

- A JSON file in one of two shapes: a walk-forward result with
  `approved_live_config.config`, or a plain settings object. A walk-forward
  result that selected **no** config stops with an error, because there is
  nothing to test.
- The same factor and ETF data as a normal run.

Expected outputs (research only):

- `logs/research_candidate_execution_stress_<name>.csv`
- `logs/research_candidate_execution_stress_<name>.json`

`<name>` is the JSON file name without `.json`, or `--candidate-name`.

Safety rules:

- The official files `signals/core_satellite_execution_stress.csv` and
  `logs/core_satellite_execution_stress.json` are **never** touched in this
  mode. The daily paper-trading gate reads those, so a candidate run cannot
  change what the gate sees.
- The JSON report is stamped `"research_candidate": true` and
  `"approves_trading": false`.
- Without the flag, the script behaves exactly as before.

Key terms:

- **Research candidate**: a strategy setting being studied, not yet approved.
- **Publish**: make a config the one the paper-trading bot actually uses. This
  option never publishes.

Offline tests: `python -m pytest tests/test_research_candidate_stress.py -q`.
