# validate_fixed_live_config.py - Fixed Live Candidate Validation

## What it does

This script validates one exact fixed core-alpha config across all outer
walkforward years.  It exists for cases where yearly nested config selection is
too noisy and a simple fixed config must prove itself before paper trading.

It:

1. Runs the fixed config on each held-out outer year.
2. Rechecks each outer year at the required cost-stress levels.
3. Applies the existing live approval thresholds for Sharpe, alpha hit rate,
   drawdown, turnover, and medium-risk review reports.
4. Writes a research JSON/CSV pair.

By default it does not publish a live config.  Publishing requires the explicit
`--publish-live-config` flag and still blocks if approval fails.

## How to run it

```bash
python3 validate_fixed_live_config.py \
  --config "h=20,ov=0.5,ma=100,vol=percentile:0.3,score=regime_adaptive,shape=top3,weighting=sticky_score,tqqq=0.0,risk=off" \
  --output-prefix wf_fixed_old_live_validation

# Only after reviewing the research result:
python3 validate_fixed_live_config.py \
  --config "h=20,ov=0.5,ma=100,vol=percentile:0.3,score=regime_adaptive,shape=top3,weighting=sticky_score,tqqq=0.0,risk=off" \
  --output-prefix wf_fixed_old_live_validation \
  --publish-live-config
```

## Inputs

- A full config signature copied from a walkforward result
- Factor data from `data/*.parquet`
- Existing medium-risk review reports in `logs/`

## Outputs

- A unique research JSON/CSV pair under `signals/`
- Optional update of `signals/core_satellite_live_configs.json` only when
  `--publish-live-config` is used and the candidate passes

## Key concepts

- **Fixed config**: one exact strategy setup used on every test year.
- **Outer fold**: one held-out yearly test that simulates unseen history.
- **Cost stress**: rerun with larger trading costs to check if edge survives
  slippage and fee assumptions.
- **Approval gate**: a threshold check that blocks unsafe live promotion.

## Safety

Research is the default.  A fixed candidate that looks good in one output file
does not become paper-trading state until an explicit publish run passes.
