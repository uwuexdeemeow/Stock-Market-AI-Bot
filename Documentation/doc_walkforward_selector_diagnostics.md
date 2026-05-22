# walkforward_selector_diagnostics.py - Selector Diagnostics

## What it does

This is a research-only script for checking whether the nested walkforward
selector is helping.  It does not publish a paper-trading config.

It has two modes:

1. `fixed` evaluates one or more fixed config signatures on every outer test
   year.  That gives a boring baseline to compare against the yearly selector.
2. `replay` evaluates candidate configs on inner folds and then on the matching
   outer year.  It reports whether configs with better inner scores also rank
   better out of sample.

## How to run it

```bash
# Fixed baseline A/B.  Repeat --config for every config you want to compare.
python3 walkforward_selector_diagnostics.py fixed \
  --label old \
  --config "h=20,ov=0.5,ma=100,vol=percentile:0.3,score=regime_adaptive,shape=top3,weighting=sticky_score,tqqq=0.0,risk=off" \
  --label low_turnover \
  --config "h=20,ov=0.25,ma=100,vol=percentile:0.3,score=regime_adaptive,shape=top3,weighting=risk_parity,tqqq=0.0,risk=off"

# Bounded replay of 16 low-turnover candidates on one outer year.
python3 walkforward_selector_diagnostics.py --start-year 2024 --end-year 2024 replay \
  --grid low-turnover \
  --max-configs 16 \
  --output-prefix wf_selector_replay_2024
```

## Inputs

- The factor panel loaded from `data/*.parquet`
- Config signatures copied from nested walkforward JSON/CSV outputs
- The same candidate grid builders used by
  `core_satellite_nested_walkforward.py`

## Outputs

Each run writes a new JSON/CSV pair under `signals/`.  If an output prefix is
reused, the shared research writer adds a timestamp so older runs stay intact.

- Fixed mode JSON includes summary stats for every fixed config.
- Fixed mode CSV has one row per fixed config and outer year.
- Replay mode JSON includes yearly and pooled rank correlations.
- Replay mode CSV has one row per candidate and outer year.

## Key terms

- **Fixed config**: one exact strategy setup tested year after year.
- **Selector**: the logic that chooses a config from the candidate grid.
- **Inner fold**: historical validation year used to score a candidate before
  the outer test year.
- **Outer fold**: the year that simulates the unseen out-of-sample test.
- **Rank correlation**: whether the order of candidates by inner score matches
  their later order by out-of-sample quality.

## Safety

This script always writes research results only.  It never changes
`signals/core_satellite_live_configs.json`.
