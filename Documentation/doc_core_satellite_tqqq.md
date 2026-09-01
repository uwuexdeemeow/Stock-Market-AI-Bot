# Core-Satellite TQQQ Research

## What it does

`core_satellite_tqqq.py` researches whether a controlled TQQQ allocation adds
value to the core-satellite strategy. It supports one configuration, a weight
grid, a historical backtest, and cost stress. It does not submit broker orders.

Research trade rows include the exact ETF and stock target-weight maps. That
evidence lets `quant_performance_audit.py` rebuild daily profit/loss from raw
prices. Standalone TQQQ live signal generation remains disabled.

## How to run it

Use `python3 core_satellite_tqqq.py --backtest`, or add `--grid` to compare
weights. Optional inputs include `--tqqq-weight`, `--holding-days`, and
`--cost-stress`. It reads local ETF data and writes/prints research metrics and
signals; stale data is rejected unless `--ignore-stale` is intentionally used.

## Key terms

- **Leveraged ETF:** a fund targeting a multiple of daily index moves.
- **Grid:** a list of candidate settings tested consistently.
- **Cost stress:** rerun with worse trading-cost assumptions.

## Boundary-Safe Fold Mode

The backtest accepts optional `evaluation_start` and `evaluation_end` inputs
for nested research. It uses older data only as indicator history, starts the
fold with cash, creates a fresh rebalance schedule, and drops any position that
would exit after the fold. Boundary evidence is returned with the metrics.
