# Cross-Sectional Features

## What it does

`cross_sectional_features.py` ranks each stock against its sector and the broad
universe on the same date. It writes percentile features back to local ticker
parquets so training can learn relative leadership instead of isolated moves.

## How to use it

Import `apply_cross_sectional_rank_features(tickers)`. Inputs are ticker names
with existing parquet files. Expected outputs are `xs_rank_sector_*` and
`xs_rank_market_*` columns plus a summary. Small sector groups fall back to a
market rank rather than producing misleading ranks.

## Key terms

- **Cross-sectional:** comparing many stocks at the same moment.
- **Percentile:** position from 0 (lowest) to 1 (highest).
- **Sector:** a peer group such as technology or healthcare.
