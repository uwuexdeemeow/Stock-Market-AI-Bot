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

**Failed companies are left out after their failure date.** Some failed
companies (First Republic, Bed Bath & Beyond) kept trading for years at
fractions of a cent. Rows after the date in `settings.SURVIVORSHIP_FAILURE_DATES`
are not used in any ranking, so they can't push healthy stocks' ranks up or
down. The failed company's own later rows get the neutral rank 0.5. Pass
`failure_dates={}` to turn this off (tests only).

## Key terms

- **Cross-sectional:** comparing many stocks at the same moment.
- **Percentile:** position from 0 (lowest) to 1 (highest).
- **Sector:** a peer group such as technology or healthcare.
