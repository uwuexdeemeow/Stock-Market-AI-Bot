# quant_common.py

## What it does

`quant_common.py` holds three small helpers that several quant scripts share:

| Name | What it is | Used by |
|---|---|---|
| `INITIAL_CAPITAL` | Starting cash for every simulated portfolio ($10,000) | `core_satellite_alpha.py`, `alpha_factor_backtest.py`, `core_satellite_drawdown_throttle.py`, `core_satellite_tqqq.py` |
| `_load_etf_price_frame` | Loads ETF closing prices, lines them up on the dates you ask for, and scales each one to start at 1.0 | `core_satellite_alpha.py`, `alpha_factor_backtest.py` |
| `_newey_west_tstat` | A t-statistic for "is the average return really above zero?" that allows for returns being linked from day to day | `alpha_factor_backtest.py`, `factor_decay_monitor.py`, `quant_performance_audit.py` |

These used to live inside `backtest.py`, the old machine-learning
backtester. Importing them from there also loaded the whole legacy ML layer
(`train.py`, xgboost, scikit-learn, matplotlib). On 2026-10-25 they were moved
here unchanged, and `backtest.py` moved to `legacy/`. See
`LEGACY_ML_CLEANUP_PLAN.md`.

## How to run it

You don't run it. Other scripts import it:

```python
from quant_common import INITIAL_CAPITAL, _load_etf_price_frame, _newey_west_tstat
```

- **Input:** `_load_etf_price_frame` reads `data/<SYMBOL>.parquet`. If a
  file is missing, it downloads the prices from Yahoo Finance.
- **Output:** a table with one column per symbol, each scaled to 1.0 on the
  first date.
- **When prices are missing:** if a requested date has no valid price
  before it, the helper raises an error. It never fills a gap with a later
  price, because that would leak the future into a backtest.

## Proof that the move changed nothing

Before and after the move, the helpers were run on the real price data:
- 3 symbol sets × 4 date grids (full, last 600 days, every 20th day,
  reversed) × 2 warm-up settings;
- t-statistics at lags 1, 5 and 10.

All 122 results matched exactly. `factor_decay_monitor.py` also wrote
identical output before and after. `tests/test_submission_history_guards.py`
tests the price loader directly.

## Key terms

- **Cache:** a memory of prices already loaded, so repeat calls in one run
  are fast. The cache key includes each data file's size and change time,
  so a data refresh never serves old prices.
- **Warm-up (`warmup_calendar_days`):** extra earlier history loaded so
  long moving averages have enough past data on the first requested date.
- **Newey-West t-statistic:** a plain t-statistic assumes every day is
  independent. Returns are often a little linked from one day to the next,
  and Newey-West widens the error bar to account for that. Below 30 data
  points it returns 0.
- **Legacy layer:** the old XGBoost prediction scripts, now in `legacy/`.
  `tests/test_no_legacy_imports.py` makes sure live code (including this
  file) never imports them.
