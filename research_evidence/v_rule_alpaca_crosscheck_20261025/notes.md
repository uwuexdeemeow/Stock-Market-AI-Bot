# Notes: rule V Alpaca cross-check

**Request (owner, 2026-10-25):** use the installed Alpaca backtest skill
(pinned commit `89d0943` of `alpacahq/alpaca-skills`) to rebuild rule V
independently, and check the project's V code and data before the first V
trade on 2026-11-02.

**Purpose:** verification only. No parameters were tuned. 20% and 63 days
are fixed by the pre-registration (`Documentation/V_RULE_FORWARD_TEST.md`).

## Strategy as implemented (`run.py`)

- **Symbols:** QQQ and BIL (US ETFs).
- **Data:** Alpaca CLI 0.0.14, `data multi-bars`, 1Day bars, feed `sip`,
  adjustment `all` (splits and dividends folded into prices, so dividends
  count as reinvested). Data runs 2016-01-04 to 2026-10-23.
- **Calendar:** `alpaca calendar`. The signal day is the last NYSE session of each month.
- **Signal:** take the last 63 simple daily returns of the adjusted close,
  up to and including the signal day. vol = sample std (N−1) × √252.
  QQQ weight = min(1, 0.20 ÷ vol). BIL weight = 1 − QQQ weight.
- **Warm-up:** no signal until 63 returns exist. The first signal is 2016-04-29.
- **Fill model:** `next_open`. The signal is computed at the month-end close
  and filled at the next session's open. Sells go first, then buys.
  Fractional shares are allowed. Between signal days the bot holds and the
  weights drift with prices.
- **Sizing:** target value = equity at the fill-day open × weight. A small
  cash buffer keeps cash from going below zero. This is a documented change
  from the skill's default of sizing at the signal-day close: it matches the
  live bot, which sizes from live equity on the trade day.
- **Friction:** 5 bps slippage per side on every fill (bar-based proxy; no quotes).
- **Fees** (Alpaca fee schedule PDF, revised 2026-10-12, extracted 2026-10-25):
  - Modelled:
    - SEC fee: $0.0000206 × sell value.
    - FINRA CAT: $0.000003 a share, on buys and sells.
    - FINRA TAF: read as $0 a share for equities.
  - Not modelled: commissions (none on this account type), borrow and margin, account fees.
- **Benchmarks:**
  - QQQ buy-and-hold.
  - BIL buy-and-hold.
  - 50/50 QQQ+BIL with no rebalancing.
  - All three buy at the first V fill day's open with the same friction.
- **Metrics:** the daily equity curve. Sharpe uses sample std ×√252, with rf
  = 0 (skill default) and also with BIL's daily return (the project's
  convention).

## Cross-check (`compare.py`)

The comparison is a 2 × 2:

| | Project data | Alpaca data |
|---|---|---|
| **Project code** | ✓ | ✓ |
| **`run.py`** | — | ✓ |

The project code is `research_evidence/index_risk_20261016/index_risk_check.py`.
`compare.py` also checks:
- the live helper `v_rule_helper.qqq_weight`, weight by weight, against `run.py`;
- the live calendar `v_rule_signal.month_end_schedule` against Alpaca's calendar.

## How to reproduce

```
ALP=<alpaca binary> bash fetch.sh     # needs paper keys in ALPACA_API_KEY / ALPACA_SECRET_KEY
python run.py
cd ../.. && python research_evidence/v_rule_alpaca_crosscheck_20261025/compare.py
```

The CLI was installed from the GitHub release `v0.0.14` (`cli_0.0.14_windows_amd64.zip`),
SHA-256 checked against `checksums.txt`. `alpaca doctor` showed the paper endpoint.
No orders were placed.

## Caveats

See `report.md`. Backtests are hypothetical and are not investment advice.
More detail: https://alpaca.markets/disclosures
