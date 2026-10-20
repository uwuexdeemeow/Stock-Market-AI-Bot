# v_rule_signal.py — the daily paper signal for rule V

## What it does, in plain language

From the 2026-10-30 signal, the Alpaca paper account follows **rule V**, the
volatility-target rule that passed the pre-registered H-index-risk test. The
rules and the pass/fail test for the paper run are in
`V_RULE_FORWARD_TEST.md`.

The rule:

1. On the **last trading day of each month** (the "signal day"), look at
   QQQ's last 63 daily returns and turn their jumpiness into a yearly
   volatility.
2. QQQ share = 20% ÷ that volatility, never more than 100%. The rest goes in
   **BIL**, a T-bill fund that is basically cash.
3. Trade at the next session. Then hold until the next month end.

This script runs every trading day inside `daily_run.py` (only when
`PAPER_MODE_STRATEGY=v_rule`). It does the arithmetic and writes one row to
`signals/core_satellite_alpha_signal.csv`, the same file the broker script
already reads. The broker's existing calendar does the rest:

- `last_scheduled_rebalance_date` is the latest signal day;
- `scheduled_rebalance_today` is true when the newest completed session
  **is** that signal day;
- on every other day the broker holds. If a trading run fails, the next run
  catches up, still using the signal day's weights.

It never contacts the broker and never sends an order.

**Safety details:**
- It uses **completed sessions only**. A price source can already show
  today's unfinished bar while the market is open, and the rule must not
  decide on a half-day close.
- It uses the same weight function as the tested rule (`v_rule_helper.py`,
  checked against the research script by `tests/test_v_rule_helper.py`).
- If QQQ or BIL prices are not current, or there are fewer than 64 closes,
  the row says `paper_ready = False`. The script exits with code 1 and the
  broker refuses to trade.

## How to run

From the project root, after `refresh_etf_data.py --refresh` has updated
`data/QQQ.parquet` and `data/BIL.parquet`:

```bash
python v_rule_signal.py            # write the signal file and print a summary
python v_rule_signal.py --print    # also print every field
```

**Example output (2026-10-20):**

```
Rule V signal -> signals/core_satellite_alpha_signal.csv
  data up to 2026-10-19, signal day 2026-09-30 (trade today: False)
  QQQ 100.0%, BIL 0.0%, next signal day 2026-10-30
```

The paper run is controlled by `PAPER_MODE` in
`.github/workflows/daily_paper_trading.yml`. In `v_rule` mode the workflow
skips the core-alpha factor snapshot and robustness steps, because V reads
only QQQ/BIL prices. It also moves the two account-wide sell-everything
halts outside the tested range: drawdown halt 40% and one-day loss halt
−12%.

Tests: `python -m pytest tests/test_v_rule_signal.py -q`.

## Key terms

- **Signal day:** the day whose closing prices decide the trade. Orders go
  in on the next trading session.
- **Volatility:** how much the price jumps around, as a yearly percentage.
- **Volatility target:** holding less of something when it gets jumpier, so
  the swings of the whole account stay near the target (20%).
- **BIL:** an ETF of 1–3 month US Treasury bills.
- **Hold day:** a day with no strategy orders. The weights drift with
  prices, on purpose.
- **Catch-up:** if the trading run on the session after a signal day fails,
  the next run trades the same signal.
