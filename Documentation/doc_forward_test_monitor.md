# forward_test_monitor.py — monthly forward-test numbers

## What it does, in plain language

The paper strategy is judged only on how it does **from now on**, by
Forward Test Rule F1 (`Documentation/FORWARD_TEST_RULE.md`). The rule fixes
three measures in advance. This script computes them from the files the
daily paper run publishes (the `signals/latest` branch):

- **M1, return vs a fair yardstick:** the paper account's return minus
  what holding QQQ would have made, at the same invested share each day
  (the rest in BIL, a T-bill fund). Chained day by day and annualised. With
  fewer than 20 days it shows "n/a" plus the raw returns, because
  annualising a few days is meaningless.
- **M2, real trading cost:** for each fill since the epoch start, how far
  the fill price was from that day's Open, in basis points (1 bp = 0.01%).
  Sells are sign-flipped so positive always means "worse than the Open".
  Shows the mean and a 95% range.
- **M3, can it trade?:** how many scheduled rebalance dates were blocked
  by the safety gate. The first daily run on the trade day (the session
  after the signal day) decides. It also flags the **T-block** trigger (4 or
  more of any 6 scheduled dates in a row blocked).

It only reads data. It never trades and never changes a gate. It doesn't
judge anything either: the rule's checkpoints at 12 and 24 periods do.

## How to run

From the project root (needs git access to the repo and internet for
prices):

```bash
python forward_test_monitor.py                 # print the numbers and the log line
python forward_test_monitor.py --append-log    # also add the line to FORWARD_TEST_RULE.md
```

Options: `--as-of YYYY-MM-DD` (measure up to a date), `--signals-dir
<folder>` (read a local copy instead of the branch), `--note "..."` (text
for the log line's Note column).

- **Output:** prints a JSON summary and a ready-made log line, and saves
  `signals/forward_test_check.json` (not in Git).
- **When:** once a month, around the 3rd. Commit the updated
  `FORWARD_TEST_RULE.md` afterwards.

Tests: `python -m pytest tests/test_forward_test_monitor.py -q`.

## Data it reads

From `signals/latest`: `paper_validation_epoch.json` (epoch start),
`paper_rebalance_period.json` (the rebalance calendar),
`alpaca_paper_equity.csv` (daily equity and invested amount),
`alpaca_paper_log.csv` (fills) and `alpaca_submit_outcomes.csv` (each run's
status). Prices for QQQ, BIL and the traded stocks come from the project's
`data_provider.py`.

**Known limit:** the equity snapshots are taken during the 10:00 ET run,
while the yardstick uses daily closes. Over a year or two this timing noise
averages out, but single months can look better or worse than they were.

## Key terms

- **Forward test:** judging a strategy only on data that didn't exist when
  it was designed.
- **Epoch:** the stretch of paper trading with one frozen version of the
  strategy. Changing the trading logic starts a new epoch and restarts the
  clock.
- **Period:** one 20-trading-day stretch between scheduled rebalances.
- **Yardstick (benchmark):** what a simple alternative would have earned
  with the same amount of money invested.
- **Basis point (bp):** 0.01%. 20 bps = 0.20%.
- **Annualised:** turned into a "per year" rate so periods of different
  length can be compared.
