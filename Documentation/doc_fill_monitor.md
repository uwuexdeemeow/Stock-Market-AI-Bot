# fill_monitor.py — Order Fill Watcher

## What It Does

`fill_monitor.py` checks the recent paper-trading order log and looks for orders
that did not cleanly fill. It catches cancelled, rejected, partial, open, or
missing broker orders before the next daily trade cycle.

An order recorded as `skipped` by a pre-submit safety guard was never sent to
the broker. These audit rows are excluded from fill counts and fill-rate
calculations instead of being mislabeled as unknown fills.

A first-stage limit that is deliberately cancelled after its refreshed quote
breaches the spread guard is also excluded when the log explicitly records
`execution_stage=stage2_blocked` and a `spread_guard:` reason. Generic broker
cancellations and rejections remain fill problems and still block trading.

The historical `client_order_id must be unique` submission failure is excluded
after the run-specific recovery-ID repair: Alpaca rejected it before creating a
broker order, so there is no fill to verify. Other submission failures remain
blocking problems.

The monitor accepts the current Alpaca paper log shape (`side`, `quantity`,
`filled_qty`) as well as older logs that used `action`, `broker_qty`, and
`broker_dealt_qty`.

It always writes `signals/fill_monitor.json`, even when there are no recent
orders. That file is the heartbeat proof that the monitor ran.

## How To Run It

```bash
python3 fill_monitor.py
python3 fill_monitor.py --days 2
python3 fill_monitor.py --quiet
```

Inputs:

- `signals/alpaca_paper_log.csv` — Alpaca paper order log.
- `--days` — how far back to check.
- `--quiet` — suppress normal output unless there are problems.

Outputs:

- `signals/fill_monitor.json` — latest fill-check report.
- Warning notification when problematic fills are found.

## Key Terms

- **Fill** — broker completed the order.
- **Partial fill** — broker filled only part of the requested shares.
- **Rejected/cancelled** — broker did not execute the order.
- **Heartbeat file** — a small JSON file proving the monitor ran recently.
