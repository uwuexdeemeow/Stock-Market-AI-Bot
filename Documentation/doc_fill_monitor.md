# fill_monitor.py — Order Fill Watcher

## What It Does

`fill_monitor.py` checks the recent paper-trading order log and looks for orders
that did not cleanly fill. It catches cancelled, rejected, partial, open, or
missing broker orders before the next daily trade cycle.

It always writes `signals/fill_monitor.json`, even when there are no recent
orders. That file is the heartbeat proof that the monitor ran.

## How To Run It

```bash
python3 fill_monitor.py
python3 fill_monitor.py --days 2
python3 fill_monitor.py --quiet
```

Inputs:

- `signals/paper_trades.csv` — local paper order log.
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
