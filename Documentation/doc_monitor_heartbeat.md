# monitor_heartbeat.py — Monitor Watchdog

The monitor maintains `signals/operational_incident_ledger.csv` with severity,
incident ID, first/latest observation, occurrence count, and resolution time.
It sends only new-failure and recovery alerts. The independent GitHub watchdog
covers the case where the whole daily workflow never starts.

## What It Does

`monitor_heartbeat.py` checks whether the bot's safety monitors are still
producing fresh output files. If a monitor file is missing or stale, it sends a
warning so a silent failure does not go unnoticed.

It watches files such as `signals/fill_monitor.json`,
`signals/broker_health.json`, `signals/alpaca_execution_scorecard.json`,
`signals/broker_truth.json`, `signals/regime_history.json`, and the latest
daily run log.

## How To Run It

```bash
python3 monitor_heartbeat.py
python3 monitor_heartbeat.py --json
python3 monitor_heartbeat.py --max-age 24
```

Inputs:

- Monitor output files under `signals/`.
- Daily run logs under `logs/`.
- `--max-age` controls how old a file can be before it is stale.

Outputs:

- `signals/monitor_heartbeat.json` — latest watchdog report.
- Warning notification when any watched file is missing or stale.

## Key Terms

- **Monitor** — a script that checks one safety area, such as fills or broker
  connectivity.
- **Heartbeat** — proof that a monitor recently ran.
- **Stale file** — a file that exists but is too old to trust.
- **Watchdog** — a script that checks other scripts are alive.
