# monitor.py - Central Health Monitor

## What it does

This script checks model drift, pipeline failures, portfolio drawdown, and
signal anomalies. Every alert is written to local log files. It can also send
the alert to Slack when `SLACK_WEBHOOK_URL` is configured.

Email delivery has been removed. Daily trading alerts use Telegram through
`notifications.py`.

## How to run it

```bash
python3 monitor.py
python3 monitor.py --check drift
python3 monitor.py --check pipeline
python3 monitor.py --check drawdown
python3 monitor.py --check signals
```

Inputs include the project signal files, pipeline logs, and optional Slack
webhook setting. Outputs are written to `logs/monitor.log` and
`logs/alerts.jsonl`.

## Key terms

- **Monitor**: a check that looks for a problem automatically.
- **Drift**: live data behaving differently from the data used in research.
- **Drawdown**: the decline from the portfolio's previous highest value.
- **Deduplication**: suppressing repeated copies of the same alert for a set
  period.
- **Webhook**: a private web address that accepts automated Slack messages.
