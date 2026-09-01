# config_health.py - Configuration Health Check

## What It Does

`config_health.py` checks whether the local paper-trading environment has the
Python packages and key environment settings the bot expects.

Plain language: it is a quick "is this machine configured correctly?" check
before relying on the daily trading pipeline.

## How To Run It

```bash
python config_health.py
python config_health.py --json
python config_health.py --skip-pip-check
```

Expected output:

- Terminal summary
- `logs/config_health.json`

The JSON report is written atomically, so dashboard or runbook checks never
read a half-written config-health file.

## Key Concepts

- Package: installed Python library, such as `yfinance`.
- Requirement: version rule a package must satisfy.
- Environment variable: setting loaded from `.env` or the shell.
- Pip check: Python's dependency-conflict checker.

The local interpreter must include `alpaca-py==0.43.4`, matching CI. Install
the project requirements with `python3 -m pip install -r requirements.txt` if
the Alpaca check says `missing`.
