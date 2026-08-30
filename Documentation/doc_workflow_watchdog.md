# `workflow_watchdog.py`

## What it does

This independent watchdog asks GitHub whether the factor-data, daily paper,
shadow-paper, and post-market workflows completed after their New York
deadlines. It can report that the main job never started. It understands NYSE
holidays and daylight saving time, deduplicates repeated failures, and sends a
recovery message. It has no Alpaca credentials.

## How to run it

Set `GITHUB_REPOSITORY`, `GITHUB_TOKEN`, and optional Telegram variables:

```bash
python workflow_watchdog.py
python workflow_watchdog.py --json
```

GitHub runs `.github/workflows/independent_workflow_watchdog.yml` hourly on
weekdays. Reports and alert state are stored under `signals/`.

## Key terms

- **Independent watchdog:** a separate job that notices another job never ran.
- **Deadline:** the latest expected completion time in New York.
- **Deduplication:** one alert for a continuing problem.
- **Recovery alert:** confirmation that the problem cleared.

