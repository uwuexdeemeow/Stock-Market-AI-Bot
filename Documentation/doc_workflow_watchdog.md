# `workflow_watchdog.py`

## What it does

This independent watchdog asks GitHub whether the factor-data, daily paper,
shadow-paper, and post-market workflows completed after their New York
deadlines. It can report that the main job never started. It understands NYSE
holidays and daylight saving time, deduplicates repeated failures, and sends a
recovery message. When GitHub drops a cron event, it can dispatch that same
workflow once inside a strict safe-time window. It has no Alpaca credentials.

## How to run it

Set `GITHUB_REPOSITORY`, `GITHUB_TOKEN`, and optional Telegram variables:

```bash
python workflow_watchdog.py
python workflow_watchdog.py --json
```

GitHub attempts `.github/workflows/independent_workflow_watchdog.yml` six times
per hour on weekdays. Repeated attempts reduce the chance that GitHub drops the
watchdog itself. Reports and alert state are stored under `signals/`.

The daily-paper fallback is allowed only from 9:45 through 10:25 AM New York.
It never supplies the emergency override. Factor, shadow, and post-market
fallbacks are also time-bounded and keep their original workflow safeguards.
Each workflow gets at most one watchdog dispatch per New York session. The
read-only shadow workflow may use that one dispatch to retry a failed or
cancelled run. A manually launched daily dry run does not count as a successful
real paper session.
The GitHub query reads twenty recent runs so daylight-saving duplicates and
manual diagnostics cannot hide the intended scheduled run.

## Key terms

- **Independent watchdog:** a separate job that notices another job never ran.
- **Deadline:** the latest expected completion time in New York.
- **Deduplication:** one alert for a continuing problem.
- **Recovery alert:** confirmation that the problem cleared.
- **Fallback dispatch:** a guarded manual start when the normal cron is absent.
- **Incident key:** a stable workflow identity used to avoid false recovery
  messages when only the failure explanation changes.
