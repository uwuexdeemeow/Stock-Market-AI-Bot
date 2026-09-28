# core_satellite_calendar_robustness.py - Rebalance Calendar Robustness

## What It Does

The core-satellite strategy rebalances every 20 trading days. Which 20 days
depends only on the first day of the data, and that choice is arbitrary.
Starting one day later picks almost the same stocks but enters and exits on
different days, and the headline results can change a lot.

This script reruns the backtest once for every possible start day (an
"offset" of 0 to 19 days) and reports the spread: median, worst quarter
(p25), worst and best. Offset 0 is the calendar the live strategy uses.

It is research-only. No gate reads its output, and it never approves or
blocks trading.

### Why it exists (2026-09-28 finding)

With the approved config on local data, the 2023-2026 holdout alpha versus
QQQ was:

| | Usual calendar (offset 0) | Across 19 offsets |
|---|---|---|
| Holdout alpha vs QQQ | +58% | median +26%, range -72% to +72% |
| Offsets with holdout alpha at or below 0 | | 5 of 19 |
| Full-history total return | 4607% | 2701% to 7246% |
| Max drawdown | -30.5% | -29% to -34% |

So the usual calendar was 5th luckiest of 19. The edge looks real (median
still positive) but much noisier than one backtest suggests. The failing
"one day late" execution stress is the same effect: identical picks, with
three periods making most of the difference.

## How To Run It

```bash
python core_satellite_calendar_robustness.py
python core_satellite_calendar_robustness.py --offsets 5
python core_satellite_calendar_robustness.py --telegram   # also send summary
python core_satellite_calendar_robustness.py --candidate-json my_config.json --candidate-name lowturn_v1
```

Inputs:

- The approved config in `signals/core_satellite_alpha_metrics.json`, or a
  research candidate JSON (walk-forward result or plain settings file)
- Factor data in `data/`

Expected outputs:

- `logs/core_satellite_calendar_robustness.csv` - one row per offset
- `logs/core_satellite_calendar_robustness.json` - rows plus summary
- Candidate runs write `logs/research_candidate_calendar_robustness_<name>.*`
  instead, stamped `research_candidate: true`.

Each backtest takes about 5-10 seconds, so 20 offsets take a few minutes. An
offset that hits a data problem is recorded with `status: error` and the
rest still run.

## Weekly Automation

`.github/workflows/weekly_calendar_robustness.yml` runs this script once a
week. Like every workflow here it has no GitHub cron: a cron-job.org job
calls its `workflow_dispatch` every Saturday at 15:00 UTC (it can also be
started from the Actions tab). It:

1. Restores the same checked factor-data snapshot the daily paper run uses.
2. Runs `python3 core_satellite_calendar_robustness.py --telegram`.
3. Publishes to the `signals/latest` branch:
   - `logs/core_satellite_calendar_robustness.json` / `.csv` (latest week)
   - `logs/calendar_robustness_history/YYYY-MM-DD.json` (one file per week,
     so the trend can be compared over months)

It never talks to the broker and no gate reads its output. The workflow is
a new file, not one of the files frozen in `paper_version_lock.json`, so
adding it does not disturb a running paper epoch.

`--telegram` sends a short summary (median, worst and best holdout alpha,
the live calendar's rank, drawdown). A failed message does not fail the
report.

Why not a gate yet: with results ranging from -72% to +72% on one run, any
threshold would be a guess. Collect a few months of weekly history first.

## Reading The Summary

- `holdout_alpha_vs_qqq_pct.median` - the typical 2023-2026 result. Trust
  this more than the offset-0 number.
- `holdout_alpha_vs_qqq_pct.p25` / `min` - how bad an unlucky calendar is.
- `holdout_alpha_vs_qqq_nonpositive_share` - share of calendars that did not
  beat QQQ in 2023-2026.
- `usual_calendar_holdout_rank` - where the live calendar ranks (1 = best).
  Near 1 means the headline number is flattered by luck.

## Key Concepts

- Rebalance calendar: the fixed list of days when the strategy trades.
- Offset: how many days later the calendar starts.
- Holdout: the 2023-2026 period kept aside to check the strategy on recent
  data.
- Alpha: return above a benchmark such as QQQ.
- Median: the middle value; half the calendars did better and half worse.

Offline tests: `python -m pytest tests/test_calendar_robustness.py -q`.
