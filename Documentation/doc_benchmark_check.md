# benchmark_check.py — H-benchmark fair-yardstick test

Script: `research_evidence/benchmark_20260929/benchmark_check.py`
Tests: `tests/test_top_names_and_benchmark.py` (fake data only)
Rules: "Hypothesis H-benchmark" in `DELAY_STRESS_PAPER_ADVISORY.md`
(pre-registered 2026-09-29, before any run).

## What it does, in plain language

The strategy is often only partly in the market, so "beat QQQ by X points"
mixes stock-picking skill with how much money was invested. This script
compares the unchanged strategy (point-in-time list, H-edge test S) with two
fairer yardsticks:

- **Exposure-matched (B-exp):** each 20-day period, QQQ at the same
  investment level the strategy had, with the rest in cash (BIL).
- **Beta-adjusted (B-beta):** a daily regression of the strategy on QQQ.
  The part left over is alpha. A Newey–West t-stat of 2 or more means it is
  unlikely to be noise.

It also measures control E (the core with no stock picks) the same way, for
context. Verdict: "edge holds against a fair benchmark", "beats a fair
benchmark, but not reliably", or "headline alpha is mostly market exposure
and timing of the core".

## How to run it

```bash
python research_evidence/benchmark_20260929/benchmark_check.py
# smoke test only (not a result):
python research_evidence/benchmark_20260929/benchmark_check.py --offsets 2
```

It makes 60 engine runs. Output:
`research_evidence/benchmark_20260929/benchmark_check.json`.

**Daily returns** are rebuilt from each period's holdings: bought at the
entry Open, valued at each Close, with the period's cost taken off the first
day. QQQ and cash use the same rule, so the regression compares like with
like. The overnight gap between one period's exit and the next entry is not
counted. Each row reports `daily_rebuild_mean_abs_gap_pct`, how far the
rebuilt period returns are from the engine's own. A smoke run showed about
0.3% per period.

## Key terms

- **Gross exposure:** the share of the account invested (core ETFs + stocks).
- **Beta:** how much the strategy moves when QQQ moves 1%.
- **Information ratio:** yearly alpha divided by the yearly wobble of what
  QQQ doesn't explain.
- **Newey–West t-stat:** a t-stat that allows for returns on nearby days
  being related.
