# top_names_check.py — H-top-names concentration test

Script: `research_evidence/top_names_20260929/top_names_check.py`
Tests: `tests/test_top_names_and_benchmark.py` (fake data only)
Rules: "Hypothesis H-top-names" in `DELAY_STRESS_PAPER_ADVISORY.md`
(pre-registered 2026-09-29, before any run).

## What it does, in plain language

The strategy holds only 3 stocks at a time, so a great backtest could come
from a few lucky names or a few giant winners instead of real skill. This
script runs the unchanged strategy on the point-in-time S&P 500 list (H-edge
test S) and asks:

- **N1 / N3:** what if the 1 or 3 stocks that earned the most had never been
  on the list? Those stocks are removed, and the engine picks the next best.
- **P1:** what if, in every 20-day period, the best of the picks had only
  earned what a typical stock earned that period?
- **MP:** the same P1 cut applied to 100 random-pick runs, so the harsh cut
  is judged against luck, not against zero.

Verdict: "edge is broad", "edge rests on a few names", and/or "edge rests
on each period's one big winner".

## How to run it

Run on the project computer (it needs `data/`, `signals/` and internet once
for the S&P 500 membership history):

```bash
python research_evidence/top_names_20260929/top_names_check.py
# smoke test only (not a result):
python research_evidence/top_names_20260929/top_names_check.py --offsets 2 --monkeys 4
```

It makes 220 engine runs. Output:
`research_evidence/top_names_20260929/top_names_check.json`, which holds
every run row, the chosen top tickers and their share of profit, the gates
and the verdict. `--membership-csv PATH` uses a saved copy of the membership
file.

## Key terms

- **Contribution:** a stock's weight × its 20-day return, added up over
  every period it was held.
- **Neutral stand-in:** the median 20-day return of all stocks that had a
  score that day.
- **Hindsight test:** removing names after seeing results. It measures
  concentration; it is not a rule anyone could have traded.
