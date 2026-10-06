# newedge_round3.py — H-newedge-3, three ideas built on real earnings dates

Script: `research_evidence/newedge_20261007/newedge_round3.py`
Tests: `tests/test_newedge_round3.py` (fake data only, no network)
Rules: "Hypothesis H-newedge-3" in `DELAY_STRESS_PAPER_ADVISORY.md`
(pre-registered 2026-10-07, before any run).

## What it does, in plain language

The nine ideas of the earlier rounds all re-used the same price and volume
numbers. This round adds one new kind of information: the days on which
companies really published their results ("earnings").

The project never had those dates. The panel's earnings columns hold a
placeholder ("60 days" on every row), because `USE_EARNINGS_DATA` is off.
The script gets the dates for free from the SEC's EDGAR service: a company
that publishes results files a form **8-K** with item **2.02**, and the SEC
records the exact time it arrived.

Three ideas use the dates:

- **P, earnings drift:** buy the stocks whose price jumped most after their
  latest report. Good news often keeps working for weeks, so buying one day
  late should cost little.
- **X, real earnings blackout:** the current strategy already has a rule
  "don't buy a new stock within 5 days before its earnings". Without dates
  it never did anything. Here it gets real dates.
- **PS, blended score:** half the current score, half the earnings score.

Only panel columns are changed, after the normal panel is built. The
engine file is never edited, so the paper lock stays valid.

### How a filing becomes a number

1. **Reaction day (E):** a filing before 16:00 New York time moves the price
   the same day. A later one (most are after the close), or one on a weekend,
   moves it on the next trading day.
2. **Reaction (R):** the stock's return from the close before E to the close
   after E, minus the median of the same two-day return over all panel
   stocks. This removes the market's own move.
3. **No peeking:** R needs the close of E+1, so it is first used on E+1.
   It stays valid for 40 trading days.

### Steps the script takes

1. **Step 0, coverage gate C0:** download the dates, then count how many
   ticker-quarters of 2012–2022 have at least one usable report. Under 90%
   means the round stops before any backtest.
2. **Decision gates:** the current strategy (S) and each idea run over 20
   calendar start days, on time and one day late, and are judged on
   2013–2022.
3. **Final exam:** only an idea that passes its gates takes the project's
   normal execution stress test (late fills, extra costs, 2023–2026). It
   must pass with no exceptions.

## How to run it

```bash
# Step 0 only: download the dates and print the coverage gate
python research_evidence/newedge_20261007/newedge_round3.py --membership-csv PATH --coverage-only
# the full round
python research_evidence/newedge_20261007/newedge_round3.py --membership-csv PATH
# smoke test only (not a result):
python research_evidence/newedge_20261007/newedge_round3.py --membership-csv PATH --offsets 2 --no-exam
```

- **Inputs:** `data/` and `signals/` (the normal research data), and the
  saved S&P 500 membership CSV. Without `--membership-csv` it is downloaded.
- **Network:** only the SEC, and only for tickers missing from
  `earnings_events.json`. The SEC asks scripts to identify themselves; set
  `SEC_USER_AGENT` to change the default text.
- **Outputs:**
  - `research_evidence/newedge_20261007/earnings_events.json`: every filing
    time used, per ticker, with the SEC company numbers.
  - `research_evidence/newedge_20261007/newedge_round3.json`: all runs, the
    gate numbers and the verdicts.
  - Exam reports (if any): `logs/research_candidate_execution_stress_newedge3_<idea>.*`.
- **Time:** 200 engine runs, about 10 minutes, plus up to 3 exams.

## Key terms

- **Earnings:** the results a company publishes every three months.
- **8-K, item 2.02:** the SEC form and item number a company uses when it
  publishes results.
- **Earnings drift:** after surprisingly good results, a stock tends to keep
  rising for some weeks.
- **Blackout:** a period in which the strategy may not open a new position.
- **Percentile rank:** where a value stands among that day's stocks, from 0
  (lowest) to 1 (highest).
- **Predecessor number:** when a company reorganises, the SEC gives it a new
  company number and its older filings stay under the old one. The script
  lists those old numbers so early years are not lost.
- **Coverage gate:** a check that the data is complete enough **before** any
  result is looked at.
- **Decision window / final exam:** ideas are judged on 2013–2022 first.
  2023–2026 is touched only once, at the end, so it can't be tuned to.
