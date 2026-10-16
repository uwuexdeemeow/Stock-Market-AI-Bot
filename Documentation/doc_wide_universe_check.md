# wide_universe_check.py — H-wide-universe, data step

Script: `research_evidence/wide_universe_20261016/wide_universe_check.py`
Rules: "Hypothesis H-wide-universe" and its implementation notes in
`Documentation/DELAY_STRESS_PAPER_ADVISORY.md` (written before any download).

## What it does, in plain language

The strategy has only been tested on 62 big companies picked in 2026. That
flatters it, because companies that shrank or failed between 2013 and 2022
were never in the test. H-wide-universe re-tests the strategy on every
company that was in the S&P 500 back then. First the price data has to be
good enough, and that is what this script checks.

1. **`--step wiki`** reads a public Kaggle copy of the free Quandl WIKI
   price archive and keeps the rows for every
   company that was in the S&P 500 at any time from 2012-10-01 to
   2022-12-31 (709 companies, about 748 ticker spellings). WIKI stopped
   updating in March 2018, but it still holds many companies that no
   longer trade.
2. **`--step coverage`** joins ("stitches") WIKI with the Tiingo price files
   already downloaded for H-full-universe, then applies gate **C1**: at
   least 85% of the companies that left the index, and 95% of current
   members, must have prices on at least 90% of the days they were members.
   If C1 fails, the test stops: "not testable with this source".

3. **`--step sectors`** gives every company a sector from its SEC SIC code.
   Companies that no longer trade are found at the SEC by name; the name
   comes from Wikipedia's "Historical components of the S&P 500" table.
4. **`--step runs`** builds the project's features from the stitched
   prices and runs the unchanged strategy:
   - **R0** (project data) and **R0n** (new data, project recipe) first;
     gate **C2** stops everything if they disagree by more than 25%.
   - **U**: the 62 most-traded S&P 500 members each month (fair version of
     today's list). **W**: every S&P 500 member that month (~494).
   - **MU / MW**: 100 random-pick runs in each universe (luck check).
   - **WN3**: W without its 3 biggest winners (breadth check).
   - **UD / WD**: a −30% hit when a held company stops trading (diagnostic).

## How to run

One-time setup: download the Kaggle copy of WIKI (about 463 MB, no login
needed) to `research_evidence/wide_universe_20261016/data/kaggle_wiki/wiki.zip`:

```bash
curl -L -o research_evidence/wide_universe_20261016/data/kaggle_wiki/wiki.zip https://www.kaggle.com/api/v1/datasets/download/marketneutral/quandl-wiki-prices-us-equites
```

The official WIKI API on data.nasdaq.com no longer works with a free
account, so the Kaggle copy (same table, frozen 2018-03-27) is used.

Then, from the project root:

```bash
python research_evidence/wide_universe_20261016/wide_universe_check.py --step wiki
python research_evidence/wide_universe_20261016/wide_universe_check.py --step coverage
python research_evidence/wide_universe_20261016/wide_universe_check.py --step sectors
python research_evidence/wide_universe_20261016/wide_universe_check.py --step runs
```

- `--step runs` takes about an hour and needs ~10 GB of memory. Run it in
  a normal terminal; if it stops, run it again and it skips finished runs.
  Output: `research_evidence/wide_universe_20261016/wide_universe_check.json`.
- Smoke test (not a valid result): add `--offsets 2 --monkeys 4 --out <other file>`.

- `--step wiki` takes a few minutes: it reads ~15 million rows and saves
  one small file per needed ticker. It records the zip's SHA-256, row count
  and last date.
- **Output:** `research_evidence/wide_universe_20261016/wide_universe_coverage.json`
  with the C1 verdict, per-company coverage and which source covered each
  company. The console prints a short summary.
- **Downloads** go to `research_evidence/wide_universe_20261016/data/`,
  which Git ignores. The project's real `data/` folder is never touched.

Tests: `python -m pytest tests/test_wide_universe_check.py -q`.

## How the join works

- Tiingo is used wherever it has a price.
- WIKI only fills the days **before** Tiingo's first price, or every day if
  Tiingo has nothing for that company.
- WIKI prices are multiplied by one number so they line up with Tiingo's
  prices. Daily returns stay exactly the same.
- If the two sources share fewer than 20 days, or their daily returns on
  shared days don't move together (correlation below 0.95), they may be two
  different securities with the same ticker. The WIKI part is then dropped
  and listed in the output.

## Result (2026-10-16)

C1 and C2 passed. **U:** "edge real but mostly list hindsight" (keeps
about 16% of the hand-picked list's alpha). **W:** "real but too small or
too narrow" (beats random picks but loses to QQQ). Details in
`DELAY_STRESS_PAPER_ADVISORY.md`.

## Key terms

- **Survivorship bias:** testing only on companies that survived to today.
  It makes a backtest look better than real life, because the failures are
  missing.
- **Point-in-time membership:** who was really in the S&P 500 on each past
  date, not who is in it today.
- **Adjusted price:** a price corrected for splits and dividends so that
  returns over time are correct.
- **Coverage gate (C1):** a data check run before any backtest. If too many
  companies have no prices, any result would be unreliable, so the test
  stops.
- **Stitching:** joining two price sources into one history for a company.
- **Random-pick run ("monkey"):** the same strategy with its scores
  replaced by random numbers. If the real strategy can't beat these, its
  results could be luck.
- **Hindsight list:** a stock list chosen today from companies that turned
  out well. Backtesting on it overstates how good a strategy would have been.
- **Pre-registered:** the pass/fail rules were written down before the data
  or results were seen, so they can't be bent to fit the results.
