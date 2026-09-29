# full_universe_check.py — H-full-universe survivorship test

Script: `research_evidence/full_universe_20260929/full_universe_check.py`
Tests: `tests/test_full_universe_check.py` (fake data only)
Rules: "Hypothesis H-full-universe" in `DELAY_STRESS_PAPER_ADVISORY.md`
(pre-registered 2026-09-29, before any data was downloaded).

## What it does, in plain language

The incumbent strategy picks stocks from a list of 62 big companies that was
written in 2026. Companies that shrank, went bankrupt or were bought between
2013 and 2022 are not on it, so the backtest never has to pick among them.
That makes the backtest look better than reality.

This script builds a fairer list and runs the unchanged strategy on it:

1. **Downloads prices** from Tiingo for every company that was in the S&P 500
   at any time from 2012-10-01 to 2022-12-31 (about 730 companies, including
   about 290 that later left the index).
2. **Checks coverage (gate C1)** before any backtest. At least 85% of the
   companies that left, and 95% of current members, need prices for at least
   90% of the days they were in the index. If not, it stops.
3. **Builds the "U" list**: on the first trading day of each month, the 62
   index members with the highest median daily dollar volume over the
   previous 63 trading days.
4. **Runs the strategy** on the U list (on time and one day late, from all 20
   start days), on today's list (R0) for comparison, and 100 times with
   random scores (MU). A diagnostic row (UD) also charges −30% on stocks
   that stop trading while held.
5. **Judges** with the pre-registered rules (U1, U2) and writes one verdict.

It never writes to `data/`, never changes a config or gate, and never trades.

## How to run it

It needs the project computer (with `data/` and `signals/`), internet, and a
Tiingo key in `.env` as `TIINGO_KEY=...`.

```bash
# Step 1: download + coverage gate. Resumable: run it again until it says done.
python research_evidence/full_universe_20260929/full_universe_check.py --step coverage

# Step 2: only if C1 passed.
python research_evidence/full_universe_20260929/full_universe_check.py --step runs
```

- **Tiingo free plan limits:** 50 requests an hour and 500 different tickers
  a month. The script waits between requests (`--per-hour 45` by default), so
  the download takes about a day. It needs two calendar months on the free
  plan (a paid month removes that limit). When Tiingo says the limit is
  reached, the script saves its progress and stops; run the same command
  again later.
  Tiingo reports the monthly limit as a normal "200 OK" reply with a
  message instead of prices; the script recognises that message too.
- **Outputs:**
  - `research_evidence/full_universe_20260929/full_universe_coverage.json`:
    the C1 result, per-company coverage, sector counts, renames used.
  - `research_evidence/full_universe_20260929/full_universe_check.json`:
    every run row, the gates and the verdict.
  - `research_evidence/full_universe_20260929/data/`: downloaded prices,
    feature files and SEC answers. Git ignores this folder, and Tiingo's
    free terms say the data is for personal use only.
- **Smoke test** (not a result): `--step runs --offsets 2 --monkeys 4` writes
  `valid_full_test: false`.

## How the pieces work

- **Membership** comes from the free fja05680/sp500 history (the same source
  as H-edge). A company counts as a member on a day if the latest snapshot on
  or before that day lists it under any of its tickers.
- **Ticker changes:** Tiingo stores a company's history under its latest
  ticker, so `RENAMES` maps old S&P tickers to that ticker (for example
  BK → BNY, FB → META, UTX → RTX). If a pair were wrong, the coverage step
  would show it as not covered.
- **Reused tickers:** some tickers now belong to a different company (ALTR
  was Altera; Tiingo's ALTR is Altair). Coverage compares dates, so a ticker
  whose Tiingo history starts after the old company's index time counts as
  not covered, instead of silently using the wrong company.
- **Sectors:** every company, old or new, gets its sector from its SEC
  industry code (SIC) through a fixed table in the script. Companies that
  still trade are found by ticker. For companies that stopped, the script
  asks Tiingo for the name and searches the SEC by name. No code →
  sector `OTHER`.
- **Features:** the project's own feature code (`pipeline_shared.py`)
  builds them offline. Stock prices come from Tiingo (adjusted), and SPY, QQQ
  and sector ETFs come from `data/`. Macro, VIX, news and valuation inputs are
  left neutral, because the incumbent's score doesn't use them. The
  cross-sectional ranks (`xs_rank_*`) are recomputed across the whole pool.
  A check on AAPL matched the project's own feature file on normal days.
- **Stopped stocks:** if a held stock stops trading before its 20-day exit,
  it is sold at its last Close. The script adds flat price bars after the
  last real day so the backtest can price the sale. Those added rows can
  never be bought.

## Key terms

- **Survivorship bias:** testing only on companies that survived, which
  hides the ones that failed.
- **Point-in-time:** using only what was known on each past date.
- **Dollar volume:** price × shares traded in a day, a simple size measure
  that is known on the day.
- **Coverage:** the share of a company's index days for which we have prices.
- **Pre-registration:** writing the pass/fail rule down before running, so
  the result can't bend the rule.
- **Random picks (monkeys):** the same strategy with random scores. The real
  strategy has to beat nearly all of them to show skill.
