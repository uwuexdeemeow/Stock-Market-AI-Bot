# Survivorship Check of the Stock List — 2026-09-26

**Question:** how much does the stock list (`settings.WATCHLIST`, 62 names)
lean on hindsight?

**Script:** `research_evidence/survivorship_check_20260926/watchlist_membership_check.py`
(tests: `tests/test_watchlist_membership_check.py`). Output:
`watchlist_membership_check.json` in the same folder.

**Source:** the free, MIT-licensed community history of S&P 500 membership
([fja05680/sp500](https://github.com/fja05680/sp500), snapshots up to
2026-08-18). The project already treats this list as an **unverified
candidate** (see `doc_audit_evidence_recovery.md`). It is good enough to
measure the problem, but not to pass the point-in-time membership gate.
This check changes no gate, config or trading universe.

## Findings

| | Jan 2010 (backtest start) | Jan 2013 (bake-off decision window) |
|---|---|---|
| S&P 500 members | 499 | 497 |
| …no longer in the index by 2026 | 226 (45%) | 199 (40%) |
| …of those, on our list | **0** | **0** |
| Our names not yet in the index | 9 | 6 |

The 45% includes plain ticker changes (for example UTX → RTX), so the real
share of companies that left is lower. It is still large.

Names on the list that were **not** in the S&P 500 in January 2010, with the
date they joined: CCI (2012-03), NFLX (2010-12), ABBV (2013-01, spun off from
ABT), META (2013-12, as FB), AVGO (2014-05), EQIX (2015-03), CHTR (2016-09),
LYV (2019-12), TSLA (2020-12). RTX and LIN were in the index under older
tickers (UTX, PX), so they are not hindsight picks.

**The incumbent's own trades** (`signals/core_satellite_alpha_trades.csv`,
630 stock holdings from 2010 to 2026):

- **152 (24%)** were in those 9 names, which a 2010 investor had no reason
  to put on a large-company list;
- **53 (8.4%)** were held **before the company joined the S&P 500**: TSLA 23
  periods (2011–2020), EQIX 17, LYV 7, CHTR 6.

## What this means

- The stock-picking history is flattered twice. None of the roughly 200+
  companies that fell out of the index can be picked. And about a quarter of
  the incumbent's picks were stocks we only know to include because they
  later became large.
- It supports the bake-off's survivorship rule. Ideas A and B pick from this
  list, but idea C (sector ETFs) doesn't, so C wins if it gets close
  (`DELAY_STRESS_PAPER_ADVISORY.md`, H-bakeoff).
- The real fix is unchanged. Build a point-in-time universe that includes
  the companies that later left, with price files for delisted stocks. The
  missing piece is a free, lawful price source for delisted stocks. It can't
  be done from this cloud session: only GitHub and PyPI are reachable here.

## Key terms

- **Survivorship bias:** testing only on companies that still exist (or are
  still big) today, which hides the ones that failed.
- **Hindsight pick:** a stock on the list only because of what happened later.
- **Point-in-time universe:** the set of stocks an investor could actually
  have chosen from on each past date.
- **Ticker change:** the same company trading under a new symbol (FB → META).
