# Forward Test Rule FV: rule V on the paper account (owner-approved 2026-10-21)

**Why this exists:** core-alpha's approval ran out on 2026-10-19 and could not
be renewed. With every year scored it fails the cost-stress check, and it
passes or fails depending on tiny data differences (see
`DELAY_STRESS_PAPER_ADVISORY.md`, "2026-10-21 revalidation"). The owner moved the
paper account to rule **V**, which passed the pre-registered H-index-risk test
on 2026-10-16, including its one-time 2023–2026 exam. This file states, **before
the first V trade**, how the paper run will be judged. If the rule is written
after the results come in, a bad year is easy to explain away.

**Key term — forward test:** judging a strategy only on data that did not
exist when it was designed.

## What is traded (fixed; no tuning during the test)

- **Signal day:** the last NYSE session of each month, using that day's
  adjusted close (`data/QQQ.parquet`, refreshed by `refresh_etf_data.py`).
- **Weights:** QQQ share = min(1, 20% ÷ QQQ's annualised volatility of its
  last 63 daily returns). The rest goes in **BIL**. These are the same numbers
  and the same code (`vol_weight`) as the H-index-risk test.
- **Trade:** the next session's run, inside the usual 09:35–10:30 New York
  window, with the existing two-stage limit orders. If that run fails, the
  next run catches up, still using the signal day's weights.
- **Between signal days:** hold. Weights drift with prices on purpose.
- **No stops.** Core and stock trailing stops stay off, as in the test.
- **Safety halts moved outside the tested range.** The account-wide halts
  sell everything, which the tested rule never does:
  - Drawdown halt (`ALPACA_DD_HALT_PCT`): 12% → **40%**. The worst drawdown
    in the test was −38% (2007–2022).
  - One-day P&L halt (`GUARD_PNL_HALT_PCT`): −8% → **−12%**. QQQ's worst
    day since 1999 is about −12%, and V holds less QQQ when markets are
    jumpy.
  - If either halt fires, something is outside anything the test saw. Treat
    it as an incident, not a strategy result.
- **First trade:** signal on 2026-10-30, orders on 2026-11-02. Until then the
  account keeps its 2026-10-15 core-alpha positions and places no orders.
- **Paper only.** Nothing here approves real money.

## The measures

The clock starts at the first V fill. The data is the account's own daily
equity (`signals/alpaca_paper_equity.csv` on `signals/latest`) and its fills
(`signals/alpaca_paper_log.csv`). The yardstick **B** is buy-and-hold QQQ over
exactly the same days: adjusted closes, starting from the close of the day
before the first fill.

| | Measure | How |
|---|---|---|
| **V1** | Worst fall | Max drawdown of the account's daily equity vs B's max drawdown. |
| **V2** | Return per unit of risk | Sharpe of daily returns, annualised (√252), with BIL's daily return as the risk-free rate, for the account and for B. |
| **V3** | Real trading cost | For every rebalance fill: (fill ÷ that day's Open − 1), sign flipped for sells. Mean in bps, weighted by trade value. |
| **V4** | Tracking (information only) | Each month: account return vs the simulated V return over the same month (same weights, Open-to-Open). Not a pass/fail measure; it shows whether the bot does what the rule says. |

## Checkpoint (after 12 completed V rebalances, about November 2027)

- **PASS** only if all three hold (these are the H-index-risk exam rules):
  - **V1:** the account's max drawdown is not worse than B's.
  - **V2:** the account's Sharpe ≥ B's Sharpe − **0.20**.
  - **V3:** mean cost ≤ **20 bps**.
- **FAIL** otherwise. Stop the V paper run and report. Don't change the 20% /
  63-day settings and try again; that would be tuning after seeing results.
- A pass allows a real-money discussion. It still needs the owner, and it is
  not an automatic switch.

## Early operational stops (not strategy verdicts)

Pause and investigate, without judging V, if:
- V3's mean is above **40 bps** after the first 3 rebalances, or
- V4's monthly gap is above **2 points** in 3 months of any 6, or
- a rebalance is still not done 3 sessions after its signal day.

These mean the plumbing is wrong, not the rule. After a fix the clock
continues, unless the fix changes what is traded.

## Monthly log

Filled in automatically on the 3rd of each month by the Forward Test Monitor workflow
(`python forward_test_monitor.py --rule v --append-log`).

| Month | Rebalances done | Account DD | QQQ DD | Account Sharpe | QQQ Sharpe | Mean cost (bps) | V4 gap (pts) | Note |
|---|---|---|---|---|---|---|---|---|
| 2026-10 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | Monthly check (forward_test_monitor.py). |
