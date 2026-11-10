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
| **V1** | Worst fall | Max drawdown of the account's daily equity vs B's max drawdown. Since 2026-11-10 the account is valued at the official close (amendment A2). |
| **V2** | Return per unit of risk | Sharpe of daily returns, annualised (√252), with BIL's daily return as the risk-free rate, for the account and for B. |
| **V3** | Real trading cost | For every rebalance fill: (fill ÷ that day's Open − 1), sign flipped for sells. Mean in bps, weighted by trade value. Since 2026-11-10: QQQ and BIL fills only (amendment A1). |
| **V4** | Tracking (information only) | Each month: account return vs the simulated V return over the same month (same weights, Open-to-Open). Not a pass/fail measure; it shows whether the bot does what the rule says. |

## Amendments (2026-11-10, owner-approved)

Both were written **after** the first 6 trading days, so they are not part of
the original pre-registration. They change only **how the account is
measured**. What is traded, the 20% / 63-day settings and the three pass
lines (V1, V2 − 0.20, V3 ≤ 20 bps) are unchanged. The old-style numbers stay
in the monitor's JSON output so the two can always be compared.

- **A1 — V3 counts only QQQ and BIL fills.**
  - *Problem:* on 2026-11-02 the account also sold its leftover core-alpha
    stocks (AMD, AMAT, CAT). Those sales were 136, 89 and 61 bps below the
    Open, mostly because prices fell between the 09:30 Open and the 10:02
    fill. With them V3 read 48.3 bps; the one real V trade (the QQQ buy) was
    −13.1 bps.
  - *Why it matters:* those sales are about half of all money traded so far.
    They could keep the 12-rebalance mean above 20 bps even if every V trade
    is cheap, failing V for something rule V never does.
  - *Change:* V3 uses fills in QQQ and BIL only. The all-fills figure is still
    reported as `all_fills_information_only`.
- **A2 — V1, V2 and V4 value the account at the official close.**
  - *Problem:* `alpaca_paper_equity.csv` is written at 17:15 New York with
    after-hours prices; yardstick B uses the 16:00 close. On 2026-11-05 this
    made the account's fall −7.03% against QQQ's −6.49% while the account
    held only QQQ.
  - *Why it matters:* V1 passes only if the account's fall is not worse than
    B's. At 100% QQQ the two are equal by construction, so clock noise alone
    could decide the test.
  - *Change:* each day's account value = cash (from the equity file) + shares
    × that day's raw official close. Shares come from
    `alpaca_daily_status.json` walked back through the fill log. Valued this
    way the same fall is −6.43%. If a price is missing the monitor falls back
    to the 17:15 snapshot and says so in `equity_basis`.
  - *Known small limit:* between a fund's ex-dividend day and its pay day the
    account looks lower by the dividend (QQQ about 0.15% a quarter).

The 2026-11 log row written on 2026-11-03 used the old measures. The row
dated 2026-11-10 below is the same month re-measured.

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
| 2026-11 | 1 | -1.27% | -1.36% | -1.906 | -3.0 | 48.3 | n/a | Monthly check (forward_test_monitor.py). |
| 2026-11 | 1 | -6.43% | -6.49% | -0.96 | -0.834 | -13.1 | n/a | Re-measured after amendments A1 and A2 (2026-11-10). |
