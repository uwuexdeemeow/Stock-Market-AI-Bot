# Forward Test Rule F1 (owner-approved 2026-10-02)

**Why this exists:** the backtests are done. Over 2013–2022 they show a real
but concentrated edge (H-edge, H-benchmark, H-top-names), and a weak one
since 2023. The only question left is whether the strategy works **from now
on**. Paper trading answers that, but only if the pass/fail rule is written
down **before** the results come in. Otherwise a bad year is easy to explain
away.

**Key term — forward test:** judging a strategy only on data that did not
exist when it was designed.

## What is tested

- **Strategy:** the incumbent exactly as frozen in paper epoch
  `paper-v20260926T140402Z` (20-day calendar, no trailing stops).
- **Clock:** counts completed 20-day rebalance periods from the epoch
  start. A change to the strategy's trading logic starts a new epoch **and
  restarts this clock**. Bug fixes that keep the epoch don't restart it.
- **Data:** the paper account's own daily equity
  (`signals/paper_equity.csv` on the `signals/latest` branch), its fills
  (`signals/alpaca_paper_log.csv`), and the daily heartbeats.
- **Blocked days count.** If the safety gate blocks a rebalance, the account
  keeps its old positions and that period is judged as it really went. A
  strategy that can't trade is part of the result.

## The three measures

| | Measure | How |
|---|---|---|
| **M1** | Return vs a fair yardstick | Paper account return minus the exposure-matched benchmark (the same yardstick as H-benchmark): each day, QQQ × the account's actual invested share (`current_gross_exposure`) + BIL × the rest. Chained over all days, annualised. |
| **M2** | Real trading cost | For every fill: (fill price ÷ that day's Open − 1), with sign flipped for sells. Positive = worse than the Open. Mean in bps with a 95% confidence interval. |
| **M3** | Can it actually trade? | Share of scheduled rebalance dates that the safety gate blocked. |

**Numbers the rule is built on (from the backtests):**

- Expected M1 if the backtest edge is real: about **+9.6% a year**, with a
  yearly tracking error of about **10.3%** (H-benchmark, one-day-late runs:
  information ratio 0.93).
- The research assumes trading costs of **10 bps per trade**, and stress
  tests use **20 bps** (cost_stress 2.0).

## Checkpoints

**Checkpoint 1 — after 12 completed periods (about 1 year): stop or carry on**

- **STOP** if either:
  - M1 is below **−10.3% a year**. That is one tracking error below zero:
    if the backtest edge were real, this would happen only about 3% of the
    time.
  - M2's mean is above **+40 bps**, and the lower end of its 95% interval
    is above **+20 bps** (fills clearly cost more than the stress test
    assumes).
- Otherwise **CARRY ON**. One year can catch a disaster, but it can't prove
  an edge.

**Checkpoint 2 — after 24 completed periods (about 2 years): verdict**

- **PASS** only if M1 is above **0** **and** M2's mean is at most **+20
  bps**. A pass allows a real-money discussion. It still needs the owner,
  every existing gate (including the stress test with no advisory) and a
  small first size. It never approves real money by itself.
- **FAIL** otherwise. The incumbent stays paper-only, and the
  "new edge" research below starts if it hasn't already.
- **Honest power note:** if the true edge is +9.6% a year, a 2-year M1
  comes out above 0 about 91% of the time. If the true edge is zero, it
  still comes out above 0 about 50% of the time. So a pass is a "not
  disproven", not proof.

## New-edge research runs in parallel, starting now

The owner decided (2026-10-02) not to wait for a trigger. New-edge research
starts **now**, alongside paper trading, under the rules below. The checks
below don't start that research any more; they decide what happens to the
**incumbent**:

- **T-block:** 4 or more of any 6 scheduled rebalance dates in a row are
  blocked by the safety gate. The recent edge doesn't survive realistic costs
  in practice, so new-edge research becomes the top priority.
- **T-stop / T-fail:** checkpoint 1 says STOP or checkpoint 2 says FAIL. The
  incumbent is retired from consideration for real money.

A new idea that passes its own pre-registered research gets its **own**
paper epoch and its own forward test under this same rule. Its clock starts
when it starts paper trading.

## Rules for that new-edge research (so it doesn't fool us)

1. Each idea is written into `DELAY_STRESS_PAPER_ADVISORY.md` with its
   pass/fail rule **before** it is run, the same way as H-bakeoff.
2. Ideas are chosen and tuned on **2013–2022 only**. 2023–2026 is used once,
   as the final exam, and is not reused for tuning.
3. Each idea must pass the existing execution stress test (one day late,
   +25 bps) **without** any paper advisory, and must hold more than 3 names
   or show it doesn't rely on a few winners (the H-top-names test).
4. Keep the number of ideas small (at most 3 per round), and report the
   failures too.

## Who checks and when

- Monthly: Claude (or the owner) computes M1, M2 and M3 and adds one line
  to the log below. Nothing is judged before a checkpoint, except the T-block
  trigger.
- At each checkpoint: apply the rule exactly as written, record the result
  here, and don't change the thresholds.

## Log

| Date | Periods done | M1 (annualised) | M2 mean (bps, 95% CI) | M3 blocked share | Note |
|---|---|---|---|---|---|
| 2026-10-03 | 0 | n/a | −20 bps (−100 to +61), 16 fills, all before this epoch | 0 of 0 scheduled dates | Daily runs blocked 9-29 to 10-02 (stress `delay_1d_extra_25bps` loses to the blend by about 2.3 points), but no rebalance was due. Next scheduled rebalance: 2026-10-14. Owner: leave the gate as is. |
