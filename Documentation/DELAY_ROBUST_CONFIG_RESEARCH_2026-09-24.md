# One-day-delay configuration research — 2026-09-24

## Question

Can a less concentrated version of the existing paper strategy keep its
performance when an order is filled one trading day later, without relaxing
any safety check?

## Rules fixed before this search

- Use refreshed ticker features and ETF prices through the completed
  2026-09-23 New York session. A failed current data-health check blocks
  the final stress test and any promotion. A pre-2023-only exploratory
  walk-forward may still run if its affected ticker history through 2022
  passes the same structural check; label this screening provisional.
- Search only the repository's existing `--stable-grid` candidate family.
  This tests broader stock baskets and the already-defined weighting,
  volatility, and small TQQQ choices. Do not add candidates in response to
  the results of this run.
- Select with nested walk-forward outer years ending in 2022. Use the
  program's existing inner-selection and approval rules. Do not publish a
  live configuration during research.
- Then run the selected candidate through the existing full-history
  one-day-delay and extra-cost scenarios, plus the unchanged survivorship,
  feature-health, and factor-decay checks. Record every result, including
  rejection.
- The 2023–2026 period has already been examined for the incumbent's
  failure. Treat performance there as a **previously viewed diagnostic**,
  not a pristine holdout or proof of future returns. A historical pass alone
  does not justify live promotion; require additional forward paper evidence.
- The current feature-health profile is built from the latest full dataset
  and then reused in historical folds. Thus pre-2023 folds are useful for
  screening configurations, but are not a perfectly point-in-time replay of
  feature selection. Report this limitation rather than calling them clean
  out-of-sample proof.
- Do not change approval thresholds, bypass the daily guard, place orders,
  or approve real capital as part of this research.

## Why this search is limited

Trying many configurations until one happens to pass the known failing
period would make a reassuring backtest without credible evidence. The
existing stable grid provides one bounded hypothesis: diversification may
reduce dependence on the exact entry day. It may also reduce returns; the
walk-forward and stress tests decide whether the idea survives.

## Results

The current strict data-health check rejected NEE's 2026-09-23 Yahoo bar:
its reported Open is 79.04 while High is 79.00. NEE history through 2022
has no structural issue. The independent Alpaca IEX bar also disagrees with
Yahoo's Open. No price has been guessed, clipped, or approved; current-data
stress testing remains blocked until this provider conflict is resolved.

The provisional 2013–2022 nested walk-forward completed all 10 folds. It
selected the `top5` sticky-score family with 10% TQQQ in five folds, but the
**authoritative approval rejected it** for four independent reasons:

- Worst OOS turnover was 654.0%, above the unchanged 600% cap.
- Selector QQQ-alpha correlation was -0.146, below the required positive
  value. Higher inner scores were not reliably picking better later alpha.
- The 2022 fold needed a relaxed selection fallback.
- Selector Sharpe uplift versus the control was -0.175, below zero.

Mean OOS Sharpe (1.496) and CAGR (33.24%) alone do not erase these failures.
No configuration was promoted, and no delayed-fill stress was run on this
rejected candidate. The analyzer's older five-check summary incorrectly
printed “Safe to deploy” for this run; its recommendation has been corrected
to include the authoritative rejection and to never equate diagnostics with
deployment authorization. No safety thresholds were changed. The live
configuration remains the incumbent; current data health remains blocked on
the unresolved NEE bar.
