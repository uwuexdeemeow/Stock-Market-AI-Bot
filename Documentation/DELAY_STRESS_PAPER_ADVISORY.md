# One-Day-Delay Stress: Paper-Only Advisory (2026-09-26)

## What happened

Daily Paper Trading run #298 (2026-09-25) was safety-blocked with
`execution_stress_review_failed`. Commit `cfa0723` made the execution stress
test use the **complete** approved configuration, including the per-regime
weights. Before that commit, the test had been stressing a slightly
different portfolio. With the real portfolio, every one-day-late scenario
loses the 2023–2026 edge over QQQ:

| Scenario | 2023–2026 alpha vs QQQ | Full-history alpha vs QQQ |
|---|---|---|
| base | +41.7% | +3146% |
| delay_1d | −8.2% | +2298% |
| delay_1d + 10 bps | −11.1% | +2116% |
| delay_1d + 25 bps | −15.4% | +1859% |

The history-based gate is very unlikely to pass again unless the strategy
changes, so the block would effectively be permanent.

## Decision (owner-approved "option 4")

1. **Paper trading continues** with this known weakness shown as a loud
   warning, so forward evidence keeps accumulating.
2. **A delay-robust replacement is researched** in parallel (see the plan
   below).
3. **Real capital stays blocked** until a strategy passes every stress
   scenario outright.

## Exactly what was relaxed (and what was not)

`robustness_review.py` treats a failed stress row as a *paper advisory*
only when **all** of these are true:

- the row is a delayed-entry scenario (`entry_delay_days > 0`);
- its only failed gate is `holdout_2023_2026_vs_qqq_pass`;
- its full-history alpha versus QQQ **and** the SPY/QQQ blend is still positive.

Everything else still blocks paper orders: base and cost-only scenarios,
any other failed gate, negative full-history alpha, a stressed drawdown
worse than −35%, and missing or mismatched reports.

Advisory rows:

- are listed in `execution_stress_review.paper_advisory_scenarios`;
- set `execution_stress_review.capital_approval_pass = false`;
- add `execution_stress_capital_evidence_incomplete` to the validation
  bundle's reasons and make `capital_approval_eligible` false;
- print a `PAPER-ONLY advisory` warning annotation on every run
  (`robustness_snapshot_gate.py`).

**Key term — "paper advisory":** a warning that is recorded and shown but
does not stop simulated (paper) orders. It can never count as permission to
trade real money.

## Activating it (owner step)

`robustness_review.py` and `validation_bundle.py` both sit in the reviewed
paper release. After you review and commit the change, re-freeze the
release, otherwise the daily run refuses to trade:

```bash
python -m pytest -q
git add robustness_review.py robustness_snapshot_gate.py validation_bundle.py \
    tests/test_medium_gap_fixes.py Documentation/DELAY_STRESS_PAPER_ADVISORY.md
git commit -m "Paper-only advisory for one-day-delay holdout stress"
python paper_validation_epoch.py --freeze-current
python paper_validation_epoch.py --check-lock
git add paper_version_lock.json && git commit -m "Freeze delay-advisory paper release" && git push
```

`--freeze-current` keeps the current epoch start. The trading logic did not
change, only the rule that decides whether the gate blocks. If you would
rather start a clean evidence period, run
`python paper_validation_epoch.py --invalidate-current --reason delay_stress_advisory`
instead.

## Replacement research plan

The rules below are fixed **before** running anything, so the results
can't steer the search:

- **Why the incumbent is fragile:** nested walk-forward selection never
  looks at late fills. It picks configurations that are tuned to the
  exact entry day.
- **Hypothesis:** configurations that are more diversified and trade less
  (the `--low-turnover-grid`: overlay 0.25/0.50, broader shapes) keep more
  of their edge when fills arrive a day late.
- **Selection:** outer years up to 2022 only, using the existing inner
  selection and approval rules. Don't publish a live config during
  research.
- **Acceptance:** every scenario in `core_satellite_execution_stress.py`
  must pass outright, **with no advisories**, plus the unchanged
  survivorship, feature-health and factor-decay checks. 2023–2026 has
  already been seen, so treat it as a diagnostic, not a clean holdout.
- **If nothing passes:** record the rejection and keep the incumbent on the
  paper advisory. Don't widen the grid in response to results.

- **Delay-aware selection (added 2026-09-26):** every inner fold is also
  replayed with fills one trading day late, and the candidate keeps the
  **worse** of its two scores (lower alpha, higher turnover). Candidates that
  need perfect timing lose inside the selector, before the final stress
  test ever sees them. TQQQ candidates are rejected in this mode, because
  the TQQQ engine can't replay late fills.

```bash
python core_satellite_nested_walkforward.py --low-turnover-grid --end-year 2022 \
    --selection-entry-delay-days 1 --no-publish-live-config \
    --output-prefix wf_delay_robust_lowturnover_20260926
```

The Colab notebook (`Colab/stockbot_walkforward.ipynb`) already uses these
settings. The notebook checks out the exact commit recorded in the data
snapshot. Make the snapshot with `python prepare_colab_walkforward.py`
**after** this change is pushed, otherwise Colab runs the old code without
the flag.

## Result: delay-aware low-turnover run (2026-09-26) — REJECTED

Run `wf_delay_robust_lowturnover_20260926` finished on Colab. The result
was unpacked into `Colab/result_20260926/signals/`. The delay flag was active:
every fold has `selection_entry_delay_days = 1`.

- **Folds:** 10 of 10 valid (outer years 2013–2022), no fallback folds.
- **Out-of-sample (OOS) performance:** 714.8% compound return, mean Sharpe
  1.41, beat QQQ in 9 of 10 years, mean alpha vs QQQ +7.3%, worst drawdown
  -16.6%.
- **Leading family:** `score=regime_adaptive, shape=top3,
  weighting=sticky_score, risk=off, tqqq=0` (chosen in 6 of 10 folds).
- **Inside the selector:** in each fold, 32 of the 64 candidates were
  TQQQ-based and were dropped because TQQQ can't replay late fills. Another
  20 failed the cost-stress gate, which left 12 valid configs.

**Approval: NOT approved** by the unchanged walk-forward approval rules:

1. `selector_alpha_correlation = -0.531` (must be > 0). A higher inner
   score went with *lower* OOS alpha vs QQQ, so the selector's ranking
   doesn't predict results.
2. `selector_sharpe_uplift = -0.258` (must be >= 0). The selector did worse
   than the frozen baseline (mean OOS Sharpe 1.41 vs 1.67).

The analyzer also flagged concentration vulnerability (FAIL): years with a
more concentrated portfolio had mean alpha of +2.3%, versus +12.4% in years
with a less concentrated one.

The `medium_risk_review` block inside the result JSON was copied from the
snapshot's existing logs. It describes the **incumbent**, not this
candidate, so it is not evidence for this candidate.

**Decision:** rejected. The incumbent stays on the paper advisory. Per the
fixed plan, the grid will not be widened in response to this result.

## Diagnostic: rebalance-day ("timing") luck (2026-09-26)

**Question:** is the one-day-delay failure really about late fills, or about
*which day* the 20-day rebalance calendar happens to fall on?

**Method:** the incumbent config was run 40 times on the current local data:
the 20-day calendar started on each of the 20 possible first days (offsets
0–19), each with on-time fills and with fills one day late. Every run ended on
the same date (30 sessions before the data ends, so all labels are complete).
Nothing was selected or tuned. Script and raw output:
`research_evidence/phase_luck_20260926/`.

| 2023–2026 alpha vs QQQ | min | median | mean | max | negative |
|---|---|---|---|---|---|
| on-time fills | −72.2 | +26.5 | +21.4 | +74.2 | 5 of 20 |
| one day late | −63.0 | +27.6 | +19.4 | +68.4 | 6 of 20 |

- The **start day** moves the 2023–2026 result by about 146 points.
- A one-day **delay** costs only about 2 points on average at the same start
  day.
- Offset 0 is +57.7 on time and +8.1 late. It is ranked 6th of 20, so the
  official stress row compares a lucky day with a less lucky one.

**Key term — "timing luck":** with only 3 stocks held for 20 days and traded
on one fixed calendar, the result depends heavily on which days you happen to
trade. The late-fill stress test mostly measures that luck, not a real
weakness in fill timing.

## Hypothesis H-tranche (pre-registered 2026-09-26, before any tranche run)

**Hypothesis:** splitting the incumbent into 4 *tranches* removes most of the
timing luck, and the resulting book passes the one-day-delay stress on merit.
Each tranche holds 25% of capital with the unchanged incumbent config, and the
tranches rebalance 5 sessions apart.

**Key term — "tranche":** a slice of the portfolio. Four slices each
rebalance every 20 days, but on different days (day 0, 5, 10, 15), so no single
day decides the result. The signal, shape and costs don't change. Only the
calendar is spread out. This also spreads the book over up to 12 names
instead of 3, which addresses the concentration FAIL above.

**Nothing else changes:** no new grid, no selector, no threshold changes. The
config is the incumbent's, fixed in advance.

### Step 1 — preview (research only, no locked file touched)

A 4-tranche book is approximated by averaging the equity of single-calendar
runs at offsets {k, k+5, k+10, k+15}. Each slice starts at 25% and drifts;
there is no netting between slices, so costs are slightly overstated
(conservative). This is done for k = 0–4 (the 5 distinct tranche calendars),
on time and one day late: 10 books.

**Preview passes only if both are true:**

1. all 10 books have positive 2023–2026 alpha vs QQQ, **and**
2. the spread (max − min) of 2023–2026 alpha vs QQQ across the 5 on-time
   books is under 73 points (half the single-calendar spread of 146).

2023–2026 has been seen before, so this is a diagnostic, not a clean holdout.

### Step 2 — only if the preview passes (owner decision needed)

Tranche support must be added to `core_satellite_alpha.py`, which is a
**locked** file, so the owner must agree before it is built. After that, the
tranche config goes through the unchanged gates: nested walk-forward in fixed
mode (outer years to 2022), `core_satellite_execution_stress.py
--candidate-json`, and the survivorship audit. Acceptance is as before: every
stress scenario must pass outright, with no advisory.

### If the preview fails

Record the rejection here. Conclusion: the incumbent's 2023–2026 edge can't
be separated from timing luck. The incumbent stays on the paper advisory, and
the next research step is a new signal, not a new grid.

### Result: H-tranche preview (2026-09-26) — REJECTED

Run with `research_evidence/phase_luck_20260926/tranche_preview.py`; the raw
output is in `tranche_preview.json` in the same folder. Data ends 2026-08-12.

| Tranche calendar k | 2023–2026 alpha vs QQQ, on time | one day late | 2013–2022 alpha vs QQQ, on time |
|---|---|---|---|
| 0 (offsets 0/5/10/15) | +12.1 | −2.7 | +647 |
| 1 (1/6/11/16) | +2.7 | +4.8 | +635 |
| 2 (2/7/12/17) | +10.2 | +7.8 | +790 |
| 3 (3/8/13/18) | +4.9 | −3.1 | +712 |
| 4 (4/9/14/19) | −12.4 | +3.7 | +569 |

- Rule 2 (spread under 73 points): **passed**, 24.6 points. Tranches do
  remove most of the timing luck.
- Rule 1 (all 10 books beat QQQ over 2023–2026): **failed**. 3 of 10 are
  negative.

**What this means:** with the timing luck averaged out, the incumbent's
2023–2026 edge over QQQ is small, about +3 points in the median book over
roughly 3.6 years, and not reliably positive. 2013–2022 alpha stays large in
every book, so the edge has faded recently. The failing delay row was mostly
timing luck on top of a thin recent edge.

**Known limit (not a reason to re-run):** the preview slices drift from 2010
without re-balancing between them. By 2023 the book leans toward whichever
slice had grown most. A real tranche engine would behave the same way unless
it re-balanced slices, which is a different design.

**Decision:** H-tranche rejected by its pre-registered rule. The incumbent
stays on the paper advisory. Tranche support will **not** be added to the
locked engine for this hypothesis. Next research step: a new signal, or
refreshing the existing one for the recent regime, pre-registered here
before any run. Not a wider grid.

## Hypothesis H-bakeoff (pre-registered 2026-09-26, before any run)

**Owner request:** "find which is the best" of the ideas in
`Documentation/SIGNAL_IDEAS_2026-09-26.md`. This section fixes the test and
the rule for "best" **before** anything is run. Script:
`research_evidence/signal_bakeoff_20260926/signal_bakeoff.py` (tests:
`tests/test_signal_bakeoff.py`, fake data only).

**Question:** which new stock-overlay signal best survives timing luck and
one-day-late fills?

**What stays fixed for every idea:** the incumbent's ETF core, regime switch,
overlay size per regime, 20-day holding, exit-rank floor, costs and cost
stress. Only the overlay's score, and the basket, change.

| Idea | Role | Score | Basket |
|---|---|---|---|
| A | candidate | average daily rank of `factor_mom_12_1` and `factor_resid_mom_sector_12_1` | top 10, equal weight, max 2 per sector |
| B | candidate | each year, features re-chosen from a fixed pool of 28 using only data whose one-day-late 20-day label ended before that year: late IC \|t\| ≥ 2, same sign as on-time IC, keeps ≥ half of it, max 8 | top 10, equal weight, max 2 per sector |
| C | candidate | 11 sector ETFs: average rank of 6-month and 12-1 month return; ETFs below their 200-day average can't be picked | top 3 ETFs, equal weight |
| R0 | control | incumbent, unchanged | top 3, sticky score |
| R1 | control | incumbent score | top 10, equal weight |
| E | control | no overlay (overlay gross 0 in every regime) | none |

**Idea D (earnings drift) is excluded before running:** `settings.py` has
`USE_EARNINGS_DATA = False`, so the local feature files hold placeholder
earnings values, not history. It can't be tested honestly.

**Runs:** each idea 40 times: calendar start offsets 0–19, each on time and
one day late. Same end date for every run (30 sessions before the data ends).

**Decision window:** 2013–2022 alpha vs QQQ. 2023–2026 is printed as a
diagnostic only and decides nothing.

**A candidate is eligible only if all are true:**

- G0: all 40 runs finished with data;
- G1: every one of the 40 runs beats QQQ over 2013–2022;
- G2: on-time spread (max − min over the 20 start days) is under 73 points;
- G3: mean delay cost (on-time minus late at the same start day) is at most 5 points;
- G4: median late alpha is above control E's (stock picks must add value).

**Best:** the eligible candidate with the highest median one-day-late alpha.
**Survivorship tie rule:** if C is eligible and its median late alpha is at
least half of the best eligible stock idea's (A or B), C is chosen instead,
because its history has no survivorship bias and A/B's does.

**If no candidate is eligible:** record "no winner"; the incumbent stays on
the paper advisory.

**What "best" does NOT mean:** it is not approval. The winner still has to go
through the unchanged gates (fixed-mode nested walk-forward to 2022,
`core_satellite_execution_stress.py --candidate-json`, survivorship audit),
and adding its score to the locked engine needs the owner's agreement first.

**How to run (project computer, which has `data/`):**

```bash
python refresh_etf_data.py --symbols XLK XLY XLF XLV XLE XLI XLP XLU XLRE XLB XLC --refresh
python research_evidence/signal_bakeoff_20260926/signal_bakeoff.py
```

It makes 240 engine runs (about 20 minutes on a cloud machine) and writes
`research_evidence/signal_bakeoff_20260926/signal_bakeoff.json`.

**Dry run (2026-09-26, fake data, no result):**
`python research_evidence/signal_bakeoff_20260926/dry_run_fake_data.py` copies
the project to a temporary folder, fills it with random prices, and runs the
bake-off with `--offsets 2`. It passed: all 24 engine runs finished and every
candidate was judged. It only proves the pipeline runs; random prices say
nothing about which idea is best. The real `data/` folder is never touched.
`--offsets 2` is a quick smoke test only; its output says
`valid_full_test: false` and must not be judged.

### Result: H-bakeoff (2026-09-26) — NO WINNER

Run on the project computer after refreshing the 11 sector ETFs, with the
engine as of `main` 4e4661f (includes the M3/M4 cost and daily-drawdown
changes). Full test: 20 start days × on time/late, 240 runs, data to
2026-08-12. Raw output: `research_evidence/signal_bakeoff_20260926/signal_bakeoff.json`.

Alpha vs QQQ in % points, added up over the decision window 2013–2022.
2023–2026 is a diagnostic only.

| Idea | Median late | Worst run | Start-day spread | Mean delay cost | 2023–26 median | Gates failed |
|---|---|---|---|---|---|---|
| A slow momentum | +70.7 | +14.4 | 168.1 | 3.1 | −44.9 (40 of 40 negative) | G2 |
| B yearly re-picked features | −63.4 | −119.3 | 104.7 | 1.8 | −80.3 | G1, G2 |
| C sector ETFs | −122.1 | −160.2 | 58.0 | 1.9 | −71.8 | G1 |
| R0 incumbent (control) | +567.4 | +298.5 | 655.3 | 7.2 | +26.3 (11 of 40 negative) | — |
| R1 incumbent score, top 10 (control) | +85.9 | +23.1 | 186.1 | 6.4 | −41.1 | — |
| E no stock overlay (control) | −146.1 | −177.1 | 44.6 | 0.1 | −75.3 | — |

**Decision (by the pre-registered rule):** no candidate passed every gate,
so there is no winner. The incumbent stays on the paper advisory.

**What the numbers say:**

- **Beating QQQ is a high bar for this design.** With no stock picks
  (control E), the core loses 146 points to QQQ over 2013–2022, because it is
  only partly invested (core gross 0.50–0.75) and holds SPY in weaker regimes.
  Any stock overlay has to earn all of that back first.
- **The incumbent's score is far stronger than every new idea** over
  2013–2022, and it is the only one whose median is still positive in
  2023–2026. But its start-day spread is huge (655 points), so its result
  depends heavily on timing luck.
- **Its edge sits in the top 3 names.** R1 uses the same score with 10 names:
  the spread drops to 186, but alpha falls to +86 and 2023–2026 turns
  negative.
- **Idea A came closest.** It failed only G2. Even so, it lost to QQQ in all
  40 runs over 2023–2026, so it would not be a useful replacement.

**Lesson for the next pre-registration (not a reason to re-judge this run):**
the G2 limit of 73 points was copied from the timing-luck study, which
measured 2023–2026 (about 3.6 years). This bake-off judges 10-year totals,
where spreads are naturally several times larger. A future spread limit
should be set relative to the window length, for example as a share of the
median alpha. It must be written down before that run.

**Open question for the owner:** none of the three new ideas beats the
incumbent. The bigger question is whether the incumbent's concentrated
top-3 edge is real or mostly timing luck plus survivorship: 24% of its
holdings were in stocks that joined the index after 2010 (see
`SURVIVORSHIP_WATCHLIST_CHECK_2026-09-26.md`).

## Hypothesis H-edge (pre-registered 2026-09-26, before any run)

**Owner request:** check whether the incumbent's edge is real before fixing
H1 (paper trading follows different rules from the backtest). Script:
`research_evidence/edge_check_20260926/edge_check.py`.

**Question:** is the incumbent's 2013–2022 edge over QQQ real, or mostly
(1) hindsight in the stock list and (2) luck? And does the live-only 8%
trailing stop help?

**What stays fixed:** the incumbent config, unchanged. The measure is the same
as H-bakeoff: alpha vs QQQ in % points, added up over the decision window
2013–2022. 2023–2026 is printed as a diagnostic only. Every run ends on the
same date (30 sessions before the data ends).

**Point-in-time list:** a stock may be picked on a date only if it (or its
known earlier ticker: META←FB, RTX←UTX, LIN←PX) was in the S&P 500 on that
date. Membership comes from the free community history (fja05680/sp500, MIT
licence, the same source as `SURVIVORSHIP_WATCHLIST_CHECK_2026-09-26.md`),
using the latest snapshot on or before the date. **Limit:** this removes
"picked before it joined the index" hindsight, but it cannot add companies
that later left the index (there is no price data for them). So it is a
partial survivorship test that still leans in the strategy's favour.

**Runs:**

| Set | What | Runs |
|---|---|---|
| R0 | incumbent, current list | 20 start days × on time/late = 40 |
| S | incumbent, point-in-time list | 40 |
| M | "random picks": S with its three score columns replaced by random numbers (seed 0–99), only where a real score exists, start day = seed mod 20, on time | 100 |
| T | S on-time runs with a simulated 8% trailing stop | 20 (no extra engine runs) |

**How T simulates the stop:** each stock trade is bought at the next day's
Open, as in the engine. The highest price since entry is tracked from daily
Highs. If a day's Low touches 92% of that high, the stock is sold at that
level, or at the Open if it gapped below. Its money then stays in cash until
the next 20-day date. Each stop exit is charged one extra stock trade at the
engine's calibrated cost. Drawdowns for T and its comparison are measured on
20-day period ends.

**Gates:**

- **S1, survivorship:** the median one-day-late alpha of S is above 0 **and**
  at least 50% of R0's median one-day-late alpha.
- **L1, luck:** the median on-time alpha of S is above the 95th percentile of
  the 100 M runs.
- **Edge verdict:** "edge shown" only if S1 and L1 both pass. Otherwise
  "edge not shown".
- **Stop verdict:** keep the 8% stop only if, over the 20 on-time S runs,
  (a) the median alpha with the stop is at least the median without it minus
  10% of the absolute value of the median without it, **and** (b) the median
  max drawdown with the stop is at least 1.0 point shallower. Otherwise the
  recommendation is to drop the stop.

**What the verdict means:** it is evidence for the owner's H1 decision, not
approval. No gate, threshold or live config changes because of it.
