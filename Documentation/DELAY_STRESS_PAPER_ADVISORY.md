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

### Add-on H-edge-core-stop (pre-registered 2026-09-26, before any run)

**Why:** while planning the H1 fix, it turned out the live account also puts
**5% trailing stops on the core ETFs** (SPY, QQQ; TQQQ 10%), set by
`GUARD_CORE_STOP` in `alpaca_protection.py`. H-edge above only tests the 8%
stock stop. The core is most of the portfolio (gross 0.50–0.75), so the H1
design needs the same evidence for the core stop.

**Test (script `research_evidence/edge_check_20260926/edge_core_stop.py`):**
the same 20 on-time S runs (point-in-time list). The core ETF holding of each
20-day period is replayed on daily bars with the live trail (5% SPY/QQQ, 10%
TQQQ). The engine buys core ETFs at the Close of the entry day, so the
replay starts from that Close and checks the stop from the next day on, with
the same gap and high-water rules as the stock stop. Stopped ETF money stays
in cash until the next 20-day date. Each stop exit pays one extra ETF trade
at the engine's ETF cost. Stock stops are **off** in this test, so the core
stop is judged on its own.

**Rule (same as the stock stop):** keep the core stop only if the median
2013–2022 alpha with it is at least the median without it minus 10% of that
median's absolute value, **and** the median period max drawdown is at least
1.0 point shallower. Otherwise the recommendation is to drop the core stop.

### Result: H-edge (2026-09-26) — EDGE SHOWN; DROP THE 8% STOCK STOP

Full test (20 start days, 100 random-pick runs), data to 2026-08-12. Raw
output: `research_evidence/edge_check_20260926/edge_check.json` (membership
source SHA-256 recorded there). The point-in-time list removed 4.4% of the
panel rows.

Alpha vs QQQ in % points, added up over 2013–2022:

| Set | Runs | Worst | Median | Best |
|---|---|---|---|---|
| R0 incumbent, current list, on time | 20 | +298 | +570 | +954 |
| R0 incumbent, current list, one day late | 20 | +335 | +567 | +897 |
| S point-in-time list, on time | 20 | +279 | +468 | +671 |
| S point-in-time list, one day late | 20 | +343 | +445 | +642 |
| M random picks (point-in-time list, on time) | 100 | −220 | −76 | +163 |
| T = S on time + simulated 8% stock stop | 20 | −53 | +9 | +159 |

**Gates:**

- **S1 survivorship: PASS.** S keeps 78% of R0's median late alpha
  (+445 vs +567; at least 50% needed).
- **L1 luck: PASS.** S's median (+468) beats all 100 random-pick runs
  (95th percentile +26). S's *worst* run beats the *best* random run.
- **Stock stop: DROP.** The stop makes drawdowns shallower (median −21.6% →
  −17.6%) but cuts alpha from +468 to +9. It fired on about 49% of stock
  positions.

**Edge verdict: edge shown.** 2023–2026 diagnostic medians: incumbent +25,
random picks −55, with the stock stop −31.

**Caveats (recorded, not re-judged):**

- **The survivorship test is partial.** It can't add companies that left the
  index, because there is no price data for them, so the true edge is lower
  than S shows.
- **Random picks pay more costs.** New random scores every period mean about
  2.7× the incumbent's turnover. That costs tens of points over 10 years,
  far smaller than the gap measured here.
- **Timing luck is still large.** S's on-time runs range from +279 to +671
  depending on the start day.

### Result: H-edge-core-stop (2026-09-26) — DROP THE CORE ETF STOPS

Raw output: `research_evidence/edge_check_20260926/edge_core_stop.json`.
Over the 20 on-time S runs, the 5% SPY/QQQ stops cut the median alpha from
+468 to +198 (the rule allowed at most a 10% giveback). They made the median
drawdown shallower (−21.6% → −16.8%). They fired in about 44% of 20-day core
holding periods. 2023–2026 diagnostic: +25 without them, −8 with them.

**What this means for H1:** the tested strategy is the 20-day calendar with
**no** trailing stops, on the stocks or on the ETFs. Both stops cost far more
return than the drawdown they save. The emergency brakes (the −12% drawdown
halt and the −8% one-day halt) are different: they rarely fire, and they
stay.

## Hypothesis H-full-universe (pre-registered 2026-09-29, before any data download or run)

**Owner request:** close the survivorship hole left open by H-edge. H-edge's
point-in-time test (S) only removed "picked before it joined the index"
hindsight. It could not add the roughly 289 companies that were in the
S&P 500 at some point in 2013–2022 but are no longer in it (count from the
fja05680/sp500 history, some of them plain ticker changes). None of them can
be picked today, because there are no price files for them, and the 62-name
`WATCHLIST` was chosen in hindsight from companies that stayed big. Script
(to be written): `research_evidence/full_universe_20260929/full_universe_check.py`.

**Question:** does the incumbent keep its 2013–2022 edge over QQQ when it
picks from a rule-based, point-in-time list of large S&P 500 companies that
includes the ones that later left the index?

**What stays fixed:** the incumbent config, the engine, the ETF core and
regime switch, costs, the feature shortlist (`logs/feature_ic_shortlist.csv`
as committed today) and the membership source used in H-edge. The measure is
the same as H-edge: alpha vs QQQ in % points, added up over 2013–2022.
2023–2026 is printed as a diagnostic only. Every run ends on the same date
(30 sessions before the data ends).

### Step 0 — data coverage gate (judged before any engine run)

The price source for delisted stocks is **not chosen yet**. Yahoo and Stooq
both returned nothing on 2026-09-29, even for AAPL, so coverage could not be
checked in advance. The owner chooses the source. The script records its
name, the download date and a SHA-256 of every price file. Prices must be
split-adjusted daily OHLCV. Paid sources are fine if the owner agrees.
Downloads go to a separate folder (`research_evidence/full_universe_20260929/data/`,
not in Git). The real `data/` folder is never written.

**Pool:** every ticker that was an S&P 500 member on any snapshot date from
2012-10-01 to 2022-12-31 (the extra quarter gives the first universe
ranking its trailing data). Known ticker changes are merged into one company
using the predecessor map; new pairs found during download are added to the
map and listed in the output.

**Coverage gate (C1):** at least **85%** of the pool companies that are no
longer members must have prices for at least **90%** of the NYSE sessions
they were members during 2013–2022. The same must hold for at least 95% of
current members. If C1 fails, stop: the result is recorded as "not testable
with this source", and no engine run is made or looked at.

**Sector:** each company gets one of the 11 sector labels from its SEC SIC
code (EDGAR, which keeps filings of delisted companies), using a fixed
SIC→sector table written into the script before any download. Current
`SECTOR_MAP` entries are **not** used for any name, so old and new names
are treated the same way. No SIC code → sector `OTHER`.

**Features:** built by the project's own feature code over the whole pool
in the separate folder, including the cross-sectional (`xs_rank_*`) ranks,
which are recomputed across that pool. The fixed feature list is not
re-chosen.

### Universe rule (U)

On the first NYSE session of each month, the eligible list is the **62**
point-in-time S&P 500 members (the same size as `WATCHLIST`) with the
highest median daily dollar volume (Close × Volume) over the previous 63
sessions, using data up to the day before. A stock needs 63 sessions of
history to be ranked. The list gates new picks only: rows of names off the
list are removed, the same way test S removed non-member rows.

**Delisting inside a hold:** if a held stock's prices end before its 20-day
exit, it is sold at its last Close. A diagnostic row also re-runs this with
a −30% return applied at that last Close (a standard rough allowance for
delistings caused by failure). The diagnostic row decides nothing.

### Runs

| Set | What | Runs |
|---|---|---|
| R0 | incumbent, current list and data (reference) | 20 start days × on time/late = 40 |
| U | incumbent on the rule-based full universe | 40 |
| MU | random picks in U: three score columns replaced by random numbers (seed 0–99), start day = seed mod 20, on time | 100 |
| UD | U on-time runs with the −30% delisting allowance (diagnostic) | 20 |

### Gates

- **C1 (data):** as above. Fail → stop, "not testable with this source".
- **U1 (survivorship):** the median one-day-late alpha of U is above 0
  **and** at least **50%** of R0's median one-day-late alpha.
- **U2 (luck):** the median on-time alpha of U is above the 95th percentile
  of the 100 MU runs.

**Verdicts:**

- U1 and U2 pass → **"edge survives the full survivorship test"**.
- U2 passes but U1 fails → **"edge real but mostly list hindsight"**: the
  signal beats random picks, but most of the backtest's size comes from the
  hand-picked list. Backtest alpha should then not be used to set
  expectations.
- U2 fails → **"edge not shown"** on an unbiased universe.

**Recorded but not judged:** the start-day spread (the H-bakeoff lesson says
10-year spreads are large, and this is an edge test, not a candidate
selection), the mean delay cost, the 2023–2026 medians, the UD row, the
number of delisting exits, and the share of U's picks that are later-removed
companies.

**Limits (written down now):**

- The feature shortlist and the engine settings were chosen while looking at
  the 62-name list's history. A U failure may partly mean "tuned to that
  list", not only survivorship. Both mean the same thing for expectations.
- The SIC→sector map is rougher than GICS, so sector ranks and sector caps
  differ a little from live.
- A free source may miss some delisted names. C1 limits this but cannot
  rule out that the missing names are the worst ones.

**What the verdict means:** it is evidence, not approval. No gate, threshold,
universe, feature list or live config changes because of it. Switching the
live universe to the U rule would need its own pre-registered hypothesis and
the owner's agreement, and it would touch locked files.

**How to run (project computer, after the owner picks the source):**

```bash
python research_evidence/full_universe_20260929/full_universe_check.py --step coverage
# only if C1 passes:
python research_evidence/full_universe_20260929/full_universe_check.py --step runs
```

Output: `research_evidence/full_universe_20260929/full_universe_check.json`.
A smoke test (`--offsets 2 --monkeys 4`) is marked `valid_full_test: false`
and must not be judged.

### Implementation notes for H-full-universe (2026-09-29, before the coverage step)

Written when the script was built, before any coverage number or backtest
result was seen. They fill in details the rules above leave open; none of
them changes a gate.

- **Source chosen by the owner:** Tiingo end-of-day API (adjusted OHLCV for
  features and returns; raw Close × raw Volume for the dollar-volume
  ranking). A spot check of Tiingo's public ticker list found 227 of the 297
  later-removed tickers by plain ticker and date, before any renames.
- **Rename pairs** are listed in `RENAMES` in the script (35 hand-checked
  pairs such as BK→BNY, FB→META, UTX→RTX, CBS/VIAC→PARA). Acquired or
  bankrupt companies are **not** mapped to their buyer.
- **"Traded recently":** besides 63 sessions of history, a stock must have
  traded in the last 5 sessions to be ranked in a month.
- **Scores for U** are ranked within that month's 62-name list. The live
  signal ranks within its own list the same way. The ML score is empty for U
  (the incumbent's regime score doesn't use it); a run fails if it picks by
  a column outside the three regime scores.
- **Features** that need data beyond prices and SPY/QQQ/sector ETFs (VIX,
  macro, news, valuation) are neutral for U. The incumbent's 42 score
  features don't use them.
- **Sector lookup:** SEC ticker list for companies still trading. Otherwise,
  the company name from Tiingo is searched on SEC EDGAR, and the best name
  match is taken. Each match is saved for review.

Code: `research_evidence/full_universe_20260929/full_universe_check.py`,
tests: `tests/test_full_universe_check.py`, doc:
`Documentation/doc_full_universe_check.md`.

## Hypothesis H-top-names (pre-registered 2026-09-29, before any run)

**Owner request:** find out whether the incumbent's edge is spread across
many picks or rests on a few lucky names or a few big winners. NVDA alone
was 20% of overlay profit in the walk-forward report. Script (to be
written): `research_evidence/top_names_20260929/top_names_check.py`.

**Base:** the point-in-time list from H-edge (test S: a stock can be picked
only while it was in the S&P 500), the incumbent config unchanged, the same
measure (alpha vs QQQ in % points added up over 2013–2022; 2023–2026 printed
only), the same end date rule (30 sessions before the data ends), and the
same 20 start days × on time/one day late. It runs after H-full-universe
finishes and does not depend on its verdict.

**Tests:**

| Set | What | Runs |
|---|---|---|
| S | base, unchanged (reference) | 40 |
| N1 | S with the single biggest-contributing ticker removed from the list | 40 |
| N3 | S with the 3 biggest-contributing tickers removed from the list | 40 |
| P1 | S on-time runs, with each 20-day period's best-returning pick replaced by a neutral stand-in | 20 (no extra engine runs) |
| MP | 100 random-pick runs (as H-edge test M: seeds 0–99, start day = seed mod 20, on time), with the same P1 replacement | 100 |

- **Biggest contributors** are chosen once, from the 20 on-time S runs: for
  each ticker, add up weight × 20-day holding return over every period it
  was held in 2013–2022, average across the 20 runs, and rank. The chosen
  tickers are written to the output before N1/N3 run. N1 and N3 remove those
  tickers' rows from the panel, so the engine picks the next best stocks.
- **Holding return** is the panel's on-time 20-day label for that stock on
  the period's decision date (buy at the next Open, sell at the Close 20
  sessions later).
- **P1 stand-in:** the best pick's return in each period is swapped for the
  median 20-day label of all stocks that had a score on that decision date.
  Its weight stays the same, and the period return changes by weight ×
  (median − best). The equity curve is rebuilt from the adjusted period
  returns.

**Gates:**

- **T1 (names):** the median one-day-late alpha of N3 is above 0 **and** at
  least **50%** of S's median one-day-late alpha.
- **T2 (big winners):** the median alpha of P1 is above the **95th
  percentile** of the 100 MP runs. The same removal is applied to random
  picks, so this checks skill after the harsh cut instead of expecting a
  positive number that any strategy would struggle to reach.

**Verdicts:**

- T1 and T2 pass → **"edge is broad"**.
- T1 fails → **"edge rests on a few names"** (named in the output).
- T2 fails → **"edge rests on each period's one big winner"**.
- Both fail → both labels.

**Recorded, not judged:** N1 numbers, the chosen tickers and their share of
S's overlay profit, spread across start days, the delay cost, and 2023–2026.

**Limit (written now):** removing the top names after seeing S's results is
deliberately a hindsight test. It asks "what if those names had not
existed", not "could an investor have known". It is evidence about how
concentrated the edge is, not a tradable rule.

**What the verdict means:** evidence only. No gate, config or stock list
changes because of it.

## Hypothesis H-benchmark (pre-registered 2026-09-29, before any run)

**Owner request:** judge the incumbent against a fairer yardstick. The core
is only partly invested (gross 0.50–0.75, with SPY or cash in weaker
regimes), so "alpha vs QQQ" mixes stock-picking skill with how much money
is in the market. With no stock picks at all, the core loses about 146
points to QQQ over 2013–2022 (H-bakeoff control E). Script (to be written):
`research_evidence/benchmark_20260929/benchmark_check.py`.

**Base:** the same as H-top-names: test S, incumbent config unchanged, 20
start days × on time/one day late, decision window 2013–2022, 2023–2026 printed
only, and the same end-date rule. The only extra engine runs are 20 on-time
runs of control E (overlay gross 0, as in H-bakeoff), which are diagnostic
only.

**Two fair yardsticks:**

1. **Exposure-matched benchmark (B-exp):** for each 20-day period, the
   benchmark earns g × QQQ return + (1 − g) × cash return, where g is the
   strategy's actual gross exposure that period (`gross_exposure` in the
   trades output) and cash is BIL's return (0 before BIL data starts). If
   g > 1, the extra is borrowed at the cash return. The benchmark pays no
   trading costs. Alpha = strategy minus B-exp, in % points, added up over
   2013–2022 like every other test.
2. **Beta-adjusted alpha (B-beta):** regress the strategy's daily returns on
   QQQ's daily returns over 2013–2022 (both minus cash). The intercept,
   annualised, is the alpha. Its significance is the Newey–West t-stat from
   `backtest._newey_west_tstat` with the helper's default lag. The
   information ratio is the annualised intercept divided by the annualised
   residual volatility.

**Gates:**

- **F1:** the median one-day-late B-exp alpha is above **0**.
- **F2:** the median one-day-late B-beta Newey–West t-stat is at least
  **2.0**.

**Verdicts:**

- F1 and F2 pass → **"edge holds against a fair benchmark"**.
- F1 passes, F2 fails → **"beats a fair benchmark, but not reliably"**.
- F1 fails → **"headline alpha is mostly market exposure and timing of the
  core, not stock-picking skill"**.

**Recorded, not judged:** the median beta, information ratio, annualised
alpha, the on-time numbers, spread across start days, and 2023–2026. Control E
(no overlay) is also measured against B-exp, to show how much the core
timing alone earns on this yardstick.

**What the verdict means:** evidence only. This does not replace "alpha vs
QQQ" in any existing gate or report. Changing an official measure would be
its own decision for the owner.
