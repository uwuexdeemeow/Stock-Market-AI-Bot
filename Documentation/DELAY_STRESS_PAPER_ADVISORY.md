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
