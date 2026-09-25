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

```bash
python run_walkforward_batched.py --help   # memory-safe driver
python core_satellite_nested_walkforward.py --low-turnover-grid --end-year 2022 \
    --output-prefix wf_delay_robust_lowturnover_20260926
```

A stronger follow-up: add a delay-aware inner selection option, so that
walk-forward scores each candidate on its one-day-late results and
delay-robustness becomes part of how a candidate gets picked, not only a
final check.
