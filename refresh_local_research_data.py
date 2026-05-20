"""
refresh_local_research_data.py — bring the local research data up to date.

PLAIN ENGLISH:
The GitHub Actions workflows (`factor_data_refresh.yml` and
`daily_paper_trading.yml`) keep the live trading bot fresh.  But those
workflows use GitHub's internal `actions/cache` for the heavy data
files (the per-ticker parquets, the feature quality report, the
factor-decay state), and they do NOT commit any of that back to git.

The consequence: your laptop's copy of `data/*.parquet` and the
per-feature research outputs get stale fast.  If you try to run
anything LOCALLY that depends on fresh research data — the nested
walkforward, a training pass, an ad-hoc backtest — it would silently
use whatever data was on disk the last time you refreshed.

This script is the local equivalent of the factor_data_refresh
workflow.  It runs the four refresh steps in the right order, halts
on any critical failure, and tells you whether the data is now
healthy enough for further research:

  1. `research.py --incremental`           — pull new OHLC bars + factor
                                              features for the watchlist
  2. `feature_quality_diagnostic.py`       — re-grade per-feature live IC
  3. `feature_research.py --top 24`        — refresh the quarantine IC
                                              CSV used by feature_health
  4. `factor_data_health.py --strict`      — final pass/fail gate

Each step runs in a fresh subprocess so any memory leak in one script
doesn't carry into the next.  Output streams to your terminal exactly
as if you ran the scripts by hand.

Usage:
    # Refresh everything (default — what you want before research)
    python refresh_local_research_data.py

    # Quarterly mode (adds the slower feature_research --pairs sweep)
    python refresh_local_research_data.py --pairs

    # Skip the slow feature_research step (data + quality only)
    python refresh_local_research_data.py --skip-feature-research

    # Skip the panel refresh entirely (already done today)
    python refresh_local_research_data.py --skip-research

    # Show what would run without executing
    python refresh_local_research_data.py --dry-run

Exit codes:
    0  every requested step succeeded
    1  a critical step failed; later steps were skipped
    2  argparse / configuration error
"""
from __future__ import annotations

# `subprocess` lets us spawn each refresh script as its own process.
# Running in a separate process resets the Python heap between steps,
# which protects later steps from any memory leak in earlier ones.
import subprocess
# `sys` gives us the Python interpreter path (sys.executable) so we
# launch the same Python that's running THIS script.  That avoids
# accidentally calling system Python when we meant a conda/venv one.
import sys
# `argparse` parses our own CLI flags.
import argparse
# `time` is just for elapsed-seconds reporting.
import time
# `pathlib.Path` for cleaner file existence checks at the end.
from pathlib import Path

from safe_io import run_utf8


# Each Step is just a name + the argv to run + whether a failure here
# should abort the whole script.  Keeping these as plain tuples keeps
# the script easy to scan and skip-flag.
class Step:
    """One refresh step.

    PLAIN ENGLISH: A small bundle of `what to call` and `what to do if
    it fails`.  We instantiate one per script we want to run, then
    iterate them in order.
    """
    def __init__(self, name: str, argv: list[str], description: str,
                 critical: bool = True, timeout_seconds: int | None = None):
        self.name = name
        self.argv = argv
        self.description = description
        # If a critical step fails we stop the whole refresh.  Non-
        # critical steps just print a warning and let the script
        # continue (e.g. feature_research's --top 24 is nice-to-have
        # but the panel refresh is not blocked on it).
        self.critical = critical
        # Per-step timeout in seconds.  None = no timeout (use only
        # for steps you trust to terminate on their own).
        self.timeout_seconds = timeout_seconds


def _run_step(step: Step, dry_run: bool) -> tuple[bool, float]:
    """Spawn one refresh step.

    Returns (success_bool, elapsed_seconds).  In dry-run mode we just
    print what would run and return success without actually invoking.
    """
    print(f"\n{'='*72}")
    print(f"  [{step.name}] {step.description}")
    print(f"  cmd: {' '.join(step.argv)}")
    print(f"{'='*72}")

    if dry_run:
        print(f"  (dry-run — not executing)")
        return True, 0.0

    started_at = time.time()
    try:
        # check=False so we can inspect the return code ourselves —
        # critical-vs-non-critical handling lives in the caller.
        result = run_utf8(
            step.argv,
            check=False,
            timeout=step.timeout_seconds,
        )
        elapsed = time.time() - started_at
        if result.returncode == 0:
            print(f"\n  ✓ {step.name} completed in {elapsed:.0f}s")
            return True, elapsed
        print(f"\n  ✗ {step.name} exited with code {result.returncode} "
              f"after {elapsed:.0f}s")
        return False, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started_at
        print(f"\n  ✗ {step.name} timed out after {elapsed:.0f}s "
              f"(limit was {step.timeout_seconds}s)")
        return False, elapsed
    except FileNotFoundError as e:
        elapsed = time.time() - started_at
        print(f"\n  ✗ {step.name} could not find script: {e}")
        return False, elapsed


def _verify_outputs(verbose: bool = True) -> bool:
    """Sanity-check that key research artifacts exist.

    Returns True if every expected file is present.  This is a soft
    check — missing files mean further research won't have what it
    needs, but the refresh script doesn't try to retroactively fix
    them.  The user re-runs with the right flag instead.
    """
    expected_files = [
        ("data/QQQ.parquet", "benchmark price data"),
        ("data/SPY.parquet", "benchmark price data"),
        ("signals/feature_quality_report.json", "feature quality grading"),
        ("signals/feature_quality_summary.csv", "feature quality grading"),
        ("signals/feature_research_summary.csv", "IC decay quarantine data"),
        ("signals/feature_health_profile.json", "feature health profile"),
        ("signals/factor_data_health.json", "panel freshness report"),
    ]
    missing = []
    for path_str, purpose in expected_files:
        path = Path(path_str)
        if not path.exists():
            missing.append((path_str, purpose))
    if not verbose:
        return not missing
    if missing:
        print("\n  Missing artifacts (downstream research will degrade):")
        for path_str, purpose in missing:
            print(f"    - {path_str}  ({purpose})")
    else:
        print("\n  All expected research artifacts are present.")
    return not missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh local research data the way factor_data_refresh.yml does on CI.",
    )
    # --skip-research: useful when you've already pulled fresh parquets
    # via some other path (e.g. you copied them from another machine)
    # and just want to rebuild the feature reports.
    parser.add_argument(
        "--skip-research", action="store_true",
        help="Skip the research.py panel refresh.  Use when data/*.parquet is already up to date.",
    )
    # --skip-feature-research: feature_research.py is the slowest step
    # (~5-10 min) and you only need to re-run it quarterly.  Skip on
    # routine refreshes between quarters.
    parser.add_argument(
        "--skip-feature-research", action="store_true",
        help="Skip feature_research.py (the IC decay CSV is good for ~3 months).",
    )
    # --pairs: turn on the slower pairwise interaction analysis in
    # feature_research.py.  Default is --skip-pairs since the pairs
    # data isn't consumed by anything in the current pipeline.
    parser.add_argument(
        "--pairs", action="store_true",
        help="Include the pairwise interaction analysis in feature_research.py (much slower).",
    )
    # --top N for feature_research.  24 is the default that matches the
    # CI runbook; raise it if you want broader coverage.
    parser.add_argument(
        "--top", type=int, default=24,
        help="Number of features to analyse in feature_research.py (default 24).",
    )
    # --dry-run: print the plan without actually running anything.
    # Handy when wiring this into a larger orchestration script.
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print each step's command without executing it.",
    )
    args = parser.parse_args()

    # Build the per-step argv lists.  Each script is launched via the
    # same Python interpreter that's running this wrapper.
    research_argv = [sys.executable, "research.py", "--incremental"]
    fqd_argv = [sys.executable, "feature_quality_diagnostic.py", "--top", "48"]
    fr_argv = [sys.executable, "feature_research.py", "--top", str(args.top)]
    if not args.pairs:
        fr_argv.append("--skip-pairs")
    fdh_argv = [sys.executable, "factor_data_health.py", "--strict"]

    # Step list — order matters.  Each later step reads outputs the
    # earlier ones write.
    steps: list[Step] = []
    if not args.skip_research:
        steps.append(Step(
            "research", research_argv,
            "Incremental panel + factor refresh (downloads new bars only)",
            critical=True,
            timeout_seconds=7200,  # 2h cap for full first-time refresh
        ))
    else:
        print("[plan] --skip-research: panel refresh skipped on user request.")
    steps.append(Step(
        "feature_quality", fqd_argv,
        "Re-grade per-feature live IC against the refreshed panel",
        critical=True,
        timeout_seconds=900,  # 15 min cap
    ))
    if not args.skip_feature_research:
        steps.append(Step(
            "feature_research", fr_argv,
            "Refresh IC-decay summary that drives feature_health quarantine",
            critical=False,  # quarantine still works on prior CSV if this fails
            timeout_seconds=1800,  # 30 min cap (60+ if --pairs)
        ))
    else:
        print("[plan] --skip-feature-research: IC decay CSV not refreshed this run.")
    steps.append(Step(
        "factor_data_health", fdh_argv,
        "Strict pass/fail gate — confirms the refreshed panel is healthy",
        critical=True,
        timeout_seconds=300,
    ))

    print(f"\n[plan] {len(steps)} step(s) to run (dry_run={args.dry_run})")
    for s in steps:
        flag = "critical" if s.critical else "best-effort"
        print(f"  - {s.name} [{flag}]")

    overall_started = time.time()
    fail_critical = False
    summaries: list[tuple[str, bool, float]] = []
    for step in steps:
        if fail_critical:
            print(f"\n[skip] {step.name}: a prior critical step failed.")
            summaries.append((step.name, False, 0.0))
            continue
        ok, elapsed = _run_step(step, args.dry_run)
        summaries.append((step.name, ok, elapsed))
        if not ok and step.critical:
            fail_critical = True
            print(f"\n[abort] {step.name} is critical — stopping further steps.")

    overall_elapsed = time.time() - overall_started
    print(f"\n{'='*72}")
    print(f"  Refresh summary  (total {overall_elapsed:.0f}s)")
    print(f"{'='*72}")
    for name, ok, elapsed in summaries:
        status = "✓" if ok else "✗"
        print(f"  {status} {name:<22s} {elapsed:>7.0f}s")

    # Even on full success, verify the downstream files actually got
    # written.  A non-zero exit from research.py could leave a half-
    # built panel in place and we want the user to notice.
    if not args.dry_run:
        artifacts_ok = _verify_outputs(verbose=True)
    else:
        artifacts_ok = True

    if fail_critical:
        return 1
    if not artifacts_ok:
        print("\n  Note: some artifacts are missing — review the step logs above.")
        return 1
    print("\n  ✓ local research data is ready for further work "
          "(walkforward, training, ad-hoc backtests).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
