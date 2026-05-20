"""
ci_check_feature_report.py — Sanity check + rollback for the feature
quality report after a CI rebuild.

PLAIN ENGLISH: The factor refresh on GitHub Actions sometimes runs
`research.py --incremental` against a partial panel (mid-rebuild on a
fresh runner, network hiccup, etc.).  When that happens, the follow-up
`feature_quality_diagnostic.py` grades only a handful of features
instead of the full 30-50.  Loading that report later collapses the
cluster gate (2 clusters @ 50% weight) and blocks trading.

This script inspects the freshly-rebuilt report and, if it's too small,
restores the git baseline (snapshotted by the YAML step as `.gitbase`
sidecars).  Exit code 0 either way — the run is OK as long as either
the new report or the baseline is in place.

Usage:
    python3 ci_check_feature_report.py --min-features 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPORT = Path("signals/feature_quality_report.json")
SUMMARY = Path("signals/feature_quality_summary.csv")
SHORTLIST = Path("logs/feature_ic_shortlist.csv")

# `.gitbase` snapshots are written by the YAML step before the rebuild.
REPORT_BASE = Path("signals/feature_quality_report.json.gitbase")
SUMMARY_BASE = Path("signals/feature_quality_summary.csv.gitbase")
SHORTLIST_BASE = Path("logs/feature_ic_shortlist.csv.gitbase")


def _read_n_features() -> int:
    """Return the number of graded features in the current report, or 0."""
    try:
        data = json.loads(REPORT.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        print(f"Could not read {REPORT}: {exc}")
        return 0
    return int(data.get("n_features", 0) or 0)


def _restore_from_baseline() -> bool:
    """Copy each .gitbase sidecar over its live counterpart.

    Returns True if the main report file was restored (the only one that
    matters for trading); False if no baseline exists.
    """
    restored = False
    if REPORT_BASE.exists():
        shutil.copy(REPORT_BASE, REPORT)
        restored = True
        print(f"  Restored {REPORT} from .gitbase")
    if SUMMARY_BASE.exists():
        shutil.copy(SUMMARY_BASE, SUMMARY)
        print(f"  Restored {SUMMARY} from .gitbase")
    if SHORTLIST_BASE.exists():
        shutil.copy(SHORTLIST_BASE, SHORTLIST)
        print(f"  Restored {SHORTLIST} from .gitbase")
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-features",
        type=int,
        default=20,
        help="Minimum graded features required to accept the rebuilt report.",
    )
    args = parser.parse_args()

    n = _read_n_features()
    if n >= args.min_features:
        print(f"✓ Rebuilt feature quality report OK ({n} graded features).")
        return 0

    print(
        f"⚠ Rebuilt feature quality report has only {n} graded features "
        f"(min {args.min_features}). Restoring git baseline."
    )
    if _restore_from_baseline():
        # Verify the restored report is healthy.
        n_restored = _read_n_features()
        print(f"  Baseline has {n_restored} features.")
        if n_restored < args.min_features:
            print(
                f"  WARNING: even baseline has only {n_restored} features — "
                "the cluster gate will likely still fail.  Investigate before "
                "trading."
            )
            # Exit 0 anyway so the workflow proceeds and the runtime guard
            # in core_satellite_alpha.py emits the canonical error.
        return 0

    print(
        "✗ No git baseline available to restore.  daily_run.py will fail at "
        "the feature health gate; investigate `feature_quality_diagnostic.py` "
        "and the underlying panel state on this runner."
    )
    # Still exit 0 — let the runtime guard surface the canonical error.
    return 0


if __name__ == "__main__":
    sys.exit(main())
