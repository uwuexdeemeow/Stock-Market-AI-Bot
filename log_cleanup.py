"""log_cleanup.py — Archive old logs and cap disk usage.

PLAIN ENGLISH: Over time, the logs/ and signals/ directories accumulate
dated JSON/CSV files (one per day).  Left unchecked, they'll fill your disk.
This script removes files older than a retention period and warns if total
disk usage exceeds a threshold.

HOW TO RUN:
    python3 log_cleanup.py                    # preview what would be cleaned
    python3 log_cleanup.py --execute          # actually delete old files
    python3 log_cleanup.py --retention 14     # keep only 14 days (default: 30)
    python3 log_cleanup.py --check-disk       # just check disk usage, no cleanup

WHEN TO RUN:
    Weekly via cron, or add to daily_run.py as a non-critical step.

KEY CONCEPTS:
  - Retention: how many days to keep files.  Older files get deleted.
  - Dry run: shows what WOULD happen without actually deleting anything.
    Always dry-run first to make sure you won't lose something important.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from settings import LOG_DIR, SIGNAL_DIR

LOGS = Path(LOG_DIR)
SIGNALS = Path(SIGNAL_DIR)

# Default retention period in days
DEFAULT_RETENTION_DAYS = 30

# Disk usage warning threshold (fraction of total disk)
DISK_WARN_THRESHOLD = 0.80

# File patterns that are safe to age-out (dated files only)
# PLAIN ENGLISH: We only delete files that have a date in their name
# (like daily_run_20260101.json or paper_health_20260101.json).
# Files without dates (like paper_state.json) are never touched.
DATED_PATTERN_PREFIXES = [
    "daily_run_",
    "paper_health_",
    "alpaca_paper_gauntlet_",
    "daily_paper_check_",
    "moomoo_paper_gauntlet_",
]

# Signal files that accumulate and can be cleaned
SIGNAL_DATED_PREFIXES = [
    "paper_health_",  # dated health snapshots
]


def check_disk_usage() -> dict:
    """Check disk usage and return status.

    PLAIN ENGLISH: Uses the operating system to check how full your disk is.
    Returns the percentage used and whether it's above the warning threshold.
    """
    total, used, free = shutil.disk_usage("/")
    pct_used = used / total
    return {
        "total_gb": round(total / (1024**3), 1),
        "used_gb": round(used / (1024**3), 1),
        "free_gb": round(free / (1024**3), 1),
        "pct_used": round(pct_used * 100, 1),
        "warning": pct_used > DISK_WARN_THRESHOLD,
    }


def find_old_files(directory: Path, prefixes: list[str], retention_days: int) -> list[Path]:
    """Find files older than retention_days that match our dated prefixes.

    PLAIN ENGLISH: Scans the directory for files that start with one of our
    known prefixes AND are older than the retention period.  Returns a list
    of file paths that are candidates for deletion.
    """
    if not directory.exists():
        return []

    now = time.time()
    threshold = retention_days * 86400
    old_files = []

    for f in directory.iterdir():
        if not f.is_file():
            continue
        # Only consider files matching our safe-to-delete prefixes
        if not any(f.name.startswith(prefix) for prefix in prefixes):
            continue
        # Check age
        age = now - f.stat().st_mtime
        if age > threshold:
            old_files.append(f)

    return sorted(old_files)


def cleanup(*, retention_days: int = DEFAULT_RETENTION_DAYS, execute: bool = False) -> dict:
    """Find and optionally delete old log files.

    PLAIN ENGLISH: Scans logs/ and signals/ for dated files older than
    retention_days.  If execute=True, actually deletes them.  If False,
    just reports what it would delete (dry run).
    """
    log_files = find_old_files(LOGS, DATED_PATTERN_PREFIXES, retention_days)
    signal_files = find_old_files(SIGNALS, SIGNAL_DATED_PREFIXES, retention_days)
    all_files = log_files + signal_files

    total_size = sum(f.stat().st_size for f in all_files)
    deleted = 0

    if execute and all_files:
        for f in all_files:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass

    disk = check_disk_usage()

    # Alert if disk is getting full
    if disk["warning"]:
        try:
            from notifications import send_alert
            send_alert(
                f"Disk usage at {disk['pct_used']}% ({disk['free_gb']}GB free). "
                f"Consider cleaning up data/ directory.",
                title="Disk Usage",
                priority="warning",
            )
        except Exception:
            pass

    return {
        "retention_days": retention_days,
        "candidates": len(all_files),
        "deleted": deleted,
        "freed_mb": round(total_size / (1024**2), 1) if execute else 0,
        "would_free_mb": round(total_size / (1024**2), 1),
        "disk": disk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up old log files and check disk usage."
    )
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete files (default: dry run)")
    parser.add_argument("--retention", type=int, default=DEFAULT_RETENTION_DAYS,
                        help=f"Days to keep files (default: {DEFAULT_RETENTION_DAYS})")
    parser.add_argument("--check-disk", action="store_true",
                        help="Only check disk usage, no cleanup")
    args = parser.parse_args()

    if args.check_disk:
        disk = check_disk_usage()
        print(f"Disk Usage: {disk['pct_used']}% used "
              f"({disk['used_gb']}GB / {disk['total_gb']}GB, "
              f"{disk['free_gb']}GB free)")
        if disk["warning"]:
            print(f"  ⚠ WARNING: Disk usage above {DISK_WARN_THRESHOLD*100:.0f}% threshold!")
        else:
            print(f"  ✓ Disk usage healthy")
        return

    result = cleanup(retention_days=args.retention, execute=args.execute)

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"Log Cleanup ({mode})")
    print("─" * 40)
    print(f"  Retention: {result['retention_days']} days")
    print(f"  Candidates for deletion: {result['candidates']}")
    if args.execute:
        print(f"  Deleted: {result['deleted']} files ({result['freed_mb']:.1f} MB)")
    else:
        print(f"  Would free: {result['would_free_mb']:.1f} MB")
        if result['candidates'] > 0:
            print(f"\n  Run with --execute to actually delete these files.")

    disk = result["disk"]
    print(f"\n  Disk: {disk['pct_used']}% used ({disk['free_gb']}GB free)")
    if disk["warning"]:
        print(f"  ⚠ Disk usage above threshold!")


if __name__ == "__main__":
    main()
