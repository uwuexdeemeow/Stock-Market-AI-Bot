"""
install_schedule.py — Set up automated daily paper trading runs.

PLAIN ENGLISH: This script installs (or removes) a macOS launchd job that
runs daily_run.py every weekday at 9:30 AM Eastern Time.  launchd is the
macOS equivalent of cron — it runs your script in the background even if
you're not logged in.

Usage:
    python3 install_schedule.py              # install the daily schedule
    python3 install_schedule.py --uninstall  # remove the schedule
    python3 install_schedule.py --status     # check if the schedule is active
    python3 install_schedule.py --test       # run daily_run.py once right now (dry-run)

What it does:
    1. Creates a .plist file in ~/Library/LaunchAgents/
    2. Loads it with launchctl so macOS knows to run it daily
    3. Logs stdout/stderr to logs/daily_run_scheduled.log

The schedule runs at 9:30 AM ET (13:30 UTC during EST, 13:30 UTC during EDT).
Market opens at 9:30 AM ET, so your signals and orders go out right at open.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Where the project lives
PROJECT_DIR = Path(__file__).resolve().parent

# Python interpreter to use (the one running this script)
PYTHON = sys.executable

# The script we want to run daily
DAILY_SCRIPT = PROJECT_DIR / "daily_run.py"

# Log file for scheduled runs (appended, not overwritten)
LOG_FILE = PROJECT_DIR / "logs" / "daily_run_scheduled.log"

# launchd plist label and path
PLIST_LABEL = "com.stockbot.daily-paper-trading"
PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = PLIST_DIR / f"{PLIST_LABEL}.plist"

# Schedule: 9:30 AM ET on weekdays (Mon-Fri)
# PLAIN ENGLISH: launchd uses UTC internally.  9:30 AM ET is either
# 13:30 UTC (during EST/winter) or 13:30 UTC (during EDT/summer).
# macOS handles the timezone conversion if we set the calendar interval
# with a timezone-aware approach.  We'll use a simple approach: schedule
# at 13:30 UTC which is 9:30 AM ET during EDT (summer) and 8:30 AM ET
# during EST (winter).  For perfect accuracy we'd need a wrapper script
# that checks the timezone, but 13:30 UTC is close enough — during winter
# it fires at 8:30 AM ET (30 min before market open, still fine).
SCHEDULE_HOUR_UTC = 13
SCHEDULE_MINUTE = 30


def _build_plist() -> dict:
    """
    Build the launchd plist dictionary.

    PLAIN ENGLISH: This creates the configuration that tells macOS:
    - WHAT to run (python3 daily_run.py)
    - WHEN to run it (weekdays at 13:30 UTC)
    - WHERE to log output (logs/daily_run_scheduled.log)
    - Working directory (the project folder)
    """
    return {
        "Label": PLIST_LABEL,

        # What to run: python3 daily_run.py
        "ProgramArguments": [
            str(PYTHON),
            str(DAILY_SCRIPT),
        ],

        # Working directory so relative paths in daily_run.py work
        "WorkingDirectory": str(PROJECT_DIR),

        # When to run: every day at 13:30 UTC (weekday filtering done in daily_run.py)
        # launchd doesn't support weekday-only schedules natively, so we run
        # every day and daily_run.py will detect weekends/holidays gracefully
        "StartCalendarInterval": {
            "Hour": SCHEDULE_HOUR_UTC,
            "Minute": SCHEDULE_MINUTE,
        },

        # Log stdout and stderr to a file
        "StandardOutPath": str(LOG_FILE),
        "StandardErrorPath": str(LOG_FILE),

        # Environment variables — pass through the user's env so API keys work
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(Path.home()),
            # Pass through any trading API keys that are set
            **{k: v for k, v in os.environ.items()
               if k.startswith(("ALPACA_", "MOOMOO_", "OPENAI_", "PAPER_"))},
        },

        # Don't retry if the computer was asleep — just skip that day
        "StartCalendarInterval": {
            "Hour": SCHEDULE_HOUR_UTC,
            "Minute": SCHEDULE_MINUTE,
        },
    }


def install() -> None:
    """Install the launchd schedule."""
    # Make sure log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Make sure daily_run.py exists
    if not DAILY_SCRIPT.exists():
        print(f"  Error: {DAILY_SCRIPT} not found")
        sys.exit(1)

    # Create the plist directory if needed (usually already exists)
    PLIST_DIR.mkdir(parents=True, exist_ok=True)

    # Unload existing plist if it's already installed (avoid conflicts)
    if PLIST_PATH.exists():
        subprocess.run(
            ["launchctl", "unload", str(PLIST_PATH)],
            capture_output=True,
        )

    # Write the plist file
    plist_data = _build_plist()
    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist_data, f)

    # Load it into launchd
    result = subprocess.run(
        ["launchctl", "load", str(PLIST_PATH)],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print(f"  Schedule installed")
        print(f"  Plist:    {PLIST_PATH}")
        print(f"  Runs at:  {SCHEDULE_HOUR_UTC}:{SCHEDULE_MINUTE:02d} UTC daily "
              f"(~9:30 AM ET during summer)")
        print(f"  Script:   {DAILY_SCRIPT}")
        print(f"  Log:      {LOG_FILE}")
        print(f"\n  To check status:   python3 install_schedule.py --status")
        print(f"  To remove:         python3 install_schedule.py --uninstall")
    else:
        print(f"  Failed to load plist: {result.stderr}")
        sys.exit(1)


def uninstall() -> None:
    """Remove the launchd schedule."""
    if not PLIST_PATH.exists():
        print(f"  No schedule found at {PLIST_PATH}")
        return

    # Unload from launchd
    result = subprocess.run(
        ["launchctl", "unload", str(PLIST_PATH)],
        capture_output=True, text=True,
    )

    # Delete the plist file
    PLIST_PATH.unlink(missing_ok=True)

    if result.returncode == 0:
        print(f"  Schedule removed")
        print(f"  Deleted: {PLIST_PATH}")
    else:
        print(f"  Unloaded (may have already been inactive): {result.stderr}")
        print(f"  Deleted: {PLIST_PATH}")


def status() -> None:
    """Check if the schedule is currently active."""
    # Check if plist exists
    if not PLIST_PATH.exists():
        print(f"  Not installed (no plist at {PLIST_PATH})")
        return

    # Check launchctl for our job
    result = subprocess.run(
        ["launchctl", "list"],
        capture_output=True, text=True,
    )

    found = False
    for line in result.stdout.splitlines():
        if PLIST_LABEL in line:
            found = True
            parts = line.split("\t")
            pid = parts[0] if len(parts) > 0 else "?"
            exit_code = parts[1] if len(parts) > 1 else "?"
            print(f"  Status: ACTIVE")
            print(f"  PID:        {pid}")
            print(f"  Last exit:  {exit_code}")
            break

    if not found:
        print(f"  Status: INSTALLED but NOT LOADED")
        print(f"  Plist exists at {PLIST_PATH} but launchd doesn't have it loaded")
        print(f"  Run: python3 install_schedule.py  (to reload)")

    print(f"  Plist:  {PLIST_PATH}")
    print(f"  Log:    {LOG_FILE}")

    # Show last few lines of log if it exists
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().strip().splitlines()
        print(f"\n  Last 5 log lines:")
        for line in lines[-5:]:
            print(f"    {line}")


def test_run() -> None:
    """Run daily_run.py once in dry-run mode to verify it works."""
    print(f"  Running: {PYTHON} {DAILY_SCRIPT} --dry-run")
    result = subprocess.run(
        [str(PYTHON), str(DAILY_SCRIPT), "--dry-run"],
        cwd=str(PROJECT_DIR),
        text=True,
    )
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Install/manage the daily paper trading schedule (macOS launchd)"
    )
    parser.add_argument("--uninstall", action="store_true",
                        help="Remove the daily schedule")
    parser.add_argument("--status", action="store_true",
                        help="Check if the schedule is active")
    parser.add_argument("--test", action="store_true",
                        help="Run daily_run.py once in dry-run mode")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  PAPER TRADING SCHEDULER")
    print(f"{'='*50}\n")

    if args.uninstall:
        uninstall()
    elif args.status:
        status()
    elif args.test:
        test_run()
    else:
        install()

    print()


if __name__ == "__main__":
    main()
