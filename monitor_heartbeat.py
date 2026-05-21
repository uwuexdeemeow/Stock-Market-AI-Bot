"""monitor_heartbeat.py — Watchdog that checks if other monitors are alive.

PLAIN ENGLISH: Your pipeline has several monitors (paper_health, fill_monitor,
regime_monitor, broker_health, execution_guard).  Each one writes a JSON file
when it runs.  But what if a monitor itself dies silently — cron stops, the
script crashes before writing, the disk is full?  You'd never know.

This script checks the TIMESTAMPS on those output files.  If any file is older
than 36 hours (meaning the monitor hasn't run in over a day), it sends a
Telegram/macOS alert so you can investigate.

HOW TO RUN:
    python3 monitor_heartbeat.py           # check all monitors
    python3 monitor_heartbeat.py --json    # output as JSON
    python3 monitor_heartbeat.py --max-age 24  # custom threshold (hours)

WHEN TO RUN:
    Add to cron at a different time than daily_run (e.g. 8pm):
    0 20 * * 1-5  cd ~/Code/Stock_Market_AI_Bot && python3 monitor_heartbeat.py

KEY CONCEPTS:
  - Heartbeat: a periodic signal that says "I'm still alive."  Here, the
    signal is the file modification time of each monitor's output.
  - Watchdog: something that watches the watchers — if a monitor stops
    producing output, this script notices.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from safe_io import atomic_write_json
from settings import SIGNAL_DIR, LOG_DIR

SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# MONITORED FILES — each monitor writes one of these when it runs successfully
# ─────────────────────────────────────────────────────────────────────────────

MONITORED_FILES: dict[str, Path] = {
    # paper_health.py writes to alpaca_paper_health.json when --broker alpaca
    # (which is the default in GitHub Actions).  Check both paths — whichever
    # exists and is newer wins.
    "paper_health": SIGNALS / "alpaca_paper_health.json",
    "fill_monitor": SIGNALS / "fill_monitor.json",
    "regime_monitor": SIGNALS / "regime_history.json",
    "broker_health": SIGNALS / "broker_health.json",
    # execution_guard.py writes to guard_intraday_state.json, not execution_guard_state.json
    "execution_guard": SIGNALS / "guard_intraday_state.json",
    "daily_run": LOGS / "daily_run_{today}.json",  # special — uses today's date
}

# How old (hours) before we consider a monitor "dead"
DEFAULT_MAX_AGE_HOURS = 36


def check_monitors(*, max_age_hours: float = DEFAULT_MAX_AGE_HOURS) -> dict:
    """Check freshness of all monitor output files.

    PLAIN ENGLISH: For each monitor, look at when its output file was last
    modified.  If it's older than max_age_hours, flag it as stale.
    Returns a summary dict with per-monitor status.
    """
    now = time.time()
    threshold_seconds = max_age_hours * 3600
    results: dict[str, dict] = {}
    stale: list[str] = []
    missing: list[str] = []

    for name, path_template in MONITORED_FILES.items():
        # Some monitors write to different paths depending on broker flag.
        # When path_template is a list, pick whichever exists and is newest.
        if isinstance(path_template, list):
            candidates = [p for p in path_template if p.exists()]
            if candidates:
                path = max(candidates, key=lambda p: p.stat().st_mtime)
            else:
                path = path_template[0]  # use first as display path when none exist
        elif "{today}" in str(path_template):
            # Handle the daily_run special case (date in filename)
            today_str = datetime.now().strftime("%Y%m%d")
            yesterday_str = datetime.fromtimestamp(now - 86400).strftime("%Y%m%d")
            path_today = Path(str(path_template).replace("{today}", today_str))
            path_yesterday = Path(str(path_template).replace("{today}", yesterday_str))
            # Use whichever exists and is newer
            path = path_today if path_today.exists() else path_yesterday
        else:
            path = path_template

        entry = {
            "path": str(path),
            "exists": path.exists(),
            "age_hours": None,
            "status": "unknown",
        }

        if not path.exists():
            entry["status"] = "missing"
            missing.append(name)
        else:
            mtime = path.stat().st_mtime
            age_hours = (now - mtime) / 3600
            entry["age_hours"] = round(age_hours, 1)
            if age_hours > max_age_hours:
                entry["status"] = "stale"
                stale.append(name)
            else:
                entry["status"] = "fresh"

        results[name] = entry

    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "max_age_hours": max_age_hours,
        "all_ok": len(stale) == 0 and len(missing) == 0,
        "stale_monitors": stale,
        "missing_monitors": missing,
        "monitors": results,
    }

    # Save heartbeat results
    SIGNALS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary, SIGNALS / "monitor_heartbeat.json")

    # Alert if anything is stale or missing
    problems = stale + missing
    if problems:
        try:
            from notifications import send_alert
            lines = []
            for name in stale:
                age = results[name]["age_hours"]
                lines.append(f"• {name}: stale ({age:.0f}h old)")
            for name in missing:
                lines.append(f"• {name}: output file missing")
            send_alert(
                f"Monitor heartbeat FAILED:\n" + "\n".join(lines),
                title="Monitor Heartbeat",
                priority="warning",
            )
        except Exception:
            pass

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check if pipeline monitors are still producing fresh output."
    )
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_HOURS,
                        help=f"Max file age in hours before alerting (default: {DEFAULT_MAX_AGE_HOURS})")
    args = parser.parse_args()

    summary = check_monitors(max_age_hours=args.max_age)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("Monitor Heartbeat Check")
        print("─" * 40)
        for name, info in summary["monitors"].items():
            if info["status"] == "fresh":
                icon = "✓"
                detail = f"{info['age_hours']:.0f}h ago"
            elif info["status"] == "stale":
                icon = "✗"
                detail = f"STALE — {info['age_hours']:.0f}h ago (threshold: {args.max_age}h)"
            else:
                icon = "?"
                detail = "MISSING — file not found"
            print(f"  {icon} {name:20s} {detail}")

        if summary["all_ok"]:
            print(f"\n  ✓ All monitors healthy")
        else:
            problems = summary["stale_monitors"] + summary["missing_monitors"]
            print(f"\n  ⚠ Problems: {', '.join(problems)}")
        print(f"  Saved → {SIGNALS / 'monitor_heartbeat.json'}")


if __name__ == "__main__":
    main()
