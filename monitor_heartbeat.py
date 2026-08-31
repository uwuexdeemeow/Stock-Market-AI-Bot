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
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from safe_io import atomic_write_csv, atomic_write_json
from run_evidence import enrich_payload
from settings import SIGNAL_DIR, LOG_DIR

SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# MONITORED FILES — each monitor writes one of these when it runs successfully
# ─────────────────────────────────────────────────────────────────────────────

MONITORED_FILES: dict[str, Path] = {
    # paper_health.py writes the Alpaca health summary on each daily run.
    "paper_health": SIGNALS / "alpaca_paper_health.json",
    "execution_scorecard": SIGNALS / "alpaca_execution_scorecard.json",
    "broker_truth": SIGNALS / "broker_truth.json",
    "fill_monitor": SIGNALS / "fill_monitor.json",
    "regime_monitor": SIGNALS / "regime_history.json",
    "broker_health": SIGNALS / "broker_health.json",
    # execution_guard.py writes to guard_intraday_state.json, not execution_guard_state.json
    "execution_guard": SIGNALS / "guard_intraday_state.json",
    "daily_run": LOGS / "daily_run_{today}.json",  # special — uses today's date
}

# How old (hours) before we consider a monitor "dead"
DEFAULT_MAX_AGE_HOURS = 36
OPERATIONAL_INCIDENT_LEDGER = SIGNALS / "operational_incident_ledger.csv"


def _update_operational_incidents(problems: list[dict], *, now: datetime) -> dict:
    """Open, refresh, and resolve deduplicated operational incidents."""
    ledger_path = SIGNALS / OPERATIONAL_INCIDENT_LEDGER.name
    columns = [
        "incident_id", "incident_key", "severity", "status", "first_seen_at",
        "latest_seen_at", "resolved_at", "occurrences", "latest_reason", "run_id",
    ]
    try:
        ledger = pd.read_csv(ledger_path, dtype=object) if ledger_path.exists() else pd.DataFrame(columns=columns)
    except Exception:
        ledger = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in ledger:
            ledger[column] = ""
    ledger = ledger[columns].astype(object)
    observed_keys = {str(item["key"]) for item in problems}
    opened: list[str] = []
    resolved: list[str] = []
    timestamp = now.isoformat(timespec="seconds")

    for problem in problems:
        key = str(problem["key"])
        mask = ledger["incident_key"].astype(str).eq(key) & ledger["status"].astype(str).eq("open")
        if mask.any():
            index = ledger.index[mask][-1]
            ledger.at[index, "latest_seen_at"] = timestamp
            ledger.at[index, "latest_reason"] = str(problem["reason"])
            ledger.at[index, "occurrences"] = int(float(ledger.at[index, "occurrences"] or 0)) + 1
            ledger.at[index, "run_id"] = os.environ.get("STOCKBOT_RUN_ID", "")
        else:
            incident_id = hashlib.sha256(f"{key}|{timestamp}".encode("utf-8")).hexdigest()[:16]
            ledger.loc[len(ledger)] = {
                "incident_id": incident_id,
                "incident_key": key,
                "severity": str(problem.get("severity", "warning")),
                "status": "open",
                "first_seen_at": timestamp,
                "latest_seen_at": timestamp,
                "resolved_at": "",
                "occurrences": 1,
                "latest_reason": str(problem["reason"]),
                "run_id": os.environ.get("STOCKBOT_RUN_ID", ""),
            }
            opened.append(incident_id)

    open_mask = ledger["status"].astype(str).eq("open")
    for index in ledger.index[open_mask]:
        if str(ledger.at[index, "incident_key"]) not in observed_keys:
            ledger.at[index, "status"] = "resolved"
            ledger.at[index, "resolved_at"] = timestamp
            resolved.append(str(ledger.at[index, "incident_id"]))

    atomic_write_csv(ledger, ledger_path, index=False)
    return {
        "path": str(ledger_path),
        "open_count": int(ledger["status"].astype(str).eq("open").sum()),
        "new_incident_ids": opened,
        "resolved_incident_ids": resolved,
    }


def check_monitors(
    *,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    run_log_prefix: str = "daily_run",
) -> dict:
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
            # A local health refresh is not a real daily trading run.  Its
            # startup stub and final report use ``local_health`` so the
            # watchdog can verify that run without pretending orders ran.
            template = str(path_template)
            if name == "daily_run" and run_log_prefix != "daily_run":
                template = template.replace("daily_run_{today}", f"{run_log_prefix}_{{today}}")
            path_today = Path(template.replace("{today}", today_str))
            path_yesterday = Path(template.replace("{today}", yesterday_str))
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

    operational_problems = [
        {"key": f"monitor:{name}", "severity": "critical", "reason": f"{name} is stale"}
        for name in stale
    ] + [
        {"key": f"monitor:{name}", "severity": "critical", "reason": f"{name} output is missing"}
        for name in missing
    ]
    # Canonical readiness can contribute alignment, data, version-lock, and
    # execution failures to the same durable operational ledger.
    health_path = SIGNALS / "alpaca_paper_health.json"
    try:
        health = json.loads(health_path.read_text(encoding="utf-8"))
    except Exception:
        health = {}
    readiness = health.get("readiness", {}) or {}
    expected_run = os.environ.get("STOCKBOT_RUN_ID", "").strip()
    evidence_runs = readiness.get("evidence_run_ids", {}) or {}
    readiness_is_current = bool(
        not expected_run
        or (
            str(health.get("run_id", "")) == expected_run
            and all(not value or str(value) == expected_run for value in evidence_runs.values())
        )
    )
    if readiness_is_current:
        for blocker in readiness.get("blockers", []) or []:
            operational_problems.append({
                "key": f"readiness:{str(blocker).split(':', 1)[0]}",
                "severity": "critical",
                "reason": str(blocker),
            })

    incident_summary = _update_operational_incidents(
        operational_problems,
        now=datetime.now(timezone.utc),
    )
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "max_age_hours": max_age_hours,
        "all_ok": len(stale) == 0 and len(missing) == 0,
        "stale_monitors": stale,
        "missing_monitors": missing,
        "monitors": results,
        "operational_incidents": incident_summary,
    }
    summary = enrich_payload(summary)

    # Save heartbeat results
    SIGNALS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary, SIGNALS / "monitor_heartbeat.json")

    # Alert if anything is stale or missing
    problems = stale + missing
    if problems and incident_summary.get("new_incident_ids"):
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
    if incident_summary.get("resolved_incident_ids"):
        try:
            from notifications import send_alert
            send_alert(
                f"Recovered incidents: {', '.join(incident_summary['resolved_incident_ids'])}",
                title="Monitor Recovery",
                priority="info",
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
    parser.add_argument(
        "--run-log-prefix",
        choices=("daily_run", "local_health"),
        default="daily_run",
        help="Which run log proves this watchdog invocation is alive.",
    )
    args = parser.parse_args()

    summary = check_monitors(
        max_age_hours=args.max_age,
        run_log_prefix=args.run_log_prefix,
    )

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
