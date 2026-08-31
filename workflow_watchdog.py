"""Independently verify that scheduled paper workflows actually ran.

PLAIN ENGLISH: A heartbeat inside the trading job cannot speak when that job
never starts. This separate watchdog asks GitHub directly about four workflow
runs after their New York deadlines, sends one deduplicated alert, and sends a
recovery message when they become healthy. It has no Alpaca credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from safe_io import atomic_write_json


STATE_FILE = Path("signals/workflow_watchdog_state.json")
REPORT_FILE = Path("signals/workflow_watchdog.json")
WORKFLOWS = {
    # Each tuple contains: workflow file, normal start, fallback-dispatch time,
    # fallback cutoff, and final success deadline. The daily cutoff stays five
    # minutes inside the bot's 10:30 AM New York execution window.
    "factor_data": ("factor_data_refresh.yml", time(7, 30), time(7, 45), time(18, 0), time(9, 15)),
    "daily_paper": ("daily_paper_trading.yml", time(9, 35), time(9, 45), time(10, 25), time(11, 0)),
    "shadow_paper": ("shadow_paper_journal.yml", time(9, 55), time(10, 5), time(16, 0), time(11, 30)),
    "execution_quality": ("post_market_execution_quality.yml", time(17, 15), time(17, 30), time(20, 0), time(19, 0)),
}

# Manual fallback inputs preserve normal safety rules. In particular, the
# watchdog never passes the emergency override and never asks for a late trade.
RECOVERY_INPUTS = {
    "factor_data": {"xs_only": "false"},
    "daily_paper": {"force": "false", "dry_run": "false"},
    "shadow_paper": {"force": "false", "ignore_stale": "false", "fractional_initial_equity": "400"},
    "execution_quality": {},
}


def _read_json(path: Path) -> dict:
    """Read optional state without crashing a watchdog run."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _is_nyse_session(day: datetime) -> bool:
    """Use the exchange calendar so holidays do not create false alarms."""
    try:
        import exchange_calendars as xcals
        import pandas as pd

        return bool(xcals.get_calendar("XNYS").is_session(pd.Timestamp(day.date())))
    except Exception:
        return day.weekday() < 5


def _github_runs(repository: str, workflow_file: str, token: str) -> list[dict]:
    """Read recent runs from GitHub's Actions API."""
    url = f"https://api.github.com/repos/{repository}/actions/workflows/{workflow_file}/runs?per_page=5"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "stockbot-independent-watchdog",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("workflow_runs", []) or []


def _dispatch_workflow(
    repository: str,
    workflow_file: str,
    token: str,
    *,
    inputs: dict[str, str],
) -> bool:
    """Ask GitHub to run one missed workflow from the protected main branch.

    PLAIN ENGLISH: GitHub sometimes drops a cron event. This sends the same
    workflow a manual start signal. The called workflow keeps all of its own
    market-window, holiday, duplicate-order, and paper-account protections.
    """
    url = f"https://api.github.com/repos/{repository}/actions/workflows/{workflow_file}/dispatches"
    body: dict[str, object] = {"ref": "main"}
    if inputs:
        body["inputs"] = inputs
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "stockbot-independent-watchdog",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status == 204
    except Exception:
        return False


def _scheduled_run_for_today(runs: list[dict], *, clock: datetime, expected_start: time) -> dict:
    """Select today's intended cron run or one watchdog recovery dispatch."""
    expected_minutes = expected_start.hour * 60 + expected_start.minute
    for run in runs:
        event = str(run.get("event", ""))
        if event not in {"schedule", "workflow_dispatch"}:
            continue
        try:
            created = datetime.fromisoformat(str(run.get("created_at", "")).replace("Z", "+00:00"))
        except Exception:
            continue
        created_ny = created.astimezone(ZoneInfo("America/New_York"))
        observed_minutes = created_ny.hour * 60 + created_ny.minute
        same_day = created_ny.date() == clock.date()
        intended_schedule = event == "schedule" and abs(observed_minutes - expected_minutes) <= 20
        watchdog_recovery = event == "workflow_dispatch" and observed_minutes >= expected_minutes
        if same_day and (intended_schedule or watchdog_recovery):
            return run
    return {}


def _send_telegram(message: str) -> bool:
    """Send a plain watchdog message without loading the trading application."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    body = json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status == 200
    except Exception:
        return False


def check_workflows(*, now: datetime | None = None) -> dict:
    """Check every workflow whose New York completion deadline has passed."""
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not repository or not token:
        raise RuntimeError("GITHUB_REPOSITORY and GITHUB_TOKEN are required")
    clock = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    session = _is_nyse_session(clock)
    checks: dict[str, dict] = {}
    problems: list[str] = []
    dispatched: list[str] = []

    for name, (workflow_file, expected_start, fallback_at, fallback_cutoff, deadline) in WORKFLOWS.items():
        due = bool(session and clock.time() >= deadline)
        recovery_dispatched = False
        try:
            runs = _github_runs(repository, workflow_file, token)
            latest = _scheduled_run_for_today(
                runs,
                clock=clock,
                expected_start=expected_start,
            )
            created = datetime.fromisoformat(str(latest.get("created_at", "")).replace("Z", "+00:00")) if latest else None
            created_ny = created.astimezone(ZoneInfo("America/New_York")) if created else None
            ran_today = bool(created_ny and created_ny.date() == clock.date())
            conclusion = str(latest.get("conclusion") or latest.get("status") or "missing")
            run_pending = conclusion in {"queued", "in_progress", "requested", "waiting", "pending"}

            # Recover a missing cron only inside its safe time range. Once any
            # intended run exists, let it finish instead of creating duplicates.
            recovery_due = bool(
                session
                and fallback_at <= clock.time() <= fallback_cutoff
                and not ran_today
            )
            recovery_dispatched = bool(
                recovery_due
                and _dispatch_workflow(
                    repository,
                    workflow_file,
                    token,
                    inputs=RECOVERY_INPUTS.get(name, {}),
                )
            )
            if recovery_dispatched:
                dispatched.append(name)
            healthy = bool(not due or (ran_today and conclusion == "success"))
            if recovery_dispatched:
                reason = "fallback_dispatched"
            elif healthy:
                reason = "not_due" if not due else "ok"
            elif run_pending:
                reason = f"run_pending:{conclusion}"
            else:
                reason = f"latest_today={ran_today},conclusion={conclusion}"
        except Exception as exc:
            latest = {}
            ran_today = False
            conclusion = "query_failed"
            healthy = False
            reason = f"github_query_failed:{type(exc).__name__}"
        checks[name] = {
            "workflow_file": workflow_file,
            "expected_start_new_york": expected_start.isoformat(timespec="minutes"),
            "fallback_at_new_york": fallback_at.isoformat(timespec="minutes"),
            "fallback_cutoff_new_york": fallback_cutoff.isoformat(timespec="minutes"),
            "deadline_new_york": deadline.isoformat(timespec="minutes"),
            "due": due,
            "healthy": healthy,
            "ran_today": ran_today,
            "conclusion": conclusion,
            "run_url": latest.get("html_url", ""),
            "recovery_dispatched": recovery_dispatched,
            "reason": reason,
        }
        if not healthy:
            problems.append(f"{name}:{reason}")

    previous = _read_json(STATE_FILE)
    previous_problems = set(previous.get("problems", []) or [])
    current_problems = set(problems)
    newly_failed = sorted(current_problems - previous_problems)
    recovered = sorted(previous_problems - current_problems)
    if newly_failed:
        _send_telegram("Paper workflow watchdog FAILED\n" + "\n".join(f"- {item}" for item in newly_failed))
    if recovered:
        _send_telegram("Paper workflow watchdog RECOVERED\n" + "\n".join(f"- {item}" for item in recovered))

    report = {
        "schema_version": 1,
        "checked_at": clock.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "new_york_time": clock.isoformat(timespec="seconds"),
        "nyse_session": session,
        "status": "fail" if problems else "pass",
        "problems": problems,
        "new_problems": newly_failed,
        "recovered_problems": recovered,
        "recovery_dispatches": dispatched,
        "checks": checks,
        "trading_actions_allowed": False,
        "real_capital_approved": False,
    }
    atomic_write_json(report, REPORT_FILE)
    atomic_write_json({"updated_at": report["checked_at"], "problems": problems}, STATE_FILE)
    return report


def main() -> int:
    """Run the external watchdog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the full report.")
    args = parser.parse_args()
    report = check_workflows()
    print(json.dumps(report, indent=2) if args.json else f"Workflow watchdog: {report['status']} ({len(report['problems'])} problems)")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
