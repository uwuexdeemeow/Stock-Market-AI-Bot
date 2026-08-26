"""
monitor.py — Centralized alerting and health-check module.

What this does
--------------
Every other script in this project writes to log files, but nothing was watching
those files and shouting when something went wrong.  This module fixes that.

It provides three things:

1. get_logger(name) — a single function that all other scripts can call instead
   of setting up their own logging.  This makes every module's log messages go
   through one consistent formatter and to one consistent place.

2. send_alert(title, body, severity) — sends a notification to Slack when a
webhook is configured. Includes a dedup
   mechanism so the same alert is not sent more than once per hour.

3. check_*() functions + main() — health checks you can run on a cron schedule
   (e.g. every 5 minutes) to automatically catch:
     • Model drift detected by drift_monitor.py
     • Pipeline step failures logged by orchestration.py
     • Equity curve drawdown breaching the risk limit
     • Signal anomalies (e.g. 0 actionable signals, or 100 % one-directional)

How to configure alerts
-----------------------
Add any/all of the following to your .env file (or export as shell env vars):

  # Slack
  SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ

If Slack is not configured the alerts are still written to logs/monitor.log so
nothing is silently dropped.

How to run the health checks
-----------------------------
  python monitor.py                    # run all checks once
  python monitor.py --check drift      # only check drift
  python monitor.py --check pipeline   # only check pipeline
  python monitor.py --check drawdown   # only check equity drawdown
  python monitor.py --check signals    # only check signal anomalies
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import fcntl as _fcntl
except ImportError:  # Windows
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:  # POSIX
    _msvcrt = None

# We use requests for the Slack webhook call.  It is already a dependency of
# many other modules so it should always be available.
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

import pandas as pd
from dotenv import load_dotenv

from settings import (
    LOG_DIR,
    SIGNAL_DIR,
    MAX_DRAWDOWN_HALT_PCT,
    DRIFT_PSI_ALERT,
    DRIFT_KS_ALERT,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_MONITOR_LOG   = os.path.join(LOG_DIR, "monitor.log")
_ALERT_LOG     = os.path.join(LOG_DIR, "alerts.jsonl")
_DEDUP_FILE    = os.path.join(LOG_DIR, "alert_dedup.json")
_DEDUP_LOCK_FILE = _DEDUP_FILE + ".lock"
_DRIFT_LOG     = os.path.join(LOG_DIR, "drift.jsonl")
_PIPELINE_LOG  = os.path.join(LOG_DIR, "pipeline_runs.jsonl")
_EQUITY_CSV    = os.path.join(SIGNAL_DIR, "portfolio_equity.csv")
_SIGNALS_CSV   = os.path.join(SIGNAL_DIR, "signals.csv")

os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ALERT THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

# How many hours must pass before the same alert key can fire again.
# This stops a single bad event from flooding your inbox/Slack.
DEDUP_WINDOW_HOURS = 1.0

# If a pipeline step failed this many times in a row we escalate to CRITICAL.
PIPELINE_FAILURE_ESCALATION_COUNT = 3

# Signal anomaly: if this fraction (or more) of today's actionable signals
# all point the same direction, something unusual is happening.
SIGNAL_ONE_DIRECTION_THRESHOLD = 0.90

# Signal anomaly: if there have been zero actionable signals for this many
# consecutive trading days, the pipeline may have stalled.
SIGNAL_SILENCE_DAYS = 3

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY LEVELS  (mirrors Python logging levels for clarity)
# ─────────────────────────────────────────────────────────────────────────────

INFO     = "INFO"
WARNING  = "WARNING"
ERROR    = "ERROR"
CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# CENTRALIZED LOGGER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

# A shared formatter so every module uses the same timestamp + name format.
_FORMATTER = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")

# The root handler writes to monitor.log and to stdout.
# We set it up once and reuse it for every child logger.
_root_fh: Optional[logging.FileHandler] = None


def _get_root_handler() -> logging.FileHandler:
    """Create (or return the cached) shared file handler for monitor.log."""
    global _root_fh
    if _root_fh is None:
        _root_fh = logging.FileHandler(_MONITOR_LOG, encoding="utf-8")
        _root_fh.setFormatter(_FORMATTER)
        _root_fh.setLevel(logging.DEBUG)
    return _root_fh


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to both stdout and logs/monitor.log.

    Usage (in any other module):
        from monitor import get_logger
        log = get_logger(__name__)
        log.info("hello")

    Parameters
    ----------
    name  : module name, e.g. "scanner" or __name__
    level : minimum log level (default INFO)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return as-is so we don't add duplicate handlers
        # if the same module is imported more than once.
        return logger

    logger.setLevel(level)

    # Write to the shared monitor.log file
    logger.addHandler(_get_root_handler())

    # Also print to the terminal so the user can watch progress live
    sh = logging.StreamHandler()
    sh.setFormatter(_FORMATTER)
    sh.setLevel(level)
    logger.addHandler(sh)

    # Prevent the message from also going to the root logger (avoids duplicates)
    logger.propagate = False
    return logger


_log = get_logger("monitor")


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION  (so the same alert can't fire more than once per hour)
# ─────────────────────────────────────────────────────────────────────────────

def _load_dedup() -> dict:
    """Load the dedup cache from disk.  Returns {} if the file doesn't exist."""
    if not os.path.exists(_DEDUP_FILE):
        return {}
    try:
        with open(_DEDUP_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_dedup(cache: dict) -> None:
    dedup_dir = os.path.dirname(_DEDUP_FILE) or "."
    tmp_path = os.path.join(dedup_dir, f".{os.path.basename(_DEDUP_FILE)}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, _DEDUP_FILE)


def _append_alert_log(record: dict) -> None:
    with open(_ALERT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@contextlib.contextmanager
def _exclusive_file_lock(lock_f):
    """Cross-platform advisory lock for the small alert dedup cache."""
    if _fcntl is not None:
        _fcntl.flock(lock_f.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(lock_f.fileno(), _fcntl.LOCK_UN)
        return

    if _msvcrt is not None:
        lock_f.seek(0)
        _msvcrt.locking(lock_f.fileno(), _msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            lock_f.seek(0)
            _msvcrt.locking(lock_f.fileno(), _msvcrt.LK_UNLCK, 1)
        return

    yield


def _is_suppressed(alert_key: str) -> bool:
    """
    Return True if this exact alert was already sent within DEDUP_WINDOW_HOURS.

    alert_key is a short string that uniquely identifies the alert type, e.g.
    "drift:DRIFT" or "drawdown:portfolio".
    """
    cache = _load_dedup()
    last_ts = cache.get(alert_key)
    if last_ts is None:
        return False
    elapsed_hours = (time.time() - last_ts) / 3600.0
    return elapsed_hours < DEDUP_WINDOW_HOURS


def _claim_alert(alert_key: str, record: dict) -> bool:
    """Atomically claim an alert key and write the local alert record."""
    lock_dir = os.path.dirname(_DEDUP_LOCK_FILE) or "."
    os.makedirs(lock_dir, exist_ok=True)
    with open(_DEDUP_LOCK_FILE, "a") as lock_f:
        with _exclusive_file_lock(lock_f):
            if _is_suppressed(alert_key):
                return False

            _append_alert_log(record)

            cache = _load_dedup()
            cache[alert_key] = time.time()
            _prune_and_save_dedup(cache)
            return True


def _prune_and_save_dedup(cache: dict) -> None:
    # Prune entries older than 24 hours to keep the file tidy
    cutoff = time.time() - 24 * 3600
    cache = {k: v for k, v in cache.items() if v > cutoff}
    _save_dedup(cache)


def _mark_sent(alert_key: str) -> None:
    """Record that this alert was just sent so we can suppress it for a while."""
    cache = _load_dedup()
    cache[alert_key] = time.time()
    _prune_and_save_dedup(cache)


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERY CHANNELS
# ─────────────────────────────────────────────────────────────────────────────

def _send_slack(title: str, body: str, severity: str) -> bool:
    """
    Post a message to Slack via an Incoming Webhook URL.

    Returns True on success and False on failure.

    Slack webhooks are the simplest way to post messages — you just send a
    JSON payload to a special URL and the message appears in a channel.
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False
    if not _REQUESTS_OK:
        _log.warning("requests library not installed — cannot send Slack alert")
        return False

    # Emoji prefix makes severity obvious at a glance in the Slack channel
    emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "🔴", "CRITICAL": "🚨"}.get(severity, "📢")
    payload = {
        "text": f"{emoji} *[{severity}] {title}*\n```{body}```"
    }
    try:
        from http_retry import retry_request
        # Slack webhooks can return 429 or 500 — retry with backoff
        resp = retry_request("POST", webhook_url, json=payload, timeout=10)
        if resp is not None and resp.status_code == 200:
            return True
        if resp is not None:
            _log.warning("Slack webhook returned HTTP %s: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        _log.warning("Slack delivery failed: %s", exc)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ALERT API
# ─────────────────────────────────────────────────────────────────────────────

def send_alert(
    title: str,
    body: str,
    severity: str = WARNING,
    alert_key: Optional[str] = None,
) -> None:
    """
    Send an alert to Slack when its webhook is configured.

    The alert is always written to logs/alerts.jsonl regardless of whether
    Slack is configured, so there is always a local record.

    Parameters
    ----------
    title      : Short headline, e.g. "Drift detected on AAPL features"
    body       : Full explanation with numbers/details
    severity   : One of INFO / WARNING / ERROR / CRITICAL
    alert_key  : Dedup key.  If omitted, defaults to the title.  Two calls
                 with the same key within DEDUP_WINDOW_HOURS are suppressed.
    """
    key = alert_key or title

    # Always write to the local JSONL log first — this is the ground truth
    record = {
        "sent_at":  datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "title":    title,
        "body":     body,
        "key":      key,
    }

    if not _claim_alert(key, record):
        _log.debug("alert suppressed (dedup): %s", key)
        return

    # Log to monitor.log at the matching level
    level_map = {INFO: logging.INFO, WARNING: logging.WARNING,
                 ERROR: logging.ERROR, CRITICAL: logging.CRITICAL}
    _log.log(level_map.get(severity, logging.WARNING), "[ALERT] %s — %s", title, body)

    # Try remote channels; report which ones succeeded
    slack_ok = _send_slack(title, body, severity)

    if not slack_ok:
        _log.warning(
            "No remote channels delivered alert '%s'. "
            "Set SLACK_WEBHOOK_URL to enable remote monitor alerts.",
            title,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _metric_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _worst_drift_feature(features) -> tuple[str, float, float]:
    if not isinstance(features, dict) or not features:
        return "unavailable", 0.0, 0.0

    worst_feat = None
    worst_metrics = None
    worst_psi = float("-inf")
    for name, metrics in features.items():
        if not isinstance(metrics, dict):
            continue
        psi = _metric_float(metrics.get("psi"), 0.0)
        if psi > worst_psi:
            worst_feat = name
            worst_metrics = metrics
            worst_psi = psi

    if worst_feat is None or worst_metrics is None:
        return "unavailable", 0.0, 0.0

    return str(worst_feat), worst_psi, _metric_float(worst_metrics.get("ks_stat"), 0.0)


def check_drift() -> dict:
    """
    Read the latest entry in logs/drift.jsonl and alert if status is "drift".

    drift_monitor.py runs PSI and KS tests on feature distributions and writes
    one JSON line per run.  A status of "drift" means the model's input data
    looks significantly different from what it was trained on — predictions may
    no longer be reliable.

    Returns a dict with keys: status, fired (bool), message
    """
    if not os.path.exists(_DRIFT_LOG):
        return {"status": "no_data", "fired": False, "message": "drift.jsonl not found"}

    # Read only the last line of the JSONL file (most recent run)
    last_line = ""
    with open(_DRIFT_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if not last_line:
        return {"status": "no_data", "fired": False, "message": "drift.jsonl is empty"}

    try:
        entry = json.loads(last_line)
    except json.JSONDecodeError:
        return {"status": "parse_error", "fired": False, "message": "Could not parse last drift entry"}

    status   = entry.get("status", "unknown")
    run_at   = entry.get("run_at", "unknown")
    features = entry.get("features", {})

    # Pull out the worst offending feature to make the alert body informative.
    worst_feat, worst_psi, worst_ks = _worst_drift_feature(features)

    if status == "drift":
        body = (
            f"Drift detected at {run_at}\n"
            f"Worst feature: {worst_feat}  PSI={worst_psi:.3f} (alert>{DRIFT_PSI_ALERT})  "
            f"KS={worst_ks:.3f} (alert>{DRIFT_KS_ALERT})\n"
            f"Model predictions may be unreliable.  Consider retraining."
        )
        send_alert("Model drift detected", body, severity=ERROR, alert_key="drift:DRIFT")
        return {"status": status, "fired": True, "message": body}

    if status == "caution":
        body = (
            f"Drift caution at {run_at}\n"
            f"Worst feature: {worst_feat}  PSI={worst_psi:.3f}  KS={worst_ks:.3f}\n"
            f"Monitor closely — not yet at alert threshold."
        )
        send_alert("Model drift caution", body, severity=WARNING, alert_key="drift:CAUTION")
        return {"status": status, "fired": True, "message": body}

    return {"status": status, "fired": False, "message": f"Drift stable as of {run_at}"}


def check_pipeline() -> dict:
    """
    Read logs/pipeline_runs.jsonl and alert if the most recent run failed.

    orchestration.py writes one JSON line per pipeline run containing the
    return code and stderr tail for each step.  A non-zero return code means
    a script crashed or timed out.

    Escalates to CRITICAL if the last PIPELINE_FAILURE_ESCALATION_COUNT runs
    all failed (i.e. the pipeline is stuck, not just a transient glitch).
    """
    if not os.path.exists(_PIPELINE_LOG):
        return {"status": "no_data", "fired": False, "message": "pipeline_runs.jsonl not found"}

    lines = []
    with open(_PIPELINE_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    if not lines:
        return {"status": "no_data", "fired": False, "message": "pipeline_runs.jsonl is empty"}

    # Parse the most recent N runs for escalation check
    recent_entries = []
    for raw in lines[-PIPELINE_FAILURE_ESCALATION_COUNT:]:
        try:
            recent_entries.append(json.loads(raw))
        except json.JSONDecodeError:
            pass

    if not recent_entries:
        return {"status": "parse_error", "fired": False, "message": "Could not parse pipeline log"}

    latest = recent_entries[-1]
    steps  = latest.get("steps", [])

    # Find failed steps (any step with rc != 0)
    failed_steps = [s for s in steps if s.get("rc", 0) != 0]

    if not failed_steps:
        return {"status": "ok", "fired": False, "message": "Last pipeline run succeeded"}

    # Check if all recent runs failed (escalation condition)
    all_recent_failed = all(
        any(s.get("rc", 0) != 0 for s in e.get("steps", []))
        for e in recent_entries
    )
    severity = CRITICAL if all_recent_failed else ERROR

    step_lines = "\n".join(
        f"  step={s['step']}  rc={s['rc']}  stderr: {s.get('stderr_tail','')[:300]}"
        for s in failed_steps
    )
    body = (
        f"Pipeline run at {latest.get('started_at','?')} had {len(failed_steps)} failed step(s):\n"
        f"{step_lines}"
    )
    if all_recent_failed:
        body += f"\n\n⚠️  All last {PIPELINE_FAILURE_ESCALATION_COUNT} runs failed — pipeline may be stuck."

    send_alert("Pipeline step failure", body, severity=severity, alert_key="pipeline:failure")
    return {"status": "failed", "fired": True, "message": body}


def check_equity_drawdown(equity_csv: str = _EQUITY_CSV) -> dict:
    """
    Read an equity curve CSV and alert if the current drawdown breaches
    MAX_DRAWDOWN_HALT_PCT (set in settings.py, default 15 %).

    A drawdown is how far the portfolio has fallen from its all-time peak.
    For example, if the portfolio peaked at $12,000 and is now $9,800 that is
    a drawdown of (12000 - 9800) / 12000 = 18.3 %, which would trigger this
    alert.

    Parameters
    ----------
    equity_csv : path to a CSV with columns [date, equity].  Defaults to the
                 portfolio equity output from backtest.py / orchestration.py.
    """
    if not os.path.exists(equity_csv):
        fname = os.path.basename(equity_csv)
        return {"status": "no_data", "fired": False, "message": f"{fname} not found"}

    try:
        df = pd.read_csv(equity_csv, parse_dates=["date"]).sort_values("date")
    except Exception as exc:
        return {"status": "error", "fired": False, "message": f"Could not read equity CSV: {exc}"}

    if "equity" not in df.columns or df.empty:
        return {"status": "no_data", "fired": False, "message": "equity column missing or file empty"}

    equity = df["equity"].astype(float)
    peak   = equity.cummax()
    dd     = (equity / peak - 1.0)
    current_dd = float(dd.iloc[-1])
    current_eq = float(equity.iloc[-1])
    peak_eq    = float(peak.iloc[-1])

    if current_dd <= -MAX_DRAWDOWN_HALT_PCT:
        body = (
            f"Equity drawdown of {current_dd*100:.1f}% breaches the halt threshold "
            f"of -{MAX_DRAWDOWN_HALT_PCT*100:.0f}%.\n"
            f"Current equity: ${current_eq:,.2f}  Peak: ${peak_eq:,.2f}\n"
            f"Consider halting live trading and reviewing the model."
        )
        send_alert(
            f"Equity drawdown breach ({current_dd*100:.1f}%)",
            body,
            severity=CRITICAL,
            alert_key="drawdown:breach",
        )
        return {"status": "breach", "fired": True, "message": body, "drawdown_pct": current_dd * 100}

    return {
        "status": "ok",
        "fired": False,
        "message": f"Drawdown {current_dd*100:.1f}% within limit",
        "drawdown_pct": current_dd * 100,
    }


def check_signal_anomaly(signals_csv: str = _SIGNALS_CSV) -> dict:
    """
    Read signals.csv and alert on two types of anomaly:

    1. One-directional day: ≥ 90 % of today's actionable signals are all LONG
       or all SHORT.  In a real market this is suspicious — it may indicate a
       feature bug or data error rather than a genuine edge.

    2. Signal silence: no actionable signals have been generated for
       SIGNAL_SILENCE_DAYS consecutive trading days.  This usually means the
       pipeline stalled or confidence thresholds are too strict.
    """
    if not os.path.exists(signals_csv):
        return {"status": "no_data", "fired": False, "message": "signals.csv not found"}

    try:
        df = pd.read_csv(signals_csv, parse_dates=["date"])
    except Exception as exc:
        return {"status": "error", "fired": False, "message": f"Could not read signals CSV: {exc}"}

    if df.empty or "date" not in df.columns:
        return {"status": "no_data", "fired": False, "message": "signals.csv empty or missing date column"}

    df = df.sort_values("date")
    results = []

    # ── Check 1: one-directional signals today ────────────────────────────────
    # "Actionable" means confidence was above the dynamic threshold
    actionable_col = next((c for c in df.columns if "actionable" in c.lower()), None)
    signal_col     = next((c for c in df.columns if c.lower() in ("signal", "direction")), None)

    if signal_col:
        latest_date = df["date"].max()
        today_df    = df[df["date"] == latest_date]
        if actionable_col:
            today_df = today_df[today_df[actionable_col].astype(bool)]

        if len(today_df) >= 3:  # only meaningful with at least 3 signals
            long_frac  = float((today_df[signal_col] == "LONG").mean())
            short_frac = float((today_df[signal_col] == "SHORT").mean())
            dominant   = max(long_frac, short_frac)
            direction  = "LONG" if long_frac > short_frac else "SHORT"

            if dominant >= SIGNAL_ONE_DIRECTION_THRESHOLD:
                body = (
                    f"On {latest_date.date()}, {dominant*100:.0f}% of actionable signals "
                    f"are {direction} ({len(today_df)} total).\n"
                    f"This extreme one-directionality may indicate a feature bug or data issue."
                )
                send_alert(
                    f"Signal anomaly: {dominant*100:.0f}% {direction}",
                    body,
                    severity=WARNING,
                    alert_key="signal:one_direction",
                )
                results.append({"check": "one_direction", "fired": True, "message": body})
            else:
                results.append({"check": "one_direction", "fired": False,
                                 "message": f"Direction mix OK: {long_frac*100:.0f}% LONG / {short_frac*100:.0f}% SHORT"})

    # ── Check 2: signal silence ───────────────────────────────────────────────
    # Look at the last SIGNAL_SILENCE_DAYS unique dates in the file
    recent_dates = sorted(df["date"].dt.normalize().unique())[-SIGNAL_SILENCE_DAYS:]

    if len(recent_dates) >= SIGNAL_SILENCE_DAYS:
        recent_df = df[df["date"].dt.normalize().isin(recent_dates)]
        if actionable_col:
            has_any = bool(recent_df[actionable_col].astype(bool).any())
        else:
            has_any = not recent_df.empty

        if not has_any:
            body = (
                f"No actionable signals generated in the last {SIGNAL_SILENCE_DAYS} trading days "
                f"({recent_dates[0].date()} → {recent_dates[-1].date()}).\n"
                f"Check that the pipeline is running and confidence thresholds are not too strict."
            )
            send_alert(
                f"Signal silence: {SIGNAL_SILENCE_DAYS} days with no signals",
                body,
                severity=ERROR,
                alert_key="signal:silence",
            )
            results.append({"check": "silence", "fired": True, "message": body})
        else:
            results.append({"check": "silence", "fired": False, "message": "Recent signals present"})

    if not results:
        return {"status": "no_data", "fired": False, "message": "Not enough data to check signal anomalies"}

    any_fired = any(r["fired"] for r in results)
    return {"status": "anomaly" if any_fired else "ok", "fired": any_fired, "checks": results}


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run health checks and send alerts")
    parser.add_argument(
        "--check",
        choices=["all", "drift", "pipeline", "drawdown", "signals"],
        default="all",
        help="Which health check to run (default: all)",
    )
    parser.add_argument(
        "--equity-csv",
        default=_EQUITY_CSV,
        help="Path to equity curve CSV for the drawdown check",
    )
    args = parser.parse_args()

    run_all     = args.check == "all"
    any_problem = False

    if run_all or args.check == "drift":
        _log.info("── drift check ──")
        r = check_drift()
        _log.info("drift: %s | fired=%s", r["status"], r["fired"])
        if r["fired"]:
            any_problem = True

    if run_all or args.check == "pipeline":
        _log.info("── pipeline check ──")
        r = check_pipeline()
        _log.info("pipeline: %s | fired=%s", r["status"], r["fired"])
        if r["fired"]:
            any_problem = True

    if run_all or args.check == "drawdown":
        _log.info("── equity drawdown check ──")
        r = check_equity_drawdown(args.equity_csv)
        _log.info("drawdown: %s | fired=%s", r.get("message", ""), r["fired"])
        if r["fired"]:
            any_problem = True

    if run_all or args.check == "signals":
        _log.info("── signal anomaly check ──")
        r = check_signal_anomaly()
        _log.info("signals: %s | fired=%s", r["status"], r["fired"])
        if r["fired"]:
            any_problem = True

    if any_problem:
        _log.warning("One or more health checks fired — review logs/alerts.jsonl for details")
    else:
        _log.info("All health checks passed")


if __name__ == "__main__":
    main()
