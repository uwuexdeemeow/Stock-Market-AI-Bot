"""
daily_run.py — Run all paper trading steps in one command.

PLAIN ENGLISH: Instead of running 6+ commands every trading day, this script
chains them all together. Critical upstream failures stop later broker
submission steps so stale signals do not get traded.

Usage:
    python3 daily_run.py              # run everything
    python3 daily_run.py --dry-run    # show what would run without executing
    python3 daily_run.py --alpaca     # explicit Alpaca run
    python3 daily_run.py --stress     # also run stress tests (decay, drawdown, execution, survivorship)

Daily workflow (runs in order):
    1.  refresh_etf_data.py --refresh --force --strict → download/validate latest ETF price data
    2.  research.py --incremental          → refresh factor panel (optional in CI)
    3.  feature_quality_diagnostic.py      → refresh live feature quality report (optional in CI)
    4.  fill_monitor.py --days 2           → verify yesterday's fills before placing new orders
    5.  broker_health.py                   → pre-flight ping of Alpaca (alerts if down)
    6.  core_satellite_alpha.py            → generate the Alpaca paper signal
    7.  alpaca_paper_trading.py --submit   → submit to Alpaca
    8.  alpaca_paper_trading.py --reconcile → check if Alpaca orders filled
    9.  execution_guard.py --once          → repair ETF stops, stale orders, P&L guard
    10. broker_truth.py                    → reconcile broker positions, logs, plan, targets, stops
    11. paper_health.py                    → build deep health summary (slippage, drift, risk)
    12. execution_scorecard.py             → grade fill quality and throttled buys
    13. alpaca_paper_gauntlet.py           → check Alpaca health
    14. regime_monitor.py                  → detect and alert on regime changes

Schedule with cron (9:30 AM ET on weekdays):
    30 9 * * 1-5 cd "/path/to/Stock Market AI Bot" && python3 daily_run.py >> logs/daily_run.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from safe_io import atomic_write_json, popen_utf8
from settings import LOG_DIR, SIGNAL_DIR

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS = Path(LOG_DIR)

# Files copied from the GitHub Actions `signals/latest` branch for local
# health-only checks.  PLAIN ENGLISH: GitHub does the real daily trading run;
# this list brings the resulting signal/dashboard files back onto your laptop
# so local status.py and monitor_heartbeat.py read the same state.
SIGNALS_LATEST_REF = "origin/signals/latest"
SIGNALS_LATEST_BRANCH = "signals/latest"
GITHUB_SIGNAL_SYNC_FILES = (
    "signals/core_satellite_alpha_signal.csv",
    "signals/core_satellite_alpha_orders.csv",
    "signals/core_satellite_alpha_metrics.json",
    "signals/alpaca_paper_equity.csv",
    "signals/alpaca_paper_log.csv",
    # PLAIN ENGLISH: the target-weight check needs Alpaca's saved positions,
    # values, and equity. Without this file, a local health refresh cannot
    # independently confirm that the account matches the latest signal.
    "signals/alpaca_daily_status.json",
    "signals/factor_data_health.json",
    "signals/factor_decay_monitor.csv",
    "logs/etf_data_health.json",
    "signals/fill_monitor.json",
    "signals/broker_health.json",
    "signals/alpaca_paper_health.json",
    "signals/alpaca_execution_scorecard.json",
    "signals/alpaca_submit_outcome.json",
    "signals/alpaca_submit_outcomes.csv",
    "signals/execution_cost_calibration.json",
    "signals/paper_validation_epoch.json",
    "signals/paper_validation_epoch_status.json",
    "signals/broker_truth.csv",
    "signals/broker_truth.json",
    "signals/shadow_paper_journal.csv",
    "signals/guard_intraday_state.json",
    "signals/regime_history.json",
    "signals/regime_changes_log.csv",
    "signals/monitor_heartbeat.json",
)
GITHUB_SIGNAL_SYNC_PREFIXES = (
    "logs/daily_run_",
    "logs/alpaca_paper_health_",
    "logs/alpaca_execution_scorecard_",
    "logs/broker_truth_",
)


def _configure_console_output() -> None:
    """Keep Windows cp1252 consoles from crashing on existing Unicode status text."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_console_output()


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION HELPERS — alert the user when something goes wrong
# ─────────────────────────────────────────────────────────────────────────────

def notify_failures(results: list[dict], total_time: float) -> None:
    """
    Send notifications if any step failed.

    PLAIN ENGLISH: After all steps finish, this checks if anything broke.
    If yes, it sends alerts via desktop and Telegram (if configured)
    through the shared notifications module.
    If everything passed, it stays quiet — no spam on good days.
    """
    failures = [r for r in results if r["status"] not in ("ok", "no_action", "skipped")]
    if not failures:
        return

    from notifications import send_alert

    # Build a short summary for notifications
    failed_names = ", ".join(r["name"] for r in failures)
    ok_count = sum(1 for r in results if r["status"] in ("ok", "no_action"))
    total = len(results)

    # Longer message with error details
    lines = [
        f"Daily Paper Trading Run — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Passed: {ok_count}/{total}",
        f"Failed: {len(failures)}",
        f"Total time: {total_time:.1f}s",
        "",
        "FAILURES:",
    ]
    for r in failures:
        lines.append(f"  [{r['name']}] status={r['status']}")
        if r.get("stderr_tail"):
            lines.append(f"    {r['stderr_tail'][:300]}")
        if r.get("error"):
            lines.append(f"    {r['error'][:300]}")

    message = "\n".join(lines)
    send_alert(message, title="Daily Run", priority="warning")


# ─────────────────────────────────────────────────────────────────────────────
# STEP DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Step:
    name: str
    cmd: list[str]
    description: str
    critical: bool = False
    timeout_seconds: int | None = None
    always_run: bool = False
    env: dict[str, str] | None = None


# Data refresh steps — run BEFORE signal generation so factors/ETFs are fresh.
# PLAIN ENGLISH: These scripts download the latest stock and ETF data from the
# internet and save it to disk.  Without fresh data, the signal generator would
# use stale prices and factor scores, which defeats the purpose of daily trading.
ETF_REFRESH_STEP = Step(
    "refresh_etf_data",
    [sys.executable, "refresh_etf_data.py", "--refresh", "--force", "--strict"],
    "Download latest ETF price data (SPY, QQQ, TQQQ, etc.)",
    critical=True,
)

# Factor refresh can be slow on ephemeral GitHub runners.  The CI daily trading
# workflow skips these steps and relies on the latest successful factor refresh
# workflow; core_satellite_alpha.py still blocks if the restored factor data is
# too stale.
FACTOR_REFRESH_STEPS = [
    Step(
        "refresh_factor_data",
        [sys.executable, "research.py", "--incremental"],
        "Refresh factor panel data incrementally (only download new days since last run)",
        critical=True,
        timeout_seconds=1800,
        env={
            "SENTIMENT_ENGINE_LEVEL": os.environ.get("SENTIMENT_ENGINE_LEVEL", "vader"),
            "STOCKBOT_PRICE_PROVIDER_ORDER": os.environ.get(
                "STOCKBOT_PRICE_PROVIDER_ORDER",
                "yahooquery,yfinance,stooq",
            ),
            "STOCKBOT_SKIP_RESEARCH_SENTIMENT": os.environ.get(
                "STOCKBOT_SKIP_RESEARCH_SENTIMENT",
                "1",
            ),
        },
    ),
    Step(
        "refresh_feature_quality",
        [sys.executable, "feature_quality_diagnostic.py", "--top", "48"],
        "Refresh live feature quality report used by core_satellite_alpha.py",
        critical=True,
        timeout_seconds=600,
    ),
]
DATA_REFRESH_STEPS = [ETF_REFRESH_STEP, *FACTOR_REFRESH_STEPS]

# Fill verification — runs BEFORE new signals/orders to check yesterday's fills.
# PLAIN ENGLISH: Before submitting new orders, check if yesterday's orders
# actually filled.  If something got cancelled or partially filled, you want
# to know BEFORE placing new trades.
FILL_MONITOR_STEP = Step(
    "fill_monitor",
    [sys.executable, "fill_monitor.py", "--days", "2"],
    "Verify recent order fills (check for cancelled/partial orders)",
    critical=True,
    always_run=True,
)

BROKER_HEALTH_STEP = Step(
    "broker_health",
    [sys.executable, "broker_health.py"],
    "Pre-flight Alpaca connectivity check",
    critical=True,
)

CORE_SATELLITE_SIGNAL_STEP = Step(
    "core_satellite_signal",
    [sys.executable, "core_satellite_alpha.py"],
    "Generate core-satellite signal for Alpaca paper trading",
    critical=True,
)

FACTOR_DATA_HEALTH_STEP = Step(
    "factor_data_health",
    [sys.executable, "factor_data_health.py", "--strict"],
    "Validate restored factor data cache before signal generation",
    critical=True,
)

DRIFT_MONITOR_STEP = Step(
    "drift_monitor",
    [sys.executable, "drift_monitor.py", "--run-name", "pooled"],
    "Compare recent model inputs with the training-time feature baseline",
)

ALPACA_STATUS_STEP = Step(
    "alpaca_status",
    [sys.executable, "alpaca_paper_trading.py", "--status"],
    "Sync Alpaca paper account status/equity without submitting orders",
)

# Alpaca steps. core_satellite_alpha_signal.csv includes TQQQ weight when the nested
# walkforward grid search determines it helps on a risk-adjusted basis).
ALPACA_STEPS = [
    Step(
        "alpaca_submit",
        [sys.executable, "alpaca_paper_trading.py", "--submit"],
        "Submit orders to Alpaca paper trading (auto-snapshots equity + status)",
        critical=True,
    ),
    Step(
        "alpaca_reconcile",
        [sys.executable, "alpaca_paper_trading.py", "--reconcile"],
        "Reconcile Alpaca order fill statuses",
        always_run=True,
    ),
    Step(
        "alpaca_execution_guard",
        [sys.executable, "execution_guard.py", "--once"],
        "Repair ETF protection, cancel stale Alpaca orders, and check intraday P&L",
        always_run=True,
    ),
    Step(
        "broker_truth",
        [sys.executable, "broker_truth.py"],
        "Reconcile Alpaca positions, local logs, order plan, target weights, and stops",
        always_run=True,
    ),
    Step(
        "alpaca_paper_health",
        [sys.executable, "paper_health.py"],
        "Build deep health summary for Alpaca (slippage, drift vs walkforward, risk)",
        always_run=True,
    ),
    Step(
        "alpaca_execution_scorecard",
        [sys.executable, "execution_scorecard.py"],
        "Grade Alpaca execution quality and execution-risk throttle outcomes",
        always_run=True,
    ),
    Step(
        "execution_cost_calibration",
        [sys.executable, "execution_cost_calibration.py"],
        "Calibrate simulated trading costs from recent Alpaca fills",
        always_run=True,
    ),
    Step(
        "alpaca_gauntlet",
        [sys.executable, "alpaca_paper_gauntlet.py"],
        "Run Alpaca paper gauntlet health check",
        always_run=True,
    ),
    Step(
        "paper_validation_epoch",
        [sys.executable, "paper_validation_epoch.py", "--status"],
        "Measure progress in the clean paper-validation epoch",
        always_run=True,
    ),
]

# Local health-only mode must not submit fresh orders.  It can still sync
# broker state, reconcile existing orders, and rebuild health dashboards.
ALPACA_HEALTH_ONLY_STEPS = [
    ALPACA_STATUS_STEP,
    ALPACA_STEPS[1],  # alpaca_reconcile
    ALPACA_STEPS[3],  # broker_truth
    ALPACA_STEPS[4],  # alpaca_paper_health
    ALPACA_STEPS[5],  # alpaca_execution_scorecard
    ALPACA_STEPS[6],  # execution_cost_calibration
    ALPACA_STEPS[7],  # alpaca_gauntlet
    ALPACA_STEPS[8],  # paper_validation_epoch
]

# Regime change monitor runs after the Alpaca signal is generated.
# PLAIN ENGLISH: Checks if the market regime (risk_on/neutral/risk_off)
# changed since yesterday and sends you a notification if it did.
REGIME_MONITOR_STEP = Step(
    "regime_monitor",
    [sys.executable, "regime_monitor.py"],
    "Check for regime changes and alert if detected",
)

MONITOR_HEARTBEAT_STEP = Step(
    "monitor_heartbeat",
    [sys.executable, "monitor_heartbeat.py"],
    "Check if all monitors produced fresh output (watchdog)",
    always_run=True,
)

LOG_CLEANUP_STEP = Step(
    "log_cleanup",
    [sys.executable, "log_cleanup.py", "--check-disk"],
    "Check disk usage and warn if nearing capacity",
    always_run=True,
)

# Stress test steps — optional, run with --stress flag
# PLAIN ENGLISH: These are research/safety scripts that check whether the
# strategy's edge is decaying, whether drawdown throttles would have fired,
# whether execution costs could hurt us, and whether our backtest is biased
# by survivorship.  They don't change any signals — they just report.
STRESS_STEPS = [
    Step(
        "factor_decay",
        [sys.executable, "factor_decay_monitor.py"],
        "Check if factor overlay IC and alpha are decaying",
    ),
    Step(
        "drawdown_throttle",
        [sys.executable, "core_satellite_drawdown_throttle.py"],
        "Stress test drawdown throttle scenarios",
    ),
    Step(
        "execution_stress",
        [sys.executable, "core_satellite_execution_stress.py"],
        "Stress test execution costs (delayed fills, extra slippage)",
    ),
    Step(
        "survivorship_audit",
        [sys.executable, "core_satellite_survivorship_audit.py"],
        "Audit survivorship bias in backtest universe",
    ),
]


def _read_fresh_submit_outcome(started_at: datetime) -> tuple[dict, str]:
    """Read this run's Alpaca outcome and reject missing or stale reports."""
    path = Path(SIGNAL_DIR) / "alpaca_submit_outcome.json"
    if not path.exists():
        return {}, "submit_outcome_missing"
    try:
        if path.stat().st_mtime + 1.0 < started_at.timestamp():
            return {}, "submit_outcome_stale"
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"submit_outcome_invalid:{exc.__class__.__name__}"
    status = str(payload.get("status", ""))
    if status not in {"executed", "no_action", "blocked", "failed"}:
        return payload, f"submit_outcome_unknown_status:{status or 'missing'}"
    return payload, ""


def run_step(
    name: str,
    cmd: list[str],
    description: str,
    dry_run: bool = False,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> dict:
    """
    Run a single pipeline step and capture the result.

    PLAIN ENGLISH: Runs a command, waits for it to finish (up to 5 minutes),
    and captures whether it succeeded or failed.  If it fails, the error is
    logged but doesn't stop the rest of the pipeline.

    Returns a dict with: name, status, elapsed_seconds, error (if any)
    """
    print(f"\n{'─'*60}")
    print(f"  [{name}] {description}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print(f"  ⏭  DRY RUN — skipped")
        return {"name": name, "status": "skipped", "elapsed": 0.0}

    start = datetime.now()
    # Stream output live so user sees progress instead of waiting for a
    # silent capture to finish.  Buffer the tail in memory for the final
    # summary line (failures still print stderr).
    stdout_tail: list[str] = []
    stderr_tail: list[str] = []
    TAIL_SIZE = 200  # keep last 200 lines of each stream for summary/diagnostics

    try:
        proc = popen_utf8(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # line-buffered
            cwd=str(Path(__file__).parent),
            env=env,
        )
        # Use background threads for both streams so a quiet or long-running
        # child process is still governed by the timeout below.
        import threading

        def _drain(stream, sink: list[str], prefix: str):
            for line in stream:
                line = line.rstrip()
                sink.append(line)
                if len(sink) > TAIL_SIZE:
                    sink.pop(0)
                print(f"  {prefix}{line}", flush=True)

        out_thread = threading.Thread(
            target=_drain, args=(proc.stdout, stdout_tail, ""), daemon=True
        )
        err_thread = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_tail, "ERR: "), daemon=True
        )
        out_thread.start()
        err_thread.start()

        try:
            proc.wait(timeout=max(0.0, float(timeout)))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            out_thread.join(timeout=2.0)
            err_thread.join(timeout=2.0)
            elapsed = (datetime.now() - start).total_seconds()
            print(f"  ✗ TIMEOUT after {timeout}s")
            return {"name": name, "status": "timeout", "elapsed": round(elapsed, 1)}

        out_thread.join(timeout=2.0)
        err_thread.join(timeout=2.0)
        elapsed = (datetime.now() - start).total_seconds()
        returncode = proc.returncode

        submit_outcome: dict = {}
        submit_outcome_error = ""
        if name == "alpaca_submit":
            submit_outcome, submit_outcome_error = _read_fresh_submit_outcome(start)

        if returncode != 0:
            print(f"  ✗ FAILED (exit code {returncode})")
            result = {
                "name": name,
                "status": "failed",
                "exit_code": returncode,
                "elapsed": round(elapsed, 1),
                "stderr_tail": "\n".join(stderr_tail[-10:]),
            }
            if submit_outcome:
                result["execution_outcome"] = submit_outcome
            if submit_outcome_error:
                result["execution_outcome_error"] = submit_outcome_error
            return result

        if name == "alpaca_submit":
            if submit_outcome_error:
                print(f"  ✗ FAILED ({submit_outcome_error})")
                return {
                    "name": name,
                    "status": "failed",
                    "elapsed": round(elapsed, 1),
                    "error": submit_outcome_error,
                }
            execution_status = str(submit_outcome.get("status"))
            if execution_status in {"blocked", "failed"}:
                print(
                    f"  ✗ {execution_status.upper()}: "
                    f"{submit_outcome.get('reason_code', 'unknown')}"
                )
                return {
                    "name": name,
                    "status": "failed",
                    "elapsed": round(elapsed, 1),
                    "execution_outcome": submit_outcome,
                }
            if execution_status == "no_action":
                print(f"  ○ NO ACTION ({submit_outcome.get('reason_code', 'unknown')})")
                return {
                    "name": name,
                    "status": "no_action",
                    "elapsed": round(elapsed, 1),
                    "execution_outcome": submit_outcome,
                }

        print(f"  ✓ OK ({elapsed:.1f}s)")
        result = {"name": name, "status": "ok", "elapsed": round(elapsed, 1)}
        if submit_outcome:
            result["execution_outcome"] = submit_outcome
        return result

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  ✗ ERROR: {e}")
        return {"name": name, "status": "error", "elapsed": round(elapsed, 1), "error": str(e)}


def build_steps(
    *,
    skip_refresh: bool,
    skip_factor_refresh: bool = False,
    run_alpaca: bool = True,
    health_only: bool = False,
    stress: bool = False,
) -> list[Step]:
    steps: list[Step] = []
    if health_only:
        steps.append(FILL_MONITOR_STEP)
        if run_alpaca:
            steps.append(Step(
                "broker_health",
                [sys.executable, "broker_health.py"],
                "Pre-flight Alpaca connectivity check",
                critical=False,
            ))
        if run_alpaca:
            steps.extend(ALPACA_HEALTH_ONLY_STEPS)
        steps.append(REGIME_MONITOR_STEP)
        steps.append(MONITOR_HEARTBEAT_STEP)
        steps.append(LOG_CLEANUP_STEP)
        if stress:
            steps.extend(STRESS_STEPS)
        return steps

    if not skip_refresh:
        steps.append(ETF_REFRESH_STEP)
        if not skip_factor_refresh:
            steps.extend(FACTOR_REFRESH_STEPS)
    steps.append(FILL_MONITOR_STEP)
    if run_alpaca:
        steps.append(Step(
            "broker_health",
            [sys.executable, "broker_health.py"],
            "Pre-flight Alpaca connectivity check",
            critical=False,
        ))
    steps.append(FACTOR_DATA_HEALTH_STEP)
    steps.append(DRIFT_MONITOR_STEP)
    steps.append(CORE_SATELLITE_SIGNAL_STEP)
    if run_alpaca:
        steps.extend(ALPACA_STEPS)
    steps.append(REGIME_MONITOR_STEP)
    # Watchdog + housekeeping — non-critical, run last
    steps.append(MONITOR_HEARTBEAT_STEP)
    steps.append(LOG_CLEANUP_STEP)
    if stress:
        steps.extend(STRESS_STEPS)
    return steps


def _git_output(args: list[str]) -> subprocess.CompletedProcess:
    """Run git and capture bytes so CSV/JSON files copy exactly.

    PLAIN ENGLISH: `git show branch:path` gives us the file as it exists on
    GitHub's signals/latest branch.  We save those bytes locally.
    """
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )


def _latest_remote_signal_files(ref: str = SIGNALS_LATEST_REF) -> list[str]:
    """List recent daily logs and health files available on the remote branch."""
    proc = _git_output(["ls-tree", "-r", "--name-only", ref])
    if proc.returncode != 0:
        return []
    all_files = proc.stdout.decode("utf-8", errors="replace").splitlines()
    selected = set(GITHUB_SIGNAL_SYNC_FILES)
    for prefix in GITHUB_SIGNAL_SYNC_PREFIXES:
        matches = sorted(path for path in all_files if path.startswith(prefix))
        selected.update(matches[-5:])
    return sorted(selected)


def sync_latest_github_signals(*, dry_run: bool = False, fetch: bool = True) -> dict:
    """Copy latest GitHub Actions signal/dashboard files into the local tree.

    PLAIN ENGLISH: Your laptop did not run the real trading job, GitHub did.
    This function fetches the `signals/latest` branch and copies the latest
    signal files, health JSONs, and daily run log into local `signals/`/`logs/`
    so the local dashboard reflects the automated run.
    """
    start = datetime.now()
    copied: list[str] = []
    missing: list[str] = []
    errors: list[str] = []

    print(f"\n{'─'*60}")
    print("  [github_signal_sync] Download latest signal/dashboard files from GitHub")
    print(f"  Source: {SIGNALS_LATEST_BRANCH}")

    if dry_run:
        print("  ⏭  DRY RUN — skipped")
        return {"name": "github_signal_sync", "status": "skipped", "elapsed": 0.0}

    if fetch:
        fetch_proc = _git_output([
            "fetch",
            "--quiet",
            "origin",
            f"+{SIGNALS_LATEST_BRANCH}:refs/remotes/{SIGNALS_LATEST_REF}",
        ])
        if fetch_proc.returncode != 0:
            message = fetch_proc.stderr.decode("utf-8", errors="replace").strip() or "git fetch failed"
            print(f"  ✗ FAILED fetch: {message}")
            return {
                "name": "github_signal_sync",
                "status": "failed",
                "elapsed": round((datetime.now() - start).total_seconds(), 1),
                "error": message[:500],
            }

    remote_files = _latest_remote_signal_files()
    if not remote_files:
        message = f"no files found on {SIGNALS_LATEST_REF}; run GitHub Actions once or check the branch"
        print(f"  ✗ FAILED: {message}")
        return {
            "name": "github_signal_sync",
            "status": "failed",
            "elapsed": round((datetime.now() - start).total_seconds(), 1),
            "error": message,
        }

    for rel_path in remote_files:
        show_proc = _git_output(["show", f"{SIGNALS_LATEST_REF}:{rel_path}"])
        if show_proc.returncode != 0:
            missing.append(rel_path)
            continue
        try:
            out_path = PROJECT_ROOT / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(show_proc.stdout)
            copied.append(rel_path)
        except OSError as exc:
            errors.append(f"{rel_path}: {exc}")

    elapsed = round((datetime.now() - start).total_seconds(), 1)
    if errors:
        print(f"  ✗ FAILED ({len(errors)} write errors, {len(copied)} copied)")
        return {
            "name": "github_signal_sync",
            "status": "failed",
            "elapsed": elapsed,
            "copied_count": len(copied),
            "missing_count": len(missing),
            "error": "; ".join(errors[:3]),
        }

    print(f"  ✓ OK ({elapsed:.1f}s) — copied {len(copied)} files")
    if missing:
        print(f"  ⚠ Missing on {SIGNALS_LATEST_REF}: {', '.join(missing[:5])}")
    return {
        "name": "github_signal_sync",
        "status": "ok",
        "elapsed": elapsed,
        "copied_count": len(copied),
        "missing_count": len(missing),
        "copied_files": copied,
    }


def run_steps(steps: list[Step], *, dry_run: bool, timeout: int) -> list[dict]:
    results: list[dict] = []
    blocked_by: str | None = None
    for step in steps:
        if blocked_by and not step.always_run:
            print(f"\n{'─'*60}")
            print(f"  [{step.name}] BLOCKED by critical failure: {blocked_by}")
            results.append({
                "name": step.name,
                "status": "blocked",
                "blocked_by": blocked_by,
                "elapsed": 0.0,
            })
            continue
        effective_timeout = int(step.timeout_seconds or timeout)
        if blocked_by and step.always_run:
            print(f"\n{'─'*60}")
            print(f"  [{step.name}] running despite earlier blocker: {blocked_by}")
        result = run_step(
            step.name,
            step.cmd,
            step.description,
            dry_run=dry_run,
            timeout=effective_timeout,
            env=step.env,
        )
        if blocked_by and step.always_run:
            # Keep the original blocker visible in the JSON log while still
            # letting watchdog/cleanup style steps refresh their output files.
            result["upstream_blocked_by"] = blocked_by
        results.append(result)
        if step.critical and result["status"] in ("failed", "error", "timeout") and blocked_by is None:
            blocked_by = step.name
    return results


def _write_startup_stubs(now: datetime, steps: list[Step], *, write_daily_stub: bool = True) -> None:
    """Write fresh watchdog files before the first step runs.

    PLAIN ENGLISH: `monitor_heartbeat.py` runs before `daily_run.py` writes the
    final daily summary.  These tiny "pending/running" files prove the pipeline
    started, and they get overwritten later by the real reports.
    """
    # Write a "running" stub so monitor_heartbeat (which executes BEFORE the
    # final results are written at the end of main()) doesn't false-positive
    # with "daily_run: output file missing".  The stub is overwritten with
    # the real summary when the run completes.
    if write_daily_stub:
        stub_path = LOGS / f"daily_run_{now.strftime('%Y%m%d')}.json"
        try:
            atomic_write_json(
                {
                    "timestamp": now.isoformat(),
                    "status": "running",
                    "steps_total": len(steps),
                    "steps_ok": 0,
                    "steps_failed": 0,
                    "results": [],
                },
                stub_path,
            )
        except Exception as exc:
            # Stub-writing is best-effort — don't block the run on it.
            print(f"  ⚠ Could not write run stub ({exc}); continuing anyway.")

    # Same idea for fill_monitor.json: if fill_monitor.py crashes (e.g. a
    # broker module fails to import on a fresh runner), monitor_heartbeat
    # would alert "fill_monitor: output file missing".  Write an empty
    # placeholder up front — fill_monitor.py atomic-writes the real result
    # immediately afterwards if it succeeds.
    #
    # IMPORTANT: use the SAME absolute path that fill_monitor.py and
    # monitor_heartbeat.py use (SIGNAL_DIR from settings).  Before this we
    # had Path("signals") which was relative to whatever CWD the runner
    # used — on GitHub Actions that could resolve to a different directory
    # than the absolute SIGNAL_DIR.  Result: stub written to one place,
    # monitor_heartbeat looking in another.
    try:
        signals_dir = Path(SIGNAL_DIR)
        signals_dir.mkdir(parents=True, exist_ok=True)
        fm_stub = signals_dir / "fill_monitor.json"
        existed_before = fm_stub.exists()
        atomic_write_json(
            {
                "checked_at": now.isoformat(),
                "status": "pending",
                "lookback_days": 2,
                "total_checked": 0,
                "filled": 0,
                "cancelled": 0,
                "partial": 0,
                "unknown": 0,
                "not_submitted": 0,
                "problems": [],
                "fill_rate": 0,
                "note": "stub written by daily_run.py before fill_monitor.py ran",
                "stub_path": str(fm_stub),
                "stub_existed_before": existed_before,
            },
            fm_stub,
        )
        print(f"  ✓ Wrote fill_monitor stub → {fm_stub} (existed_before={existed_before})")
    except Exception as exc:
        print(f"  ⚠ Could not write fill_monitor stub ({exc}); continuing anyway.")


def _is_us_market_holiday(dt: datetime) -> bool:
    """
    Check if a date is a US stock market (NYSE) holiday.

    PLAIN ENGLISH: Instead of manually computing each holiday, we use the
    'exchange_calendars' library which has the official NYSE calendar.  This
    handles early closes, special closures, and future holiday changes
    automatically — no manual updates needed.
    """
    import exchange_calendars as xcals

    nyse = xcals.get_calendar("XNYS")
    d = dt.date() if hasattr(dt, "date") else dt
    # exchange_calendars uses pd.Timestamp for date lookups
    ts = pd.Timestamp(d)
    # A date is a holiday if it's a weekday but NOT a valid trading session
    if d.weekday() >= 5:
        return True  # weekend — not a holiday per se, but market is closed
    try:
        return not nyse.is_session(ts)
    except Exception:
        # If date is out of range for the calendar, fall back to weekend check
        return d.weekday() >= 5


def main():
    parser = argparse.ArgumentParser(
        description="Run all paper trading steps in one command"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--alpaca", action="store_true",
                        help="Run Alpaca paper-trading steps (also the default)")
    parser.add_argument("--stress", action="store_true",
                        help="Also run stress tests (factor decay, drawdown throttle, execution, survivorship)")
    parser.add_argument("--health-only", action="store_true",
                        help="Local dashboard refresh only: sync GitHub signals/latest, then run health checks without submitting orders")
    parser.add_argument("--no-github-sync", action="store_true",
                        help="With --health-only, skip downloading signal/dashboard files from GitHub")
    parser.add_argument("--skip-refresh", action="store_true",
                        help="Skip data refresh steps (use existing factor/ETF data as-is)")
    parser.add_argument("--skip-factor-refresh", action="store_true",
                        help="Refresh ETF data, but use latest existing factor data/report")
    parser.add_argument("--force", action="store_true",
                        help="Run even on weekends and US market holidays")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Max seconds per step (default: 300)")
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)

    # ── PID lock — prevents cron + manual overlap from double-submitting ──
    # PLAIN ENGLISH: If you accidentally run daily_run.py twice at the same time
    # (e.g. cron fires while you're running it manually), the second instance
    # will see the lock file is held and exit immediately instead of submitting
    # duplicate orders.  The lock is automatically released when the process ends.
    _lock_path = LOGS / "daily_run.lock"
    _lock_file = open(_lock_path, "w")
    try:
        if sys.platform == "win32":
            # Windows: use msvcrt for file locking (no fcntl available)
            import msvcrt
            msvcrt.locking(_lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            # macOS / Linux: use fcntl for file locking
            import fcntl
            fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print("⛔ Another daily_run.py is already running — exiting to prevent double-orders.")
        sys.exit(1)

    # Weekend & holiday guard — skip if market is closed today
    # PLAIN ENGLISH: No point downloading data or submitting orders when
    # the stock market is closed.  We check if today is a weekday and not
    # a US market holiday.  Use --force to override (e.g. for testing).
    if not args.dry_run and not args.force and not args.health_only:
        today = datetime.now()
        # Saturday = 5, Sunday = 6
        if today.weekday() >= 5:
            print(f"{'═'*60}")
            print(f"  DAILY PAPER TRADING RUN")
            print(f"  {today.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  ⏭ Skipping — today is {today.strftime('%A')} (market closed)")
            print(f"  Use --force to run anyway")
            print(f"{'═'*60}\n")
            sys.exit(0)

        # Check US market holidays (NYSE observed holidays)
        # PLAIN ENGLISH: These are the days the NYSE is closed every year.
        # We check the current year's dates.  Not exhaustive for special
        # closures but covers the standard calendar.
        if _is_us_market_holiday(today):
            print(f"{'═'*60}")
            print(f"  DAILY PAPER TRADING RUN")
            print(f"  {today.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  ⏭ Skipping — today is a US market holiday")
            print(f"  Use --force to run anyway")
            print(f"{'═'*60}\n")
            sys.exit(0)

    # ── Pre-flight: validate that broker API keys are configured ──
    # PLAIN ENGLISH: Check that required environment variables exist BEFORE
    # spending 5 minutes on signal generation only to fail at order submission.
    # This catches copy-paste mistakes (empty strings, missing .env lines).
    _missing_keys: list[str] = []
    if not args.dry_run:
        import os as _os
        if not _os.environ.get("ALPACA_API_KEY", "").strip():
            _missing_keys.append("ALPACA_API_KEY")
        if not _os.environ.get("ALPACA_SECRET_KEY", "").strip():
            _missing_keys.append("ALPACA_SECRET_KEY")
        if _missing_keys:
            print(f"⛔ Missing required environment variables: {', '.join(_missing_keys)}")
            print("  Set them in your .env file or shell profile before running.")
            sys.exit(1)

    # PLAIN ENGLISH: This repo uses Alpaca for paper execution now. The
    # explicit --alpaca flag remains accepted for workflow readability.
    run_alpaca = True

    steps = build_steps(
        skip_refresh=bool(args.skip_refresh),
        skip_factor_refresh=bool(args.skip_factor_refresh),
        run_alpaca=bool(run_alpaca),
        health_only=bool(args.health_only),
        stress=bool(args.stress),
    )

    # Header
    now = datetime.now()
    print(f"{'═'*60}")
    print(f"  DAILY PAPER TRADING RUN")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Steps: {len(steps)}")
    if run_alpaca:
        print(f"  Alpaca: core-satellite unified signal")
    if args.dry_run:
        print(f"  ⚠ DRY RUN MODE — nothing will execute")
    if args.health_only:
        print("  Mode: health-only local dashboard refresh (no order submission)")
        if not args.no_github_sync:
            print("  GitHub sync: enabled — pulling signals/latest first")
    if args.skip_factor_refresh and not args.skip_refresh:
        print("  Factor refresh: skipped — using latest restored factor data/report")
    print(f"{'═'*60}")

    pre_results: list[dict] = []
    if args.health_only and not args.no_github_sync:
        pre_results.append(sync_latest_github_signals(dry_run=bool(args.dry_run)))

    if not args.dry_run:
        _write_startup_stubs(now, steps, write_daily_stub=not bool(args.health_only))

    # Run each step
    results = pre_results + run_steps(steps, dry_run=bool(args.dry_run), timeout=int(args.timeout))

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    no_action = sum(1 for r in results if r["status"] == "no_action")
    failed = sum(1 for r in results if r["status"] == "failed")
    errors = sum(1 for r in results if r["status"] in ("error", "timeout"))
    blocked = sum(1 for r in results if r["status"] == "blocked")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    total_time = sum(r.get("elapsed", 0) for r in results)

    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  ✓ Completed: {ok + no_action}/{len(results)}")
    if no_action:
        print(f"  ○ No action: {no_action}")
    if failed:
        print(f"  ✗ Failed:  {failed}")
    if errors:
        print(f"  ✗ Errors:  {errors}")
    if blocked:
        print(f"  ⛔ Blocked: {blocked}")
    if skipped:
        print(f"  ⏭ Skipped: {skipped}")
    print(f"  Total time: {total_time:.1f}s")

    # Show failures
    for r in results:
        if r["status"] not in ("ok", "no_action", "skipped"):
            print(f"\n  FAILED: [{r['name']}] — {r['status']}")
            if r.get("stderr_tail"):
                print(f"    {r['stderr_tail'][:200]}")

    print(f"\n{'═'*60}\n")

    if not args.dry_run:
        # Save run log
        import json
        log_prefix = "local_health" if args.health_only else "daily_run"
        log_path = LOGS / f"{log_prefix}_{now.strftime('%Y%m%d')}.json"
        run_log = {
            "timestamp": now.isoformat(),
            "mode": "health_only" if args.health_only else "daily_run",
            "steps_total": len(results),
            "steps_ok": ok,
            "steps_no_action": no_action,
            "steps_failed": failed + errors + blocked,
            "total_elapsed_seconds": round(total_time, 1),
            "results": results,
        }
        atomic_write_json(run_log, log_path)
        print(f"  Run log saved → {log_path}")

    # Send notifications if any step failed
    # PLAIN ENGLISH: Send a desktop and Telegram notification so you
    # know something broke without having to check the logs.
    if not args.dry_run:
        notify_failures(results, total_time)

        # ── Send a daily summary notification (even on success) ──────────
        # PLAIN ENGLISH: Even when everything works, you want a quick
        # "heartbeat" message confirming the pipeline ran.  This way,
        # silence = something is wrong (cron died, machine off, etc.)
        try:
            from notifications import notify_info
            status_emoji = "✓" if not (failed or errors or blocked) else "⚠"
            notify_info(
                f"{status_emoji} Daily run complete: "
                f"{ok + no_action}/{len(results)} completed, {total_time:.0f}s. "
                f"{now.strftime('%Y-%m-%d %H:%M')}"
            )
        except Exception:
            pass

        # ── Send signal files to Telegram ────────────────────────────────
        # PLAIN ENGLISH: After a successful run, push the signal CSVs
        # (target weights + orders) to your Telegram chat so you can review
        # them on your phone without logging into GitHub.  Only sends when
        # the signal step actually succeeded (no point sending stale files).
        if not (failed or errors or blocked):
            try:
                from notifications import send_signal_summary_telegram
                send_signal_summary_telegram()
            except Exception as exc:
                print(f"  ⚠ Telegram signal delivery failed: {exc}")

    # Exit with failure code if any step failed
    if failed or errors or blocked:
        sys.exit(1)


if __name__ == "__main__":
    main()
