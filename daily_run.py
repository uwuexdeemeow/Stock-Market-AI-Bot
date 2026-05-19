"""
daily_run.py — Run all paper trading steps in one command.

PLAIN ENGLISH: Instead of running 6+ commands every trading day, this script
chains them all together. Critical upstream failures stop later broker
submission steps so stale signals do not get traded.

Usage:
    python3 daily_run.py              # run everything
    python3 daily_run.py --dry-run    # show what would run without executing
    python3 daily_run.py --moomoo     # only run Moomoo steps
    python3 daily_run.py --alpaca     # only run Alpaca steps
    python3 daily_run.py --report     # also run the side-by-side performance report
    python3 daily_run.py --stress     # also run stress tests (decay, drawdown, execution, survivorship)

Daily workflow (runs in order):
    1.  refresh_etf_data.py --refresh --force → download latest ETF price data
    2.  research.py --incremental          → refresh factor panel (optional in CI)
    3.  feature_quality_diagnostic.py      → refresh live feature quality report (optional in CI)
    4.  fill_monitor.py --days 2           → verify yesterday's fills before placing new orders
    5.  broker_health.py                   → pre-flight ping of Alpaca + Moomoo (alerts if down)
    6.  core_satellite_alpha.py            → generate unified signal (both brokers)
    7.  moomoo_paper_trading.py --submit   → submit to Moomoo (auto-syncs fills, trails stops)
    8.  moomoo_paper_trading.py --status   → sync equity/positions, save daily status
    9.  moomoo_paper_trading.py --execution-guard → repair Moomoo ETF stops
    10. paper_health.py                    → build deep health summary (slippage, drift, risk)
    11. paper_gauntlet.py                  → check Moomoo gauntlet gates
    12. daily_paper_check.py --skip-status --skip-sync  → read-only verdict
    13. alpaca_paper_trading.py --submit   → submit to Alpaca (reads same signal as Moomoo)
    14. alpaca_paper_trading.py --reconcile → check if Alpaca orders filled
    15. execution_guard.py --once          → repair ETF stops, stale orders, P&L guard
    16. alpaca_paper_gauntlet.py           → check Alpaca health
    17. regime_monitor.py                  → detect and alert on regime changes
    18. paper_report.py                    → side-by-side strategy comparison (optional, --report)

Schedule with cron (9:30 AM ET on weekdays):
    30 9 * * 1-5 cd "/path/to/Stock Market AI Bot" && python3 daily_run.py >> logs/daily_run.log 2>&1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from safe_io import atomic_write_json
from settings import LOG_DIR, SIGNAL_DIR

LOGS = Path(LOG_DIR)


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
    If yes, it sends alerts via macOS, email, AND Telegram (if configured)
    through the shared notifications module.
    If everything passed, it stays quiet — no spam on good days.
    """
    failures = [r for r in results if r["status"] not in ("ok", "skipped")]
    if not failures:
        return

    from notifications import send_alert

    # Build a short summary for notifications
    failed_names = ", ".join(r["name"] for r in failures)
    ok_count = sum(1 for r in results if r["status"] == "ok")
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


# Data refresh steps — run BEFORE signal generation so factors/ETFs are fresh.
# PLAIN ENGLISH: These scripts download the latest stock and ETF data from the
# internet and save it to disk.  Without fresh data, the signal generator would
# use stale prices and factor scores, which defeats the purpose of daily trading.
ETF_REFRESH_STEP = Step(
    "refresh_etf_data",
    [sys.executable, "refresh_etf_data.py", "--refresh", "--force"],
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
    "Pre-flight broker connectivity check (Alpaca + Moomoo)",
    critical=False,  # don't block pipeline — just alert if a broker is down
)

CORE_SATELLITE_SIGNAL_STEP = Step(
    "core_satellite_signal",
    [sys.executable, "core_satellite_alpha.py"],
    "Generate unified core-satellite signal for all brokers",
    critical=True,
)

FACTOR_DATA_HEALTH_STEP = Step(
    "factor_data_health",
    [sys.executable, "factor_data_health.py", "--strict"],
    "Validate restored factor data cache before signal generation",
    critical=True,
)

# Steps for Moomoo core-satellite strategy
MOOMOO_STEPS = [
    Step(
        "moomoo_submit",
        [sys.executable, "moomoo_paper_trading.py", "--submit"],
        "Submit orders to Moomoo paper trading (auto-syncs fills)",
    ),
    Step(
        "moomoo_status",
        [sys.executable, "moomoo_paper_trading.py", "--status"],
        "Sync Moomoo equity/positions and save daily status",
    ),
    Step(
        "moomoo_execution_guard",
        [sys.executable, "moomoo_paper_trading.py", "--execution-guard"],
        "Repair Moomoo core ETF stop-limit protection",
    ),
    Step(
        "moomoo_health",
        [sys.executable, "paper_health.py"],
        "Build deep Moomoo health summary (slippage, concentration, equity risk, P&L)",
    ),
    Step(
        "moomoo_gauntlet",
        [sys.executable, "paper_gauntlet.py"],
        "Run Moomoo paper gauntlet health check",
    ),
    Step(
        "moomoo_daily_check",
        [sys.executable, "daily_paper_check.py", "--skip-status", "--skip-sync"],
        "Read-only verdict check (status/sync already done above)",
    ),
]

# Steps for Alpaca strategy (reads the same unified signal as Moomoo —
# core_satellite_alpha_signal.csv includes TQQQ weight when the nested
# walkforward grid search determines it helps on a risk-adjusted basis).
ALPACA_STEPS = [
    Step(
        "alpaca_submit",
        [sys.executable, "alpaca_paper_trading.py", "--submit"],
        "Submit orders to Alpaca paper trading (auto-snapshots equity + status)",
    ),
    Step(
        "alpaca_reconcile",
        [sys.executable, "alpaca_paper_trading.py", "--reconcile"],
        "Reconcile Alpaca order fill statuses",
    ),
    Step(
        "alpaca_execution_guard",
        [sys.executable, "execution_guard.py", "--once"],
        "Repair ETF protection, cancel stale Alpaca orders, and check intraday P&L",
    ),
    Step(
        "alpaca_paper_health",
        [sys.executable, "paper_health.py", "--broker", "alpaca"],
        "Build deep health summary for Alpaca (slippage, drift vs walkforward, risk)",
    ),
    Step(
        "alpaca_gauntlet",
        [sys.executable, "alpaca_paper_gauntlet.py"],
        "Run Alpaca paper gauntlet health check",
    ),
]

# Regime change monitor — runs after BOTH strategies generate signals.
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


def run_step(
    name: str,
    cmd: list[str],
    description: str,
    dry_run: bool = False,
    timeout: int = 300,
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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            cwd=str(Path(__file__).parent),
        )
        # Use a background thread for stderr so we don't deadlock when one
        # stream fills its OS pipe before the other gets drained.
        import threading

        def _drain(stream, sink: list[str], prefix: str):
            for line in stream:
                line = line.rstrip()
                sink.append(line)
                if len(sink) > TAIL_SIZE:
                    sink.pop(0)
                print(f"  {prefix}{line}", flush=True)

        err_thread = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_tail, "ERR: "), daemon=True
        )
        err_thread.start()
        _drain(proc.stdout, stdout_tail, "")
        err_thread.join(timeout=2.0)

        try:
            proc.wait(timeout=timeout - (datetime.now() - start).total_seconds())
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed = (datetime.now() - start).total_seconds()
            print(f"  ✗ TIMEOUT after {timeout}s")
            return {"name": name, "status": "timeout", "elapsed": round(elapsed, 1)}

        elapsed = (datetime.now() - start).total_seconds()
        returncode = proc.returncode

        if returncode != 0:
            print(f"  ✗ FAILED (exit code {returncode})")
            return {
                "name": name,
                "status": "failed",
                "exit_code": returncode,
                "elapsed": round(elapsed, 1),
                "stderr_tail": "\n".join(stderr_tail[-10:]),
            }

        print(f"  ✓ OK ({elapsed:.1f}s)")
        return {"name": name, "status": "ok", "elapsed": round(elapsed, 1)}

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  ✗ ERROR: {e}")
        return {"name": name, "status": "error", "elapsed": round(elapsed, 1), "error": str(e)}


def build_steps(
    *,
    skip_refresh: bool,
    skip_factor_refresh: bool = False,
    run_moomoo: bool,
    run_alpaca: bool,
    stress: bool = False,
    report: bool = False,
) -> list[Step]:
    steps: list[Step] = []
    if not skip_refresh:
        steps.append(ETF_REFRESH_STEP)
        if not skip_factor_refresh:
            steps.extend(FACTOR_REFRESH_STEPS)
    steps.append(FILL_MONITOR_STEP)
    if run_moomoo or run_alpaca:
        # Pass broker flags so broker_health.py only checks the brokers
        # we're actually using.  Without this, it imports moomoo_api even
        # when running Alpaca-only (breaks on CI where moomoo isn't installed).
        health_flags = []
        if run_alpaca:
            health_flags.append("--alpaca")
        if run_moomoo:
            health_flags.append("--moomoo")
        steps.append(Step(
            "broker_health",
            [sys.executable, "broker_health.py"] + health_flags,
            f"Pre-flight broker connectivity check ({', '.join(health_flags)})",
            critical=False,
        ))
        steps.append(FACTOR_DATA_HEALTH_STEP)
        steps.append(CORE_SATELLITE_SIGNAL_STEP)
    if run_moomoo:
        steps.extend(MOOMOO_STEPS)
    if run_alpaca:
        steps.extend(ALPACA_STEPS)
    steps.append(REGIME_MONITOR_STEP)
    # Watchdog + housekeeping — non-critical, run last
    steps.append(MONITOR_HEARTBEAT_STEP)
    steps.append(LOG_CLEANUP_STEP)
    if stress:
        steps.extend(STRESS_STEPS)
    if report:
        steps.append(Step(
            "performance_report",
            [sys.executable, "paper_report.py"],
            "Generate side-by-side Moomoo vs Alpaca performance report",
        ))
    return steps


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
        result = run_step(step.name, step.cmd, step.description, dry_run=dry_run, timeout=effective_timeout)
        if blocked_by and step.always_run:
            # Keep the original blocker visible in the JSON log while still
            # letting watchdog/cleanup style steps refresh their output files.
            result["upstream_blocked_by"] = blocked_by
        results.append(result)
        if step.critical and result["status"] in ("failed", "error", "timeout") and blocked_by is None:
            blocked_by = step.name
    return results


def _write_startup_stubs(now: datetime, steps: list[Step]) -> None:
    """Write fresh watchdog files before the first step runs.

    PLAIN ENGLISH: `monitor_heartbeat.py` runs before `daily_run.py` writes the
    final daily summary.  These tiny "pending/running" files prove the pipeline
    started, and they get overwritten later by the real reports.
    """
    # Write a "running" stub so monitor_heartbeat (which executes BEFORE the
    # final results are written at the end of main()) doesn't false-positive
    # with "daily_run: output file missing".  The stub is overwritten with
    # the real summary when the run completes.
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
    parser.add_argument("--moomoo", action="store_true",
                        help="Only run Moomoo steps")
    parser.add_argument("--alpaca", action="store_true",
                        help="Only run Alpaca steps")
    parser.add_argument("--report", action="store_true",
                        help="Also run the side-by-side performance report at the end")
    parser.add_argument("--stress", action="store_true",
                        help="Also run stress tests (factor decay, drawdown throttle, execution, survivorship)")
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
    if not args.dry_run and not args.force:
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
        if (args.alpaca or (not args.alpaca and not args.moomoo)):
            if not _os.environ.get("ALPACA_API_KEY", "").strip():
                _missing_keys.append("ALPACA_API_KEY")
            if not _os.environ.get("ALPACA_SECRET_KEY", "").strip():
                _missing_keys.append("ALPACA_SECRET_KEY")
        if _missing_keys:
            print(f"⛔ Missing required environment variables: {', '.join(_missing_keys)}")
            print("  Set them in your .env file or shell profile before running.")
            sys.exit(1)

    # Determine which steps to run
    # If neither --moomoo nor --alpaca specified, run both
    run_moomoo = args.moomoo or (not args.moomoo and not args.alpaca)
    run_alpaca = args.alpaca or (not args.moomoo and not args.alpaca)

    steps = build_steps(
        skip_refresh=bool(args.skip_refresh),
        skip_factor_refresh=bool(args.skip_factor_refresh),
        run_moomoo=bool(run_moomoo),
        run_alpaca=bool(run_alpaca),
        stress=bool(args.stress),
        report=bool(args.report),
    )

    # Header
    now = datetime.now()
    print(f"{'═'*60}")
    print(f"  DAILY PAPER TRADING RUN")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Steps: {len(steps)}")
    if run_moomoo:
        print(f"  Moomoo: core-satellite")
    if run_alpaca:
        print(f"  Alpaca: core-satellite unified signal")
    if args.dry_run:
        print(f"  ⚠ DRY RUN MODE — nothing will execute")
    if args.skip_factor_refresh and not args.skip_refresh:
        print("  Factor refresh: skipped — using latest restored factor data/report")
    print(f"{'═'*60}")

    _write_startup_stubs(now, steps)

    # Run each step
    results = run_steps(steps, dry_run=bool(args.dry_run), timeout=int(args.timeout))

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "failed")
    errors = sum(1 for r in results if r["status"] in ("error", "timeout"))
    blocked = sum(1 for r in results if r["status"] == "blocked")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    total_time = sum(r.get("elapsed", 0) for r in results)

    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  ✓ Passed:  {ok}/{len(results)}")
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
        if r["status"] not in ("ok", "skipped"):
            print(f"\n  FAILED: [{r['name']}] — {r['status']}")
            if r.get("stderr_tail"):
                print(f"    {r['stderr_tail'][:200]}")

    print(f"\n{'═'*60}\n")

    # Save run log
    import json
    log_path = LOGS / f"daily_run_{now.strftime('%Y%m%d')}.json"
    run_log = {
        "timestamp": now.isoformat(),
        "steps_total": len(results),
        "steps_ok": ok,
        "steps_failed": failed + errors + blocked,
        "total_elapsed_seconds": round(total_time, 1),
        "results": results,
    }
    atomic_write_json(run_log, log_path)
    print(f"  Run log saved → {log_path}")

    # Send notifications if any step failed
    # PLAIN ENGLISH: Pop a macOS notification and optionally email you so you
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
                f"{ok}/{len(results)} passed, {total_time:.0f}s. "
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
