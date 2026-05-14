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
    1.  refresh_etf_data.py --refresh      → download latest ETF price data
    2.  research.py                        → refresh factor panel (stock prices + factor scores)
    3.  fill_monitor.py --days 2           → verify yesterday's fills before placing new orders
    4.  broker_health.py                   → pre-flight ping of Alpaca + Moomoo (alerts if down)
    5.  core_satellite_alpha.py            → generate unified signal (both brokers)
    6.  moomoo_paper_trading.py --submit   → submit to Moomoo (auto-syncs fills, trails stops)
    7.  moomoo_paper_trading.py --status   → sync equity/positions, save daily status
    8.  moomoo_paper_trading.py --execution-guard → repair Moomoo ETF stops
    9.  paper_health.py                    → build deep health summary (slippage, drift, risk)
    10. paper_gauntlet.py                  → check Moomoo gauntlet gates
    11. daily_paper_check.py --skip-status --skip-sync  → read-only verdict
    12. alpaca_paper_trading.py --submit   → submit to Alpaca (reads same signal as Moomoo)
    13. alpaca_paper_trading.py --reconcile → check if Alpaca orders filled
    14. execution_guard.py --once          → repair ETF stops, stale orders, P&L guard
    15. alpaca_paper_gauntlet.py           → check Alpaca health
    16. regime_monitor.py                  → detect and alert on regime changes
    17. paper_report.py                    → side-by-side strategy comparison (optional, --report)

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

from settings import LOG_DIR

LOGS = Path(LOG_DIR)


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


# Data refresh steps — run BEFORE signal generation so factors/ETFs are fresh.
# PLAIN ENGLISH: These scripts download the latest stock and ETF data from the
# internet and save it to disk.  Without fresh data, the signal generator would
# use stale prices and factor scores, which defeats the purpose of daily trading.
DATA_REFRESH_STEPS = [
    Step(
        "refresh_etf_data",
        [sys.executable, "refresh_etf_data.py", "--refresh"],
        "Download latest ETF price data (SPY, QQQ, TQQQ, etc.)",
        critical=True,
    ),
    Step(
        "refresh_factor_data",
        [sys.executable, "research.py"],
        "Refresh factor panel data (download stock prices, compute factor scores)",
        critical=True,
    ),
]

# Fill verification — runs BEFORE new signals/orders to check yesterday's fills.
# PLAIN ENGLISH: Before submitting new orders, check if yesterday's orders
# actually filled.  If something got cancelled or partially filled, you want
# to know BEFORE placing new trades.
FILL_MONITOR_STEP = Step(
    "fill_monitor",
    [sys.executable, "fill_monitor.py", "--days", "2"],
    "Verify recent order fills (check for cancelled/partial orders)",
    critical=True,
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
        "Submit orders to Alpaca paper trading (auto-snapshots equity)",
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
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent),
        )

        elapsed = (datetime.now() - start).total_seconds()

        # Print stdout (last 20 lines to keep it readable)
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 20:
                print(f"  ... ({len(lines) - 20} lines omitted)")
            for line in lines[-20:]:
                print(f"  {line}")

        if result.returncode != 0:
            print(f"  ✗ FAILED (exit code {result.returncode})")
            if result.stderr:
                # Show last 10 lines of stderr
                err_lines = result.stderr.strip().split("\n")
                for line in err_lines[-10:]:
                    print(f"  ERR: {line}")
            return {
                "name": name,
                "status": "failed",
                "exit_code": result.returncode,
                "elapsed": round(elapsed, 1),
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            }

        print(f"  ✓ OK ({elapsed:.1f}s)")
        return {"name": name, "status": "ok", "elapsed": round(elapsed, 1)}

    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  ✗ TIMEOUT after {timeout}s")
        return {"name": name, "status": "timeout", "elapsed": round(elapsed, 1)}

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  ✗ ERROR: {e}")
        return {"name": name, "status": "error", "elapsed": round(elapsed, 1), "error": str(e)}


def build_steps(
    *,
    skip_refresh: bool,
    run_moomoo: bool,
    run_alpaca: bool,
    stress: bool = False,
    report: bool = False,
) -> list[Step]:
    steps: list[Step] = []
    if not skip_refresh:
        steps.extend(DATA_REFRESH_STEPS)
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
        steps.append(CORE_SATELLITE_SIGNAL_STEP)
    if run_moomoo:
        steps.extend(MOOMOO_STEPS)
    if run_alpaca:
        steps.extend(ALPACA_STEPS)
    steps.append(REGIME_MONITOR_STEP)
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
        if blocked_by:
            print(f"\n{'─'*60}")
            print(f"  [{step.name}] BLOCKED by critical failure: {blocked_by}")
            results.append({
                "name": step.name,
                "status": "blocked",
                "blocked_by": blocked_by,
                "elapsed": 0.0,
            })
            continue
        result = run_step(step.name, step.cmd, step.description, dry_run=dry_run, timeout=timeout)
        results.append(result)
        if step.critical and result["status"] in ("failed", "error", "timeout"):
            blocked_by = step.name
    return results


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
    parser.add_argument("--force", action="store_true",
                        help="Run even on weekends and US market holidays")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Max seconds per step (default: 300)")
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)

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

    # Determine which steps to run
    # If neither --moomoo nor --alpaca specified, run both
    run_moomoo = args.moomoo or (not args.moomoo and not args.alpaca)
    run_alpaca = args.alpaca or (not args.moomoo and not args.alpaca)

    steps = build_steps(
        skip_refresh=bool(args.skip_refresh),
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
    print(f"{'═'*60}")

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
    log_path.write_text(json.dumps(run_log, indent=2, default=str), encoding="utf-8")
    print(f"  Run log saved → {log_path}")

    # Send notifications if any step failed
    # PLAIN ENGLISH: Pop a macOS notification and optionally email you so you
    # know something broke without having to check the logs.
    if not args.dry_run:
        notify_failures(results, total_time)

    # Exit with failure code if any step failed
    if failed or errors or blocked:
        sys.exit(1)


if __name__ == "__main__":
    main()
