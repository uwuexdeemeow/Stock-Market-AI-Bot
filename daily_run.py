"""
daily_run.py — Run all paper trading steps in one command.

PLAIN ENGLISH: Instead of running 6+ commands every trading day, this script
chains them all together.  If one step fails, it logs the error and continues
with the rest — one bad signal generation won't prevent the other strategy
from trading.

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
    3.  core_satellite_alpha.py            → generate Moomoo signal
    4.  moomoo_paper_trading.py --submit   → submit to Moomoo (auto-syncs fills)
    5.  moomoo_paper_trading.py --status   → sync equity/positions, save daily status
    6.  paper_health.py                    → build deep health summary (slippage, concentration, risk)
    7.  paper_gauntlet.py                  → check Moomoo gauntlet gates
    8.  daily_paper_check.py --skip-status --skip-sync  → read-only verdict (status/sync already done)
    9.  core_satellite_tqqq.py             → generate Alpaca TQQQ signal
    10. alpaca_paper_trading.py --submit   → submit to Alpaca (auto-snapshots equity)
    11. alpaca_paper_trading.py --reconcile → check if Alpaca orders filled
    12. alpaca_paper_gauntlet.py           → check Alpaca health
    13. paper_report.py                    → side-by-side strategy comparison (optional, --report)

Schedule with cron (9:30 AM ET on weekdays):
    30 9 * * 1-5 cd "/path/to/Stock Market AI Bot" && python3 daily_run.py >> logs/daily_run.log 2>&1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from settings import LOG_DIR

LOGS = Path(LOG_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION HELPERS — alert the user when something goes wrong
# ─────────────────────────────────────────────────────────────────────────────

def _macos_notification(title: str, message: str) -> None:
    """
    Show a native macOS notification banner.

    PLAIN ENGLISH: Uses the built-in AppleScript command to pop a
    notification on your Mac.  Works without any extra packages.
    If it fails (e.g. running on Linux), it just prints a warning.
    """
    try:
        escaped_title = title.replace('"', '\\"')
        escaped_msg = message.replace('"', '\\"')
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "{escaped_msg}" with title "{escaped_title}"',
            ],
            capture_output=True,
            timeout=5,
        )
    except Exception as e:
        print(f"  ⚠ Could not send macOS notification: {e}")


def _send_failure_email(subject: str, body: str) -> None:
    """
    Send a failure alert email via SMTP (optional).

    PLAIN ENGLISH: If you've set up SMTP credentials in environment variables,
    this sends you an email when the daily run has failures.  If no credentials
    are set, it silently skips — no crash.

    Required env vars (all optional — if missing, email is skipped):
        SMTP_HOST     — e.g. smtp.gmail.com
        SMTP_PORT     — e.g. 587
        SMTP_USER     — your email login
        SMTP_PASSWORD — your email password or app password
        ALERT_EMAIL   — where to send alerts (defaults to SMTP_USER)
    """
    import os as _os
    smtp_host = _os.environ.get("SMTP_HOST", "")
    smtp_port = _os.environ.get("SMTP_PORT", "587")
    smtp_user = _os.environ.get("SMTP_USER", "")
    smtp_pass = _os.environ.get("SMTP_PASSWORD", "")
    alert_to = _os.environ.get("ALERT_EMAIL", smtp_user)

    if not all([smtp_host, smtp_user, smtp_pass, alert_to]):
        # No SMTP configured — skip silently
        return

    try:
        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = alert_to

        with smtplib.SMTP(smtp_host, int(smtp_port), timeout=15) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [alert_to], msg.as_string())

        print(f"  📧 Alert email sent to {alert_to}")
    except Exception as e:
        print(f"  ⚠ Could not send alert email: {e}")


def notify_failures(results: list[dict], total_time: float) -> None:
    """
    Send notifications if any step failed.

    PLAIN ENGLISH: After all steps finish, this checks if anything broke.
    If yes, it sends a macOS notification banner AND an email (if configured).
    If everything passed, it stays quiet — no spam on good days.
    """
    failures = [r for r in results if r["status"] not in ("ok", "skipped")]
    if not failures:
        return

    # Build a short summary for notifications
    failed_names = ", ".join(r["name"] for r in failures)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    total = len(results)

    title = f"⚠ Daily Run: {len(failures)} step(s) failed"
    short_msg = f"Failed: {failed_names} ({ok_count}/{total} passed)"

    # Longer message for email with error details
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

    email_body = "\n".join(lines)

    # Send both notification types
    _macos_notification(title, short_msg)
    _send_failure_email(title, email_body)


# ─────────────────────────────────────────────────────────────────────────────
# STEP DEFINITIONS — each step is a (name, command, description) tuple
# ─────────────────────────────────────────────────────────────────────────────

# Data refresh steps — run BEFORE signal generation so factors/ETFs are fresh.
# PLAIN ENGLISH: These scripts download the latest stock and ETF data from the
# internet and save it to disk.  Without fresh data, the signal generator would
# use stale prices and factor scores, which defeats the purpose of daily trading.
DATA_REFRESH_STEPS = [
    (
        "refresh_etf_data",
        [sys.executable, "refresh_etf_data.py", "--refresh"],
        "Download latest ETF price data (SPY, QQQ, TQQQ, etc.)",
    ),
    (
        "refresh_factor_data",
        [sys.executable, "research.py"],
        "Refresh factor panel data (download stock prices, compute factor scores)",
    ),
]

# Fill verification — runs BEFORE new signals/orders to check yesterday's fills.
# PLAIN ENGLISH: Before submitting new orders, check if yesterday's orders
# actually filled.  If something got cancelled or partially filled, you want
# to know BEFORE placing new trades.
FILL_MONITOR_STEP = (
    "fill_monitor",
    [sys.executable, "fill_monitor.py", "--days", "2"],
    "Verify recent order fills (check for cancelled/partial orders)",
)

# Steps for Moomoo core-satellite strategy
MOOMOO_STEPS = [
    (
        "moomoo_signal",
        [sys.executable, "core_satellite_alpha.py"],
        "Generate core-satellite signal for Moomoo",
    ),
    (
        "moomoo_submit",
        [sys.executable, "moomoo_paper_trading.py", "--submit"],
        "Submit orders to Moomoo paper trading (auto-syncs fills)",
    ),
    (
        "moomoo_status",
        [sys.executable, "moomoo_paper_trading.py", "--status"],
        "Sync Moomoo equity/positions and save daily status",
    ),
    (
        "moomoo_health",
        [sys.executable, "paper_health.py"],
        "Build deep Moomoo health summary (slippage, concentration, equity risk, P&L)",
    ),
    (
        "moomoo_gauntlet",
        [sys.executable, "paper_gauntlet.py"],
        "Run Moomoo paper gauntlet health check",
    ),
    (
        "moomoo_daily_check",
        [sys.executable, "daily_paper_check.py", "--skip-status", "--skip-sync"],
        "Read-only verdict check (status/sync already done above)",
    ),
]

# Steps for Alpaca TQQQ-enhanced strategy
ALPACA_STEPS = [
    (
        "alpaca_signal",
        [sys.executable, "core_satellite_tqqq.py"],
        "Generate TQQQ-enhanced signal for Alpaca",
    ),
    (
        "alpaca_submit",
        [sys.executable, "alpaca_paper_trading.py", "--submit"],
        "Submit orders to Alpaca paper trading (auto-snapshots equity)",
    ),
    (
        "alpaca_reconcile",
        [sys.executable, "alpaca_paper_trading.py", "--reconcile"],
        "Reconcile Alpaca order fill statuses",
    ),
    (
        "alpaca_gauntlet",
        [sys.executable, "alpaca_paper_gauntlet.py"],
        "Run Alpaca paper gauntlet health check",
    ),
]

# Regime change monitor — runs after BOTH strategies generate signals.
# PLAIN ENGLISH: Checks if the market regime (risk_on/neutral/risk_off)
# changed since yesterday and sends you a notification if it did.
REGIME_MONITOR_STEP = (
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
    (
        "factor_decay",
        [sys.executable, "factor_decay_monitor.py"],
        "Check if factor overlay IC and alpha are decaying",
    ),
    (
        "drawdown_throttle",
        [sys.executable, "core_satellite_drawdown_throttle.py"],
        "Stress test drawdown throttle scenarios",
    ),
    (
        "execution_stress",
        [sys.executable, "core_satellite_execution_stress.py"],
        "Stress test execution costs (delayed fills, extra slippage)",
    ),
    (
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


def _is_us_market_holiday(dt: datetime) -> bool:
    """
    Check if a date is a US stock market (NYSE) holiday.

    PLAIN ENGLISH: The NYSE closes on ~9 holidays per year.  This function
    checks if today matches one of them.  It handles the "observed" rule:
    if a holiday falls on Saturday, Friday is the observed day off; if on
    Sunday, Monday is the observed day off.

    Covers: New Year's, MLK Day, Presidents' Day, Good Friday,
    Memorial Day, Juneteenth, Independence Day, Labor Day,
    Thanksgiving, Christmas.
    """
    from datetime import date, timedelta

    year = dt.year
    d = dt.date() if hasattr(dt, "date") else dt

    def _observed(holiday: date) -> date:
        """Shift Saturday→Friday, Sunday→Monday."""
        if holiday.weekday() == 5:  # Saturday
            return holiday - timedelta(days=1)
        if holiday.weekday() == 6:  # Sunday
            return holiday + timedelta(days=1)
        return holiday

    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """Return the nth occurrence of a weekday in a month (1-indexed)."""
        first = date(year, month, 1)
        # Days until the target weekday
        days_ahead = (weekday - first.weekday()) % 7
        first_occurrence = first + timedelta(days=days_ahead)
        return first_occurrence + timedelta(weeks=n - 1)

    def _last_weekday(year: int, month: int, weekday: int) -> date:
        """Return the last occurrence of a weekday in a month."""
        # Start from the last day of the month
        if month == 12:
            last = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(year, month + 1, 1) - timedelta(days=1)
        days_back = (last.weekday() - weekday) % 7
        return last - timedelta(days=days_back)

    def _easter(year: int) -> date:
        """Compute Easter Sunday using the anonymous Gregorian algorithm."""
        a = year % 19
        b, c = divmod(year, 100)
        d, e = divmod(b, 4)
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = divmod(c, 4)
        l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    holidays = [
        _observed(date(year, 1, 1)),                      # New Year's Day
        _nth_weekday(year, 1, 0, 3),                      # MLK Day (3rd Monday Jan)
        _nth_weekday(year, 2, 0, 3),                      # Presidents' Day (3rd Monday Feb)
        _easter(year) - timedelta(days=2),                 # Good Friday
        _last_weekday(year, 5, 0),                         # Memorial Day (last Monday May)
        _observed(date(year, 6, 19)),                      # Juneteenth
        _observed(date(year, 7, 4)),                       # Independence Day
        _nth_weekday(year, 9, 0, 1),                       # Labor Day (1st Monday Sep)
        _nth_weekday(year, 11, 3, 4),                      # Thanksgiving (4th Thursday Nov)
        _observed(date(year, 12, 25)),                     # Christmas
    ]

    return d in holidays


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

    steps = []

    # Data refresh runs FIRST — both strategies need fresh data
    # PLAIN ENGLISH: Download latest prices and factor scores before generating
    # any signals.  Skip with --skip-refresh if you already refreshed today.
    if not args.skip_refresh:
        steps.extend(DATA_REFRESH_STEPS)

    # Fill verification — check yesterday's orders before submitting new ones
    # PLAIN ENGLISH: Before placing new trades, verify that yesterday's orders
    # filled.  If something got cancelled, you want to know first.
    steps.append(FILL_MONITOR_STEP)

    if run_moomoo:
        steps.extend(MOOMOO_STEPS)
    if run_alpaca:
        steps.extend(ALPACA_STEPS)

    # Regime monitor — runs after signal generation to detect regime shifts
    # PLAIN ENGLISH: After both strategies generate their signals, check if
    # the market regime changed (risk_on ↔ neutral ↔ risk_off) and alert you.
    steps.append(REGIME_MONITOR_STEP)

    # Optional: stress tests (factor decay, drawdown, execution, survivorship)
    if args.stress:
        steps.extend(STRESS_STEPS)

    # Optional: side-by-side performance report (needs both strategies' data)
    if args.report:
        steps.append((
            "performance_report",
            [sys.executable, "paper_report.py"],
            "Generate side-by-side Moomoo vs Alpaca performance report",
        ))

    # Header
    now = datetime.now()
    print(f"{'═'*60}")
    print(f"  DAILY PAPER TRADING RUN")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Steps: {len(steps)}")
    if run_moomoo:
        print(f"  Moomoo: core-satellite")
    if run_alpaca:
        print(f"  Alpaca: TQQQ-enhanced")
    if args.dry_run:
        print(f"  ⚠ DRY RUN MODE — nothing will execute")
    print(f"{'═'*60}")

    # Run each step
    results = []
    for name, cmd, desc in steps:
        result = run_step(name, cmd, desc, dry_run=args.dry_run, timeout=args.timeout)
        results.append(result)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "failed")
    errors = sum(1 for r in results if r["status"] in ("error", "timeout"))
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
        "steps_failed": failed + errors,
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
    if failed or errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
