"""
fill_monitor.py — Check if yesterday's orders filled successfully.

PLAIN ENGLISH: After the sell-wait-buy fix, buy orders go out after sells
settle.  But sometimes orders still don't fill — maybe the limit price was
too tight, the stock was halted, or the broker rejected it.

This script reads the paper_trades.csv log and checks if any recent orders
are still unfilled (cancelled, partial, or missing).  If it finds problems,
it sends a macOS notification so you know to investigate.

Usage:
    python3 fill_monitor.py              # check recent fills
    python3 fill_monitor.py --days 3     # look back 3 days instead of 1
    python3 fill_monitor.py --quiet      # only print if there are problems

When to run:
    - Morning after trading (next day), to confirm yesterday's orders filled
    - Automatically via daily_run.py (runs before new signals/orders)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from settings import SIGNAL_DIR, LOG_DIR

SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)

PAPER_TRADES_FILE = SIGNALS / "paper_trades.csv"
FILL_MONITOR_LOG = SIGNALS / "fill_monitor.json"


def _send_fill_alert(title: str, message: str) -> None:
    """Send fill issue alert via all configured channels (macOS + Telegram + email)."""
    from notifications import send_alert
    send_alert(message, title=title, priority="warning")


def _parse_timestamp(value: object) -> pd.Timestamp | None:
    """
    Try to parse a timestamp string into a pandas Timestamp.

    PLAIN ENGLISH: Trade log timestamps come in various formats.
    This tries a few common patterns and returns None if none work.
    """
    if pd.isna(value) or str(value).strip() in ("", "nan", "None"):
        return None
    try:
        return pd.Timestamp(str(value))
    except Exception:
        return None


def check_recent_fills(*, lookback_days: int = 1, quiet: bool = False) -> dict:
    """
    Check paper_trades.csv for unfilled orders from recent days.

    PLAIN ENGLISH: Reads the trade log, filters to orders submitted in the
    last N days, and checks each one's fill_status.  Returns a summary
    with counts and details of any problems.

    Args:
        lookback_days: How many days back to check (default: 1 = yesterday)
        quiet: If True, only print if there are problems

    Returns:
        dict with keys: total_checked, filled, unfilled, cancelled, partial,
                       unknown, problems (list of problem details)
    """
    if not PAPER_TRADES_FILE.exists():
        if not quiet:
            print(f"  No paper_trades.csv found — nothing to check")
        return {"total_checked": 0, "problems": []}

    trades = pd.read_csv(PAPER_TRADES_FILE)
    if trades.empty:
        if not quiet:
            print(f"  paper_trades.csv is empty — nothing to check")
        return {"total_checked": 0, "problems": []}

    # Filter to recent trades only
    # PLAIN ENGLISH: We only care about orders from the last N days.
    # Older orders were already checked on previous runs.
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)

    # Try multiple timestamp columns (different parts of the pipeline use different names)
    ts_col = None
    for col in ("submitted_at", "predicted_at", "reconciled_at", "timestamp"):
        if col in trades.columns:
            ts_col = col
            break

    if ts_col is None:
        if not quiet:
            print(f"  No timestamp column found in paper_trades.csv — checking all rows")
        recent = trades
    else:
        parsed = trades[ts_col].apply(_parse_timestamp)
        # Make cutoff tz-naive for comparison if timestamps are tz-naive
        cutoff_naive = cutoff.tz_localize(None)
        has_ts = parsed.notna()
        # Compare tz-aware and tz-naive safely
        recent_mask = has_ts.copy()
        for i in recent_mask.index:
            if has_ts[i]:
                ts = parsed[i]
                try:
                    if ts.tzinfo is not None:
                        recent_mask[i] = ts >= cutoff
                    else:
                        recent_mask[i] = ts >= cutoff_naive
                except Exception:
                    recent_mask[i] = True  # Include if we can't compare
            else:
                recent_mask[i] = False
        recent = trades[recent_mask]

    if recent.empty:
        if not quiet:
            print(f"  No orders found in the last {lookback_days} day(s)")
        return {"total_checked": 0, "problems": []}

    # Only check BUY and SELL orders (not HOLD, SKIP, etc.)
    actionable = recent[recent["action"].isin(["BUY", "SELL"])] if "action" in recent.columns else recent
    if actionable.empty:
        if not quiet:
            print(f"  No actionable orders (BUY/SELL) in the last {lookback_days} day(s)")
        return {"total_checked": 0, "problems": []}

    # Count fill statuses
    fill_col = "fill_status" if "fill_status" in actionable.columns else None
    problems: list[dict] = []

    filled = 0
    cancelled = 0
    partial = 0
    unknown = 0
    not_submitted = 0

    for _, row in actionable.iterrows():
        ticker = str(row.get("ticker", "?"))
        action = str(row.get("action", "?"))
        status = str(row.get(fill_col, "unknown") if fill_col else "unknown").lower().strip()

        if status in ("filled",):
            filled += 1
        elif status in ("cancelled", "canceled", "failed", "rejected"):
            cancelled += 1
            problems.append({
                "ticker": ticker,
                "action": action,
                "status": status,
                "shares": row.get("delta_shares", row.get("broker_qty", "?")),
                "limit_price": row.get("limit_price", "?"),
                "reason": "Order was cancelled/rejected by broker",
            })
        elif status in ("partial",):
            partial += 1
            problems.append({
                "ticker": ticker,
                "action": action,
                "status": status,
                "shares": row.get("delta_shares", row.get("broker_qty", "?")),
                "dealt_qty": row.get("broker_dealt_qty", "?"),
                "reason": "Order only partially filled",
            })
        elif status in ("open", "pending", "submitting"):
            # Still open — might fill later, but flag it
            problems.append({
                "ticker": ticker,
                "action": action,
                "status": status,
                "shares": row.get("delta_shares", row.get("broker_qty", "?")),
                "reason": "Order still open/pending — may not have filled",
            })
            unknown += 1
        elif status in ("not_submitted_or_missing_order_id", "missing_broker_order"):
            not_submitted += 1
            problems.append({
                "ticker": ticker,
                "action": action,
                "status": status,
                "reason": "Order was not submitted or broker lost track of it",
            })
        else:
            unknown += 1

    total = len(actionable)

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback_days": lookback_days,
        "total_checked": total,
        "filled": filled,
        "cancelled": cancelled,
        "partial": partial,
        "unknown": unknown,
        "not_submitted": not_submitted,
        "problems": problems,
        "fill_rate": round(filled / total * 100, 1) if total > 0 else 0,
    }

    return result


def print_fill_report(result: dict, *, quiet: bool = False) -> None:
    """
    Print a summary of the fill check and send alerts if needed.

    PLAIN ENGLISH: Shows how many orders filled vs. failed, and sends
    a macOS notification if there are any problems.
    """
    total = result["total_checked"]
    if total == 0:
        return

    problems = result.get("problems", [])

    if not quiet or problems:
        print(f"\n  Fill Verification ({result['lookback_days']}-day lookback)")
        print(f"  {'─'*40}")
        print(f"  Total orders checked: {total}")
        print(f"  ✓ Filled:       {result['filled']}")
        if result["cancelled"]:
            print(f"  ✗ Cancelled:    {result['cancelled']}")
        if result["partial"]:
            print(f"  ⚠ Partial:      {result['partial']}")
        if result["not_submitted"]:
            print(f"  ⚠ Not submitted: {result['not_submitted']}")
        if result["unknown"]:
            print(f"  ? Unknown:      {result['unknown']}")
        print(f"  Fill rate:      {result['fill_rate']}%")

    if problems:
        print(f"\n  ⚠ PROBLEMS DETECTED:")
        for p in problems:
            print(f"    [{p['action']}] {p['ticker']}: {p['status']} — {p['reason']}")

        # Send macOS notification
        problem_tickers = ", ".join(p["ticker"] for p in problems[:5])
        _send_fill_alert(
            f"Fill Issues: {len(problems)} order(s)",
            f"Problems with: {problem_tickers}. Fill rate: {result['fill_rate']}%",
        )
    elif not quiet:
        print(f"\n  ✓ All orders filled successfully!")

    # Save the report to disk for reference
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = SIGNALS / "fill_monitor.json"
    log_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check if recent orders filled successfully"
    )
    parser.add_argument("--days", type=int, default=1,
                        help="How many days back to check (default: 1)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print if there are problems")
    args = parser.parse_args()

    if not args.quiet:
        print(f"\n{'='*50}")
        print(f"  FILL MONITOR")
        print(f"{'='*50}")

    result = check_recent_fills(lookback_days=args.days, quiet=args.quiet)
    print_fill_report(result, quiet=args.quiet)

    if not args.quiet:
        print()

    # Exit with failure code if there are unfilled orders
    if result.get("problems"):
        sys.exit(1)


if __name__ == "__main__":
    main()
