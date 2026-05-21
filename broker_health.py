"""Broker connectivity health check.

PLAIN ENGLISH: This script pings Alpaca to make sure it is reachable
BEFORE the daily pipeline tries to submit orders.  If the broker is down, you
broker is down, you get a Telegram alert immediately instead of waiting
for the order submission to time out 5 minutes later.

HOW TO RUN:
  python3 broker_health.py              # check Alpaca
  python3 broker_health.py --alpaca     # explicit Alpaca check
  python3 broker_health.py --json       # output results as JSON

KEY CONCEPTS:
  - Pre-flight check: a quick test that runs BEFORE the main task to
    catch problems early.
  - Health check: a minimal API call (get account equity) that proves
    the broker is reachable and our credentials work.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from safe_io import atomic_write_json
from settings import SIGNAL_DIR

SIGNALS = Path(SIGNAL_DIR)
HEALTH_FILE = SIGNALS / "broker_health.json"


def check_alpaca() -> dict:
    """Ping Alpaca paper account.

    PLAIN ENGLISH: Try to connect to Alpaca and fetch account equity.
    If it works, the broker is healthy.  If it fails, we capture the
    error so we can report it.
    Returns a dict with: broker, healthy (bool), equity, latency_ms, error.
    """
    result = {
        "broker": "alpaca",
        "healthy": False,
        "equity": None,
        "latency_ms": None,
        "error": None,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        start = time.monotonic()
        from alpaca_paper_trading import AlpacaBroker
        broker = AlpacaBroker()
        equity = broker.get_equity()
        latency = (time.monotonic() - start) * 1000
        result["healthy"] = True
        result["equity"] = round(equity, 2)
        result["latency_ms"] = round(latency, 1)
    except ImportError as exc:
        result["error"] = f"alpaca-trade-api not installed: {exc}"
    except Exception as exc:
        result["error"] = str(exc)[:300]
    return result


def check_all(*, alpaca: bool = True) -> dict:
    """Run the Alpaca health check when requested.

    PLAIN ENGLISH: Pings each broker, collects the results, and determines
    if any broker is down.  If a broker is unreachable, sends a Telegram
    alert immediately.

    Returns a summary dict with per-broker results and overall status.
    """
    results = {}
    if alpaca:
        results["alpaca"] = check_alpaca()
    all_healthy = all(r["healthy"] for r in results.values())
    down_brokers = [name for name, r in results.items() if not r["healthy"]]

    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "all_healthy": all_healthy,
        "down_brokers": down_brokers,
        "brokers": results,
    }

    # Save results to disk for other scripts to read
    SIGNALS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary, HEALTH_FILE)

    # Send alert if any broker is down
    if down_brokers:
        try:
            from notifications import send_alert
            errors = []
            for name in down_brokers:
                err = results[name].get("error", "unknown error")
                errors.append(f"• {name}: {err}")
            send_alert(
                f"Broker(s) DOWN: {', '.join(down_brokers)}\n" + "\n".join(errors),
                title="Broker Health",
                priority="critical",
            )
        except Exception:
            pass

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight Alpaca connectivity check."
    )
    parser.add_argument("--alpaca", action="store_true",
                        help="Check Alpaca only")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    summary = check_all(alpaca=True)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("Broker Health Check")
        print("─" * 40)
        for name, result in summary["brokers"].items():
            status = "✓ HEALTHY" if result["healthy"] else "✗ DOWN"
            latency = f"{result['latency_ms']:.0f}ms" if result["latency_ms"] else "n/a"
            equity = f"${result['equity']:,.2f}" if result["equity"] else "n/a"
            print(f"  {name:8s}  {status}  latency={latency}  equity={equity}")
            if result["error"]:
                print(f"           error: {result['error']}")
        print(f"\n  Overall: {'ALL HEALTHY' if summary['all_healthy'] else 'DEGRADED — ' + ', '.join(summary['down_brokers']) + ' DOWN'}")
        print(f"  Saved → {HEALTH_FILE}")


if __name__ == "__main__":
    main()
