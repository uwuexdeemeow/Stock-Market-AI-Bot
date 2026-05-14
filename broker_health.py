"""Broker connectivity health check.

PLAIN ENGLISH: This script pings both Alpaca and Moomoo to make sure they
are reachable BEFORE the daily pipeline tries to submit orders.  If a
broker is down, you get a Telegram alert immediately instead of waiting
for the order submission to time out 5 minutes later.

HOW TO RUN:
  python3 broker_health.py              # check both brokers
  python3 broker_health.py --alpaca     # check Alpaca only
  python3 broker_health.py --moomoo     # check Moomoo only
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


def check_moomoo() -> dict:
    """Ping Moomoo OpenD paper account.

    PLAIN ENGLISH: Try to connect to the Moomoo OpenD gateway (which must
    be running on your machine) and fetch account equity.  If OpenD is not
    running, the connection will fail.
    Returns a dict with: broker, healthy (bool), equity, latency_ms, error.
    """
    result = {
        "broker": "moomoo",
        "healthy": False,
        "equity": None,
        "latency_ms": None,
        "error": None,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ctx = None
    try:
        start = time.monotonic()
        from moomoo_paper_trading import (
            connect_moomoo_trade_context,
            fetch_moomoo_positions_and_prices,
        )
        ctx = connect_moomoo_trade_context()

        # Try to fetch positions as a connectivity test
        from moomoo import TrdEnv
        ret, data = ctx.accinfo_query(trd_env=TrdEnv.SIMULATE, refresh_cache=True)
        latency = (time.monotonic() - start) * 1000

        if ret == 0 and data is not None and not data.empty:
            # Try to extract equity from account info
            for col in ("total_assets", "net_assets", "total_equity", "equity"):
                if col in data.columns:
                    val = data[col].iloc[0]
                    if val and float(val) > 0:
                        result["equity"] = round(float(val), 2)
                        break
            result["healthy"] = True
        else:
            result["healthy"] = True  # connection worked, just no data
        result["latency_ms"] = round(latency, 1)
    except ImportError as exc:
        result["error"] = f"moomoo-api not installed: {exc}"
    except ConnectionRefusedError:
        result["error"] = "Moomoo OpenD not running — start OpenD app first"
    except Exception as exc:
        result["error"] = str(exc)[:300]
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass
    return result


def check_all(*, alpaca: bool = True, moomoo: bool = True) -> dict:
    """Run health checks for requested brokers.

    PLAIN ENGLISH: Pings each broker, collects the results, and determines
    if any broker is down.  If a broker is unreachable, sends a Telegram
    alert immediately.

    Returns a summary dict with per-broker results and overall status.
    """
    results = {}
    if alpaca:
        results["alpaca"] = check_alpaca()
    if moomoo:
        results["moomoo"] = check_moomoo()

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
    HEALTH_FILE.write_text(json.dumps(summary, indent=2))

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
        description="Pre-flight broker connectivity check. "
        "Pings Alpaca and Moomoo to verify they are reachable."
    )
    parser.add_argument("--alpaca", action="store_true",
                        help="Check Alpaca only")
    parser.add_argument("--moomoo", action="store_true",
                        help="Check Moomoo only")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    # If neither flag set, check both
    do_alpaca = args.alpaca or (not args.alpaca and not args.moomoo)
    do_moomoo = args.moomoo or (not args.alpaca and not args.moomoo)

    summary = check_all(alpaca=do_alpaca, moomoo=do_moomoo)

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
