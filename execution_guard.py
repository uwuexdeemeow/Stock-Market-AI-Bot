"""
execution_guard.py - Alpaca execution safety checks.

PLAIN ENGLISH: This script is a safety layer for the Alpaca paper account.
It does three things:
1. Repairs broker-side trailing stops for core ETFs, which keeps protection
   alive even when this laptop is asleep.
2. Cancels stale non-protective orders.
3. Watches intraday P&L while the laptop is awake and can trigger the existing
   emergency liquidation circuit breaker during market hours.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time as time_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alpaca_paper_trading import AlpacaBroker, _emergency_liquidate
from broker_interface import Order
from safe_io import atomic_write_json
from alpaca_protection import (
    CORE_PROTECTION_ENABLED,
    list_open_orders,
    order_id,
    order_qty,
    order_side,
    order_symbol,
    order_type,
    repair_core_etf_protective_stops,
)
from settings import LOG_DIR, SIGNAL_DIR


SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)
STATE_FILE = SIGNALS / "guard_intraday_state.json"
LOG_FILE = LOGS / "execution_guard.log"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# PLAIN ENGLISH: Each switch can disable one safety behavior without editing
# code. The defaults favor protection in paper trading.
CORE_ETF_STOP_ENABLED = _env_bool("GUARD_CORE_STOP", True)
STALE_CANCEL_ENABLED = _env_bool("GUARD_STALE_CANCEL", True)
STALE_ORDER_MINUTES = int(os.environ.get("GUARD_STALE_MINUTES", "90"))
STALE_REPLACE_SELLS_ENABLED = _env_bool("GUARD_REPLACE_STALE_SELLS", True)
STALE_REPLACE_SELL_OFFSET_BPS = float(os.environ.get("GUARD_REPLACE_STALE_SELL_OFFSET_BPS", "20"))
PNL_MONITOR_ENABLED = _env_bool("GUARD_PNL_MONITOR", True)
PNL_WARN_PCT = float(os.environ.get("GUARD_PNL_WARN_PCT", "-3.0"))
PNL_HALT_PCT = float(os.environ.get("GUARD_PNL_HALT_PCT", "-8.0"))
CHECK_INTERVAL_SEC = int(os.environ.get("GUARD_INTERVAL_SEC", "300"))
ALERTS_ENABLED = _env_bool("GUARD_ALERTS", True)


def _now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def _today_et() -> str:
    return _now_et().date().isoformat()


def log(message: str) -> None:
    """Append a timestamped message to the guard log and echo it."""
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} | {message}"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(f"  {message}")


def send_alert(message: str, *, priority: str = "warning") -> None:
    """Send a notification via all configured channels and write the log.

    PLAIN ENGLISH: Delegates to the shared notifications module which
    handles macOS banners, email, AND Telegram — so you get alerts on
    your phone even when your laptop is asleep.
    """
    log(f"ALERT: {message}")
    if not ALERTS_ENABLED:
        return
    # Import here to avoid circular imports at module load time
    from notifications import send_alert as _notify
    _notify(message, title="Execution Guard", priority=priority)


def load_state() -> dict:
    """Load today's guard state, resetting daily flags when the ET date changes."""
    today = _today_et()
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    else:
        state = {}
    if state.get("date") != today:
        state = {
            "date": today,
            "baseline_equity": None,
            "baseline_source": None,
            "intraday_high": None,
            "pnl_warn_sent": False,
            "pnl_halt_sent": False,
            "pnl_halt_blocked_alert_sent": False,
            "stale_alerted_order_ids": [],
            "protection_error_sent": False,
        }
    state.setdefault("stale_alerted_order_ids", [])
    return state


def save_state(state: dict) -> None:
    SIGNALS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(state, STATE_FILE)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _account_float(account: Any, field: str) -> float | None:
    raw = getattr(account, field, None)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if math.isfinite(value) and value > 0 else None


def _positive_finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) and out > 0 else None


def _stale_sell_replacement_limit(broker: AlpacaBroker, order: Any, symbol: str) -> float | None:
    """Choose a fresh marketable limit for a replacement SELL order."""
    price = None
    getter = getattr(broker, "get_last_price", None)
    if callable(getter):
        try:
            price = _positive_finite(getter(symbol))
        except Exception:
            price = None
    if price is None:
        # Fallback to the stale order's own limit only when no fresh price can
        # be read.  It is less ideal, but still recreates a broker-side exit
        # after a cancel/reject cycle without turning into a market order.
        price = _positive_finite(getattr(order, "limit_price", None))
    if price is None:
        return None
    offset = max(0.0, STALE_REPLACE_SELL_OFFSET_BPS) / 10_000.0
    return round(price * (1.0 - offset), 2)


def _replace_stale_sell_order(
    broker: AlpacaBroker,
    order: Any,
    *,
    dry_run: bool,
) -> dict | None:
    """
    Replace a cancelled stale SELL order with a fresh protective limit.

    PLAIN ENGLISH: A stale SELL is supposed to reduce risk but failed to fill.
    After cancelling it, the guard can submit a fresh sell at a current,
    marketable limit.  BUY orders are never replaced here.
    """
    if not STALE_REPLACE_SELLS_ENABLED or order_side(order) != "sell":
        return None
    symbol = order_symbol(order)
    qty_float = order_qty(order)
    qty = int(qty_float)
    if not symbol or qty <= 0:
        return {"ticker": symbol, "status": "skipped", "reason": "invalid_stale_sell_replacement"}

    limit_price = _stale_sell_replacement_limit(broker, order, symbol)
    if limit_price is None:
        log(f"skipped stale sell replacement: no usable price for {symbol}")
        return {"ticker": symbol, "status": "skipped", "reason": "replacement_price_unavailable"}

    replacement = Order(
        ticker=symbol,
        side="sell",
        quantity=qty,
        type="limit",
        limit_price=limit_price,
        client_id=f"guard_{_today_et().replace('-', '')}_{symbol}_sell_{qty}_{int(time_module.time())}"[:48],
    )
    if dry_run:
        log(f"dry-run would replace stale sell {symbol} qty={qty:g} limit=${limit_price:.2f}")
        return {"ticker": symbol, "status": "dry_run", "qty": qty, "limit_price": limit_price}

    try:
        new_oid = broker.place_order(replacement)
    except Exception as exc:
        log(f"failed stale sell replacement: {symbol} qty={qty:g} error={exc}")
        return {"ticker": symbol, "status": "failed", "qty": qty, "error": str(exc)}

    log(f"replaced stale sell {symbol} qty={qty:g} limit=${limit_price:.2f} order={str(new_oid)[:12]}")
    send_alert(f"Replaced stale sell {symbol} x{qty:g} with fresh limit ${limit_price:.2f}")
    return {"ticker": symbol, "status": "submitted", "qty": qty, "limit_price": limit_price, "order_id": new_oid}


def guard_core_etf_protection(broker: AlpacaBroker, state: dict, *, dry_run: bool) -> None:
    """
    Repair durable broker-side stops for core ETF positions.

    PLAIN ENGLISH: This is the baseline safety layer for your laptop workflow.
    If the laptop turns off, these Alpaca orders still exist at the broker.
    """
    if not CORE_ETF_STOP_ENABLED or not CORE_PROTECTION_ENABLED:
        log("core ETF protective stop guard disabled")
        return

    result = repair_core_etf_protective_stops(
        broker,
        dry_run=dry_run,
        replace=False,
        logger=log,
    )
    changed = bool(result["cancelled"] or result["submitted"])
    failed = bool(result["errors"])
    if changed:
        sent = ", ".join(
            f"{row['ticker']} x{row['qty']}" for row in result["submitted"]
        ) or "updated existing stops"
        send_alert(f"Core ETF protection repaired: {sent}")
    if failed and not state.get("protection_error_sent"):
        send_alert(f"Core ETF protection repair failed: {result['errors']}")
        state["protection_error_sent"] = True
    if not failed:
        state["protection_error_sent"] = False


def guard_stale_orders(broker: AlpacaBroker, state: dict, *, dry_run: bool) -> None:
    """
    Cancel stale open orders, but leave trailing stops alone.

    PLAIN ENGLISH: A normal buy/sell order sitting open for too long usually
    means the market moved away. Trailing stops are intentionally long-lived,
    so they are skipped.
    """
    if not STALE_CANCEL_ENABLED:
        log("stale order cancel guard disabled")
        return

    alerted = set(str(x) for x in state.get("stale_alerted_order_ids", []))
    now = datetime.now(timezone.utc)
    for order in list_open_orders(broker):
        if order_type(order) == "trailing_stop":
            continue
        oid = order_id(order)
        submitted_at = _parse_timestamp(getattr(order, "submitted_at", None) or getattr(order, "created_at", None))
        if submitted_at is None:
            continue
        age_minutes = (now - submitted_at).total_seconds() / 60.0
        if age_minutes < STALE_ORDER_MINUTES:
            continue

        symbol = order_symbol(order)
        side = order_side(order)
        qty = order_qty(order)
        if dry_run:
            cancelled = True
            status = "dry-run would cancel"
        else:
            cancelled = broker.cancel_order(oid)
            status = "cancelled" if cancelled else "cancel failed"
        log(f"{status}: stale order {side} {symbol} qty={qty:g} age={age_minutes:.1f}m order={oid[:12]}")
        if cancelled:
            _replace_stale_sell_order(broker, order, dry_run=dry_run)
        if oid not in alerted:
            send_alert(f"{status}: stale {side} {symbol} x{qty:g}, open {age_minutes:.0f} min")
            alerted.add(oid)
    state["stale_alerted_order_ids"] = sorted(alerted)


def guard_intraday_pnl(
    broker: AlpacaBroker,
    state: dict,
    *,
    dry_run: bool,
    market_open: bool,
    force_market_closed: bool,
) -> None:
    """
    Monitor current equity versus prior close and trigger halt if needed.

    PLAIN ENGLISH: The baseline is Alpaca's last_equity when available, which
    avoids accidentally treating a late laptop startup as the day's opening
    balance after losses have already happened.
    """
    if not PNL_MONITOR_ENABLED:
        log("intraday P&L guard disabled")
        return

    state.setdefault("pnl_warn_sent", False)
    state.setdefault("pnl_halt_sent", False)
    state.setdefault("pnl_halt_blocked_alert_sent", False)

    account = broker._api.get_account()
    current_equity = _account_float(account, "equity")
    if current_equity is None:
        current_equity = _positive_finite(broker.get_equity())
    if current_equity is None:
        send_alert("Intraday P&L guard skipped: invalid current broker equity")
        return

    stored_baseline = state.get("baseline_equity")
    baseline = _positive_finite(stored_baseline) if stored_baseline else None
    baseline_source = state.get("baseline_source")
    if baseline is None:
        baseline = _account_float(account, "last_equity")
        baseline_source = "alpaca_last_equity"
    if baseline is None:
        baseline = current_equity
        baseline_source = "current_equity_fallback"

    state["baseline_equity"] = round(float(baseline), 2)
    state["baseline_source"] = baseline_source

    previous_high = state.get("intraday_high")
    try:
        intraday_high = max(_positive_finite(previous_high) or 0.0, float(baseline), float(current_equity))
    except Exception:
        intraday_high = max(float(baseline), float(current_equity))
    state["intraday_high"] = round(intraday_high, 2)

    day_pnl_pct = (float(current_equity) / float(baseline) - 1.0) * 100.0
    intraday_dd_pct = (float(current_equity) / intraday_high - 1.0) * 100.0 if intraday_high > 0 else 0.0
    log(
        f"P&L check: equity=${current_equity:,.2f}, baseline=${baseline:,.2f} "
        f"({baseline_source}), day={day_pnl_pct:.2f}%, intraday_dd={intraday_dd_pct:.2f}%"
    )

    halt_action_allowed = market_open or force_market_closed
    if day_pnl_pct <= PNL_HALT_PCT:
        if halt_action_allowed:
            if not state.get("pnl_halt_sent"):
                message = f"Emergency halt: day P&L {day_pnl_pct:.2f}% <= {PNL_HALT_PCT:.2f}%"
                if dry_run:
                    log(f"DRY RUN: {message}; liquidation skipped")
                    send_alert(f"DRY RUN {message}")
                else:
                    log(message)
                    _emergency_liquidate(broker)
                    send_alert(f"{message}; liquidation requested")
                state["pnl_halt_sent"] = True
            return
        if not state.get("pnl_halt_blocked_alert_sent"):
            send_alert(
                f"Halt threshold reached while market is closed: day P&L {day_pnl_pct:.2f}% "
                f"<= {PNL_HALT_PCT:.2f}%. No market liquidation was sent."
            )
            state["pnl_halt_blocked_alert_sent"] = True
        return

    if day_pnl_pct <= PNL_WARN_PCT and not state.get("pnl_warn_sent"):
        send_alert(f"Intraday P&L warning: {day_pnl_pct:.2f}% <= {PNL_WARN_PCT:.2f}%")
        state["pnl_warn_sent"] = True


def run_once(broker: AlpacaBroker, *, dry_run: bool, force_market_closed: bool) -> None:
    state = load_state()
    try:
        market_open = bool(broker.is_market_open())
    except Exception as exc:
        market_open = False
        log(f"could not read Alpaca market clock: {exc}")

    log(f"guard cycle started: market_open={market_open}, dry_run={dry_run}")
    if CORE_ETF_STOP_ENABLED:
        guard_core_etf_protection(broker, state, dry_run=dry_run)
    if STALE_CANCEL_ENABLED:
        guard_stale_orders(broker, state, dry_run=dry_run)
    if PNL_MONITOR_ENABLED:
        guard_intraday_pnl(
            broker,
            state,
            dry_run=dry_run,
            market_open=market_open,
            force_market_closed=force_market_closed,
        )
    save_state(state)
    log("guard cycle finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca execution safety guard")
    parser.add_argument("--once", action="store_true", help="Run one guard cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without cancelling or trading")
    parser.add_argument(
        "--force-market-closed",
        action="store_true",
        help="Allow halt liquidation even when Alpaca reports the market closed",
    )
    args = parser.parse_args()

    try:
        broker = AlpacaBroker()
    except Exception as exc:
        print(f"Could not start execution guard: {exc}", file=sys.stderr)
        raise SystemExit(1)

    while True:
        run_once(broker, dry_run=args.dry_run, force_market_closed=args.force_market_closed)
        if args.once:
            break
        time_module.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
