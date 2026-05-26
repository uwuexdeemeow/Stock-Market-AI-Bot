"""
alpaca_protection.py - shared helpers for durable Alpaca ETF stop protection.

PLAIN ENGLISH: The execution guard can only run while this laptop is awake.
These helpers place broker-side trailing stop orders at Alpaca so SPY, QQQ,
and TQQQ still have basic protection after the laptop goes offline.
"""
from __future__ import annotations

import os
import re
import math
from datetime import datetime, timezone
from typing import Any, Callable

from broker_interface import Order, Position


Logger = Callable[[str], None]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_tickers(name: str, default: str) -> set[str]:
    raw = os.environ.get(name, default)
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


# PLAIN ENGLISH: These are the core ETF positions that should have durable
# broker-side trailing stops. TQQQ gets its own wider trail because it moves
# much more than SPY or QQQ.
CORE_PROTECTION_ENABLED = _env_bool("GUARD_CORE_STOP", True)
CORE_PROTECTION_TICKERS = _env_tickers("GUARD_CORE_TICKERS", "SPY,QQQ,TQQQ")
CORE_PROTECTION_TRAIL_PCT = float(os.environ.get("GUARD_CORE_TRAIL_PCT", "0.05"))
TQQQ_PROTECTION_TRAIL_PCT = float(os.environ.get("GUARD_TQQQ_TRAIL_PCT", "0.10"))
CORE_PROTECTION_CLIENT_PREFIX = os.environ.get("GUARD_CORE_STOP_CLIENT_PREFIX", "core-stop")


def core_trail_pct(ticker: str) -> float:
    """Return the trailing stop percentage for a core ETF ticker."""
    return TQQQ_PROTECTION_TRAIL_PCT if str(ticker).upper() == "TQQQ" else CORE_PROTECTION_TRAIL_PCT


def _valid_trail_pct(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) and 0.0 < out <= 1.0 else None


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def order_symbol(order: Any) -> str:
    return str(_attr(order, "symbol", "") or "").upper()


def order_side(order: Any) -> str:
    return str(_attr(order, "side", "") or "").lower()


def order_type(order: Any) -> str:
    return str(_attr(order, "type", "") or "").lower()


def order_id(order: Any) -> str:
    return str(_attr(order, "id", "") or _attr(order, "order_id", "") or "")


def order_qty(order: Any) -> float:
    raw = _attr(order, "qty", 0) or _attr(order, "quantity", 0) or 0
    try:
        return float(raw)
    except Exception:
        return 0.0


def order_trail_percent(order: Any) -> float | None:
    raw = _attr(order, "trail_percent", None)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except Exception:
        return None


def list_open_orders(broker: Any) -> list[Any]:
    """
    Return open Alpaca orders.

    PLAIN ENGLISH: Different alpaca-trade-api versions accept slightly
    different keyword arguments, so this tries the richer call first and then
    falls back to the simpler one.
    """
    try:
        return list(broker._api.list_orders(status="open", limit=500, nested=False))
    except TypeError:
        try:
            return list(broker._api.list_orders(status="open", limit=500))
        except TypeError:
            return list(broker._api.list_orders(status="open"))


def is_core_etf_protective_stop(order: Any, tickers: set[str] | None = None) -> bool:
    """True when an open order is a protective trailing sell stop for a core ETF."""
    watch = CORE_PROTECTION_TICKERS if tickers is None else {t.upper() for t in tickers}
    return (
        order_symbol(order) in watch
        and order_side(order) == "sell"
        and order_type(order) == "trailing_stop"
    )


def _client_order_id(ticker: str) -> str:
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]", "", str(ticker).upper())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{CORE_PROTECTION_CLIENT_PREFIX}-{safe_ticker}-{stamp}"[:48]


def _trail_matches(order: Any, expected_fraction: float) -> bool:
    actual_percent = order_trail_percent(order)
    if actual_percent is None:
        # Some API/entity versions omit trail_percent on returned orders. In
        # that case, avoid cancel/recreate churn if quantity already matches.
        return True
    expected_percent = float(expected_fraction) * 100.0
    return abs(actual_percent - expected_percent) <= 0.05


def _position_map(positions: list[Position], tickers: set[str]) -> dict[str, Position]:
    out: dict[str, Position] = {}
    for pos in positions:
        ticker = str(pos.ticker).upper()
        if ticker in tickers and int(pos.quantity) > 0:
            out[ticker] = Position(ticker=ticker, quantity=int(pos.quantity), avg_price=float(pos.avg_price))
    return out


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


def cancel_core_etf_protective_stops(
    broker: Any,
    *,
    tickers: set[str] | None = None,
    dry_run: bool = False,
    logger: Logger | None = None,
) -> list[dict]:
    """
    Cancel open core ETF trailing stops.

    PLAIN ENGLISH: Before a rebalance sells SPY/QQQ/TQQQ, old protective sell
    stops can reserve shares and block the rebalance. This clears those stops
    so the rebalance can proceed.
    """
    watch = CORE_PROTECTION_TICKERS if tickers is None else {t.upper() for t in tickers}
    actions: list[dict] = []
    for order in list_open_orders(broker):
        if not is_core_etf_protective_stop(order, watch):
            continue
        oid = order_id(order)
        symbol = order_symbol(order)
        qty = order_qty(order)
        if dry_run:
            status = "dry_run"
        else:
            status = "cancelled" if broker.cancel_order(oid) else "failed"
        actions.append({"action": "cancel_stop", "status": status, "ticker": symbol, "qty": qty, "order_id": oid})
        _log(logger, f"{status}: core ETF protective stop {symbol} qty={qty:g} order={oid[:12]}")
    return actions


def repair_core_etf_protective_stops(
    broker: Any,
    *,
    tickers: set[str] | None = None,
    dry_run: bool = False,
    replace: bool = False,
    logger: Logger | None = None,
) -> dict:
    """
    Ensure each held core ETF has one matching broker-side trailing stop.

    PLAIN ENGLISH: If you hold SPY, QQQ, or TQQQ, this checks whether Alpaca
    already has a matching trailing sell stop. Missing or wrong-size stops are
    cancelled/recreated. Good existing stops are left alone so their high-water
    mark is preserved.
    """
    result = {"checked": [], "cancelled": [], "submitted": [], "skipped": [], "errors": []}
    if not CORE_PROTECTION_ENABLED:
        result["skipped"].append({"reason": "disabled"})
        _log(logger, "core ETF protective stops disabled by GUARD_CORE_STOP=0")
        return result

    watch = CORE_PROTECTION_TICKERS if tickers is None else {t.upper() for t in tickers}
    positions = _position_map(broker.get_positions(), watch)
    open_orders = list_open_orders(broker)
    core_stops = [order for order in open_orders if is_core_etf_protective_stop(order, watch)]

    for ticker in sorted(watch):
        pos = positions.get(ticker)
        ticker_stops = [order for order in core_stops if order_symbol(order) == ticker]
        result["checked"].append({"ticker": ticker, "position_qty": 0 if pos is None else pos.quantity, "stop_count": len(ticker_stops)})

        if pos is None:
            # PLAIN ENGLISH: If we no longer own the ETF, any leftover stop is
            # stale and should be removed.
            for stop in ticker_stops:
                oid = order_id(stop)
                if dry_run:
                    status = "dry_run"
                else:
                    status = "cancelled" if broker.cancel_order(oid) else "failed"
                row = {"ticker": ticker, "qty": order_qty(stop), "order_id": oid, "status": status, "reason": "no_position"}
                result["cancelled"].append(row)
                _log(logger, f"{status}: stale core ETF stop {ticker} order={oid[:12]} (no position)")
                if status == "failed":
                    result["errors"].append({
                        "ticker": ticker,
                        "order_id": oid,
                        "error": "could_not_cancel_stale_protective_stop",
                    })
            continue

        expected_trail = _valid_trail_pct(core_trail_pct(ticker))
        if expected_trail is None:
            result["errors"].append({
                "ticker": ticker,
                "error": "invalid_core_trail_pct",
            })
            _log(logger, f"skipped: invalid core ETF trail percent for {ticker}")
            continue
        stop_qty = sum(order_qty(order) for order in ticker_stops)
        qty_matches = abs(stop_qty - float(pos.quantity)) < 1e-6
        trail_matches = all(_trail_matches(order, expected_trail) for order in ticker_stops)

        if ticker_stops and qty_matches and trail_matches and not replace:
            row = {"ticker": ticker, "qty": pos.quantity, "trail_pct": expected_trail, "reason": "already_protected"}
            result["skipped"].append(row)
            _log(logger, f"protected: {ticker} qty={pos.quantity} trail={expected_trail * 100:.2f}%")
            continue

        for stop in ticker_stops:
            oid = order_id(stop)
            if dry_run:
                status = "dry_run"
            else:
                status = "cancelled" if broker.cancel_order(oid) else "failed"
            row = {"ticker": ticker, "qty": order_qty(stop), "order_id": oid, "status": status, "reason": "replace"}
            result["cancelled"].append(row)
            _log(logger, f"{status}: outdated core ETF stop {ticker} order={oid[:12]}")
            if status == "failed":
                result["errors"].append({
                    "ticker": ticker,
                    "order_id": oid,
                    "error": "could_not_cancel_existing_protective_stop",
                })

        if any(row["ticker"] == ticker and row["status"] == "failed" for row in result["cancelled"]):
            _log(logger, f"skipped: submit replacement stop for {ticker} because cancel failed")
            continue

        order = Order(
            ticker=ticker,
            side="sell",
            quantity=int(pos.quantity),
            type="trailing_stop",
            trail_percent=expected_trail,
            client_id=_client_order_id(ticker),
        )
        if dry_run:
            oid = "DRY_RUN"
            status = "dry_run"
        else:
            try:
                oid = broker.place_order(order)
                status = "submitted"
            except Exception as exc:
                row = {"ticker": ticker, "qty": pos.quantity, "trail_pct": expected_trail, "error": str(exc)}
                result["errors"].append(row)
                _log(logger, f"failed: submit core ETF stop {ticker} qty={pos.quantity}: {exc}")
                continue
        row = {"ticker": ticker, "qty": pos.quantity, "trail_pct": expected_trail, "order_id": oid, "status": status}
        result["submitted"].append(row)
        _log(logger, f"{status}: core ETF stop {ticker} qty={pos.quantity} trail={expected_trail * 100:.2f}% order={oid[:12]}")

    return result
