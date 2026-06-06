"""broker_truth.py - reconcile live Alpaca state with local trading records.

PLAIN ENGLISH: A trading bot can have several "truths" at once: the signal
target weights, the order plan, the local CSV log, the broker's live positions,
and broker-side trailing stops.  This script joins those sources into one table
so differences are visible before they become silent risk.

How to run:
    python broker_truth.py
    python broker_truth.py --json
    python broker_truth.py --strict
    python broker_truth.py --offline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from safe_io import atomic_write_csv, atomic_write_json, configure_console_output
from settings import LOG_DIR, SIGNAL_DIR

configure_console_output()

SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)

SIGNAL_FILE = SIGNALS / "core_satellite_alpha_signal.csv"
ORDER_PLAN_FILE = SIGNALS / "core_satellite_alpha_orders.csv"
PAPER_LOG_FILE = SIGNALS / "alpaca_paper_log.csv"
STATUS_FILE = SIGNALS / "alpaca_daily_status.json"

BROKER_TRUTH_CSV = SIGNALS / "broker_truth.csv"
BROKER_TRUTH_JSON = SIGNALS / "broker_truth.json"

# PLAIN ENGLISH: these tolerances decide how big a mismatch must be before the
# script complains.  A few cents or a fractional share can happen from rounding.
QTY_TOLERANCE = float(os.environ.get("BROKER_TRUTH_QTY_TOLERANCE", "0.001"))
WEIGHT_TOLERANCE = float(os.environ.get("BROKER_TRUTH_WEIGHT_TOLERANCE", "0.02"))
REQUIRE_LIVE_OPEN_ORDERS = os.environ.get("BROKER_TRUTH_REQUIRE_LIVE_ORDERS", "0").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
    "on",
}

ETF_TICKERS = {"SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD"}
CORE_PROTECTION_TICKERS = {
    ticker.strip().upper()
    for ticker in os.environ.get("GUARD_CORE_TICKERS", "SPY,QQQ,TQQQ").split(",")
    if ticker.strip()
}
OVERLAY_TRAILING_STOP_ENABLED = os.environ.get("ALPACA_TRAILING_STOP", "1").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
    "on",
}

TRUTH_COLUMNS = [
    "ticker",
    "target_weight",
    "broker_qty",
    "broker_value",
    "broker_weight",
    "planned_side",
    "planned_quantity",
    "planned_buy_qty",
    "planned_sell_qty",
    "planned_current_qty",
    "planned_target_qty",
    "planned_target_weight",
    "submitted_quantity",
    "accepted_quantity",
    "filled_quantity",
    "failed_quantity",
    "skipped_quantity",
    "open_quantity",
    "latest_fill_status",
    "expected_qty_from_log",
    "quantity_gap",
    "open_buy_qty",
    "open_sell_qty",
    "trailing_stop_qty",
    "trailing_stop_count",
    "stop_required",
    "issue_severity",
    "issues",
]


def _now_utc() -> datetime:
    """Return one timezone-aware timestamp for all generated output."""
    return datetime.now(timezone.utc)


def _float_or_none(value: Any) -> float | None:
    """Convert common CSV/JSON values into a float, or None when blank/bad."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Float helper that gives a default instead of raising."""
    out = _float_or_none(value)
    return default if out is None else out


def _ticker(value: Any) -> str:
    """Normalize ticker text so CSV, JSON, and Alpaca symbols match."""
    return str(value or "").strip().upper()


def _read_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a CSV without crashing on missing or empty files."""
    meta = {"path": str(path), "exists": path.exists(), "rows": 0, "error": ""}
    if not path.exists():
        return pd.DataFrame(), meta
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        meta["error"] = "empty_csv"
        return pd.DataFrame(), meta
    except Exception as exc:
        meta["error"] = str(exc)
        return pd.DataFrame(), meta
    meta["rows"] = int(len(df))
    return df, meta


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a JSON object without crashing on missing or malformed files."""
    meta = {"path": str(path), "exists": path.exists(), "error": ""}
    if not path.exists():
        return {}, meta
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        meta["error"] = str(exc)
        return {}, meta
    if not isinstance(data, dict):
        meta["error"] = "json_root_not_object"
        return {}, meta
    return data, meta


def _parse_json_map(value: Any) -> dict[str, float]:
    """Parse a JSON weight map such as {"MU": 0.2}."""
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = json.loads(str(value))
        except Exception:
            return {}
    weights: dict[str, float] = {}
    for key, val in raw.items():
        symbol = _ticker(key)
        weight = _float_or_none(val)
        if symbol and weight is not None:
            weights[symbol] = float(weight)
    return weights


def _date_from_value(value: Any) -> pd.Timestamp | None:
    """Parse a date-like value and return its UTC calendar day."""
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def load_target_weights(signal_path: Path = SIGNAL_FILE) -> tuple[dict[str, float], dict[str, Any]]:
    """Load target weights from the latest core-satellite signal CSV."""
    df, meta = _read_csv(signal_path)
    weights: dict[str, float] = {}
    meta["as_of"] = ""
    if df.empty:
        return weights, meta

    # PLAIN ENGLISH: the signal file has one row.  Columns like
    # target_qqq_weight become ticker QQQ.  overlay_weights_json contains the
    # stock picks and their weights.
    row = df.iloc[-1].to_dict()
    meta["as_of"] = str(row.get("predicted_at") or row.get("as_of") or "")
    for col, value in row.items():
        name = str(col).lower()
        if not name.startswith("target_") or not name.endswith("_weight"):
            continue
        middle = name.removeprefix("target_").removesuffix("_weight")
        if middle in {"cash", "overlay", "core"}:
            continue
        weight = _float_or_none(value)
        if weight is not None:
            weights[middle.upper()] = float(weight)

    overlay_weights = _parse_json_map(row.get("overlay_weights_json"))
    weights.update(overlay_weights)
    meta["target_count"] = len(weights)
    return weights, meta


def load_broker_status(status_path: Path = STATUS_FILE) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Load Alpaca position truth from alpaca_daily_status.json."""
    data, meta = _read_json(status_path)
    equity = _safe_float(data.get("account_equity"))
    cash = _safe_float(data.get("account_cash"))
    positions: dict[str, dict[str, float]] = {}

    # PLAIN ENGLISH: position_details has the richest data, but older status
    # files may only have positions and position_values maps.  Support both.
    for item in data.get("position_details", []) or []:
        if not isinstance(item, dict):
            continue
        symbol = _ticker(item.get("ticker") or item.get("symbol"))
        if not symbol:
            continue
        qty = _safe_float(item.get("quantity") or item.get("qty"))
        value = _safe_float(item.get("market_value"))
        positions[symbol] = {
            "quantity": qty,
            "market_value": value,
            "weight": value / equity if equity > 0 else 0.0,
        }

    raw_positions = data.get("positions", {}) or {}
    raw_values = data.get("position_values", {}) or {}
    if isinstance(raw_positions, dict):
        for symbol_raw, qty_raw in raw_positions.items():
            symbol = _ticker(symbol_raw)
            if not symbol:
                continue
            qty = _safe_float(qty_raw)
            value = _safe_float(raw_values.get(symbol) if isinstance(raw_values, dict) else 0.0)
            current = positions.get(symbol, {})
            positions[symbol] = {
                "quantity": current.get("quantity", qty),
                "market_value": current.get("market_value", value),
                "weight": current.get("weight", value / equity if equity > 0 else 0.0),
            }

    meta.update(
        {
            "generated_at": data.get("generated_at", ""),
            "equity": equity,
            "cash": cash,
            "position_count": len(positions),
        }
    )
    return positions, meta


def _sum_by_side(rows: pd.DataFrame, side: str, quantity_col: str = "quantity") -> float:
    """Sum a quantity column for one side of a grouped order table."""
    if rows.empty or quantity_col not in rows.columns or "side" not in rows.columns:
        return 0.0
    side_mask = rows["side"].astype(str).str.lower().eq(side)
    return float(pd.to_numeric(rows.loc[side_mask, quantity_col], errors="coerce").fillna(0.0).sum())


def load_order_plan(plan_path: Path = ORDER_PLAN_FILE) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load the latest order plan and summarize it ticker-by-ticker."""
    df, meta = _read_csv(plan_path)
    plan: dict[str, dict[str, Any]] = {}
    if df.empty or "ticker" not in df.columns:
        return plan, meta

    clean = df.copy()
    clean["ticker"] = clean["ticker"].map(_ticker)
    if "side" not in clean.columns:
        clean["side"] = ""
    clean["side"] = clean["side"].astype(str).str.lower()
    for col in ["quantity", "current_qty", "target_qty", "target_weight", "current_weight", "trade_value"]:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    for symbol, rows in clean.groupby("ticker", dropna=True):
        if not symbol:
            continue
        sides = [str(x) for x in rows.get("side", pd.Series(dtype=str)).dropna().unique() if str(x)]
        plan[symbol] = {
            "planned_side": "|".join(sides),
            "planned_quantity": float(rows.get("quantity", pd.Series(dtype=float)).fillna(0.0).sum()),
            "planned_buy_qty": _sum_by_side(rows, "buy"),
            "planned_sell_qty": _sum_by_side(rows, "sell"),
            "planned_current_qty": _safe_float(rows.get("current_qty", pd.Series([None])).iloc[-1]),
            "planned_target_qty": _safe_float(rows.get("target_qty", pd.Series([None])).iloc[-1]),
            "planned_target_weight": _safe_float(rows.get("target_weight", pd.Series([None])).iloc[-1]),
            "planned_current_weight": _safe_float(rows.get("current_weight", pd.Series([None])).iloc[-1]),
            "planned_trade_value": _safe_float(rows.get("trade_value", pd.Series([None])).sum()),
        }
    return plan, meta


def _latest_log_slice(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Keep the latest submitted_at date so old logs do not distort today."""
    if df.empty or "submitted_at" not in df.columns:
        return df, ""
    submitted = pd.to_datetime(df["submitted_at"], errors="coerce", utc=True)
    if submitted.notna().sum() == 0:
        return df, ""
    latest_day = submitted.dropna().dt.normalize().max()
    return df.loc[submitted.dt.normalize().eq(latest_day)].copy(), str(latest_day.date())


def _status_bucket(row: pd.Series) -> str:
    """Classify one log row into accepted, failed, skipped, filled, or open."""
    status = str(row.get("fill_status", "") or "").strip().lower()
    order_id = str(row.get("order_id", "") or "").strip()
    if order_id.startswith("ERROR") or status in {"submission_failed", "rejected", "failed"}:
        return "failed"
    if order_id.startswith("SKIPPED") or status == "skipped":
        return "skipped"
    if status == "filled":
        return "filled"
    if status in {"partially_filled", "open", "accepted", "new", "pending"}:
        return "open"
    if order_id:
        return "accepted"
    return "failed"


def _filled_qty(row: pd.Series) -> float:
    """Use filled_qty when present, otherwise infer full fill from quantity."""
    filled = _float_or_none(row.get("filled_qty"))
    if filled is not None:
        return max(0.0, filled)
    if str(row.get("fill_status", "")).strip().lower() == "filled":
        return max(0.0, _safe_float(row.get("quantity")))
    return 0.0


def load_paper_log(log_path: Path = PAPER_LOG_FILE) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load the most recent local Alpaca order log rows."""
    df, meta = _read_csv(log_path)
    log: dict[str, dict[str, Any]] = {}
    if df.empty or "ticker" not in df.columns:
        return log, meta

    latest_df, latest_day = _latest_log_slice(df)
    meta["latest_submitted_date"] = latest_day
    clean = latest_df.copy()
    clean["ticker"] = clean["ticker"].map(_ticker)
    if "side" not in clean.columns:
        clean["side"] = ""
    if "quantity" not in clean.columns:
        clean["quantity"] = 0.0
    clean["side"] = clean["side"].astype(str).str.lower()
    clean["quantity"] = pd.to_numeric(clean["quantity"], errors="coerce").fillna(0.0)
    clean["filled_qty_normalized"] = clean.apply(_filled_qty, axis=1)
    clean["bucket"] = clean.apply(_status_bucket, axis=1)

    for symbol, rows in clean.groupby("ticker", dropna=True):
        if not symbol:
            continue
        accepted_rows = rows[~rows["bucket"].isin({"failed", "skipped"})]
        filled_rows = rows[rows["bucket"].eq("filled")]
        failed_rows = rows[rows["bucket"].eq("failed")]
        skipped_rows = rows[rows["bucket"].eq("skipped")]
        open_rows = rows[rows["bucket"].eq("open")]
        filled_buy_qty = _sum_by_side(filled_rows.rename(columns={"filled_qty_normalized": "filled"}), "buy", "filled")
        filled_sell_qty = _sum_by_side(filled_rows.rename(columns={"filled_qty_normalized": "filled"}), "sell", "filled")
        latest_status = str(rows.get("fill_status", pd.Series([""])).iloc[-1] or "")
        log[symbol] = {
            "submitted_quantity": float(rows["quantity"].sum()),
            "accepted_quantity": float(accepted_rows["quantity"].sum()),
            "filled_quantity": float(filled_rows["filled_qty_normalized"].sum()),
            "filled_buy_qty": filled_buy_qty,
            "filled_sell_qty": filled_sell_qty,
            "failed_quantity": float(failed_rows["quantity"].sum()),
            "skipped_quantity": float(skipped_rows["quantity"].sum()),
            "open_quantity": float(open_rows["quantity"].sum()),
            "latest_fill_status": latest_status,
        }
    return log, meta


def collect_live_open_orders() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read live Alpaca open orders when credentials are available."""
    _, _, open_orders, open_meta = collect_live_alpaca_state()
    return open_orders, open_meta


def collect_live_alpaca_state() -> tuple[dict[str, dict[str, float]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Read live Alpaca positions and open orders in one read-only pass."""
    position_meta = {
        "available": False,
        "attempted": True,
        "source": "alpaca_api",
        "error": "",
        "equity": 0.0,
        "cash": 0.0,
        "position_count": 0,
    }
    meta = {"available": False, "count": 0, "error": "", "source": "alpaca_api"}
    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_SECRET_KEY"):
        error = "missing_alpaca_credentials"
        position_meta["error"] = error
        meta["error"] = error
        return {}, position_meta, [], meta

    # PLAIN ENGLISH: imports stay inside this function so normal offline tests
    # can run without connecting to Alpaca.
    try:
        from alpaca_paper_trading import AlpacaBroker
        from alpaca_protection import (
            list_open_orders,
            order_id,
            order_qty,
            order_side,
            order_symbol,
            order_trail_percent,
            order_type,
        )

        broker = AlpacaBroker()
        equity = _safe_float(broker.get_equity())
        cash = _safe_float(broker.get_cash())
        raw_positions = broker._api.list_positions()
        raw_orders = list_open_orders(broker)
    except Exception as exc:
        error = str(exc)
        position_meta["error"] = error
        meta["error"] = error
        return {}, position_meta, [], meta

    positions: dict[str, dict[str, float]] = {}
    for pos in raw_positions:
        symbol = _ticker(getattr(pos, "symbol", ""))
        if not symbol:
            continue
        qty = _safe_float(getattr(pos, "qty", 0.0))
        value = _safe_float(getattr(pos, "market_value", 0.0))
        positions[symbol] = {
            "quantity": qty,
            "market_value": value,
            "weight": value / equity if equity > 0 else 0.0,
        }

    orders: list[dict[str, Any]] = []
    for order in raw_orders:
        orders.append(
            {
                "ticker": order_symbol(order),
                "side": order_side(order),
                "type": order_type(order),
                "quantity": order_qty(order),
                "trail_percent": order_trail_percent(order),
                "order_id": order_id(order),
                "status": str(getattr(order, "status", "") or ""),
            }
        )
    position_meta.update(
        {
            "available": True,
            "generated_at": _now_utc().isoformat(timespec="seconds"),
            "equity": equity,
            "cash": cash,
            "position_count": len(positions),
        }
    )
    meta["available"] = True
    meta["count"] = len(orders)
    return positions, position_meta, orders, meta


def summarize_open_orders(open_orders: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group live open orders by ticker for row-level reconciliation."""
    out: dict[str, dict[str, Any]] = {}
    for row in open_orders:
        symbol = _ticker(row.get("ticker"))
        if not symbol:
            continue
        side = str(row.get("side", "") or "").lower()
        order_type = str(row.get("type", "") or "").lower()
        qty = max(0.0, _safe_float(row.get("quantity")))
        entry = out.setdefault(
            symbol,
            {
                "open_buy_qty": 0.0,
                "open_sell_qty": 0.0,
                "trailing_stop_qty": 0.0,
                "trailing_stop_count": 0,
                "open_order_ids": [],
            },
        )
        if side == "buy":
            entry["open_buy_qty"] += qty
        if side == "sell":
            entry["open_sell_qty"] += qty
        if side == "sell" and order_type == "trailing_stop":
            entry["trailing_stop_qty"] += qty
            entry["trailing_stop_count"] += 1
        if row.get("order_id"):
            entry["open_order_ids"].append(str(row["order_id"]))
    return out


def _stop_required(symbol: str, broker_qty: float, target_weight: float) -> bool:
    """Return True when a held symbol should have a broker-side stop."""
    if broker_qty <= QTY_TOLERANCE:
        return False
    if symbol in CORE_PROTECTION_TICKERS:
        return True
    if symbol in ETF_TICKERS:
        return False
    return OVERLAY_TRAILING_STOP_ENABLED and target_weight > 0


def _row_severity(issues: list[tuple[str, str]]) -> str:
    """Collapse issue severities into one label for the CSV row."""
    severities = {severity for severity, _ in issues}
    if "fail" in severities:
        return "fail"
    if "warning" in severities:
        return "warning"
    return "pass"


def _status_from_rows(
    rows: list[dict[str, Any]],
    global_issues: list[tuple[str, str]],
    *,
    status_meta: dict[str, Any],
) -> str:
    """Pick the overall broker-truth status."""
    if not status_meta.get("exists"):
        return "collecting"
    severities = {severity for severity, _ in global_issues}
    severities.update(str(row.get("issue_severity", "")) for row in rows)
    if "fail" in severities:
        return "fail"
    if "warning" in severities:
        return "warning"
    return "pass"


def build_broker_truth(
    *,
    signal_path: Path = SIGNAL_FILE,
    plan_path: Path = ORDER_PLAN_FILE,
    log_path: Path = PAPER_LOG_FILE,
    status_path: Path = STATUS_FILE,
    open_orders: list[dict[str, Any]] | None = None,
    open_orders_meta: dict[str, Any] | None = None,
    live_positions: dict[str, dict[str, float]] | None = None,
    live_positions_meta: dict[str, Any] | None = None,
    collect_open_orders_fn: Callable[[], tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
    collect_live_state_fn: Callable[[], tuple[dict[str, dict[str, float]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]] | None = None,
    include_live_open_orders: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the full reconciliation payload without writing files."""
    clock = now or _now_utc()
    targets, signal_meta = load_target_weights(signal_path)
    positions, status_meta = load_broker_status(status_path)
    plan, plan_meta = load_order_plan(plan_path)
    log, log_meta = load_paper_log(log_path)

    live_positions_meta = live_positions_meta or {
        "available": False,
        "attempted": False,
        "source": "not_requested",
        "error": "",
    }
    if open_orders is None and include_live_open_orders and collect_open_orders_fn is None:
        collector = collect_live_state_fn or collect_live_alpaca_state
        live_positions, live_positions_meta, open_orders, open_orders_meta = collector()
    elif open_orders is None:
        if include_live_open_orders:
            collector = collect_open_orders_fn or collect_live_open_orders
            open_orders, open_orders_meta = collector()
        else:
            open_orders = []
            open_orders_meta = {"available": False, "count": 0, "error": "offline_mode", "source": "offline"}
    open_orders_meta = open_orders_meta or {"available": False, "count": len(open_orders), "error": "", "source": "injected"}

    if live_positions:
        # PLAIN ENGLISH: when the API read succeeds, the live broker account
        # beats the saved JSON snapshot.  The saved file remains a fallback for
        # offline laptop checks.
        positions = live_positions
        status_meta = {
            **status_meta,
            **live_positions_meta,
            "path": str(status_path),
            "exists": True,
            "file_generated_at": status_meta.get("generated_at", ""),
            "source": "alpaca_api",
        }
    open_summary = summarize_open_orders(open_orders)

    global_issues: list[tuple[str, str]] = []
    for name, meta in [
        ("signal", signal_meta),
        ("order_plan", plan_meta),
        ("paper_log", log_meta),
        ("broker_status", status_meta),
    ]:
        if meta.get("error"):
            global_issues.append(("warning", f"{name}_read_error:{meta['error']}"))
        elif not meta.get("exists"):
            severity = "warning" if name != "broker_status" else "fail"
            global_issues.append((severity, f"{name}_missing"))

    if REQUIRE_LIVE_OPEN_ORDERS and not open_orders_meta.get("available"):
        global_issues.append(("fail", f"live_open_orders_unavailable:{open_orders_meta.get('error', '')}"))
    elif not open_orders_meta.get("available"):
        global_issues.append(("warning", f"live_open_orders_unavailable:{open_orders_meta.get('error', '')}"))
    if live_positions_meta.get("attempted") and not live_positions_meta.get("available"):
        global_issues.append(("warning", f"live_positions_unavailable:{live_positions_meta.get('error', '')}"))

    # PLAIN ENGLISH: a local paper log can be older than the broker snapshot.
    # Old failed orders should stay in the audit log, but they should not make
    # today's reconciliation fail after the account has moved on.
    log_day = _date_from_value(log_meta.get("latest_submitted_date"))
    status_day = _date_from_value(status_meta.get("generated_at"))
    plan_log_stale = False
    if log_meta.get("exists") and log_day is not None and status_day is not None:
        plan_log_stale = abs((status_day - log_day).days) > 1
    if plan_log_stale:
        log_meta["stale_vs_broker_status"] = True
        global_issues.append(("warning", "paper_log_stale_vs_broker_status"))
        effective_plan: dict[str, dict[str, Any]] = {}
        effective_log: dict[str, dict[str, Any]] = {}
    else:
        log_meta["stale_vs_broker_status"] = False
        effective_plan = plan
        effective_log = log

    tickers = sorted(set(targets) | set(positions) | set(effective_plan) | set(effective_log) | set(open_summary))
    rows: list[dict[str, Any]] = []

    for symbol in tickers:
        target_weight = float(targets.get(symbol, 0.0))
        pos = positions.get(symbol, {})
        plan_row = effective_plan.get(symbol, {})
        log_row = effective_log.get(symbol, {})
        open_row = open_summary.get(symbol, {})

        broker_qty = _safe_float(pos.get("quantity"))
        broker_value = _safe_float(pos.get("market_value"))
        broker_weight = _safe_float(pos.get("weight"))
        planned_current_qty = _safe_float(plan_row.get("planned_current_qty"))
        expected_qty = None
        if symbol in effective_plan and symbol in effective_log:
            expected_qty = planned_current_qty + _safe_float(log_row.get("filled_buy_qty")) - _safe_float(log_row.get("filled_sell_qty"))
        quantity_gap = "" if expected_qty is None else round(broker_qty - expected_qty, 6)

        issues: list[tuple[str, str]] = []
        abs_weight_gap = abs(broker_weight - target_weight)
        planned_qty = _safe_float(plan_row.get("planned_quantity"))
        submitted_qty = _safe_float(log_row.get("submitted_quantity"))
        failed_qty = _safe_float(log_row.get("failed_quantity"))
        skipped_qty = _safe_float(log_row.get("skipped_quantity"))
        open_qty = _safe_float(log_row.get("open_quantity"))
        open_buy_qty = _safe_float(open_row.get("open_buy_qty"))
        open_sell_qty = _safe_float(open_row.get("open_sell_qty"))
        trailing_stop_qty = _safe_float(open_row.get("trailing_stop_qty"))
        trailing_stop_count = int(_safe_float(open_row.get("trailing_stop_count")))
        stop_required = _stop_required(symbol, broker_qty, target_weight)

        if planned_qty > QTY_TOLERANCE and submitted_qty <= QTY_TOLERANCE and log_meta.get("exists"):
            issues.append(("warning", "planned_order_not_seen_in_latest_log"))
        if failed_qty > QTY_TOLERANCE:
            issues.append(("fail", "latest_logged_order_failed"))
        if skipped_qty > QTY_TOLERANCE:
            issues.append(("warning", "latest_logged_order_skipped"))
        if open_qty > QTY_TOLERANCE:
            issues.append(("warning", "latest_logged_order_still_open"))
        if expected_qty is not None and abs(float(quantity_gap)) > max(QTY_TOLERANCE, 0.01):
            issues.append(("warning", "broker_qty_differs_from_latest_log_expected_qty"))
        if target_weight <= WEIGHT_TOLERANCE and broker_qty > QTY_TOLERANCE and open_sell_qty <= QTY_TOLERANCE:
            issues.append(("warning", "extra_broker_position_not_in_target"))
        if target_weight > WEIGHT_TOLERANCE and broker_qty <= QTY_TOLERANCE and open_buy_qty <= QTY_TOLERANCE:
            issues.append(("warning", "target_position_missing_at_broker"))
        if abs_weight_gap > WEIGHT_TOLERANCE and open_buy_qty + open_sell_qty <= QTY_TOLERANCE:
            issues.append(("warning", f"broker_weight_gap_{abs_weight_gap:.4f}"))

        if open_orders_meta.get("available") and stop_required:
            if trailing_stop_qty <= QTY_TOLERANCE:
                issues.append(("fail", "required_trailing_stop_missing"))
            elif trailing_stop_qty + QTY_TOLERANCE < broker_qty:
                issues.append(("fail", "trailing_stop_qty_below_position_qty"))
            elif trailing_stop_qty - broker_qty > QTY_TOLERANCE:
                issues.append(("warning", "trailing_stop_qty_above_position_qty"))
        if open_orders_meta.get("available") and not stop_required and trailing_stop_qty > QTY_TOLERANCE:
            issues.append(("warning", "stale_trailing_stop_without_required_position"))

        row = {
            "ticker": symbol,
            "target_weight": round(target_weight, 6),
            "broker_qty": broker_qty,
            "broker_value": round(broker_value, 2),
            "broker_weight": round(broker_weight, 6),
            "planned_side": plan_row.get("planned_side", ""),
            "planned_quantity": _safe_float(plan_row.get("planned_quantity")),
            "planned_buy_qty": _safe_float(plan_row.get("planned_buy_qty")),
            "planned_sell_qty": _safe_float(plan_row.get("planned_sell_qty")),
            "planned_current_qty": planned_current_qty if symbol in effective_plan else "",
            "planned_target_qty": plan_row.get("planned_target_qty", ""),
            "planned_target_weight": plan_row.get("planned_target_weight", ""),
            "submitted_quantity": submitted_qty,
            "accepted_quantity": _safe_float(log_row.get("accepted_quantity")),
            "filled_quantity": _safe_float(log_row.get("filled_quantity")),
            "failed_quantity": failed_qty,
            "skipped_quantity": skipped_qty,
            "open_quantity": open_qty,
            "latest_fill_status": log_row.get("latest_fill_status", ""),
            "expected_qty_from_log": "" if expected_qty is None else round(expected_qty, 6),
            "quantity_gap": quantity_gap,
            "open_buy_qty": open_buy_qty,
            "open_sell_qty": open_sell_qty,
            "trailing_stop_qty": trailing_stop_qty,
            "trailing_stop_count": trailing_stop_count,
            "stop_required": stop_required,
            "issue_severity": _row_severity(issues),
            "issues": ";".join(message for _, message in issues),
        }
        rows.append(row)

    status = _status_from_rows(rows, global_issues, status_meta=status_meta)
    fail_count = sum(1 for row in rows if row["issue_severity"] == "fail") + sum(1 for severity, _ in global_issues if severity == "fail")
    warning_count = sum(1 for row in rows if row["issue_severity"] == "warning") + sum(1 for severity, _ in global_issues if severity == "warning")
    score = max(0.0, 100.0 - fail_count * 30.0 - warning_count * 5.0)

    payload = {
        "schema_version": 1,
        "generated_at": clock.isoformat(timespec="seconds"),
        "status": status,
        "score": round(score, 1),
        "summary": {
            "tickers_checked": len(rows),
            "fail_count": fail_count,
            "warning_count": warning_count,
            "pass_count": sum(1 for row in rows if row["issue_severity"] == "pass"),
            "account_equity": status_meta.get("equity", 0.0),
            "account_cash": status_meta.get("cash", 0.0),
            "live_open_orders_available": bool(open_orders_meta.get("available")),
            "live_open_orders_count": int(open_orders_meta.get("count", len(open_orders))),
            "latest_log_date": log_meta.get("latest_submitted_date", ""),
        },
        "inputs": {
            "signal": signal_meta,
            "order_plan": plan_meta,
            "paper_log": log_meta,
            "broker_status": status_meta,
            "live_positions": live_positions_meta,
            "open_orders": open_orders_meta,
            "qty_tolerance": QTY_TOLERANCE,
            "weight_tolerance": WEIGHT_TOLERANCE,
        },
        "global_issues": [
            {"severity": severity, "issue": message} for severity, message in global_issues
        ],
        "rows": rows,
    }
    return payload


def write_broker_truth(
    *,
    output_csv: Path = BROKER_TRUTH_CSV,
    output_json: Path = BROKER_TRUTH_JSON,
    log_dir: Path = LOGS,
    now: datetime | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build broker truth, then write latest CSV/JSON plus a dated snapshot."""
    clock = now or _now_utc()
    payload = build_broker_truth(now=clock, **kwargs)
    rows = payload.get("rows", [])
    df = pd.DataFrame(rows, columns=TRUTH_COLUMNS)
    atomic_write_csv(df, output_csv)
    atomic_write_json(payload, output_json)

    dated = log_dir / f"broker_truth_{clock.strftime('%Y%m%d')}.json"
    atomic_write_json(payload, dated)
    return payload


def print_summary(payload: dict[str, Any]) -> None:
    """Print a small human-readable summary for terminal runs."""
    summary = payload.get("summary", {})
    print("Broker Truth Reconciliation")
    print("-" * 36)
    print(f"Status: {payload.get('status', 'unknown').upper()} | score: {payload.get('score')}")
    print(
        "Rows: "
        f"{summary.get('tickers_checked', 0)} checked, "
        f"{summary.get('fail_count', 0)} fail, "
        f"{summary.get('warning_count', 0)} warning"
    )
    print(f"Live open orders: {summary.get('live_open_orders_available')} ({summary.get('live_open_orders_count', 0)})")
    for item in payload.get("global_issues", [])[:5]:
        print(f"  {str(item.get('severity', '')).upper():7s} GLOBAL: {item.get('issue', '')}")
    problem_rows = [
        row for row in payload.get("rows", [])
        if row.get("issue_severity") in {"fail", "warning"}
    ]
    for row in problem_rows[:10]:
        print(f"  {row['issue_severity'].upper():7s} {row['ticker']}: {row.get('issues', '')}")
    if len(problem_rows) > 10:
        print(f"  ... {len(problem_rows) - 10} more rows with issues")
    print(f"Saved: {BROKER_TRUTH_CSV}")
    print(f"Saved: {BROKER_TRUTH_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Alpaca broker truth with local bot records.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when broker truth status is fail.")
    parser.add_argument("--offline", "--no-live-open-orders", dest="offline", action="store_true",
                        help="Do not call Alpaca for live open/trailing orders.")
    args = parser.parse_args()

    payload = write_broker_truth(include_live_open_orders=not args.offline)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print_summary(payload)

    if args.strict and payload.get("status") == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
