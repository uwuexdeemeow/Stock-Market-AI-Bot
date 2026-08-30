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
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from safe_io import atomic_write_csv, atomic_write_json, configure_console_output
from run_evidence import current_run_id, enrich_payload, update_rebalance_state
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
ALIGNMENT_RECOVERY_PLAN_CSV = SIGNALS / "alignment_recovery_plan.csv"
ALIGNMENT_INCIDENT_LEDGER_CSV = SIGNALS / "alignment_incident_ledger.csv"

# PLAIN ENGLISH: these tolerances decide how big a mismatch must be before the
# script complains.  A few cents or a fractional share can happen from rounding.
QTY_TOLERANCE = float(os.environ.get("BROKER_TRUTH_QTY_TOLERANCE", "0.001"))
WEIGHT_TOLERANCE = float(os.environ.get("BROKER_TRUTH_WEIGHT_TOLERANCE", "0.02"))
GROSS_EXPOSURE_TOLERANCE = float(os.environ.get("BROKER_TRUTH_GROSS_EXPOSURE_TOLERANCE", "0.05"))
ALIGNMENT_WAIT_SECONDS = float(os.environ.get("BROKER_TRUTH_ALIGNMENT_WAIT_SECONDS", "90"))
ALIGNMENT_POLL_SECONDS = float(os.environ.get("BROKER_TRUTH_ALIGNMENT_POLL_SECONDS", "5"))
REQUIRE_LIVE_OPEN_ORDERS = os.environ.get("BROKER_TRUTH_REQUIRE_LIVE_ORDERS", "0").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
    "on",
}
SIGNAL_WARN_AGE_HOURS = float(os.environ.get("BROKER_TRUTH_SIGNAL_WARN_AGE_HOURS", "36"))
SIGNAL_FAIL_AGE_HOURS = float(os.environ.get("BROKER_TRUTH_SIGNAL_FAIL_AGE_HOURS", "96"))
ORDER_PLAN_WARN_AGE_HOURS = float(os.environ.get("BROKER_TRUTH_ORDER_PLAN_WARN_AGE_HOURS", "36"))
ORDER_PLAN_FAIL_AGE_HOURS = float(os.environ.get("BROKER_TRUTH_ORDER_PLAN_FAIL_AGE_HOURS", "96"))

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
    "open_rebalance_sell_qty",
    "open_rebalance_order_count",
    "trailing_stop_qty",
    "trailing_stop_count",
    "stop_required",
    "issue_severity",
    "issues",
]

RECOVERY_COLUMNS = [
    "run_id",
    "generated_at",
    "signal_as_of",
    "ticker",
    "action",
    "current_quantity",
    "planned_target_quantity",
    "suggested_quantity",
    "quantity_basis",
    "target_weight",
    "broker_weight",
    "weight_gap",
    "target_value",
    "broker_value",
    "corrective_value",
    "reference_price",
    "review_status",
    "reason",
]

INCIDENT_COLUMNS = [
    "incident_id",
    "run_id",
    "status",
    "opened_at",
    "updated_at",
    "resolved_at",
    "duration_seconds",
    "signal_as_of",
    "initial_reason",
    "latest_reason",
    "initial_max_weight_gap",
    "maximum_observed_weight_gap",
    "latest_max_weight_gap",
    "initial_gross_exposure_gap",
    "maximum_observed_gross_gap",
    "latest_gross_exposure_gap",
    "latest_alignment_status",
    "active_rebalance_order_count",
    "recovery_plan_rows",
    "human_action_required",
    "orders_submitted",
    "resolution",
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
        meta["modified_at"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        meta["modified_at"] = ""
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
        meta["modified_at"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        meta["modified_at"] = ""
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


def _timestamp_from_value(value: Any) -> pd.Timestamp | None:
    """Parse a timestamp-like value and keep the exact UTC time."""
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _annotate_age(
    meta: dict[str, Any],
    *,
    clock: datetime,
    label: str,
    timestamp_keys: list[str],
    warn_hours: float,
    fail_hours: float,
) -> tuple[str, str]:
    """Add freshness fields to an input metadata dict.

    PLAIN ENGLISH: broker truth is only useful if the files it compares are
    from roughly the same time. This records file age and returns an optional
    global warning/fail issue.
    """
    meta["freshness_status"] = "unknown"
    meta["age_hours"] = None
    meta["timestamp_used"] = ""
    if not meta.get("exists"):
        return "", ""

    timestamp = None
    timestamp_key = ""
    for key in [*timestamp_keys, "modified_at"]:
        timestamp = _timestamp_from_value(meta.get(key))
        if timestamp is not None:
            timestamp_key = key
            break
    if timestamp is None:
        return "warning", f"{label}_timestamp_missing"

    clock_ts = pd.Timestamp(clock)
    if clock_ts.tzinfo is None:
        clock_ts = clock_ts.tz_localize("UTC")
    else:
        clock_ts = clock_ts.tz_convert("UTC")
    age_hours = (clock_ts - timestamp).total_seconds() / 3600.0
    meta["timestamp_used"] = str(timestamp_key)
    meta["timestamp_utc"] = timestamp.isoformat()
    meta["age_hours"] = round(float(age_hours), 2)

    if age_hours < -0.1 and timestamp_key == "modified_at":
        meta["freshness_status"] = "unknown"
        return "", ""
    if age_hours < -0.1:
        meta["freshness_status"] = "future"
        return "warning", f"{label}_timestamp_from_future_{abs(age_hours):.1f}h"
    if age_hours > float(fail_hours):
        meta["freshness_status"] = "fail"
        return "fail", f"{label}_stale_{age_hours:.1f}h_gt_{float(fail_hours):.1f}h"
    if age_hours > float(warn_hours):
        meta["freshness_status"] = "warning"
        return "warning", f"{label}_stale_{age_hours:.1f}h_gt_{float(warn_hours):.1f}h"
    meta["freshness_status"] = "fresh"
    return "", ""


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
                "open_rebalance_sell_qty": 0.0,
                "open_rebalance_order_count": 0,
                "trailing_stop_qty": 0.0,
                "trailing_stop_count": 0,
                "open_order_ids": [],
            },
        )
        if side == "buy":
            entry["open_buy_qty"] += qty
        if side == "sell":
            entry["open_sell_qty"] += qty
        # PLAIN ENGLISH: stop orders protect a position; they are not today's
        # rebalance waiting to complete. Keep the old all-sells total for
        # compatibility, but count only ordinary sells as alignment-changing.
        if side == "sell" and order_type not in {"stop", "stop_limit", "trailing_stop"}:
            entry["open_rebalance_sell_qty"] += qty
        if qty > QTY_TOLERANCE and (
            side == "buy"
            or (side == "sell" and order_type not in {"stop", "stop_limit", "trailing_stop"})
        ):
            entry["open_rebalance_order_count"] += 1
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


def _alignment_result(
    *,
    rows: list[dict[str, Any]],
    target_comparison_enabled: bool,
    status_meta: dict[str, Any],
    live_positions_meta: dict[str, Any],
    open_orders_meta: dict[str, Any],
    waited_seconds: float = 0.0,
    pending_timed_out: bool = False,
) -> dict[str, Any]:
    """Return the canonical target-versus-Alpaca alignment verdict.

    PLAIN ENGLISH: normal reports may say "collecting" when live proof is
    missing. The enforcing command treats every result except "pass" as a
    failure, but this function never sends orders or changes positions.
    """
    comparable = len(rows) if target_comparison_enabled else 0
    target_gross = sum(abs(_safe_float(row.get("target_weight"))) for row in rows)
    broker_gross = sum(abs(_safe_float(row.get("broker_weight"))) for row in rows)
    max_gap = (
        max(
            (abs(_safe_float(row.get("target_weight")) - _safe_float(row.get("broker_weight"))) for row in rows),
            default=0.0,
        )
        if target_comparison_enabled
        else None
    )
    gross_gap = abs(broker_gross - target_gross) if target_comparison_enabled else None
    active_orders = sum(int(_safe_float(row.get("open_rebalance_order_count"))) for row in rows)

    status = "collecting"
    reason = "target_comparison_unavailable"
    if not target_comparison_enabled:
        pass
    elif not live_positions_meta.get("available") or status_meta.get("source") != "alpaca_api":
        reason = f"live_positions_unavailable:{live_positions_meta.get('error', '')}"
    elif _safe_float(status_meta.get("equity")) <= 0:
        reason = "live_account_equity_unavailable"
    elif not open_orders_meta.get("available"):
        reason = f"live_open_orders_unavailable:{open_orders_meta.get('error', '')}"
    elif active_orders > 0 and pending_timed_out:
        status = "fail"
        reason = "alignment_pending_timeout"
    elif active_orders > 0:
        status = "pending"
        reason = "exposure_changing_orders_open"
    else:
        reasons: list[str] = []
        if max_gap is not None and max_gap > WEIGHT_TOLERANCE:
            reasons.append(f"max_weight_gap_{max_gap:.6f}_above_{WEIGHT_TOLERANCE:.6f}")
        if gross_gap is not None and gross_gap > GROSS_EXPOSURE_TOLERANCE:
            reasons.append(
                f"gross_exposure_gap_{gross_gap:.6f}_above_{GROSS_EXPOSURE_TOLERANCE:.6f}"
            )
        status = "fail" if reasons else "pass"
        reason = ";".join(reasons) if reasons else "ok"

    return {
        "status": status,
        "passed": status == "pass",
        "reason": reason,
        "comparable_tickers": comparable,
        "maximum_target_weight_gap": None if max_gap is None else round(max_gap, 6),
        "current_gross_exposure": None if not target_comparison_enabled else round(broker_gross, 6),
        "target_gross_exposure": None if not target_comparison_enabled else round(target_gross, 6),
        "gross_exposure_gap": None if gross_gap is None else round(gross_gap, 6),
        "weight_tolerance": WEIGHT_TOLERANCE,
        "gross_exposure_tolerance": GROSS_EXPOSURE_TOLERANCE,
        "active_rebalance_order_count": active_orders,
        "waited_seconds": round(max(0.0, float(waited_seconds)), 3),
    }


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
    alignment_waited_seconds: float = 0.0,
    alignment_pending_timed_out: bool = False,
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

    if live_positions_meta.get("available"):
        # PLAIN ENGLISH: when the API read succeeds, the live broker account
        # beats the saved JSON snapshot, even when Alpaca correctly reports an
        # empty all-cash account. The saved file remains an offline fallback.
        positions = live_positions or {}
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

    signal_age_issue = _annotate_age(
        signal_meta,
        clock=clock,
        label="signal",
        timestamp_keys=["as_of", "predicted_at", "generated_at"],
        warn_hours=SIGNAL_WARN_AGE_HOURS,
        fail_hours=max(SIGNAL_FAIL_AGE_HOURS, SIGNAL_WARN_AGE_HOURS),
    )
    if signal_age_issue[0]:
        global_issues.append(signal_age_issue)
    plan_age_issue = _annotate_age(
        plan_meta,
        clock=clock,
        label="order_plan",
        timestamp_keys=["generated_at", "as_of", "predicted_at"],
        warn_hours=ORDER_PLAN_WARN_AGE_HOURS,
        fail_hours=max(ORDER_PLAN_FAIL_AGE_HOURS, ORDER_PLAN_WARN_AGE_HOURS),
    )
    if plan_age_issue[0]:
        global_issues.append(plan_age_issue)

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
        effective_plan = {} if plan_meta.get("freshness_status") == "fail" else plan
        effective_log = log

    # PLAIN ENGLISH: zero parsed targets is missing evidence, not an instruction
    # to compare every Alpaca holding with a 0% target.  Treating an empty or
    # malformed signal as all-cash created false gaps as large as the biggest
    # account position (for example, a 60% QQQ holding looked 60% wrong).
    target_comparison_enabled = bool(
        signal_meta.get("exists")
        and not signal_meta.get("error")
        and int(signal_meta.get("target_count", 0) or 0) > 0
        and signal_meta.get("freshness_status") != "fail"
    )
    if signal_meta.get("exists") and signal_meta.get("rows", 0) and not targets:
        global_issues.append(("fail", "signal_has_no_target_weights"))
    plan_comparison_enabled = bool(effective_plan)
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
        open_rebalance_sell_qty = _safe_float(open_row.get("open_rebalance_sell_qty"))
        trailing_stop_qty = _safe_float(open_row.get("trailing_stop_qty"))
        trailing_stop_count = int(_safe_float(open_row.get("trailing_stop_count")))
        open_rebalance_order_count = int(_safe_float(open_row.get("open_rebalance_order_count")))
        stop_required = _stop_required(symbol, broker_qty, target_weight)

        if plan_comparison_enabled and planned_qty > QTY_TOLERANCE and submitted_qty <= QTY_TOLERANCE and log_meta.get("exists"):
            issues.append(("warning", "planned_order_not_seen_in_latest_log"))
        if failed_qty > QTY_TOLERANCE:
            issues.append(("fail", "latest_logged_order_failed"))
        if skipped_qty > QTY_TOLERANCE:
            issues.append(("warning", "latest_logged_order_skipped"))
        if open_qty > QTY_TOLERANCE:
            issues.append(("warning", "latest_logged_order_still_open"))
        if expected_qty is not None and abs(float(quantity_gap)) > max(QTY_TOLERANCE, 0.01):
            issues.append(("warning", "broker_qty_differs_from_latest_log_expected_qty"))
        if target_comparison_enabled and target_weight <= WEIGHT_TOLERANCE and broker_qty > QTY_TOLERANCE and open_rebalance_sell_qty <= QTY_TOLERANCE:
            issues.append(("warning", "extra_broker_position_not_in_target"))
        if target_comparison_enabled and target_weight > WEIGHT_TOLERANCE and broker_qty <= QTY_TOLERANCE and open_buy_qty <= QTY_TOLERANCE:
            issues.append(("warning", "target_position_missing_at_broker"))
        if target_comparison_enabled and abs_weight_gap > WEIGHT_TOLERANCE and open_buy_qty + open_rebalance_sell_qty <= QTY_TOLERANCE:
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
            "open_rebalance_sell_qty": open_rebalance_sell_qty,
            "open_rebalance_order_count": open_rebalance_order_count,
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

    # PLAIN ENGLISH: these headline numbers let health reports answer the
    # alignment question directly without recalculating it from CSV rows.
    target_gross_exposure = sum(abs(float(weight)) for weight in targets.values())
    broker_gross_exposure = sum(
        abs(_safe_float(position.get("weight"))) for position in positions.values()
    )
    maximum_target_weight_gap = (
        max((abs(float(row["target_weight"]) - float(row["broker_weight"])) for row in rows), default=0.0)
        if target_comparison_enabled
        else None
    )
    alignment = _alignment_result(
        rows=rows,
        target_comparison_enabled=target_comparison_enabled,
        status_meta=status_meta,
        live_positions_meta=live_positions_meta,
        open_orders_meta=open_orders_meta,
        waited_seconds=alignment_waited_seconds,
        pending_timed_out=alignment_pending_timed_out,
    )

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
            "signal_age_hours": signal_meta.get("age_hours"),
            "signal_freshness_status": signal_meta.get("freshness_status"),
            "order_plan_age_hours": plan_meta.get("age_hours"),
            "order_plan_freshness_status": plan_meta.get("freshness_status"),
            "target_comparison_enabled": bool(target_comparison_enabled),
            "order_plan_comparison_enabled": bool(plan_comparison_enabled),
            "target_weight_count": len(targets),
            "target_weights": {ticker: round(weight, 6) for ticker, weight in sorted(targets.items())},
            "target_gross_exposure": round(target_gross_exposure, 6),
            "broker_gross_exposure": round(broker_gross_exposure, 6),
            "maximum_target_weight_gap": (
                None if maximum_target_weight_gap is None else round(maximum_target_weight_gap, 6)
            ),
            "alignment": alignment,
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
            "gross_exposure_tolerance": GROSS_EXPOSURE_TOLERANCE,
        },
        "global_issues": [
            {"severity": severity, "issue": message} for severity, message in global_issues
        ],
        "rows": rows,
    }
    return payload


def build_alignment_recovery_plan(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build manual-review correction ideas for a settled alignment failure.

    PLAIN ENGLISH: this is a calculator, not a trader. It writes what would
    move each mismatched holding toward the target, but nothing here calls
    Alpaca or submits an order. Missing prices stay blank instead of guessing.
    """
    summary = payload.get("summary", {}) or {}
    alignment = summary.get("alignment", {}) or {}
    if alignment.get("status") != "fail":
        return []
    if int(alignment.get("active_rebalance_order_count", 0) or 0) > 0:
        # An open order may still fix the gap. Suggesting another order could
        # duplicate it, so pending-timeout incidents receive no repair rows.
        return []
    reason = str(alignment.get("reason", "") or "")
    if "weight_gap" not in reason and "gross_exposure_gap" not in reason:
        return []

    equity = _safe_float(summary.get("account_equity"))
    if equity <= 0:
        return []
    signal_as_of = str(payload.get("inputs", {}).get("signal", {}).get("as_of", "") or "")
    generated_at = str(payload.get("generated_at", "") or "")
    run_id = str(payload.get("run_id") or current_run_id())
    recovery_rows: list[dict[str, Any]] = []

    for row in payload.get("rows", []) or []:
        target_weight = _safe_float(row.get("target_weight"))
        broker_weight = _safe_float(row.get("broker_weight"))
        weight_gap = target_weight - broker_weight
        if abs(weight_gap) <= 1e-9:
            continue

        ticker = _ticker(row.get("ticker"))
        current_qty = max(0.0, _safe_float(row.get("broker_qty")))
        broker_value = _safe_float(row.get("broker_value"))
        target_value = target_weight * equity
        corrective_value = target_value - broker_value
        action = "buy" if corrective_value > 0 else "sell"

        planned_target_qty = _float_or_none(row.get("planned_target_qty"))
        planned_target_weight = _float_or_none(row.get("planned_target_weight"))
        reference_price = broker_value / current_qty if current_qty > 0 and broker_value > 0 else None
        suggested_qty: float | None = None
        quantity_basis = "fresh_quote_required"

        # The current order plan is the safest quantity source when it matches
        # this exact target. Otherwise estimate only from the saved snapshot.
        if (
            planned_target_qty is not None
            and planned_target_weight is not None
            and abs(planned_target_weight - target_weight) <= 1e-6
        ):
            suggested_qty = abs(planned_target_qty - current_qty)
            quantity_basis = "current_order_plan"
            if reference_price is None and planned_target_qty > 0 and target_value > 0:
                reference_price = target_value / planned_target_qty
        elif reference_price is not None and reference_price > 0:
            suggested_qty = abs(corrective_value) / reference_price
            quantity_basis = "saved_broker_snapshot_estimate"

        recovery_rows.append(
            {
                "run_id": run_id,
                "generated_at": generated_at,
                "signal_as_of": signal_as_of,
                "ticker": ticker,
                "action": action,
                "current_quantity": round(current_qty, 6),
                "planned_target_quantity": (
                    "" if planned_target_qty is None else round(planned_target_qty, 6)
                ),
                "suggested_quantity": "" if suggested_qty is None else round(suggested_qty, 6),
                "quantity_basis": quantity_basis,
                "target_weight": round(target_weight, 6),
                "broker_weight": round(broker_weight, 6),
                "weight_gap": round(weight_gap, 6),
                "target_value": round(target_value, 2),
                "broker_value": round(broker_value, 2),
                "corrective_value": round(corrective_value, 2),
                "reference_price": "" if reference_price is None else round(reference_price, 6),
                "review_status": "manual_review_required_not_submitted",
                "reason": reason,
            }
        )
    return recovery_rows


def update_alignment_incident_ledger(
    payload: dict[str, Any],
    *,
    ledger_path: Path,
    recovery_plan_rows: int,
) -> dict[str, Any]:
    """Open, update, or resolve durable alignment incidents.

    PLAIN ENGLISH: repeated polling updates one incident instead of adding
    duplicate rows. A later passing live check closes open incidents and keeps
    the full history. This ledger records events only; it never trades.
    """
    if ledger_path.exists() and ledger_path.stat().st_size > 0:
        try:
            ledger = pd.read_csv(ledger_path, dtype=object)
        except Exception as exc:
            raise RuntimeError(f"Alignment incident ledger is unreadable: {exc}") from exc
        for column in INCIDENT_COLUMNS:
            if column not in ledger.columns:
                ledger[column] = ""
        ledger = ledger[INCIDENT_COLUMNS].copy()
    else:
        ledger = pd.DataFrame(columns=INCIDENT_COLUMNS)
    # Pandas may infer strict Arrow string columns from the first mostly-empty
    # row. Object columns safely hold timestamps, numbers, booleans, and blanks.
    ledger = ledger.astype(object)

    summary = payload.get("summary", {}) or {}
    alignment = summary.get("alignment", {}) or {}
    alignment_status = str(alignment.get("status", "collecting") or "collecting")
    generated_at = str(payload.get("generated_at", "") or _now_utc().isoformat(timespec="seconds"))
    signal_as_of = str(payload.get("inputs", {}).get("signal", {}).get("as_of", "") or "")
    run_id = str(payload.get("run_id") or current_run_id())
    reason = str(alignment.get("reason", "") or "")
    max_gap = alignment.get("maximum_target_weight_gap")
    gross_gap = alignment.get("gross_exposure_gap")

    open_mask = (
        ledger["status"].astype(str).str.lower().eq("open")
        if not ledger.empty
        else pd.Series(dtype=bool)
    )
    same_signal_mask = (
        open_mask & ledger["signal_as_of"].astype(str).eq(signal_as_of)
        if not ledger.empty
        else pd.Series(dtype=bool)
    )
    current_incident_id = ""

    if alignment_status == "fail":
        if same_signal_mask.any():
            index = ledger.index[same_signal_mask][-1]
        else:
            incident_seed = f"{signal_as_of}|{generated_at}|{reason}"
            current_incident_id = hashlib.sha256(incident_seed.encode("utf-8")).hexdigest()[:16]
            new_row = {column: "" for column in INCIDENT_COLUMNS}
            new_row.update(
                {
                    "incident_id": current_incident_id,
                    "run_id": run_id,
                    "status": "open",
                    "opened_at": generated_at,
                    "signal_as_of": signal_as_of,
                    "initial_reason": reason,
                    "initial_max_weight_gap": max_gap,
                    "maximum_observed_weight_gap": max_gap,
                    "initial_gross_exposure_gap": gross_gap,
                    "maximum_observed_gross_gap": gross_gap,
                    "human_action_required": True,
                    "orders_submitted": False,
                }
            )
            new_frame = pd.DataFrame([new_row], columns=INCIDENT_COLUMNS).astype(object)
            ledger = pd.concat([ledger, new_frame], ignore_index=True).astype(object)
            index = ledger.index[-1]

        current_incident_id = str(ledger.at[index, "incident_id"])
        ledger.at[index, "run_id"] = run_id
        old_max_gap = _safe_float(ledger.at[index, "maximum_observed_weight_gap"])
        old_gross_gap = _safe_float(ledger.at[index, "maximum_observed_gross_gap"])
        ledger.at[index, "updated_at"] = generated_at
        ledger.at[index, "latest_reason"] = reason
        ledger.at[index, "latest_max_weight_gap"] = max_gap
        ledger.at[index, "latest_gross_exposure_gap"] = gross_gap
        ledger.at[index, "maximum_observed_weight_gap"] = max(old_max_gap, _safe_float(max_gap))
        ledger.at[index, "maximum_observed_gross_gap"] = max(old_gross_gap, _safe_float(gross_gap))
        ledger.at[index, "latest_alignment_status"] = alignment_status
        ledger.at[index, "active_rebalance_order_count"] = int(
            alignment.get("active_rebalance_order_count", 0) or 0
        )
        ledger.at[index, "recovery_plan_rows"] = int(recovery_plan_rows)
        ledger.at[index, "human_action_required"] = True
        ledger.at[index, "orders_submitted"] = False

    elif alignment_status == "pending" and open_mask.any():
        # Preserve the incident while Alpaca still has an exposure-changing
        # order. Do not resolve it until a later live alignment pass.
        index = ledger.index[open_mask][-1]
        current_incident_id = str(ledger.at[index, "incident_id"])
        ledger.at[index, "updated_at"] = generated_at
        ledger.at[index, "latest_reason"] = reason
        ledger.at[index, "latest_alignment_status"] = alignment_status
        ledger.at[index, "active_rebalance_order_count"] = int(
            alignment.get("active_rebalance_order_count", 0) or 0
        )

    elif alignment_status == "pass" and open_mask.any():
        resolved_time = pd.to_datetime(generated_at, errors="coerce", utc=True)
        for index in ledger.index[open_mask]:
            opened_time = pd.to_datetime(ledger.at[index, "opened_at"], errors="coerce", utc=True)
            duration = ""
            if pd.notna(resolved_time) and pd.notna(opened_time):
                duration = round(max(0.0, (resolved_time - opened_time).total_seconds()), 3)
            ledger.at[index, "status"] = "resolved"
            ledger.at[index, "updated_at"] = generated_at
            ledger.at[index, "resolved_at"] = generated_at
            ledger.at[index, "duration_seconds"] = duration
            ledger.at[index, "latest_reason"] = reason
            ledger.at[index, "latest_max_weight_gap"] = max_gap
            ledger.at[index, "latest_gross_exposure_gap"] = gross_gap
            ledger.at[index, "latest_alignment_status"] = alignment_status
            ledger.at[index, "active_rebalance_order_count"] = 0
            ledger.at[index, "recovery_plan_rows"] = 0
            ledger.at[index, "human_action_required"] = False
            ledger.at[index, "orders_submitted"] = False
            ledger.at[index, "resolution"] = "alignment_passed"

    atomic_write_csv(ledger, ledger_path)
    remaining_open = int(ledger["status"].astype(str).str.lower().eq("open").sum()) if not ledger.empty else 0
    return {
        "path": str(ledger_path),
        "total_incidents": int(len(ledger)),
        "open_incidents": remaining_open,
        "current_incident_id": current_incident_id,
        "orders_submitted": False,
    }


def write_broker_truth(
    *,
    output_csv: Path = BROKER_TRUTH_CSV,
    output_json: Path = BROKER_TRUTH_JSON,
    output_recovery_csv: Path | None = None,
    output_incident_ledger_csv: Path | None = None,
    log_dir: Path = LOGS,
    now: datetime | None = None,
    manage_alignment_lifecycle: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build broker truth, then write latest CSV/JSON plus a dated snapshot."""
    clock = now or _now_utc()
    recovery_path = output_recovery_csv or output_csv.parent / ALIGNMENT_RECOVERY_PLAN_CSV.name
    incident_path = (
        output_incident_ledger_csv or output_csv.parent / ALIGNMENT_INCIDENT_LEDGER_CSV.name
    )
    payload = build_broker_truth(now=clock, **kwargs)
    signal_as_of = str(payload.get("inputs", {}).get("signal", {}).get("as_of", "") or "")
    payload = enrich_payload(payload, signal_as_of=signal_as_of, now=clock)
    rows = payload.get("rows", [])
    df = pd.DataFrame(rows, columns=TRUTH_COLUMNS)
    atomic_write_csv(df, output_csv)

    if manage_alignment_lifecycle:
        # PLAIN ENGLISH: only the post-trade enforcing check may open/resolve
        # incidents or replace the recovery plan. Pre-submit reports must not
        # mistake an expected new target for a post-trade failure.
        recovery_rows = build_alignment_recovery_plan(payload)
        payload.setdefault("summary", {})["alignment_recovery_plan"] = {
            "path": str(recovery_path),
            "row_count": len(recovery_rows),
            "review_required": bool(recovery_rows),
            "orders_submitted": False,
        }
        payload["summary"]["alignment_incident_ledger"] = update_alignment_incident_ledger(
            payload,
            ledger_path=incident_path,
            recovery_plan_rows=len(recovery_rows),
        )
        recovery_df = pd.DataFrame(recovery_rows, columns=RECOVERY_COLUMNS)
        # Always write the header, even on pass. This clears yesterday's stale
        # recovery suggestions after the account returns to alignment.
        atomic_write_csv(recovery_df, recovery_path)
        alignment = payload.get("summary", {}).get("alignment", {}) or {}
        alignment_status = str(alignment.get("status", "collecting"))
        lifecycle_status = {
            "pass": "aligned",
            "pending": "submitted",
            "fail": "timed_out" if alignment.get("reason") == "alignment_pending_timeout" else "rejected",
        }.get(alignment_status, "planned")
        update_rebalance_state(
            lifecycle_status,
            run_id=payload.get("run_id"),
            signal_as_of=signal_as_of,
            details={
                "alignment_status": alignment_status,
                "reason": alignment.get("reason"),
                "active_rebalance_order_count": alignment.get("active_rebalance_order_count", 0),
                "maximum_target_weight_gap": alignment.get("maximum_target_weight_gap"),
                "gross_exposure_gap": alignment.get("gross_exposure_gap"),
            },
        )
    atomic_write_json(payload, output_json)

    dated = log_dir / f"broker_truth_{clock.strftime('%Y%m%d')}.json"
    atomic_write_json(payload, dated)
    return payload


def wait_for_required_alignment(
    *,
    wait_seconds: float = ALIGNMENT_WAIT_SECONDS,
    poll_seconds: float = ALIGNMENT_POLL_SECONDS,
    write_fn: Callable[..., dict[str, Any]] = write_broker_truth,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll live Alpaca until alignment settles or the bounded wait expires.

    PLAIN ENGLISH: an order can be accepted before Alpaca updates positions.
    We wait only for normal exposure-changing orders. Protective stops do not
    delay this check, and this function never submits a repair order.
    """
    wait_limit = max(0.0, float(wait_seconds))
    poll_interval = max(0.1, float(poll_seconds))
    started = monotonic_fn()

    while True:
        elapsed = max(0.0, monotonic_fn() - started)
        payload = write_fn(
            include_live_open_orders=True,
            alignment_waited_seconds=elapsed,
            manage_alignment_lifecycle=True,
        )
        alignment = payload.get("summary", {}).get("alignment", {})
        if alignment.get("status") != "pending":
            return payload
        if elapsed >= wait_limit:
            # Refresh once more at the deadline so a just-completed fill can
            # pass instead of being mislabeled as a timeout.
            return write_fn(
                include_live_open_orders=True,
                alignment_waited_seconds=elapsed,
                alignment_pending_timed_out=True,
                manage_alignment_lifecycle=True,
            )

        remaining = wait_limit - elapsed
        sleep_fn(min(poll_interval, remaining))


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
    alignment = summary.get("alignment", {}) or {}
    print(
        "Alignment: "
        f"{str(alignment.get('status', 'collecting')).upper()} | "
        f"max gap={alignment.get('maximum_target_weight_gap')} | "
        f"gross gap={alignment.get('gross_exposure_gap')} | "
        f"reason={alignment.get('reason', '')}"
    )
    recovery = summary.get("alignment_recovery_plan", {}) or {}
    print(
        "Recovery plan: "
        f"rows={recovery.get('row_count', 0)} | "
        f"review_required={recovery.get('review_required', False)} | "
        "orders_submitted=False"
    )
    incidents = summary.get("alignment_incident_ledger", {}) or {}
    print(
        "Alignment incidents: "
        f"open={incidents.get('open_incidents', 0)} | "
        f"total={incidents.get('total_incidents', 0)} | "
        f"current={incidents.get('current_incident_id', '')}"
    )
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
    print(f"Saved: {ALIGNMENT_RECOVERY_PLAN_CSV}")
    print(f"Saved: {ALIGNMENT_INCIDENT_LEDGER_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Alpaca broker truth with local bot records.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when broker truth status is fail.")
    parser.add_argument(
        "--require-alignment",
        action="store_true",
        help="Poll live Alpaca and exit nonzero unless settled target alignment passes.",
    )
    parser.add_argument(
        "--alignment-wait-seconds",
        type=float,
        default=ALIGNMENT_WAIT_SECONDS,
        help="Maximum seconds to wait for exposure-changing orders (default: 90).",
    )
    parser.add_argument(
        "--alignment-poll-seconds",
        type=float,
        default=ALIGNMENT_POLL_SECONDS,
        help="Seconds between live alignment checks (default: 5).",
    )
    parser.add_argument("--offline", "--no-live-open-orders", dest="offline", action="store_true",
                        help="Do not call Alpaca for live open/trailing orders.")
    args = parser.parse_args()

    if args.require_alignment and not args.offline:
        payload = wait_for_required_alignment(
            wait_seconds=args.alignment_wait_seconds,
            poll_seconds=args.alignment_poll_seconds,
        )
    else:
        payload = write_broker_truth(include_live_open_orders=not args.offline)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print_summary(payload)

    if args.strict and payload.get("status") == "fail":
        sys.exit(1)
    if args.require_alignment:
        alignment_status = payload.get("summary", {}).get("alignment", {}).get("status")
        if alignment_status != "pass":
            sys.exit(1)


if __name__ == "__main__":
    main()
