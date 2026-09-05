"""Classify broker evidence as logical rebalance orders.

PLAIN ENGLISH: one portfolio decision can create a passive ``-a1`` order and a
repriced ``-a2`` order.  Those are two attempts, but only one requested trade.
This module gives the scorecard and validation epoch the same honest counts.
"""
from __future__ import annotations

import re

import pandas as pd


NON_ACCEPTED_STATUSES = {"skipped", "submission_failed", "rejected", "error", "failed"}
TERMINAL_UNFILLED_STATUSES = {"canceled", "cancelled", "expired"}
PROTECTIVE_ORDER_TYPES = {"trailing_stop", "stop", "stop_limit"}
CHILD_SUFFIX = re.compile(r"-a([12])$")


def _text(row: pd.Series, *columns: str) -> str:
    """Return the first useful text value across new and legacy column names."""
    for column in columns:
        value = row.get(column)
        if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}:
            return str(value).strip()
    return ""


def _number(row: pd.Series, *columns: str) -> float | None:
    """Return the first usable number across new and legacy columns."""
    for column in columns:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return float(value)
    return None


def _logical_key(row: pd.Series) -> tuple[str, str]:
    """Return ``(key, source)`` using the safest available identity."""
    parent = _text(row, "parent_order_id", "parent_client_order_id")
    if parent:
        return parent, "parent_order_id"
    client_id = _text(row, "client_order_id")
    if client_id:
        return CHILD_SUFFIX.sub("", client_id), "client_order_id"

    # PLAIN ENGLISH: old logs did not store the deterministic client ID.  The
    # trading day plus requested trade details reconstructs the same idea.
    when = pd.to_datetime(_text(row, "submitted_at", "date"), errors="coerce", utc=True)
    symbol = _text(row, "ticker", "symbol").upper()
    side = _text(row, "side", "action").lower()
    quantity = _number(row, "requested_quantity", "quantity", "qty")
    if pd.isna(when) or not symbol or side not in {"buy", "sell"} or quantity is None or quantity <= 0:
        return "", "unclassifiable"
    new_york_day = when.tz_convert("America/New_York").date().isoformat()
    return f"legacy:{new_york_day}:{symbol}:{side}:{quantity:g}", "legacy_fields"


def _is_protective(row: pd.Series) -> bool:
    """Protective exits are safety orders, not normal rebalance decisions."""
    order_type = _text(row, "order_type", "submitted_order_type").lower()
    group = _text(row, "order_group", "execution_group").lower()
    return order_type in PROTECTIVE_ORDER_TYPES or group in {"protective_stop", "trailing_stop"}


def _is_accepted(row: pd.Series) -> bool:
    """An accepted row has a real broker ID and was not rejected locally/by broker."""
    status = _text(row, "fill_status", "status").lower()
    broker_id = _text(row, "order_id", "broker_order_id", "stage1_order_id", "stage2_order_id")
    return bool(
        broker_id
        and not broker_id.startswith(("ERROR", "SKIPPED"))
        and status not in NON_ACCEPTED_STATUSES
    )


def _attempt_ids(row: pd.Series) -> dict[str, set[str]]:
    """Map broker IDs to attempt slots so normal a1/a2 repricing is not duplicate."""
    attempts: dict[str, set[str]] = {}
    for column, slot in (("stage1_order_id", "a1"), ("stage2_order_id", "a2")):
        value = _text(row, column)
        if value:
            attempts.setdefault(slot, set()).add(value)
    client_id = _text(row, "client_order_id")
    order_id = _text(row, "order_id", "broker_order_id")
    suffix = CHILD_SUFFIX.search(client_id)
    if order_id and not attempts:
        attempts.setdefault(f"a{suffix.group(1)}" if suffix else "logical", set()).add(order_id)
    return attempts


def classify_logical_orders(frame: pd.DataFrame) -> dict:
    """Return logical-order observations and denominator-safe aggregate counts."""
    if frame.empty:
        return {
            "logical_orders": [],
            "rebalance_rows": 0,
            "accepted_logical_orders": 0,
            "fully_filled_logical_orders": 0,
            "any_filled_logical_orders": 0,
            "partially_filled_logical_orders": 0,
            "open_logical_orders": 0,
            "canceled_unfilled_logical_orders": 0,
            "skipped_rows": 0,
            "unclassifiable_rows": 0,
            "unclassifiable_logical_orders": 0,
            "duplicate_logical_orders": 0,
            "duplicate_child_attempts": 0,
            "child_attempts": 0,
            "complete_fill_rate": None,
            "any_fill_rate": None,
        }

    work = frame.copy()
    protective = work.apply(_is_protective, axis=1)
    work = work[~protective].copy()
    statuses = work.apply(lambda row: _text(row, "fill_status", "status").lower(), axis=1)
    ids = work.apply(lambda row: _text(row, "order_id", "broker_order_id"), axis=1)
    skipped_rows = int((statuses.eq("skipped") | ids.str.startswith("SKIPPED")).sum())
    accepted_mask = work.apply(_is_accepted, axis=1)
    accepted = work[accepted_mask].copy()

    keys = accepted.apply(_logical_key, axis=1)
    accepted["_logical_key"] = [item[0] for item in keys]
    accepted["_key_source"] = [item[1] for item in keys]
    unclassifiable_rows = int(accepted["_logical_key"].eq("").sum())
    classifiable = accepted[accepted["_logical_key"].ne("")]

    logical_orders: list[dict] = []
    duplicate_logical_orders = 0
    duplicate_child_attempts = 0
    child_attempts = 0
    for key, group in classifiable.groupby("_logical_key", sort=False):
        attempt_ids: dict[str, set[str]] = {}
        for _, row in group.iterrows():
            for slot, values in _attempt_ids(row).items():
                attempt_ids.setdefault(slot, set()).update(values)
        child_attempts += sum(len(values) for values in attempt_ids.values())
        duplicate_attempts = sum(max(0, len(values) - 1) for values in attempt_ids.values())
        duplicate_child_attempts += duplicate_attempts
        if duplicate_attempts:
            duplicate_logical_orders += 1

        requested_values = [
            # The journal's quantity is the actual order after cash clamping;
            # requested_quantity may describe the larger, pre-clamp wish.
            (_number(row, "requested_quantity", "quantity", "qty")
             if CHILD_SUFFIX.search(_text(row, "client_order_id"))
             else _number(row, "quantity", "requested_quantity", "qty"))
            for _, row in group.iterrows()
        ]
        requested = max((value for value in requested_values if value is not None), default=None)
        filled_values = [
            _number(row, "filled_qty", "broker_dealt_qty") or 0.0
            for _, row in group.iterrows()
        ]
        # Logical journal rows already contain the combined fill quantity;
        # separate child rows need summing. Taking the maximum protects the
        # normal journal shape from being counted twice.
        separate_children = bool(set(attempt_ids) & {"a1", "a2"}) and len(group) > 1
        if separate_children and not any(_text(row, "stage1_order_id", "stage2_order_id") for _, row in group.iterrows()):
            # Broker snapshots repeat cumulative fills. Count each child ID
            # once, taking its largest observed filled quantity.
            child_fills: dict[str, float] = {}
            for (_, row), value in zip(group.iterrows(), filled_values):
                child_id = _text(row, "order_id", "broker_order_id")
                child_fills[child_id] = max(child_fills.get(child_id, 0.0), value)
            filled_qty = sum(child_fills.values())
        else:
            filled_qty = max(filled_values, default=0.0)
        status_values = {
            _text(row, "fill_status", "status").lower() for _, row in group.iterrows()
        }
        fully_filled = bool(
            (requested is not None and requested > 0 and filled_qty >= requested - 1e-9)
            # Legacy logs sometimes have no fill quantity. A child saying
            # filled must never override a known shortfall in the parent.
            or ("filled" in status_values and requested is not None
                and not separate_children
                and all(_number(row, "filled_qty", "broker_dealt_qty") is None for _, row in group.iterrows()))
        )
        any_filled = bool(filled_qty > 0 or status_values & {"filled", "partial", "partially_filled"})
        partial = any_filled and not fully_filled
        canceled_unfilled = not any_filled and bool(status_values & TERMINAL_UNFILLED_STATUSES)
        open_order = not any_filled and not canceled_unfilled
        logical_orders.append({
            "logical_order_id": key,
            "identity_source": str(group["_key_source"].iloc[0]),
            "requested_quantity": requested,
            "classification_complete": requested is not None and requested > 0,
            "filled_quantity": round(float(filled_qty), 8),
            "fully_filled": fully_filled,
            "any_filled": any_filled,
            "partially_filled": partial,
            "open": open_order,
            "canceled_unfilled": canceled_unfilled,
            "attempt_count": sum(len(values) for values in attempt_ids.values()),
            "duplicate_attempts": duplicate_attempts,
        })

    accepted_count = len(logical_orders)
    full_count = sum(bool(row["fully_filled"]) for row in logical_orders)
    any_count = sum(bool(row["any_filled"]) for row in logical_orders)
    unclassifiable_logical_orders = sum(not bool(row["classification_complete"]) for row in logical_orders)
    return {
        "logical_orders": logical_orders,
        "rebalance_rows": int(len(work)),
        "accepted_logical_orders": accepted_count,
        "fully_filled_logical_orders": full_count,
        "any_filled_logical_orders": any_count,
        "partially_filled_logical_orders": sum(bool(row["partially_filled"]) for row in logical_orders),
        "open_logical_orders": sum(bool(row["open"]) for row in logical_orders),
        "canceled_unfilled_logical_orders": sum(bool(row["canceled_unfilled"]) for row in logical_orders),
        "skipped_rows": skipped_rows,
        "unclassifiable_rows": unclassifiable_rows,
        "unclassifiable_logical_orders": unclassifiable_logical_orders,
        "duplicate_logical_orders": duplicate_logical_orders,
        "duplicate_child_attempts": duplicate_child_attempts,
        "child_attempts": child_attempts,
        "complete_fill_rate": round(full_count / accepted_count, 4) if accepted_count else None,
        "any_fill_rate": round(any_count / accepted_count, 4) if accepted_count else None,
    }
