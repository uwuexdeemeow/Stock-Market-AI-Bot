"""Complete, deduplicated broker history with explicit incompleteness evidence."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class HistoryResult:
    orders: list
    complete: bool
    errors: list[str]


def field(order, name, default=None):
    return order.get(name, default) if isinstance(order, dict) else getattr(order, name, default)


def collect_order_history(api, *, after, until=None, page_size=100, maximum_pages=10000) -> HistoryResult:
    """Page backward with an overlapping timestamp; detect saturated boundaries.

    The timestamp-only API cannot disambiguate more than a full page sharing
    one time. Such a boundary is reported incomplete, never skipped by 1 ns.
    """
    since = pd.to_datetime(after, utc=True)
    until = pd.to_datetime(until, utc=True) if until is not None else pd.Timestamp.now(tz="UTC")
    requested_until = until
    until += pd.Timedelta(microseconds=1)
    found, errors = {}, []
    seen_pages = set()
    complete = False
    for _ in range(maximum_pages):
        try:
            page = list(api.list_orders(status="all", after=(since - pd.Timedelta(microseconds=1)).isoformat(), until=until.isoformat(),
                                        limit=page_size, direction="desc", nested=False))
        except Exception as exc:
            errors.append(f"history_request_failed:{type(exc).__name__}:{exc}")
            break
        identities = tuple(str(field(order, "id", "")) for order in page)
        if page and (identities in seen_pages or any(not key for key in identities)):
            errors.append("history_pagination_stalled_or_missing_id")
            break
        seen_pages.add(identities)
        timestamps = []
        for order, identity in zip(page, identities):
            timestamp = pd.to_datetime(field(order, "submitted_at", field(order, "created_at")), utc=True, errors="coerce")
            if pd.isna(timestamp):
                errors.append(f"history_timestamp_missing:{identity}")
                continue
            timestamps.append(timestamp)
            if since <= timestamp <= requested_until:
                found[identity] = order
        if len(page) < page_size:
            complete = not errors
            break
        if not timestamps:
            errors.append("history_no_usable_cursor")
            break
        oldest = min(timestamps)
        # Include the boundary again, ensuring another order at that time is not lost.
        next_until = oldest + pd.Timedelta(microseconds=1)
        if next_until >= until:
            errors.append("history_saturated_timestamp_boundary")
            break
        until = next_until
    else:
        errors.append("history_page_limit_reached")
    return HistoryResult(list(found.values()), complete, errors)


def implementation_shortfall(rows: pd.DataFrame) -> dict:
    """Use parent arrival prices for fills; missing historical quotes stay missing."""
    required = {"parent_order_id", "side", "filled_qty", "fill_price", "arrival_mid"}
    if not required.issubset(rows):
        return {"complete": False, "reason": "arrival_evidence_missing"}
    frame = rows.copy()
    # Reports contain cumulative child-order fills, not one row per execution.
    # Keep each child only once so replacement parents do not count twice.
    if "order_id" in frame:
        frame = frame.drop_duplicates("order_id", keep="last")
    for column in ("filled_qty", "fill_price", "arrival_mid"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["filled_qty", "fill_price", "arrival_mid"])
    valid = valid[(valid.arrival_mid > 0) & (valid.filled_qty > 0) & (valid.fill_price > 0) & valid.side.isin(["buy", "sell"])].copy()
    valid["shortfall_bps"] = (valid.fill_price / valid.arrival_mid - 1) * valid.side.map({"buy": 1, "sell": -1}) * 10000
    groups = {}
    for dimension in ("side", "execution_stage", "spread_bucket", "liquidity_bucket", "size_bucket", "session"):
        if dimension in valid:
            groups[dimension] = {str(key): {"count": len(group), "median_bps": float(group.shortfall_bps.median()),
                                           "p95_bps": float(group.shortfall_bps.quantile(.95))}
                                 for key, group in valid.groupby(dimension)}
    return {"complete": len(valid) == len(frame), "matched_fills": len(valid), "missing_arrival_fills": len(frame) - len(valid),
            "groups": groups, "fills": valid.to_dict("records")}


def parent_execution_summary(rows: pd.DataFrame) -> dict:
    """Sum child fills once; leave unfilled opportunity cost unknown without a mark.

    Input is one cumulative snapshot per child, including canceled/unfilled
    children. The parent quantity must come from the original decision journal.
    An opportunity mark needs its documented timestamp and chosen horizon.
    """
    required = {"order_id", "parent_order_id", "filled_qty", "side"}
    if not required.issubset(rows):
        return {"complete": False, "reason": "parent_evidence_missing", "parents": []}
    frame = rows.drop_duplicates("order_id", keep="last").copy()
    parents = []
    for parent, children in frame.groupby("parent_order_id", dropna=False):
        requested = pd.to_numeric(children.get("original_requested_quantity", pd.Series(dtype=float)), errors="coerce").dropna().unique()
        quantities = pd.to_numeric(children.filled_qty, errors="coerce")
        sides = children.side.dropna().unique()
        known = len(requested) == 1 and np.isfinite(requested[0]) and requested[0] > 0 and np.isfinite(quantities).all() and (quantities >= 0).all() and len(sides) == 1 and sides[0] in {"buy", "sell"}
        filled = float(quantities.sum()) if np.isfinite(quantities).all() else None
        known = bool(known and filled <= requested[0] + 1e-8) if known else False
        remaining = max(0., float(requested[0]) - filled) if known else None
        opportunity = None
        if known and remaining == 0:
            opportunity = 0.
        elif known and {"arrival_mid", "opportunity_price", "opportunity_timestamp", "opportunity_horizon"}.issubset(children):
            marks = children[["arrival_mid", "opportunity_price", "opportunity_timestamp", "opportunity_horizon"]].dropna().drop_duplicates()
            if len(marks) == 1:
                mark = marks.iloc[0]
                if np.isfinite([float(mark.arrival_mid), float(mark.opportunity_price)]).all() and min(float(mark.arrival_mid), float(mark.opportunity_price)) > 0:
                    opportunity = remaining * (float(mark.opportunity_price) - float(mark.arrival_mid)) * (1 if sides[0] == "buy" else -1)
        parents.append({"parent_order_id": str(parent), "child_orders": len(children), "filled_quantity": filled,
                        "original_requested_quantity": float(requested[0]) if len(requested) == 1 else None,
                        "unfilled_quantity": remaining, "quantity_complete": known,
                        "opportunity_cost_dollars": opportunity,
                        "opportunity_cost_complete": opportunity is not None,
                        "canceled_children": int(children.get("status", pd.Series(dtype=str)).isin(["canceled", "expired"]).sum()),
                        "filled_children": int((quantities > 0).sum())})
    return {"complete": bool(parents) and all(p["quantity_complete"] and p["opportunity_cost_complete"] for p in parents),
            "parent_count": len(parents), "child_order_count": len(frame), "parents": parents}
