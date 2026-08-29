"""
execution_scorecard.py - Fill-quality scorecard for Alpaca paper execution.

PLAIN ENGLISH:
The trading bot already logs what it tried to trade and how fills behaved.
This script turns those raw logs into one scorecard that answers:
"Are our orders filling well, and did the execution-risk throttle help?"

Run:
    python execution_scorecard.py
    python execution_scorecard.py --json
    python execution_scorecard.py --strict
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from safe_io import atomic_write_json, configure_console_output
from settings import LOG_DIR, SIGNAL_DIR


configure_console_output()

SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)
PAPER_LOG_FILE = SIGNALS / "alpaca_paper_log.csv"
SLIPPAGE_REPORT_FILE = SIGNALS / "alpaca_slippage_reversal_report.json"
SCORECARD_FILE = SIGNALS / "alpaca_execution_scorecard.json"

# PLAIN ENGLISH: These thresholds define "good enough" execution quality.
# They are intentionally environment variables so you can tighten or loosen
# the scorecard without editing code.
MAX_AVG_SLIPPAGE_BPS = float(os.environ.get("EXECUTION_SCORECARD_MAX_AVG_SLIPPAGE_BPS", "10"))
MAX_BAD_SLIPPAGE_RATE = float(os.environ.get("EXECUTION_SCORECARD_MAX_BAD_SLIPPAGE_RATE", "0.60"))
BAD_SLIPPAGE_BPS = float(os.environ.get("EXECUTION_SCORECARD_BAD_SLIPPAGE_BPS", "2"))
MIN_FILL_RATE = float(os.environ.get("EXECUTION_SCORECARD_MIN_FILL_RATE", "0.80"))
MAX_SKIPPED_RATE = float(os.environ.get("EXECUTION_SCORECARD_MAX_SKIPPED_RATE", "0.35"))
MAX_ADVERSE_15M_RATE = float(os.environ.get("EXECUTION_SCORECARD_MAX_ADVERSE_15M_RATE", "0.60"))
MAX_ADVERSE_60M_RATE = float(os.environ.get("EXECUTION_SCORECARD_MAX_ADVERSE_60M_RATE", "0.70"))
LOOKBACK_DAYS = int(os.environ.get("EXECUTION_SCORECARD_LOOKBACK_DAYS", "30"))
MIN_DECISION_ORDERS = int(os.environ.get("EXECUTION_SCORECARD_MIN_DECISION_ORDERS", "20"))
MIN_DECISION_COVERAGE = float(os.environ.get("EXECUTION_SCORECARD_MIN_DECISION_COVERAGE", "0.80"))
WARN_AVG_SLIPPAGE_BPS = float(os.environ.get("EXECUTION_SCORECARD_WARN_AVG_SLIPPAGE_BPS", "5"))
WARN_BAD_SLIPPAGE_RATE = float(os.environ.get("EXECUTION_SCORECARD_WARN_BAD_SLIPPAGE_RATE", "0.40"))
MIN_REBALANCE_FILLS = int(os.environ.get("EXECUTION_SCORECARD_MIN_REBALANCE_FILLS", "10"))
MIN_REBALANCE_SESSIONS = int(os.environ.get("EXECUTION_SCORECARD_MIN_REBALANCE_SESSIONS", "3"))


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV and return an empty table if it is missing or broken."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    """Read JSON and return an empty dict if it is missing or broken."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _to_float(value: object) -> float | None:
    """Convert text/numbers into a finite float, or None if unusable."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _first_nonempty(row: pd.Series | dict, *keys: str, default: object = None) -> object:
    """Return the first non-blank value from a row with possible old/new names."""
    for key in keys:
        value = row.get(key, None)
        if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}:
            return value
    return default


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    """Safe division for rates such as 7 filled / 10 accepted = 0.70."""
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _series_text(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a text column even when the CSV does not have that column."""
    series = frame.get(column, pd.Series("", index=frame.index))
    return series.fillna("").astype(str).replace({"nan": "", "None": ""}).str.strip()


def _filter_lookback(frame: pd.DataFrame, *, now: datetime, days: int) -> pd.DataFrame:
    """Keep only recent rows based on submitted_at when available."""
    if frame.empty or "submitted_at" not in frame.columns or days <= 0:
        return frame.copy()
    submitted = pd.to_datetime(frame["submitted_at"], errors="coerce", utc=True)
    cutoff = pd.Timestamp(now.astimezone(timezone.utc)) - pd.Timedelta(days=int(days))
    return frame[submitted.isna() | (submitted >= cutoff)].copy()


def _filled_mask(frame: pd.DataFrame) -> pd.Series:
    """Return True for orders that filled fully or partially."""
    if frame.empty:
        return pd.Series(dtype=bool)
    status = _series_text(frame, "fill_status").str.lower()
    filled_qty = pd.to_numeric(
        frame.get("filled_qty", frame.get("broker_dealt_qty", pd.Series(0, index=frame.index))),
        errors="coerce",
    ).fillna(0.0)
    return status.isin({"filled", "partial", "partially_filled"}) | filled_qty.gt(0)


def _accepted_mask(frame: pd.DataFrame) -> pd.Series:
    """
    Return True for rows that probably reached Alpaca.

    PLAIN ENGLISH: skipped and submission_failed rows are useful audit rows,
    but they are not accepted broker orders.  Fill rate should not count them
    as submitted to Alpaca.
    """
    if frame.empty:
        return pd.Series(dtype=bool)
    status = _series_text(frame, "fill_status").str.lower()
    order_id = _series_text(frame, "order_id")
    return ~(
        status.isin({"skipped", "submission_failed", "rejected"})
        | order_id.str.startswith("SKIPPED", na=False)
        | order_id.str.startswith("ERROR", na=False)
        | order_id.eq("")
    )


def _skipped_mask(frame: pd.DataFrame) -> pd.Series:
    """Return True for planned orders the bot intentionally did not submit."""
    if frame.empty:
        return pd.Series(dtype=bool)
    status = _series_text(frame, "fill_status").str.lower()
    order_id = _series_text(frame, "order_id")
    return status.eq("skipped") | order_id.str.startswith("SKIPPED", na=False)


def _row_slippage_bps(row: pd.Series | dict) -> float | None:
    """Compute signed slippage from one paper-log row when fill price exists."""
    recorded = _to_float(row.get("realized_slippage_bps"))
    if recorded is not None:
        return recorded
    side = str(_first_nonempty(row, "side", "action", default="")).lower().strip()
    reference = _to_float(_first_nonempty(row, "price", "limit_price"))
    fill_price = _to_float(_first_nonempty(row, "filled_avg_price", "broker_dealt_avg_price"))
    if side not in {"buy", "sell"} or reference is None or fill_price is None or reference <= 0 or fill_price <= 0:
        return None
    if side == "buy":
        return round((fill_price - reference) / reference * 10_000.0, 3)
    return round((reference - fill_price) / reference * 10_000.0, 3)


def _slippage_from_log(frame: pd.DataFrame) -> list[float]:
    """Compute slippage for all filled rows where the paper log has prices."""
    if frame.empty:
        return []
    filled = frame[_filled_mask(frame)].copy()
    values: list[float] = []
    for _, row in filled.iterrows():
        value = _row_slippage_bps(row)
        if value is not None:
            values.append(float(value))
    return values


def _throttle_summary(frame: pd.DataFrame) -> dict:
    """
    Summarize execution-risk throttled BUY orders.

    PLAIN ENGLISH: A throttled buy is a buy that got reduced because recent
    fills for that ticker looked risky.  We track how much size was avoided and
    whether those throttled buys filled cleanly.
    """
    empty = {
        "throttled_buy_orders": 0,
        "throttled_buy_filled": 0,
        "throttled_buy_fill_rate": None,
        "throttled_buy_avg_slippage_bps": None,
        "unthrottled_buy_avg_slippage_bps": None,
        "quantity_reduced": 0,
        "notional_reduced": 0.0,
        "examples": [],
    }
    if frame.empty:
        return empty

    side = _series_text(frame, "side").str.lower()
    reason = _series_text(frame, "execution_risk_reason")
    scale = pd.to_numeric(frame.get("execution_risk_buy_scale", pd.Series(1.0, index=frame.index)), errors="coerce")
    before = pd.to_numeric(
        frame.get("execution_risk_quantity_before_scale", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    after = pd.to_numeric(
        frame.get("execution_risk_quantity_after_scale", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    throttled_mask = side.eq("buy") & (reason.ne("") | scale.lt(1.0) | before.gt(after))
    throttled = frame[throttled_mask].copy()
    unthrottled_buys = frame[side.eq("buy") & ~throttled_mask].copy()
    if throttled.empty:
        return empty

    filled_throttled = throttled[_filled_mask(throttled)]
    throttled_slip = _slippage_from_log(throttled)
    unthrottled_slip = _slippage_from_log(unthrottled_buys)
    quantity_reduced = (before[throttled_mask].fillna(0) - after[throttled_mask].fillna(0)).clip(lower=0)
    prices = pd.to_numeric(throttled.get("price", pd.Series(0, index=throttled.index)), errors="coerce").fillna(0)
    examples = []
    for _, row in throttled.head(8).iterrows():
        examples.append({
            "ticker": str(row.get("ticker", "")).upper(),
            "reason": str(row.get("execution_risk_reason", "")),
            "before_qty": _to_float(row.get("execution_risk_quantity_before_scale")),
            "after_qty": _to_float(row.get("execution_risk_quantity_after_scale")),
            "fill_status": str(row.get("fill_status", "")),
        })
    return {
        "throttled_buy_orders": int(len(throttled)),
        "throttled_buy_filled": int(len(filled_throttled)),
        "throttled_buy_fill_rate": _rate(len(filled_throttled), len(throttled)),
        "throttled_buy_avg_slippage_bps": round(float(np.mean(throttled_slip)), 3) if throttled_slip else None,
        "unthrottled_buy_avg_slippage_bps": round(float(np.mean(unthrottled_slip)), 3) if unthrottled_slip else None,
        "quantity_reduced": int(quantity_reduced.sum()),
        "notional_reduced": round(float((quantity_reduced.values * prices.values).sum()), 2),
        "examples": examples,
    }


def _metric_values(rows: list[dict], key: str) -> list[float]:
    """Return real observations only, so missing future bars do not dilute rates."""
    return [
        float(value)
        for value in (_to_float((row or {}).get(key)) for row in rows)
        if value is not None
    ]


def _row_group(row: dict) -> str:
    """Separate safety exits from ordinary rebalance execution."""
    order_kind = str((row or {}).get("order_type", "")).strip().lower()
    return "protective_stop" if order_kind in {"trailing_stop", "stop", "stop_limit"} else "rebalance"


def _filter_report_rows(slippage_report: dict, *, now: datetime, days: int) -> list[dict]:
    """Apply the advertised lookback window to the Alpaca fill rows."""
    cutoff = pd.Timestamp(now.astimezone(timezone.utc)) - pd.Timedelta(days=max(0, int(days)))
    kept: list[dict] = []
    for item in slippage_report.get("orders", []) or []:
        if not isinstance(item, dict):
            continue
        filled_at = pd.to_datetime(item.get("filled_at"), errors="coerce", utc=True)
        if pd.notna(filled_at) and days > 0 and filled_at < cutoff:
            continue
        kept.append(dict(item))
    return kept


def _measurement_coverage(rows: list[dict], key: str, *, now: datetime, maturity_minutes: int = 0) -> dict:
    """Count only fills old enough to have the requested market observation."""
    eligible: list[dict] = []
    values: list[float] = []
    clock = pd.Timestamp(now.astimezone(timezone.utc))
    for row in rows:
        filled_at = pd.to_datetime(row.get("filled_at"), errors="coerce", utc=True)
        if maturity_minutes and (pd.isna(filled_at) or clock - filled_at < pd.Timedelta(minutes=maturity_minutes)):
            continue
        eligible.append(row)
        value = _to_float(row.get(key))
        if value is not None:
            values.append(float(value))
    return {
        "eligible": len(eligible),
        "measured": len(values),
        "coverage": _rate(len(values), len(eligible)),
        "values": values,
    }


def _quality_slice(rows: list[dict], *, now: datetime) -> dict:
    """Summarize one group with a separate denominator for every metric."""
    slip_metric = _measurement_coverage(rows, "slippage_bps", now=now)
    adverse_15_metric = _measurement_coverage(rows, "adverse_15m_bps", now=now, maturity_minutes=15)
    adverse_60_metric = _measurement_coverage(rows, "adverse_60m_bps", now=now, maturity_minutes=60)
    slip = slip_metric["values"]
    adverse_15 = adverse_15_metric["values"]
    adverse_60 = adverse_60_metric["values"]
    raw_bad = int(sum(value > 0 for value in slip))
    material_bad = int(sum(value > BAD_SLIPPAGE_BPS for value in slip))
    session_dates: set[str] = set()
    for row in rows:
        timestamp = pd.to_datetime(row.get("filled_at"), errors="coerce", utc=True)
        if pd.notna(timestamp):
            session_dates.add(str(timestamp.date()))
    return {
        "orders_seen": len(rows),
        "eligible_orders": len(rows),
        "measured_slippage_count": len(slip),
        "slippage_eligible_orders": slip_metric["eligible"],
        "slippage_measured_orders": slip_metric["measured"],
        "slippage_coverage_rate": slip_metric["coverage"],
        "trading_sessions": len(session_dates),
        "avg_slippage_bps": round(float(np.mean(slip)), 3) if slip else None,
        "median_slippage_bps": round(float(np.median(slip)), 3) if slip else None,
        "bad_slippage_count": material_bad,
        "bad_slippage_rate": _rate(material_bad, len(slip)),
        "bad_slippage_threshold_bps": BAD_SLIPPAGE_BPS,
        "raw_bad_slippage_count": raw_bad,
        "raw_bad_slippage_rate": _rate(raw_bad, len(slip)),
        "minor_bad_slippage_count": max(0, raw_bad - material_bad),
        "adverse_15m_count": int(sum(value > 0 for value in adverse_15)),
        "adverse_15m_observations": len(adverse_15),
        "adverse_15m_eligible_orders": adverse_15_metric["eligible"],
        "adverse_15m_measured_orders": adverse_15_metric["measured"],
        "adverse_15m_coverage_rate": adverse_15_metric["coverage"],
        "adverse_15m_rate": _rate(sum(value > 0 for value in adverse_15), len(adverse_15)),
        "adverse_60m_count": int(sum(value > 0 for value in adverse_60)),
        "adverse_60m_observations": len(adverse_60),
        "adverse_60m_eligible_orders": adverse_60_metric["eligible"],
        "adverse_60m_measured_orders": adverse_60_metric["measured"],
        "adverse_60m_coverage_rate": adverse_60_metric["coverage"],
        "adverse_60m_rate": _rate(sum(value > 0 for value in adverse_60), len(adverse_60)),
    }


def _stage_comparison(log: pd.DataFrame) -> dict:
    """Compare passive and repriced attempts when their audit columns exist."""
    if log.empty or "execution_stage" not in log.columns:
        return {}
    output: dict[str, dict] = {}
    stage_series = _series_text(log, "execution_stage").str.lower()
    for stage in sorted(set(stage_series) - {""}):
        group = log[stage_series.eq(stage)]
        filled = group[_filled_mask(group)]
        slip = _slippage_from_log(filled)
        latency = pd.to_numeric(group.get("fill_latency_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
        partial = int(_series_text(group, "stage1_status").str.lower().eq("partially_filled").sum())
        cancelled = int(_series_text(group, "stage1_cancel_status").str.lower().eq("canceled").sum())
        output[stage] = {
            "orders": int(len(group)),
            "filled_orders": int(len(filled)),
            "fill_rate": _rate(len(filled), len(group)),
            "avg_slippage_bps": round(float(np.mean(slip)), 3) if slip else None,
            "median_slippage_bps": round(float(np.median(slip)), 3) if slip else None,
            "material_bad_rate": _rate(sum(value > BAD_SLIPPAGE_BPS for value in slip), len(slip)),
            "avg_fill_latency_seconds": round(float(latency.mean()), 2) if len(latency) else None,
            "partial_fill_rate": _rate(partial, len(group)),
            "cancellation_rate": _rate(cancelled, len(group)),
        }
    return output


def _report_summary(
    slippage_report: dict,
    log: pd.DataFrame,
    *,
    now: datetime,
    lookback_days: int,
) -> tuple[dict, dict, dict]:
    """Return rebalance quality, stop quality, and non-gating timing evidence."""
    rows = _filter_report_rows(slippage_report, now=now, days=lookback_days)
    rebalance_rows = [row for row in rows if _row_group(row) == "rebalance"]
    rebalance = _quality_slice(rebalance_rows, now=now)
    protective = _quality_slice(
        [row for row in rows if _row_group(row) == "protective_stop"],
        now=now,
    )

    # Old reports sometimes contain only an aggregate summary. Preserve that
    # fallback while all newly generated reports use typed rows above.
    if not rows:
        summary = dict(slippage_report.get("summary", {}) or {})
        fallback = _slippage_from_log(log)
        count = int(_to_float(summary.get("orders_analyzed")) or len(fallback))
        avg = _to_float(summary.get("avg_slippage_bps"))
        raw_bad = int(_to_float(summary.get("slippage_bad_count")) or sum(value > 0 for value in fallback))
        submitted = pd.to_datetime(log.get("submitted_at", pd.Series(dtype=str)), errors="coerce", utc=True)
        rebalance.update({
            "orders_seen": count,
            "measured_slippage_count": count,
            "trading_sessions": int(submitted.dt.date.nunique()) if len(submitted) else 0,
            "avg_slippage_bps": round(avg, 3) if avg is not None else (round(float(np.mean(fallback)), 3) if fallback else None),
            "bad_slippage_count": raw_bad,
            "bad_slippage_rate": _rate(raw_bad, count),
            "raw_bad_slippage_count": raw_bad,
            "raw_bad_slippage_rate": _rate(raw_bad, count),
        })

    if rows:
        limit_values = _metric_values(
            [row for row in rebalance_rows if str(row.get("order_type", "")).lower() == "limit"],
            "slippage_bps",
        )
        market_values = _metric_values(
            [row for row in rebalance_rows if str(row.get("order_type", "")).lower() == "market"],
            "slippage_bps",
        )
        limit_avg = round(float(np.mean(limit_values)), 3) if limit_values else None
        market_avg = round(float(np.mean(market_values)), 3) if market_values else None
    else:
        segments = slippage_report.get("segments", {}) or {}
        limit_avg = _to_float((segments.get("limit_orders", {}) or {}).get("avg_slippage_bps"))
        market_avg = _to_float((segments.get("market_orders", {}) or {}).get("avg_slippage_bps"))
    rebalance.update({
        "orders_analyzed": int(rebalance.get("measured_slippage_count") or 0),
        "limit_avg_slippage_bps": limit_avg,
        "market_avg_slippage_bps": market_avg,
        "limit_vs_market_delta_bps": round(limit_avg - market_avg, 3) if limit_avg is not None and market_avg is not None else None,
        "source": slippage_report.get("source", "paper_log_fallback" if rebalance.get("orders_seen") else "none"),
    })
    timestamped = [
        (pd.Timestamp(pd.to_datetime(row.get("filled_at"), utc=True)), row)
        for row in rebalance_rows
        if pd.notna(pd.to_datetime(row.get("filled_at"), errors="coerce", utc=True))
    ]
    latest_rows: list[dict] = []
    latest_day = None
    if timestamped:
        latest_day = max(timestamp for timestamp, _row in timestamped).tz_convert("America/New_York").date()
        latest_rows = [
            row for timestamp, row in timestamped
            if timestamp.tz_convert("America/New_York").date() == latest_day
        ]
    latest_slip = _measurement_coverage(latest_rows, "slippage_bps", now=now)
    latest_15 = _measurement_coverage(latest_rows, "adverse_15m_bps", now=now, maturity_minutes=15)
    latest_60 = _measurement_coverage(latest_rows, "adverse_60m_bps", now=now, maturity_minutes=60)
    rebalance["latest_run_coverage"] = {
        "run_date_new_york": str(latest_day) if latest_day else None,
        "eligible_orders": len(latest_rows),
        "slippage_measured_orders": latest_slip["measured"],
        "adverse_15m_measured_orders": latest_15["measured"],
        "adverse_60m_measured_orders": latest_60["measured"],
    }
    timing = {
        "status": "advisory",
        "adverse_15m_rate": rebalance.get("adverse_15m_rate"),
        "adverse_15m_observations": rebalance.get("adverse_15m_observations", 0),
        "adverse_60m_rate": rebalance.get("adverse_60m_rate"),
        "adverse_60m_observations": rebalance.get("adverse_60m_observations", 0),
        "note": "Post-fill movement measures entry timing, not fill quality.",
    }
    return rebalance, protective, timing


def _make_check(name: str, value: float | None, operator: str, threshold: float, *, collecting_reason: str) -> dict:
    """Create one pass/fail/collecting check for the scorecard."""
    if value is None:
        return {
            "name": name,
            "status": "collecting",
            "value": None,
            "threshold": threshold,
            "operator": operator,
            "reason": collecting_reason,
        }
    passed = value <= threshold if operator == "<=" else value >= threshold
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "value": round(float(value), 4),
        "threshold": threshold,
        "operator": operator,
        "reason": "ok" if passed else f"{value:.4f} {operator} {threshold} failed",
    }


def _overall_status(
    checks: list[dict],
    *,
    sample_ready: bool,
    warning: bool,
) -> tuple[str, float | None]:
    """Turn individual checks into an overall status and 0-100 score."""
    scored = [check for check in checks if check["status"] in {"pass", "fail"}]
    if not scored:
        return "collecting", None
    passed = sum(1 for check in scored if check["status"] == "pass")
    score = round(passed / len(scored) * 100.0, 2)
    if not sample_ready:
        return "collecting", score
    if any(check["status"] == "fail" for check in scored):
        return "fail", score
    return ("warning" if warning else "pass"), score


def build_execution_scorecard(
    *,
    paper_log_path: Path = PAPER_LOG_FILE,
    slippage_report_path: Path = SLIPPAGE_REPORT_FILE,
    previous_scorecard_path: Path = SCORECARD_FILE,
    now: datetime | None = None,
    lookback_days: int = LOOKBACK_DAYS,
    min_rebalance_fills: int = MIN_REBALANCE_FILLS,
    min_rebalance_sessions: int = MIN_REBALANCE_SESSIONS,
) -> dict:
    """Build the execution scorecard payload without writing files."""
    clock = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    raw_log = _read_csv(paper_log_path)
    log = _filter_lookback(raw_log, now=clock, days=lookback_days)
    slippage_report = _read_json(slippage_report_path)
    previous = _read_json(previous_scorecard_path)

    total_rows = int(len(log))
    accepted = log[_accepted_mask(log)] if not log.empty else log
    filled = accepted[_filled_mask(accepted)] if not accepted.empty else accepted
    skipped = log[_skipped_mask(log)] if not log.empty else log
    report_metrics, protective_stop_metrics, entry_timing = _report_summary(
        slippage_report,
        log,
        now=clock,
        lookback_days=lookback_days,
    )
    throttle = _throttle_summary(log)

    accepted_orders = int(len(accepted))
    filled_orders = int(len(filled))
    skipped_orders = int(len(skipped))
    fill_rate = _rate(filled_orders, accepted_orders)
    skipped_rate = _rate(skipped_orders, total_rows)

    checks = [
        _make_check(
            "avg_slippage_bps",
            report_metrics["avg_slippage_bps"],
            "<=",
            MAX_AVG_SLIPPAGE_BPS,
            collecting_reason="need_filled_orders_with_slippage",
        ),
        _make_check(
            "bad_slippage_rate",
            report_metrics["bad_slippage_rate"],
            "<=",
            MAX_BAD_SLIPPAGE_RATE,
            collecting_reason="need_filled_orders_with_slippage",
        ),
        _make_check(
            "fill_rate",
            fill_rate,
            ">=",
            MIN_FILL_RATE,
            collecting_reason="need_accepted_orders",
        ),
        _make_check(
            "skipped_rate",
            skipped_rate,
            "<=",
            MAX_SKIPPED_RATE,
            collecting_reason="need_order_log_rows",
        ),
    ]
    measured = int(report_metrics.get("measured_slippage_count") or 0)
    sessions = int(report_metrics.get("trading_sessions") or 0)
    sample_ready = bool(
        measured >= int(min_rebalance_fills)
        and sessions >= int(min_rebalance_sessions)
        and fill_rate is not None
        and skipped_rate is not None
    )
    coverage_fields = (
        ("slippage", report_metrics.get("slippage_measured_orders"), report_metrics.get("slippage_coverage_rate")),
        ("adverse_15m", report_metrics.get("adverse_15m_measured_orders"), report_metrics.get("adverse_15m_coverage_rate")),
        ("adverse_60m", report_metrics.get("adverse_60m_measured_orders"), report_metrics.get("adverse_60m_coverage_rate")),
    )
    decision_blockers = [
        f"{name}_sample_{int(measured_count or 0)}_below_{MIN_DECISION_ORDERS}"
        for name, measured_count, _coverage in coverage_fields
        if int(measured_count or 0) < MIN_DECISION_ORDERS
    ]
    decision_blockers.extend(
        f"{name}_coverage_{float(coverage or 0.0):.4f}_below_{MIN_DECISION_COVERAGE:.4f}"
        for name, _measured_count, coverage in coverage_fields
        if coverage is None or float(coverage) < MIN_DECISION_COVERAGE
    )
    decision_eligible = not decision_blockers
    warning = bool(
        (report_metrics.get("avg_slippage_bps") is not None and report_metrics["avg_slippage_bps"] > WARN_AVG_SLIPPAGE_BPS)
        or (report_metrics.get("bad_slippage_rate") is not None and report_metrics["bad_slippage_rate"] > WARN_BAD_SLIPPAGE_RATE)
    )
    status, score = _overall_status(checks, sample_ready=sample_ready, warning=warning)

    prior_summary = previous.get("summary", {}) if isinstance(previous, dict) else {}
    prior_avg = _to_float(prior_summary.get("avg_slippage_bps"))
    current_avg = _to_float(report_metrics["avg_slippage_bps"])
    slippage_delta = (
        round(float(current_avg - prior_avg), 3)
        if prior_avg is not None and current_avg is not None
        else None
    )

    recommendations = []
    failed = [check["name"] for check in checks if check["status"] == "fail"]
    if failed:
        recommendations.append("review_failed_execution_checks:" + ",".join(failed))
    if not sample_ready:
        recommendations.append("collect_more_rebalance_fill_samples")
    if not _stage_comparison(log):
        recommendations.append("collect_two_stage_execution_samples")
    if report_metrics["limit_vs_market_delta_bps"] is not None and report_metrics["limit_vs_market_delta_bps"] < 0:
        recommendations.append("limit_orders_are_beating_market_orders")

    return {
        "schema_version": 2,
        "generated_at": clock.isoformat(timespec="seconds"),
        "measurement_cutoff_60m": (pd.Timestamp(clock) - pd.Timedelta(minutes=60)).isoformat(),
        "status": status,
        "score": score,
        "decision_eligible": decision_eligible,
        "decision_blockers": decision_blockers,
        "lookback_days": int(lookback_days),
        "summary": {
            "paper_log_rows": total_rows,
            "accepted_orders": accepted_orders,
            "filled_orders": filled_orders,
            "skipped_orders": skipped_orders,
            "fill_rate": fill_rate,
            "skipped_rate": skipped_rate,
            **report_metrics,
            "slippage_delta_vs_prior_scorecard_bps": slippage_delta,
        },
        "sample_gate": {
            "ready": sample_ready,
            "measured_rebalance_fills": measured,
            "minimum_rebalance_fills": int(min_rebalance_fills),
            "rebalance_sessions": sessions,
            "minimum_rebalance_sessions": int(min_rebalance_sessions),
        },
        "protective_stops": protective_stop_metrics,
        "entry_timing": entry_timing,
        "stage_comparison": _stage_comparison(log),
        "throttle": throttle,
        "checks": checks,
        "thresholds": {
            "max_avg_slippage_bps": MAX_AVG_SLIPPAGE_BPS,
            "max_bad_slippage_rate": MAX_BAD_SLIPPAGE_RATE,
            "bad_slippage_bps": BAD_SLIPPAGE_BPS,
            "min_fill_rate": MIN_FILL_RATE,
            "max_skipped_rate": MAX_SKIPPED_RATE,
            "max_adverse_15m_rate": MAX_ADVERSE_15M_RATE,
            "max_adverse_60m_rate": MAX_ADVERSE_60M_RATE,
            "warn_avg_slippage_bps": WARN_AVG_SLIPPAGE_BPS,
            "warn_bad_slippage_rate": WARN_BAD_SLIPPAGE_RATE,
            "minimum_rebalance_fills": int(min_rebalance_fills),
            "minimum_rebalance_sessions": int(min_rebalance_sessions),
            "min_decision_orders": MIN_DECISION_ORDERS,
            "min_decision_coverage": MIN_DECISION_COVERAGE,
        },
        "recommendations": recommendations,
        "source_files": {
            "paper_log": str(paper_log_path),
            "slippage_report": str(slippage_report_path),
            "previous_scorecard": str(previous_scorecard_path),
        },
    }


def write_execution_scorecard(
    *,
    output_path: Path = SCORECARD_FILE,
    log_dir: Path = LOGS,
    now: datetime | None = None,
) -> dict:
    """Build the scorecard and write both latest and dated JSON files."""
    clock = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    payload = build_execution_scorecard(previous_scorecard_path=output_path, now=clock)
    atomic_write_json(payload, output_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    dated_path = log_dir / f"alpaca_execution_scorecard_{clock.date().isoformat().replace('-', '')}.json"
    atomic_write_json(payload, dated_path)
    return payload


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build Alpaca paper execution scorecard")
    parser.add_argument("--json", action="store_true", help="Print the full JSON payload")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when scorecard status is fail")
    args = parser.parse_args()

    payload = write_execution_scorecard()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload.get("summary", {})
        print(
            "Execution scorecard: "
            f"status={payload.get('status')} "
            f"score={payload.get('score')} "
            f"avg_slippage_bps={summary.get('avg_slippage_bps')} "
            f"fill_rate={summary.get('fill_rate')}"
        )
        print(f"  wrote {SCORECARD_FILE}")

    if args.strict and payload.get("status") == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
