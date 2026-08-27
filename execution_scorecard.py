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


def _report_summary(slippage_report: dict, log: pd.DataFrame, *, now: datetime) -> dict:
    """Combine the API slippage report with paper-log fallback metrics."""
    summary = dict(slippage_report.get("summary", {}) or {})
    eligible_orders = int(_to_float(summary.get("eligible_orders", summary.get("orders_analyzed"))) or 0)
    orders_analyzed = int(_to_float(summary.get("slippage_measured_orders", summary.get("orders_analyzed"))) or 0)
    avg_slippage = _to_float(summary.get("avg_slippage_bps"))
    raw_bad_count = int(_to_float(summary.get("slippage_bad_count")) or 0)
    bad_count = raw_bad_count
    adverse_15m_count = int(_to_float(summary.get("adverse_15m_count")) or 0)
    adverse_60m_count = int(_to_float(summary.get("adverse_60m_count")) or 0)
    slip_values = [
        float(value)
        for value in (
            _to_float((row or {}).get("slippage_bps"))
            for row in slippage_report.get("orders", []) or []
        )
        if value is not None
    ]
    report_orders = [row for row in slippage_report.get("orders", []) or [] if isinstance(row, dict)]
    if report_orders:
        eligible_orders = len(report_orders)
    adverse_15_values = [
        float(value)
        for value in (_to_float(row.get("adverse_15m_bps")) for row in report_orders)
        if value is not None
    ]
    adverse_60_values = [
        float(value)
        for value in (_to_float(row.get("adverse_60m_bps")) for row in report_orders)
        if value is not None
    ]
    timestamped_rows: list[tuple[pd.Timestamp, dict]] = []
    for row in report_orders:
        filled_at = pd.to_datetime(row.get("filled_at"), errors="coerce", utc=True)
        if not pd.isna(filled_at):
            timestamped_rows.append((pd.Timestamp(filled_at), row))
    latest_run_coverage: dict[str, object] = {
        "run_date_new_york": None,
        "eligible_orders": 0,
        "slippage_measured_orders": 0,
        "adverse_15m_measured_orders": 0,
        "adverse_60m_measured_orders": 0,
    }
    if timestamped_rows:
        latest_day = max(timestamp for timestamp, _row in timestamped_rows).tz_convert("America/New_York").date()
        latest_rows = [
            row
            for timestamp, row in timestamped_rows
            if timestamp.tz_convert("America/New_York").date() == latest_day
        ]
        latest_run_coverage = {
            "run_date_new_york": str(latest_day),
            "eligible_orders": len(latest_rows),
            "slippage_measured_orders": sum(_to_float(row.get("slippage_bps")) is not None for row in latest_rows),
            "adverse_15m_measured_orders": sum(_to_float(row.get("adverse_15m_bps")) is not None for row in latest_rows),
            "adverse_60m_measured_orders": sum(_to_float(row.get("adverse_60m_bps")) is not None for row in latest_rows),
        }
        for field in ("slippage", "adverse_15m", "adverse_60m"):
            latest_run_coverage[f"{field}_coverage_rate"] = _rate(
                int(latest_run_coverage[f"{field}_measured_orders"]),
                int(latest_run_coverage["eligible_orders"]),
            )
    measured_fill_times = [
        timestamp
        for timestamp, row in timestamped_rows
        if any(_to_float(row.get(field)) is not None for field in ("slippage_bps", "adverse_15m_bps", "adverse_60m_bps"))
    ]
    if slip_values:
        # PLAIN ENGLISH: A fill that is 0.01 bps worse than fill-minute VWAP is
        # technically unfavorable, but it is market micro-noise. The scorecard
        # fails only on material bad slippage, while still reporting the raw
        # any-positive count for visibility.
        orders_analyzed = len(slip_values)
        raw_bad_count = int(sum(1 for value in slip_values if value > 0))
        bad_count = int(sum(1 for value in slip_values if value > BAD_SLIPPAGE_BPS))
        if avg_slippage is None:
            avg_slippage = round(float(np.mean(slip_values)), 3)
    if adverse_15_values:
        adverse_15m_count = int(sum(1 for value in adverse_15_values if value > 0))
    if adverse_60_values:
        adverse_60m_count = int(sum(1 for value in adverse_60_values if value > 0))

    # If the rich API report is missing, fall back to fill prices in the paper log.
    if orders_analyzed <= 0:
        fallback_slip_values = _slippage_from_log(log)
        orders_analyzed = len(fallback_slip_values)
        avg_slippage = round(float(np.mean(fallback_slip_values)), 3) if fallback_slip_values else None
        raw_bad_count = int(sum(1 for value in fallback_slip_values if value > 0))
        bad_count = int(sum(1 for value in fallback_slip_values if value > BAD_SLIPPAGE_BPS))

    # PLAIN ENGLISH: every measurement owns its denominator. A fill with a
    # VWAP but no 60-minute price can affect slippage, but cannot vote on the
    # 60-minute reversal check.
    slippage_eligible = int(_to_float(summary.get("slippage_eligible_orders")) or eligible_orders or orders_analyzed)
    adverse_15_eligible = int(_to_float(summary.get("adverse_15m_eligible_orders")) or eligible_orders)
    adverse_60_eligible = int(_to_float(summary.get("adverse_60m_eligible_orders")) or eligible_orders)
    legacy_report = int(_to_float(slippage_report.get("schema_version")) or 1) < 2
    adverse_15_measured = len(adverse_15_values) or int(
        _to_float(summary.get("adverse_15m_measured_orders"))
        or (orders_analyzed if legacy_report and "adverse_15m_count" in summary else 0)
    )
    adverse_60_measured = len(adverse_60_values) or int(
        _to_float(summary.get("adverse_60m_measured_orders"))
        or (orders_analyzed if legacy_report and "adverse_60m_count" in summary else 0)
    )

    segments = slippage_report.get("segments", {}) or {}
    limit_avg = _to_float((segments.get("limit_orders", {}) or {}).get("avg_slippage_bps"))
    market_avg = _to_float((segments.get("market_orders", {}) or {}).get("avg_slippage_bps"))
    return {
        "orders_analyzed": orders_analyzed,
        "eligible_orders": eligible_orders or slippage_eligible,
        "slippage_eligible_orders": slippage_eligible,
        "slippage_measured_orders": orders_analyzed,
        "slippage_coverage_rate": _rate(orders_analyzed, slippage_eligible),
        "avg_slippage_bps": round(avg_slippage, 3) if avg_slippage is not None else None,
        "bad_slippage_count": bad_count,
        "bad_slippage_rate": _rate(bad_count, orders_analyzed),
        "bad_slippage_threshold_bps": BAD_SLIPPAGE_BPS,
        "raw_bad_slippage_count": raw_bad_count,
        "raw_bad_slippage_rate": _rate(raw_bad_count, orders_analyzed),
        "minor_bad_slippage_count": max(0, int(raw_bad_count) - int(bad_count)),
        "adverse_15m_count": adverse_15m_count,
        "adverse_15m_eligible_orders": adverse_15_eligible,
        "adverse_15m_measured_orders": adverse_15_measured,
        "adverse_15m_coverage_rate": _rate(adverse_15_measured, adverse_15_eligible),
        "adverse_15m_rate": _rate(adverse_15m_count, adverse_15_measured),
        "adverse_60m_count": adverse_60m_count,
        "adverse_60m_eligible_orders": adverse_60_eligible,
        "adverse_60m_measured_orders": adverse_60_measured,
        "adverse_60m_coverage_rate": _rate(adverse_60_measured, adverse_60_eligible),
        "adverse_60m_rate": _rate(adverse_60m_count, adverse_60_measured),
        "limit_avg_slippage_bps": limit_avg,
        "market_avg_slippage_bps": market_avg,
        "limit_vs_market_delta_bps": (
            round(float(limit_avg - market_avg), 3)
            if limit_avg is not None and market_avg is not None
            else None
        ),
        "newest_measured_fill_at": max(measured_fill_times).isoformat() if measured_fill_times else None,
        "latest_run_coverage": latest_run_coverage,
        "measurement_age_minutes": (
            round((pd.Timestamp(now) - max(measured_fill_times)).total_seconds() / 60.0, 1)
            if measured_fill_times
            else None
        ),
        "source": slippage_report.get("source", "paper_log_fallback" if orders_analyzed else "none"),
    }


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


def _overall_status(checks: list[dict]) -> tuple[str, float | None]:
    """Turn individual checks into an overall status and 0-100 score."""
    scored = [check for check in checks if check["status"] in {"pass", "fail"}]
    if not scored:
        return "collecting", None
    passed = sum(1 for check in scored if check["status"] == "pass")
    score = round(passed / len(scored) * 100.0, 2)
    if any(check["status"] == "fail" for check in scored):
        return "fail", score
    return "pass", score


def build_execution_scorecard(
    *,
    paper_log_path: Path = PAPER_LOG_FILE,
    slippage_report_path: Path = SLIPPAGE_REPORT_FILE,
    previous_scorecard_path: Path = SCORECARD_FILE,
    now: datetime | None = None,
    lookback_days: int = LOOKBACK_DAYS,
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
    report_metrics = _report_summary(slippage_report, log, now=clock)
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
        _make_check(
            "adverse_15m_rate",
            report_metrics["adverse_15m_rate"],
            "<=",
            MAX_ADVERSE_15M_RATE,
            collecting_reason="need_slippage_reversal_report",
        ),
        _make_check(
            "adverse_60m_rate",
            report_metrics["adverse_60m_rate"],
            "<=",
            MAX_ADVERSE_60M_RATE,
            collecting_reason="need_slippage_reversal_report",
        ),
    ]
    status, score = _overall_status(checks)
    coverage_fields = (
        ("slippage", report_metrics["slippage_measured_orders"], report_metrics["slippage_coverage_rate"]),
        ("adverse_15m", report_metrics["adverse_15m_measured_orders"], report_metrics["adverse_15m_coverage_rate"]),
        ("adverse_60m", report_metrics["adverse_60m_measured_orders"], report_metrics["adverse_60m_coverage_rate"]),
    )
    decision_blockers = [
        f"{name}_sample_{measured}_below_{MIN_DECISION_ORDERS}"
        for name, measured, _coverage in coverage_fields
        if int(measured or 0) < MIN_DECISION_ORDERS
    ]
    decision_blockers.extend(
        f"{name}_coverage_{float(coverage or 0.0):.4f}_below_{MIN_DECISION_COVERAGE:.4f}"
        for name, _measured, coverage in coverage_fields
        if coverage is None or float(coverage) < MIN_DECISION_COVERAGE
    )
    decision_eligible = not decision_blockers

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
    if throttle["throttled_buy_orders"] == 0:
        recommendations.append("collect_more_throttled_buy_samples")
    elif throttle["throttled_buy_avg_slippage_bps"] is not None and throttle["throttled_buy_avg_slippage_bps"] > MAX_AVG_SLIPPAGE_BPS:
        recommendations.append("consider_stronger_execution_risk_buy_scale")
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
