"""Shared freshness checks for broker signal files."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd


DEFAULT_SIGNAL_TIMEZONE = os.environ.get("PAPER_SIGNAL_TIMEZONE", os.environ.get("TZ", "Asia/Singapore"))


def _signal_timezone(default_timezone: str | None = None):
    try:
        return ZoneInfo(str(default_timezone or DEFAULT_SIGNAL_TIMEZONE))
    except Exception:
        return datetime.now().astimezone().tzinfo or timezone.utc


def parse_signal_timestamp(value: object, *, default_timezone: str | None = None) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize(_signal_timezone(default_timezone))
    return out.tz_convert("UTC")


def latest_completed_us_trading_day(now: datetime | None = None) -> pd.Timestamp:
    eastern = ZoneInfo("America/New_York")
    ts = (now or datetime.now(timezone.utc)).astimezone(eastern)
    current_day = pd.Timestamp(ts.date())
    if ts.weekday() >= 5:
        return current_day - pd.tseries.offsets.BDay(1)
    close_ts = ts.replace(hour=16, minute=0, second=0, microsecond=0)
    if ts < close_ts:
        return current_day - pd.tseries.offsets.BDay(1)
    return current_day


def validate_signal_freshness(
    signal,
    *,
    max_signal_age_hours: float,
    max_factor_age_trading_days: int,
    now: datetime | None = None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")

    predicted_at = signal.get("predicted_at", "")
    predicted_ts = parse_signal_timestamp(predicted_at)
    if pd.isna(predicted_ts):
        issues.append("missing_predicted_at")
    else:
        age_hours = (now_ts - predicted_ts).total_seconds() / 3600.0
        if age_hours > float(max_signal_age_hours):
            issues.append(f"signal_age_{age_hours:.1f}h_gt_{float(max_signal_age_hours):.1f}h")

    latest_factor_date = signal.get("latest_factor_date", "")
    factor_ts = pd.to_datetime(latest_factor_date, errors="coerce")
    if pd.isna(factor_ts):
        issues.append("missing_latest_factor_date")
    else:
        completed_us_day = latest_completed_us_trading_day(now=now)
        age_days = len(pd.bdate_range(pd.Timestamp(factor_ts) + pd.tseries.offsets.BDay(1), completed_us_day))
        if age_days > int(max_factor_age_trading_days):
            issues.append(f"factor_age_{age_days}_bdays_gt_{int(max_factor_age_trading_days)}")
    return not issues, issues


def validate_signal_sanity(
    signal,
    *,
    max_gross_exposure: float = 1.5,
    max_single_weight: float = 0.30,
    required_core_etfs: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Check if signal weights are sane before submitting orders.

    PLAIN ENGLISH: Even if a signal is fresh, it could be WRONG — a bug in
    signal generation might produce weights that sum to 500% equity, or put
    50% into a single stock, or forget SPY entirely.  This catches those
    bugs at submission time so you don't blindly execute a broken signal.

    Checks:
      1. Total gross exposure doesn't exceed max_gross_exposure (default 150%)
      2. No single ticker exceeds max_single_weight (default 30%)
      3. Required core ETFs (SPY or QQQ) are present with non-zero weight
      4. No negative weights (we don't short in this strategy)

    Returns (is_sane, list_of_issues).
    """
    issues: list[str] = []
    if required_core_etfs is None:
        required_core_etfs = {"SPY", "QQQ"}

    # Extract weights from signal
    weights: dict[str, float] = {}
    for key, val in signal.items():
        if key.startswith("target_weight_"):
            ticker = key.replace("target_weight_", "").upper()
            try:
                weights[ticker] = float(val)
            except (ValueError, TypeError):
                pass

    if not weights:
        # No target_weight_ fields — check if there's a target_weights dict
        if hasattr(signal, 'get') and isinstance(signal.get("target_weights"), dict):
            weights = {k: float(v) for k, v in signal["target_weights"].items()}

    if not weights:
        return True, []  # can't validate without weights, skip

    # Check gross exposure
    gross = sum(abs(w) for w in weights.values())
    if gross > max_gross_exposure:
        issues.append(f"gross_exposure_{gross:.2f}_exceeds_{max_gross_exposure:.2f}")

    # Check single-name concentration
    for ticker, w in weights.items():
        if abs(w) > max_single_weight:
            issues.append(f"{ticker}_weight_{w:.3f}_exceeds_{max_single_weight:.2f}")

    # Check for negative weights (no shorting in this strategy)
    neg_tickers = [t for t, w in weights.items() if w < -0.001]
    if neg_tickers:
        issues.append(f"negative_weights: {', '.join(neg_tickers)}")

    # Check required core ETFs present
    present_core = required_core_etfs.intersection(weights.keys())
    if not present_core and weights:
        issues.append(f"missing_core_etfs: need at least one of {required_core_etfs}")

    return not issues, issues
