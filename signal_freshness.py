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
