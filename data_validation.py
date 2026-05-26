"""
data_validation.py — Schema + freshness guards for every dataframe leaving the feature pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _nyse_calendar():
    try:
        import exchange_calendars as xcals

        return xcals.get_calendar("XNYS")
    except Exception:
        return None


def _latest_weekday_on_or_before(day: object) -> pd.Timestamp:
    ts = pd.Timestamp(day).normalize()
    while ts.weekday() >= 5:
        ts -= pd.Timedelta(days=1)
    return ts


def _latest_session_on_or_before(day: object) -> pd.Timestamp:
    ts = pd.Timestamp(day).normalize()
    calendar = _nyse_calendar()
    if calendar is not None:
        for _ in range(14):
            if calendar.is_session(ts):
                return ts
            ts -= pd.Timedelta(days=1)
        return ts
    return _latest_weekday_on_or_before(ts)


def _latest_completed_session(now: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    eastern_ts = now_ts.to_pydatetime().astimezone(ZoneInfo("America/New_York"))
    current_day = pd.Timestamp(eastern_ts.date())
    calendar = _nyse_calendar()
    if calendar is not None:
        if not calendar.is_session(current_day):
            return _latest_session_on_or_before(current_day - pd.Timedelta(days=1))
        if now_ts < calendar.session_close(current_day):
            return _latest_session_on_or_before(current_day - pd.Timedelta(days=1))
        return current_day
    if eastern_ts.weekday() >= 5:
        return _latest_weekday_on_or_before(current_day - pd.Timedelta(days=1))
    close_ts = eastern_ts.replace(hour=16, minute=0, second=0, microsecond=0)
    if eastern_ts < close_ts:
        return _latest_weekday_on_or_before(current_day - pd.Timedelta(days=1))
    return current_day


def _count_sessions(start: object, end: object) -> int:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        return 0
    calendar = _nyse_calendar()
    if calendar is not None:
        return int(len(calendar.sessions_in_range(start_ts, end_ts)))
    return int(sum(1 for day in pd.date_range(start_ts, end_ts, freq="D") if day.weekday() < 5))


def validate_price_frame(
    df: pd.DataFrame,
    ticker: str,
    max_lag_days: int = 5,
    *,
    now: datetime | pd.Timestamp | None = None,
) -> None:
    if df is None or df.empty:
        raise ValueError(f"{ticker}: empty price frame")

    missing = [c for c in REQUIRED_PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: index is not DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: index not sorted")
    if df.index.has_duplicates:
        raise ValueError(f"{ticker}: duplicate timestamps")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{ticker}: non-positive OHLC price")
    if (df["Volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume")

    rets = df["Close"].pct_change().abs()
    if (rets > 0.5).any():
        raise ValueError(f"{ticker}: >50% single-day move — likely bad tick or split not adjusted")

    latest = pd.Timestamp(df.index[-1]).normalize()
    completed = _latest_completed_session(now=now)
    lag_days = _count_sessions(latest + pd.Timedelta(days=1), completed)
    if lag_days > int(max_lag_days):
        raise ValueError(f"{ticker}: stale data — last bar is {lag_days} trading sessions old")


def validate_feature_frame(df: pd.DataFrame, required_features: list[str]) -> None:
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise ValueError(f"feature frame missing columns: {missing}")
    if df[required_features].isna().any().any():
        bad = df[required_features].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(f"NaNs in feature frame: {bad}")
    if not np.isfinite(df[required_features].to_numpy()).all():
        raise ValueError("non-finite values in feature frame")
