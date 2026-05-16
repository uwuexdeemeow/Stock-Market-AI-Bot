"""
data_validation.py — Schema + freshness guards for every dataframe leaving the feature pipeline.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def validate_price_frame(df: pd.DataFrame, ticker: str, max_lag_days: int = 5) -> None:
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

    now_utc = datetime.now(timezone.utc)
    lag = now_utc.date() - df.index[-1].date()
    allowed_lag = timedelta(days=max_lag_days)
    if now_utc.weekday() >= 5:
        allowed_lag += timedelta(days=2)
    if lag > allowed_lag:
        raise ValueError(f"{ticker}: stale data — last bar is {lag.days} days old")


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
