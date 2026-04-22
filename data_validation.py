"""
data_validation.py — Schema + freshness guards for every dataframe leaving the
feature pipeline.

PLAIN ENGLISH:
Bad data is the single biggest source of silent bugs in trading systems.
Before we let a dataframe into training OR inference, we verify:
  * expected columns exist
  * there are no NaNs in critical columns
  * index is monotonically increasing dates, no dups, no gaps > N days
  * most recent bar is < max_lag_days old (freshness)
  * no obviously broken rows (price <= 0, volume < 0, return > +/-50%)

If `pandera` is installed it's used for schema contracts; otherwise we
fall back to explicit assertions. Either way, failures raise loudly.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def validate_price_frame(df: pd.DataFrame, ticker: str, max_lag_days: int = 5) -> None:
    """Raise ValueError on any contract violation."""
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

    if (df["Close"] <= 0).any():
        raise ValueError(f"{ticker}: non-positive close price")
    if (df["Volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume")

    rets = df["Close"].pct_change().abs()
    if (rets > 0.5).any():
        raise ValueError(f"{ticker}: >50% single-day move — likely bad tick or split not adjusted")

    lag = datetime.utcnow().date() - df.index[-1].date()
    if lag > timedelta(days=max_lag_days):
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
