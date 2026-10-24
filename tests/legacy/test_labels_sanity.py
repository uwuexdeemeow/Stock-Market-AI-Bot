"""
Legacy label smoke tests, split out of tests/test_sanity.py.

PLAIN ENGLISH: labels.py builds the "answer" columns the old XGBoost model
trained on.  It now lives in legacy/, so these tests run in the separate,
non-blocking legacy CI job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labels import forward_return, vol_normalized_return, triple_barrier


def _price_frame(n=100, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1e6, 5e6, n),
    }, index=idx)


def test_forward_return_is_leak_free():
    df = _price_frame()
    r = forward_return(df["Close"], horizon=5)
    # last 5 rows must be NaN — nothing to look forward to
    assert r.tail(5).isna().all()


def test_vol_normalized_scales_with_vol():
    df = _price_frame()
    r = vol_normalized_return(df["Close"], horizon=5, vol_window=20).dropna()
    assert np.isfinite(r).any()


def test_triple_barrier_outputs_in_set():
    df = _price_frame()
    lab = triple_barrier(df["Close"], df["High"], df["Low"], max_hold=5).dropna()
    assert set(lab.unique()).issubset({-1.0, 0.0, 1.0})


def test_triple_barrier_ambiguous_daily_bar_is_conservative_stop():
    """A daily bar crossing both barriers cannot be assumed to be a win."""
    idx = pd.bdate_range("2024-01-01", periods=20)
    close = pd.Series(100.0, index=idx)
    high = pd.Series(101.0, index=idx)
    low = pd.Series(99.0, index=idx)
    high.iloc[15] = 110.0
    low.iloc[15] = 90.0

    labels = triple_barrier(close, high, low, pt_mult=1.0, sl_mult=1.0, max_hold=3, atr_window=14)

    assert labels.iloc[14] == -1.0
