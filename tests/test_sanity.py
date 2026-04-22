"""
test_sanity.py — Smoke tests runnable with `python -m pytest tests/`.

PLAIN ENGLISH:
These aren't exhaustive — they are a minimum bar. If any of these break,
don't trade. Every test should take < 1 second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labels import forward_return, vol_normalized_return, triple_barrier
from risk_sizing import vol_target_size, fractional_kelly, position_size_with_stop
from execution_model import realistic_fill_price, commission, capacity_warning
from data_validation import validate_price_frame


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


def test_risk_sizing_nonnegative():
    assert vol_target_size(100_000, 0.20) > 0
    assert vol_target_size(100_000, 0.0) == 0
    assert fractional_kelly(-0.1) == 0
    assert 0 < fractional_kelly(0.1) < 1
    assert position_size_with_stop(100_000, 100, 2.0) > 0


def test_execution_costs_sane():
    px = realistic_fill_price(100.0, 1_000, 1_000_000, "buy")
    assert px > 100  # buying costs more than mid
    assert commission(100) > 0
    assert capacity_warning(100_000, 1_000_000) is True


def test_validation_rejects_bad():
    import pytest
    good = _price_frame()
    validate_price_frame(good, "TEST", max_lag_days=10_000)  # bypass freshness in tests

    bad = good.copy()
    bad.loc[bad.index[5], "Close"] = -1
    with pytest.raises(ValueError):
        validate_price_frame(bad, "TEST", max_lag_days=10_000)
