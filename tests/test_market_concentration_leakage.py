"""Deterministic look-ahead test for market-concentration features."""

from __future__ import annotations

import numpy as np
import pandas as pd

import fundamental_features


def _download_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a provider-shaped Close panel with different causal trends."""
    tickers = ["QQQ", "QQQE", "RSP", "SPY", "XLK"]
    data = {}
    for number, ticker in enumerate(tickers, start=1):
        values = 100.0 + number * np.linspace(0.0, 20.0 + number, len(dates))
        data[("Close", ticker)] = values
    return pd.DataFrame(data, index=dates)


def test_future_etf_prices_cannot_change_past_concentration_features(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=180, freq="B")
    original = _download_frame(dates)
    cutoff = 119

    monkeypatch.setattr(
        fundamental_features,
        "_dp_download_prices",
        lambda *args, **kwargs: original.copy(),
    )
    real = fundamental_features.build_market_concentration_features(
        dates, str(dates.min().date()), str(dates.max().date())
    )

    perturbed = original.copy()
    future_dates = dates[cutoff + 1:]
    perturbed.loc[future_dates, :] = perturbed.loc[future_dates, :].to_numpy()[::-1] * 4.0
    monkeypatch.setattr(
        fundamental_features,
        "_dp_download_prices",
        lambda *args, **kwargs: perturbed.copy(),
    )
    changed_future = fundamental_features.build_market_concentration_features(
        dates, str(dates.min().date()), str(dates.max().date())
    )

    pd.testing.assert_frame_equal(
        real.iloc[:cutoff + 1],
        changed_future.iloc[:cutoff + 1],
        check_exact=True,
    )
