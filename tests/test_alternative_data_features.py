from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from alternative_data_features import build_live_eps_revision_features


class FakeTicker:
    def __init__(self, eps_trend: pd.DataFrame, earnings_estimate: pd.DataFrame | None = None):
        self.eps_trend = eps_trend
        self.earnings_estimate = earnings_estimate if earnings_estimate is not None else pd.DataFrame()


def install_fake_yfinance(monkeypatch, eps_trend: pd.DataFrame, earnings_estimate: pd.DataFrame | None = None) -> None:
    fake = types.SimpleNamespace(Ticker=lambda _ticker: FakeTicker(eps_trend, earnings_estimate))
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def test_eps_revision_skips_zero_past_estimate(monkeypatch):
    eps_trend = pd.DataFrame(
        {
            "current": [2.0],
            "7daysAgo": [0.0],
            "30daysAgo": [1.0],
        },
        index=["0q"],
    )
    install_fake_yfinance(monkeypatch, eps_trend)

    features = build_live_eps_revision_features("AAPL")

    assert features["alt_eps_revision_7d"] == 0.0
    assert features["alt_eps_revision_30d"] == 1.0
    assert features["alt_has_estimates"] == 1.0


def test_eps_revision_skips_non_finite_inputs(monkeypatch):
    eps_trend = pd.DataFrame(
        {
            "current": [2.0],
            "7daysAgo": [np.nan],
            "30daysAgo": [np.inf],
            "60daysAgo": [-np.inf],
            "90daysAgo": [1.0],
        },
        index=["0q"],
    )
    earnings_estimate = pd.DataFrame(
        {
            "numberOfAnalysts": [np.inf],
            "yearAgoEps": [1.0],
            "avg": [np.nan],
        },
        index=["0q"],
    )
    install_fake_yfinance(monkeypatch, eps_trend, earnings_estimate)

    features = build_live_eps_revision_features("AAPL")

    assert features["alt_eps_revision_7d"] == 0.0
    assert features["alt_eps_revision_30d"] == 0.0
    assert features["alt_eps_revision_60d"] == 0.0
    assert features["alt_eps_revision_90d"] == 1.0
    assert features["alt_eps_surprise_pct"] == 0.0
    assert features["alt_n_analysts"] == 0.0
    assert all(np.isfinite(value) for value in features.values())


def test_eps_revision_normal_values_still_compute_and_clip(monkeypatch):
    eps_trend = pd.DataFrame(
        {
            "current": [4.0],
            "7daysAgo": [1.0],
            "30daysAgo": [5.0],
        },
        index=["0q"],
    )
    install_fake_yfinance(monkeypatch, eps_trend)

    features = build_live_eps_revision_features("AAPL")

    assert features["alt_eps_revision_7d"] == 1.0
    assert features["alt_eps_revision_30d"] == -0.2
