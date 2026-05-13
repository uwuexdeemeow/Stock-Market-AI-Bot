from __future__ import annotations

import pandas as pd

import intraday_features
from intraday_features import _bars_in_market_time, _compute_intraday_features_from_bars


def _regular_session_bars(day: str, *, tz_aware: bool = True, volume: int = 1) -> pd.DataFrame:
    idx_et = pd.date_range(
        f"{day} 09:30",
        f"{day} 15:59",
        freq="min",
        tz="America/New_York",
    )
    idx = idx_et.tz_convert("UTC")
    if not tz_aware:
        idx = idx.tz_localize(None)
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": volume,
            "vwap": 100.0,
        },
        index=idx,
    )


def test_volume_splits_are_correct_in_winter_est():
    feats = _compute_intraday_features_from_bars(_regular_session_bars("2026-01-15"))

    assert feats["volume_am_ratio"] == 0.385
    assert feats["volume_last_hour"] == 0.154


def test_volume_splits_are_correct_in_summer_edt():
    feats = _compute_intraday_features_from_bars(_regular_session_bars("2026-07-15"))

    assert feats["volume_am_ratio"] == 0.385
    assert feats["volume_last_hour"] == 0.154


def test_noon_volume_is_not_counted_as_morning():
    bars = _regular_session_bars("2026-07-15")
    bars_et = _bars_in_market_time(bars)
    noon_utc_index = bars_et.between_time("12:00", "12:59").index.tz_convert("UTC")
    bars.loc[noon_utc_index, "volume"] = 10

    feats = _compute_intraday_features_from_bars(bars)

    assert feats["volume_am_ratio"] == 0.161


def test_extended_hours_are_excluded_from_volume_denominator():
    bars = _regular_session_bars("2026-07-15")
    extra_idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-07-15 08:00", tz="America/New_York"),
            pd.Timestamp("2026-07-15 18:00", tz="America/New_York"),
        ]
    ).tz_convert("UTC")
    extra = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
            "vwap": 100.0,
        },
        index=extra_idx,
    )

    feats = _compute_intraday_features_from_bars(pd.concat([bars, extra]).sort_index())

    assert feats["volume_am_ratio"] == 0.385
    assert feats["volume_last_hour"] == 0.154


def test_tz_naive_bars_are_interpreted_as_utc():
    bars = _regular_session_bars("2026-07-15", tz_aware=False)
    market_bars = _bars_in_market_time(bars)

    assert str(market_bars.index.tz) == "America/New_York"
    assert market_bars.index[0].time().isoformat(timespec="minutes") == "09:30"

    feats = _compute_intraday_features_from_bars(bars)
    assert feats["volume_am_ratio"] == 0.385
    assert feats["volume_last_hour"] == 0.154


def test_fetch_groups_bars_by_new_york_market_date(monkeypatch):
    session = _regular_session_bars("2026-07-15")
    afterhours_idx = pd.date_range(
        "2026-07-15 20:00",
        periods=10,
        freq="min",
        tz="America/New_York",
    ).tz_convert("UTC")
    afterhours = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
            "vwap": 100.0,
        },
        index=afterhours_idx,
    )
    bars = pd.concat([session, afterhours]).sort_index()

    monkeypatch.setattr(intraday_features, "ALPACA_API_KEY", "test-key")
    monkeypatch.setattr(intraday_features, "ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setattr(intraday_features, "_alpaca_get_bars", lambda *_args, **_kwargs: bars)

    feats = intraday_features.fetch_intraday_features("AAPL", lookback_days=1)

    assert feats["has_intraday_data"] == 1.0
    assert feats["volume_am_ratio"] == 0.385
    assert feats["volume_last_hour"] == 0.154
