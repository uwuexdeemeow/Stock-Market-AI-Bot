from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import refresh_etf_data


def test_completed_day_skips_nyse_holiday_after_close():
    pytest.importorskip("exchange_calendars")

    now = pd.Timestamp("2024-06-19T22:00:00Z")

    assert refresh_etf_data._completed_day(now=now) == pd.Timestamp("2024-06-18")


def test_etf_age_uses_nyse_holiday_calendar(monkeypatch):
    pytest.importorskip("exchange_calendars")

    idx = pd.bdate_range(end="2024-06-18", periods=260)
    frame = pd.DataFrame({"Close": np.linspace(100.0, 125.0, len(idx))}, index=idx)
    monkeypatch.setattr(refresh_etf_data, "_completed_day", lambda: pd.Timestamp("2024-06-20"))

    report = refresh_etf_data._validate_etf_frame(frame, symbol="SPY", max_age_business_days=1)

    assert report["age_business_days"] == 1
    assert report["ok"] is True


def test_etf_refresh_force_replaces_healthy_local_data(tmp_path, monkeypatch):
    idx = pd.bdate_range(start="2025-01-01", periods=260)
    local = pd.DataFrame({"Close": np.linspace(100.0, 125.0, len(idx))}, index=idx)
    downloaded = pd.DataFrame({"Close": np.linspace(200.0, 260.0, len(idx))}, index=idx)
    local.to_parquet(tmp_path / "SPY.parquet")

    monkeypatch.setattr(refresh_etf_data, "DATA", tmp_path)
    monkeypatch.setattr(refresh_etf_data, "_completed_day", lambda: idx[-1])
    monkeypatch.setattr(refresh_etf_data, "_download", lambda symbol: downloaded)

    no_force = refresh_etf_data.validate_etfs(["SPY"], refresh=True, force=False)
    assert no_force["results"][0]["refreshed"] is False
    assert pd.read_parquet(tmp_path / "SPY.parquet")["Close"].iloc[0] == 100.0

    forced = refresh_etf_data.validate_etfs(["SPY"], refresh=True, force=True)
    assert forced["results"][0]["refreshed"] is True
    assert pd.read_parquet(tmp_path / "SPY.parquet")["Close"].iloc[0] == 200.0
