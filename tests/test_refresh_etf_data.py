from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys

import refresh_etf_data


def _etf_frame(idx: pd.DatetimeIndex, start: float = 100.0, end: float = 125.0) -> pd.DataFrame:
    close = np.linspace(start, end, len(idx))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(idx), 1_000_000),
        },
        index=idx,
    )


def test_completed_day_skips_nyse_holiday_after_close():
    pytest.importorskip("exchange_calendars")

    now = pd.Timestamp("2024-06-19T22:00:00Z")

    assert refresh_etf_data._completed_day(now=now) == pd.Timestamp("2024-06-18")


def test_etf_age_uses_nyse_holiday_calendar(monkeypatch):
    pytest.importorskip("exchange_calendars")

    idx = pd.bdate_range(end="2024-06-18", periods=260)
    frame = _etf_frame(idx)
    monkeypatch.setattr(refresh_etf_data, "_completed_day", lambda: pd.Timestamp("2024-06-20"))

    report = refresh_etf_data._validate_etf_frame(frame, symbol="SPY", max_age_business_days=1)

    assert report["age_business_days"] == 1
    assert report["ok"] is True


def test_etf_validation_blocks_missing_ohlcv_columns(monkeypatch):
    idx = pd.bdate_range(end="2024-06-18", periods=260)
    frame = pd.DataFrame({"Close": np.linspace(100.0, 125.0, len(idx))}, index=idx)
    monkeypatch.setattr(refresh_etf_data, "_completed_day", lambda: pd.Timestamp("2024-06-18"))

    report = refresh_etf_data._validate_etf_frame(frame, symbol="SPY", max_age_business_days=1)

    assert report["ok"] is False
    assert "missing_open_column" in report["issues"]
    assert "missing_high_column" in report["issues"]
    assert "missing_low_column" in report["issues"]
    assert "missing_volume_column" in report["issues"]


def test_etf_refresh_force_replaces_healthy_local_data(tmp_path, monkeypatch):
    idx = pd.bdate_range(start="2025-01-01", periods=260)
    local = _etf_frame(idx, 100.0, 125.0)
    downloaded = _etf_frame(idx, 200.0, 260.0)
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


def test_etf_data_health_report_uses_atomic_writer(monkeypatch, tmp_path):
    calls = []

    def fake_write_json(data, path, **_kwargs):
        calls.append((path.name, data["ok"]))

    monkeypatch.setattr(refresh_etf_data, "atomic_write_json", fake_write_json)

    out = refresh_etf_data.write_etf_data_health(
        {"ok": True, "results": []},
        output_path=tmp_path / "etf_data_health.json",
    )

    assert out == tmp_path / "etf_data_health.json"
    assert calls == [("etf_data_health.json", True)]


def test_etf_refresh_strict_exits_nonzero_when_report_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_etf_data, "LOGS", tmp_path)
    monkeypatch.setattr(
        refresh_etf_data,
        "validate_etfs",
        lambda symbols, refresh=False, force=False: {
            "ok": False,
            "refresh": bool(refresh),
            "force": bool(force),
            "results": [{
                "symbol": "SPY",
                "ok": False,
                "issues": ["missing_close_column"],
                "rows": 0,
                "latest_date": None,
                "age_business_days": None,
            }],
        },
    )
    monkeypatch.setattr(sys, "argv", ["refresh_etf_data.py", "--strict"])

    with pytest.raises(SystemExit) as exc:
        refresh_etf_data.main()

    assert exc.value.code == 1
