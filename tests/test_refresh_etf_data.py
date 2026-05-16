from __future__ import annotations

import numpy as np
import pandas as pd

import refresh_etf_data


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
