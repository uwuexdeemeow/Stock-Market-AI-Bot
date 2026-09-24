from __future__ import annotations

import pandas as pd
import pytest

import pipeline_shared


def test_fetch_price_data_prefers_local_parquet_when_providers_fail(tmp_path, monkeypatch):
    idx = pd.bdate_range("2026-01-01", periods=5)
    local = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1_000_000] * 5,
            "feature_col": [1.0] * 5,
        },
        index=idx,
    )
    local.to_parquet(tmp_path / "AAPL.parquet")

    monkeypatch.setattr(pipeline_shared, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        pipeline_shared,
        "_dp_download_single",
        lambda *args, **kwargs: pytest.fail("provider should not be called when local parquet is usable"),
    )

    out = pipeline_shared.fetch_price_data("AAPL", "2026-01-02", "2026-01-07")

    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 3
    assert out.index.min() == pd.Timestamp("2026-01-02")
    assert out.index.max() == pd.Timestamp("2026-01-06")


def test_fetch_price_data_refreshes_when_cache_missing_expected_bar(tmp_path, monkeypatch):
    local = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [101.0, 102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.5, 101.5, 102.5, 103.5],
            "Volume": [1_000_000] * 4,
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    local.to_parquet(tmp_path / "AAPL.parquet")

    calls: list[tuple[str, str, str]] = []

    def fake_download(ticker, start=None, end=None, accept_frame=None):
        calls.append((ticker, start, end))
        return pd.DataFrame(
            {
                "Open": [104.0],
                "High": [105.0],
                "Low": [103.0],
                "Close": [104.5],
                "Volume": [1_000_000],
            },
            index=pd.to_datetime(["2024-01-05"]),
        )

    monkeypatch.setattr(pipeline_shared, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline_shared, "_dp_download_single", fake_download)

    out = pipeline_shared.fetch_price_data("AAPL", "2024-01-01", "2024-01-06")

    assert calls == [("AAPL", "2024-01-01", "2024-01-06")]
    assert out.index.max() == pd.Timestamp("2024-01-05")


def _bad_latest_bar_frame() -> pd.DataFrame:
    # PLAIN ENGLISH: the final Open is higher than High, which no real daily
    # price bar can have. Earlier days are good comparison points.
    dates = pd.bdate_range("2026-09-16", "2026-09-23")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104, 110],
            "High": [102, 103, 104, 105, 106, 109],
            "Low": [99, 100, 101, 102, 103, 106],
            "Close": [101, 102, 103, 104, 105, 108.5],
            "Volume": [1_000_000] * 6,
        },
        index=dates,
    )


def test_invalid_cached_bar_repaired_only_with_matching_alpaca_history(tmp_path, monkeypatch):
    frame = _bad_latest_bar_frame()
    frame.to_parquet(tmp_path / "NEE.parquet")
    monkeypatch.setattr(pipeline_shared, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    def unavailable_provider(ticker, start=None, end=None, accept_frame=None):
        raise RuntimeError("primary provider unavailable")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            bars = [
                {"t": f"{day.date()}T00:00:00Z", "o": close - 1,
                 "h": close + 1, "l": close - 2, "c": close, "v": 1000}
                for day, close in zip(frame.index[:-1], frame["Close"].iloc[:-1])
            ]
            bars.append({"t": "2026-09-23T00:00:00Z", "o": 108, "h": 109,
                         "l": 106, "c": 108.5, "v": 1000})
            return {"bars": bars}

    monkeypatch.setattr(pipeline_shared, "_dp_download_single", unavailable_provider)
    monkeypatch.setattr(pipeline_shared.requests, "get", lambda *args, **kwargs: Response())

    out = pipeline_shared.fetch_price_data("NEE", "2026-09-16", "2026-09-24")

    assert out.loc[pd.Timestamp("2026-09-23"), "Open"] == 108
    assert out.loc[pd.Timestamp("2026-09-23"), "Volume"] == 1_000_000
    assert pipeline_shared.data_provider.provider_for_ticker["NEE"] == "cached+alpaca_iex"


def test_invalid_cached_bar_stays_blocked_when_backup_disagrees(tmp_path, monkeypatch):
    frame = _bad_latest_bar_frame()
    frame.to_parquet(tmp_path / "NEE.parquet")
    monkeypatch.setattr(pipeline_shared, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setattr(
        pipeline_shared, "_dp_download_single",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"bars": [
                {"t": f"{day.date()}T00:00:00Z", "o": 50, "h": 52,
                 "l": 49, "c": 51, "v": 1000}
                for day in frame.index
            ]}

    monkeypatch.setattr(pipeline_shared.requests, "get", lambda *args, **kwargs: Response())
    assert pipeline_shared.fetch_price_data("NEE", "2026-09-16", "2026-09-24").empty
