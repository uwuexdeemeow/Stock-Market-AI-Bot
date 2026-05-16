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
