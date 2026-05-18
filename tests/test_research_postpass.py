from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_research_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "research.py"
    spec = importlib.util.spec_from_file_location("stock_research_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_xs_rank_summary_rejects_write_errors():
    research = _load_research_module()

    with pytest.raises(RuntimeError) as exc:
        research._validate_xs_rank_summary({
            "updated": 1,
            "new_cols": ["xs_rank_market_mom_20d"],
            "write_errors": ["AAPL: permission denied"],
        })

    assert "write_errors" in str(exc.value)


def test_validate_xs_rank_summary_rejects_zero_updates():
    research = _load_research_module()

    with pytest.raises(RuntimeError) as exc:
        research._validate_xs_rank_summary({
            "updated": 0,
            "new_cols": ["xs_rank_market_mom_20d"],
            "write_errors": [],
        })

    assert "updated 0" in str(exc.value)


def test_validate_xs_rank_summary_rejects_empty_new_columns():
    research = _load_research_module()

    with pytest.raises(RuntimeError) as exc:
        research._validate_xs_rank_summary({
            "updated": 3,
            "new_cols": [],
            "write_errors": [],
        })

    assert "produced no xs-rank columns" in str(exc.value)


def test_research_xs_only_exits_nonzero_on_bad_summary(monkeypatch):
    research = _load_research_module()
    monkeypatch.setattr(research, "WATCHLIST", ["AAPL"])
    monkeypatch.setattr(research, "SURVIVORSHIP_TRAINING_TICKERS", [])
    monkeypatch.setattr(
        research,
        "apply_cross_sectional_rank_features",
        lambda tickers, data_dir: {"updated": 0, "new_cols": [], "skipped": [], "write_errors": []},
    )
    monkeypatch.setattr(sys, "argv", ["research.py", "--xs-only"])

    with pytest.raises(SystemExit) as exc:
        research.main()

    assert exc.value.code == 1


def test_incremental_preserves_xs_rank_postpass_columns_without_full_rebuild(tmp_path, monkeypatch):
    research = _load_research_module()
    monkeypatch.setattr(research, "DATA_DIR", str(tmp_path))

    existing = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "Close": [10.5, 11.5, 12.5],
            "xs_rank_market_ret_5d": [0.5, 0.6, 0.7],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    existing.to_parquet(tmp_path / "AAA.parquet", index=True)

    fresh = pd.DataFrame(
        {
            "Open": [11.0, 12.0, 13.0],
            "Close": [11.5, 12.5, 13.5],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    monkeypatch.setattr(research, "build_research_feature_frame", lambda ticker, start, end: fresh)

    def fail_full_rebuild(*args, **kwargs):
        raise AssertionError("post-pass xs_rank columns should not force full rebuild")

    monkeypatch.setattr(research, "research_ticker", fail_full_rebuild)

    assert research.research_ticker_incremental("AAA", "2024-01-01", "2024-01-08") is True
    out = pd.read_parquet(tmp_path / "AAA.parquet")
    assert "xs_rank_market_ret_5d" in out.columns
    assert "Open" in out.columns
