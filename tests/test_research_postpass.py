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


def test_incremental_skips_closed_market_when_latest_session_present(tmp_path, monkeypatch):
    research = _load_research_module()
    monkeypatch.setattr(research, "DATA_DIR", str(tmp_path))

    existing = pd.DataFrame(
        {"Open": [10.0], "Close": [10.5]},
        index=pd.to_datetime(["2024-01-05"]),
    )
    existing.to_parquet(tmp_path / "AAA.parquet", index=True)

    monkeypatch.setattr(
        research,
        "build_research_feature_frame",
        lambda *args, **kwargs: pytest.fail("closed-market current parquet should not rebuild"),
    )

    assert research.research_ticker_incremental("AAA", "2024-01-01", "2024-01-07") is True
    manifest = research.read_parquet_manifest(tmp_path / "AAA.parquet")
    assert manifest["ticker"] == "AAA"
    assert manifest["provider"] == "legacy_unknown"
    assert manifest["row_count"] == 1


def test_incremental_repairs_stale_manifest_for_current_parquet(tmp_path, monkeypatch):
    """A fresh cached parquet must repair stale provenance without a download."""
    research = _load_research_module()
    monkeypatch.setattr(research, "DATA_DIR", str(tmp_path))
    existing = pd.DataFrame(
        {"Open": [10.0], "Close": [10.5]},
        index=pd.to_datetime(["2024-01-05"]),
    )
    path = tmp_path / "AAA.parquet"
    existing.to_parquet(path, index=True)
    research.write_parquet_manifest(
        path,
        ticker="WRONG",
        provider="legacy_unknown",
        adjustment_mode="adjusted_ohlcv",
        frame=existing,
    )
    monkeypatch.setattr(
        research,
        "build_research_feature_frame",
        lambda *args, **kwargs: pytest.fail("current parquet should not download"),
    )

    assert research.research_ticker_incremental("AAA", "2024-01-01", "2024-01-07") is True
    assert research.read_parquet_manifest(path)["ticker"] == "AAA"


def test_closed_market_noop_detects_all_tickers_current(tmp_path, monkeypatch):
    research = _load_research_module()
    monkeypatch.setattr(research, "DATA_DIR", str(tmp_path))

    for ticker in ["AAA", "BBB"]:
        pd.DataFrame(
            {"Open": [10.0], "Close": [10.5]},
            index=pd.to_datetime(["2024-01-05"]),
        ).to_parquet(tmp_path / f"{ticker}.parquet", index=True)

    assert research._closed_market_incremental_noop(["AAA", "BBB"], "2024-01-07") is True


def test_closed_market_noop_allows_refresh_when_ticker_stale(tmp_path, monkeypatch):
    research = _load_research_module()
    monkeypatch.setattr(research, "DATA_DIR", str(tmp_path))

    pd.DataFrame(
        {"Open": [10.0], "Close": [10.5]},
        index=pd.to_datetime(["2024-01-05"]),
    ).to_parquet(tmp_path / "AAA.parquet", index=True)
    pd.DataFrame(
        {"Open": [9.0], "Close": [9.5]},
        index=pd.to_datetime(["2024-01-04"]),
    ).to_parquet(tmp_path / "BBB.parquet", index=True)

    assert research._closed_market_incremental_noop(["AAA", "BBB"], "2024-01-07") is False


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


def test_incremental_tail_fills_new_columns_without_full_rebuild(tmp_path, monkeypatch):
    research = _load_research_module()
    monkeypatch.setattr(research, "DATA_DIR", str(tmp_path))

    existing = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "Close": [10.5, 11.5, 12.5],
        },
        index=pd.to_datetime(["2022-01-03", "2024-01-03", "2024-01-04"]),
    )
    existing.to_parquet(tmp_path / "AAA.parquet", index=True)

    fresh = pd.DataFrame(
        {
            "Open": [11.0, 12.0, 13.0],
            "Close": [11.5, 12.5, 13.5],
            "sector_rel_return_60d": [0.1, 0.2, 0.3],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    monkeypatch.setattr(research, "build_research_feature_frame", lambda ticker, start, end: fresh)

    def fail_full_rebuild(*args, **kwargs):
        raise AssertionError("new columns should be tail-filled during daily incremental refresh")

    monkeypatch.setattr(research, "research_ticker", fail_full_rebuild)

    assert research.research_ticker_incremental("AAA", "2020-01-01", "2024-01-08") is True
    out = pd.read_parquet(tmp_path / "AAA.parquet")
    assert "sector_rel_return_60d" in out.columns
    assert pd.isna(out.loc[pd.Timestamp("2022-01-03"), "sector_rel_return_60d"])
    assert out.loc[pd.Timestamp("2024-01-05"), "sector_rel_return_60d"] == pytest.approx(0.3)


def test_incremental_backfill_new_columns_requests_full_rebuild(tmp_path, monkeypatch):
    research = _load_research_module()
    monkeypatch.setattr(research, "DATA_DIR", str(tmp_path))

    existing = pd.DataFrame(
        {"Open": [10.0], "Close": [10.5]},
        index=pd.to_datetime(["2024-01-04"]),
    )
    existing.to_parquet(tmp_path / "AAA.parquet", index=True)

    fresh = pd.DataFrame(
        {
            "Open": [13.0],
            "Close": [13.5],
            "sector_rel_return_60d": [0.3],
        },
        index=pd.to_datetime(["2024-01-05"]),
    )
    monkeypatch.setattr(research, "build_research_feature_frame", lambda ticker, start, end: fresh)

    calls: list[tuple[str, str, str]] = []

    def fake_full_rebuild(ticker, start, end):
        calls.append((ticker, start, end))
        return True

    monkeypatch.setattr(research, "research_ticker", fake_full_rebuild)

    assert research.research_ticker_incremental(
        "AAA",
        "2024-01-01",
        "2024-01-08",
        backfill_new_columns=True,
    ) is True
    assert calls == [("AAA", "2024-01-01", "2024-01-08")]
