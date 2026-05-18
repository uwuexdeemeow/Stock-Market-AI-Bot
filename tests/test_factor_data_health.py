from __future__ import annotations

import json
import os

import pandas as pd

import factor_data_health as fdh


def _write_parquet(path, date: str) -> None:
    df = pd.DataFrame(
        {"Close": [100.0], "factor_idio_vol_252_spy": [0.1]},
        index=pd.DatetimeIndex([pd.Timestamp(date)]),
    )
    df.to_parquet(path)


def _write_feature_quality(path, *, mtime: int = 2_000_000_000) -> None:
    path.write_text(
        json.dumps({"features": [{"feature": "mom_20d", "grade": "A"}]}),
        encoding="utf-8",
    )
    os.utime(path, (mtime, mtime))


def test_factor_data_health_trade_ready_with_fresh_required_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))

    _write_parquet(data_dir / "AAA.parquet", "2026-05-18")
    _write_parquet(data_dir / "BBB.parquet", "2026-05-18")
    _write_feature_quality(signal_dir / "feature_quality_report.json")

    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA", "BBB"],
        optional_tickers=[],
        now=pd.Timestamp("2026-05-19"),
    )

    assert manifest["factor_data_ready"] is True
    assert manifest["factor_data_fresh"] is True
    assert manifest["feature_quality"]["ready"] is True
    assert manifest["trade_ready"] is True
    assert manifest["adaptive_weights"]["adaptive_weight_status"] == "fallback"


def test_factor_data_health_blocks_missing_and_stale_required_tickers(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))

    _write_parquet(data_dir / "AAA.parquet", "2026-04-01")
    _write_feature_quality(signal_dir / "feature_quality_report.json")

    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA", "BBB"],
        optional_tickers=[],
        now=pd.Timestamp("2026-05-19"),
    )

    assert manifest["factor_data_ready"] is False
    assert manifest["trade_ready"] is False
    assert manifest["missing_tickers"] == ["BBB"]
    assert manifest["blocked_tickers"][0]["ticker"] == "AAA"
    assert "missing_factor_parquets" in manifest["reasons"]
    assert "factor_data_too_stale" in manifest["reasons"]


def test_feature_quality_stale_vs_required_factor_data_but_ignores_etf(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))

    report_path = signal_dir / "feature_quality_report.json"
    _write_feature_quality(report_path, mtime=1_700_000_000)
    _write_parquet(data_dir / "AAA.parquet", "2026-05-18")
    os.utime(data_dir / "AAA.parquet", (1_700_000_100, 1_700_000_100))
    _write_parquet(data_dir / "SPY.parquet", "2026-05-18")
    os.utime(data_dir / "SPY.parquet", (1_700_000_200, 1_700_000_200))

    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA"],
        optional_tickers=[],
        now=pd.Timestamp("2026-05-19"),
    )
    assert manifest["feature_quality"]["ready"] is False
    assert manifest["feature_quality"]["newer_factor_file_count"] == 1
    assert manifest["trade_ready"] is False

    os.utime(report_path, (1_700_000_300, 1_700_000_300))
    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA"],
        optional_tickers=[],
        now=pd.Timestamp("2026-05-19"),
    )
    assert manifest["feature_quality"]["ready"] is True
    assert manifest["trade_ready"] is True
