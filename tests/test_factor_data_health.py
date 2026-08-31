from __future__ import annotations

import json
import os

import pandas as pd
import pytest

import factor_data_health as fdh


def _write_parquet(path, date: str, *, columns: list[str] | None = None) -> None:
    selected = list(columns or fdh.REQUIRED_FACTOR_COLUMNS)
    values = {column: [100.0 if column == "Close" else 0.1] for column in selected}
    df = pd.DataFrame(values, index=pd.DatetimeIndex([pd.Timestamp(date)]))
    df.to_parquet(path)
    # Live factor data now requires a provenance sidecar. Tests create the
    # smallest valid sidecar so failures still measure the behavior named by
    # each test instead of failing earlier at the provenance gate.
    manifest_dir = path.parent / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / f"{path.stem}.json").write_text(
        json.dumps({
            "ticker": path.stem,
            "provider": "test_provider",
            "adjustment_mode": "adjusted_ohlcv",
            "row_count": len(df),
            "last_date": date,
            "quality_issues": [],
        }),
        encoding="utf-8",
    )


def _write_feature_quality(path, *, mtime: int = 2_000_000_000) -> None:
    path.write_text(
        json.dumps({"features": [{"feature": "mom_20d", "grade": "A"}]}),
        encoding="utf-8",
    )
    os.utime(path, (mtime, mtime))


def _write_feature_health(signal_dir, *, mtime: int = 2_000_000_100, gate_pass: bool = True) -> None:
    (signal_dir / "feature_health_profile.json").write_text(
        json.dumps({
            "summary": {
                "feature_health_gate_pass": gate_pass,
                "feature_health_gate_reasons": [] if gate_pass else ["too_few_clusters"],
                "active_cluster_count": 6 if gate_pass else 2,
                "max_cluster_weight": 0.16 if gate_pass else 0.5,
            },
            "features": [{"feature": "mom_20d", "health_state": "healthy"}],
        }),
        encoding="utf-8",
    )
    (signal_dir / "feature_health_profile.csv").write_text(
        "feature,health_state\nmom_20d,healthy\n",
        encoding="utf-8",
    )
    os.utime(signal_dir / "feature_health_profile.json", (mtime, mtime))
    os.utime(signal_dir / "feature_health_profile.csv", (mtime, mtime))


def test_trading_day_age_uses_nyse_holiday_calendar():
    pytest.importorskip("exchange_calendars")

    # Juneteenth 2024 was a Wednesday, but the NYSE was closed. The factor
    # cache should age by one session from Jun 18 to Jun 20, not two weekdays.
    age = fdh.trading_day_age(pd.Timestamp("2024-06-18"), now=pd.Timestamp("2024-06-20"))

    assert age == 1


def test_trading_day_age_does_not_count_unfinished_new_york_session():
    pytest.importorskip("exchange_calendars")

    # At this UTC time New York is still before Monday's opening bell. Friday
    # data is current and must not be aged by the unfinished Monday session.
    age = fdh.trading_day_age(
        pd.Timestamp("2026-08-28"),
        now=pd.Timestamp("2026-08-31T12:00:00Z"),
    )

    assert age == 0


def test_factor_data_health_does_not_warn_on_nyse_holiday_gap(tmp_path, monkeypatch):
    pytest.importorskip("exchange_calendars")

    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))

    _write_parquet(data_dir / "AAA.parquet", "2024-06-18")
    _write_feature_quality(signal_dir / "feature_quality_report.json")
    _write_feature_health(signal_dir)

    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA"],
        optional_tickers=[],
        warn_days=1,
        block_days=10,
        now=pd.Timestamp("2024-06-20"),
    )

    assert manifest["max_age_trading_days"] == 1
    assert manifest["stale_tickers"] == []
    assert manifest["trade_ready"] is True


def test_factor_data_health_trade_ready_with_fresh_required_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))

    _write_parquet(data_dir / "AAA.parquet", "2026-05-18")
    _write_parquet(data_dir / "BBB.parquet", "2026-05-18")
    _write_feature_quality(signal_dir / "feature_quality_report.json")
    _write_feature_health(signal_dir)

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
    assert manifest["feature_health"]["ready"] is True
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
    _write_feature_health(signal_dir)

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


def test_factor_data_health_blocks_required_column_gaps(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))

    _write_parquet(data_dir / "AAA.parquet", "2026-05-18", columns=["Close", "factor_idio_vol_252_spy"])
    _write_feature_quality(signal_dir / "feature_quality_report.json")
    _write_feature_health(signal_dir)

    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA"],
        optional_tickers=[],
        now=pd.Timestamp("2026-05-19"),
    )

    assert manifest["factor_data_ready"] is False
    assert manifest["signal_ready"] is False
    assert manifest["trade_ready"] is False
    assert manifest["missing_required_column_tickers"][0]["ticker"] == "AAA"
    assert "hvol_20d" in manifest["missing_required_column_tickers"][0]["missing_columns"]
    assert manifest["tickers"]["AAA"]["status"] == "missing_required_columns"
    assert "factor_data_missing_required_columns" in manifest["reasons"]


def test_factor_data_health_blocks_missing_provenance_manifest(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))
    _write_parquet(data_dir / "AAA.parquet", "2026-05-18")
    (data_dir / "manifests" / "AAA.json").unlink()
    _write_feature_quality(signal_dir / "feature_quality_report.json")
    _write_feature_health(signal_dir)

    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA"],
        optional_tickers=[],
        now=pd.Timestamp("2026-05-19"),
    )

    assert manifest["trade_ready"] is False
    assert manifest["manifest_error_tickers"][0]["ticker"] == "AAA"
    assert "factor_data_manifest_errors" in manifest["reasons"]


def test_feature_quality_stale_vs_required_factor_data_but_ignores_etf(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))

    report_path = signal_dir / "feature_quality_report.json"
    _write_feature_quality(report_path, mtime=1_700_000_000)
    _write_feature_health(signal_dir, mtime=1_700_000_300)
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
    _write_feature_health(signal_dir, mtime=1_700_000_400)
    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA"],
        optional_tickers=[],
        now=pd.Timestamp("2026-05-19"),
    )
    assert manifest["feature_quality"]["ready"] is True
    assert manifest["trade_ready"] is True


def test_factor_data_health_blocks_missing_feature_health_profile(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))

    _write_parquet(data_dir / "AAA.parquet", "2026-05-18")
    _write_feature_quality(signal_dir / "feature_quality_report.json")

    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA"],
        optional_tickers=[],
        now=pd.Timestamp("2026-05-19"),
    )

    assert manifest["feature_quality"]["ready"] is True
    assert manifest["feature_health"]["ready"] is False
    assert manifest["feature_health"]["reason"] == "missing_profile"
    assert manifest["trade_ready"] is False
    assert "feature_health_missing_profile" in manifest["reasons"]


def test_factor_data_health_blocks_failed_feature_health_gate(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    signal_dir = tmp_path / "signals"
    data_dir.mkdir()
    signal_dir.mkdir()
    monkeypatch.setattr(fdh, "ADAPTIVE_WEIGHTS_FILE", str(signal_dir / "missing_adaptive.json"))

    _write_parquet(data_dir / "AAA.parquet", "2026-05-18")
    _write_feature_quality(signal_dir / "feature_quality_report.json")
    _write_feature_health(signal_dir, gate_pass=False)

    manifest = fdh.build_factor_data_health(
        data_dir=data_dir,
        signal_dir=signal_dir,
        tickers=["AAA"],
        optional_tickers=[],
        now=pd.Timestamp("2026-05-19"),
    )

    assert manifest["feature_health"]["ready"] is False
    assert manifest["feature_health"]["reason"] == "feature_health_gate_failed"
    assert manifest["trade_ready"] is False
