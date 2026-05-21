from __future__ import annotations

import json
import os

import pandas as pd
import pytest

import core_satellite_alpha as csa


def _write_status(path, *, equity=1000.0, position_values=None):
    payload = {
        "account_equity": equity,
        "positions": {ticker: 1 for ticker in (position_values or {})},
        "position_values": position_values or {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_live_metrics() -> dict:
    return {
        "core_preset": "test_static",
        "regime_mode": "static",
        "current_regime": "static",
        "core_weights": {"QQQ": 1.0},
        "core_gross": 0.5,
        "overlay_gross": 0.5,
        "score_source": "factor_walkforward",
        "shape": "top3",
        "weighting": "sticky_score",
        "exit_rank_floor": 0.20,
        "max_per_sector": 10,
        "max_single_name_weight": 1.0,
        "holding_days": 10,
        "feature_health_gate_pass": True,
        "paper_ready": True,
        "robust_cost_stress_pass": True,
        "core_satellite_gate_results": {"all_pass": True},
    }


def _live_panel() -> pd.DataFrame:
    date = pd.Timestamp("2026-05-13")
    return pd.DataFrame({
        "date": [date, date, date, date],
        "ticker": ["AAA", "BBB", "CCC", "HELD"],
        "sector": ["XLK", "XLF", "XLI", "XLB"],
        "factor_walkforward_score": [1.0, 0.9, 0.8, 0.1],
    })


def test_live_sticky_state_uses_alpaca_status_and_filters_core_tickers(tmp_path):
    status_path = tmp_path / "alpaca_daily_status.json"
    _write_status(
        status_path,
        equity=1000.0,
        position_values={
            "US.QQQ": 700.0,
            "TQQQ": 50.0,
            "CAT": 100.0,
            "MU": 50.0,
        },
    )

    state = csa._load_live_sticky_overlay_state(
        status_path=status_path,
        signal_path=tmp_path / "missing_signal.csv",
    )

    assert state["source"] == "alpaca_daily_status"
    assert state["used"] is True
    assert state["held_tickers"] == {"CAT", "MU"}
    assert round(float(state["prev_overlay"]["CAT"]), 6) == 0.10
    assert round(float(state["prev_overlay"]["MU"]), 6) == 0.05


def test_live_sticky_state_falls_back_to_previous_signal_when_status_unusable(tmp_path):
    status_path = tmp_path / "paper_daily_status.json"
    status_path.write_text("{not-json", encoding="utf-8")
    signal_path = tmp_path / "core_satellite_alpha_signal.csv"
    pd.DataFrame([{
        "overlay_weights_json": json.dumps({"QQQ": 0.5, "CAT": 0.1, "MU": 0.05}),
    }]).to_csv(signal_path, index=False)

    state = csa._load_live_sticky_overlay_state(
        status_path=status_path,
        signal_path=signal_path,
    )

    assert state["source"] == "previous_signal"
    assert state["used"] is True
    assert state["held_tickers"] == {"CAT", "MU"}
    assert round(float(state["prev_overlay"]["CAT"]), 6) == 0.10


def test_live_sticky_state_missing_sources_is_empty(tmp_path):
    state = csa._load_live_sticky_overlay_state(
        status_path=tmp_path / "missing_status.json",
        signal_path=tmp_path / "missing_signal.csv",
    )

    assert state["source"] == "none"
    assert state["used"] is False
    assert state["held_tickers"] == set()
    assert state["prev_overlay"].empty


def test_write_paper_signal_retains_and_blends_live_sticky_holding(tmp_path, monkeypatch):
    monkeypatch.setattr(csa, "SIGNAL_DIR", str(tmp_path))
    monkeypatch.setattr(csa, "SENTIMENT_VETO_ENABLED", False)
    monkeypatch.setattr(csa, "_paper_signal_timestamp", lambda: "2026-05-13T22:00+08:00")
    _write_status(
        tmp_path / "alpaca_daily_status.json",
        equity=1000.0,
        position_values={"QQQ": 500.0, "HELD": 100.0},
    )

    out = csa.write_paper_signal(_live_panel(), _base_live_metrics())
    row = pd.read_csv(out).iloc[0]
    overlay = json.loads(str(row["overlay_weights_json"]))

    assert "HELD" in overlay
    assert "CCC" not in overlay
    assert float(overlay["HELD"]) > 0.15
    assert row["sticky_holdings_source"] == "alpaca_daily_status"
    assert bool(row["sticky_holdings_used"]) is True
    assert row["sticky_held_tickers"] == "HELD"
    sticky_prev = json.loads(str(row["sticky_prev_overlay_json"]))
    assert round(float(sticky_prev["HELD"]), 6) == 0.10


def _approved_live(cost_pass=True) -> dict:
    return {
        "approval": {"approved": True, "reasons": []},
        "source_metrics": {"cost_stress_approval_pass": cost_pass},
        "medium_risk_review": {"pass": True, "reasons": []},
    }


def test_nested_live_gate_ignores_full_sample_gate_failure_when_nested_passes():
    metrics = {
        "core_satellite_gate_results": {"all_pass": False, "holdout_2023_2026_vs_qqq_pass": False},
        "feature_health_gate_pass": True,
    }
    out = csa._apply_nested_live_approval_gates(metrics, _approved_live(cost_pass=True), {"fresh": True})

    assert out["paper_ready"] is True
    assert out["core_satellite_gate_results"]["all_pass"] is True
    assert out["full_sample_gate_all_pass"] is False
    assert out["full_sample_core_satellite_gate_results"]["holdout_2023_2026_vs_qqq_pass"] is False
    assert out["live_gate_source"] == "nested_walkforward_approval"


def test_nested_live_gate_blocks_nested_cost_stress_failure():
    out = csa._apply_nested_live_approval_gates(
        {"core_satellite_gate_results": {"all_pass": True}, "feature_health_gate_pass": True},
        _approved_live(cost_pass=False),
        {"fresh": True},
    )

    assert out["paper_ready"] is False
    assert "nested_cost_stress_approval_failed" in out["live_gate_reasons"]


def test_nested_live_gate_blocks_feature_health_failure():
    out = csa._apply_nested_live_approval_gates(
        {
            "core_satellite_gate_results": {"all_pass": True},
            "feature_health_gate_pass": False,
            "feature_health_gate_reasons": ["too_few_clusters"],
        },
        _approved_live(cost_pass=True),
        {"fresh": True},
    )

    assert out["paper_ready"] is False
    assert any("feature_health_gate_failed" in reason for reason in out["live_gate_reasons"])


def test_nested_live_gate_blocks_stale_factor_data():
    freshness = {"fresh": False, "reason": "latest_factor_date_old"}
    out = csa._apply_nested_live_approval_gates(
        {"core_satellite_gate_results": {"all_pass": True}, "feature_health_gate_pass": True},
        _approved_live(cost_pass=True),
        freshness,
    )

    assert out["paper_ready"] is False
    assert "factor_data_stale" in out["live_gate_reasons"]
    assert out["factor_data_freshness"] == freshness


def test_strict_feature_quality_missing_report_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(csa, "SIGNAL_DIR", str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        csa._load_feature_quality_filter(strict=True)

    assert "feature_quality_diagnostic.py --top 48" in str(exc.value)


def test_strict_feature_quality_invalid_report_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(csa, "SIGNAL_DIR", str(tmp_path))
    (tmp_path / "feature_quality_report.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        csa._load_feature_quality_filter(strict=True)

    assert "Invalid live feature quality report" in str(exc.value)


def test_strict_feature_quality_all_filtered_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(csa, "SIGNAL_DIR", str(tmp_path))
    (tmp_path / "feature_quality_report.json").write_text(
        json.dumps({"features": [{"feature": "mom_20d", "grade": "F"}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        csa._load_feature_quality_filter(strict=True)

    assert "filters out every graded feature" in str(exc.value)


def test_strict_feature_quality_ignores_newer_etf_parquets(tmp_path, monkeypatch):
    signal_dir = tmp_path / "signals"
    data_dir = tmp_path / "data"
    signal_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(csa, "SIGNAL_DIR", str(signal_dir))
    monkeypatch.setattr(csa, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(csa, "WATCHLIST", ["AAPL"])
    monkeypatch.setattr(csa, "SURVIVORSHIP_TRAINING_TICKERS", [])

    report_path = signal_dir / "feature_quality_report.json"
    report_path.write_text(
        json.dumps({"features": [{"feature": "mom_20d", "grade": "A"}]}),
        encoding="utf-8",
    )
    report_mtime = 1_700_000_000
    os.utime(report_path, (report_mtime, report_mtime))

    factor_path = data_dir / "AAPL.parquet"
    factor_path.write_text("factor", encoding="utf-8")
    os.utime(factor_path, (report_mtime - 100, report_mtime - 100))

    etf_path = data_dir / "SPY.parquet"
    etf_path.write_text("etf", encoding="utf-8")
    os.utime(etf_path, (report_mtime + 100, report_mtime + 100))

    assert csa._load_feature_quality_filter(strict=True) == {"mom_20d"}


def test_live_feature_validator_requires_xs_rank_features():
    specs = [{"feature": "mom_20d"}]
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-05-13")], "ticker": ["AAA"], "mom_20d": [1.0]})

    with pytest.raises(SystemExit) as exc:
        csa._validate_live_feature_inputs(specs, panel)

    assert "xs_rank_market" in str(exc.value)


def test_live_feature_validator_requires_selected_columns_present():
    specs = [{"feature": "xs_rank_market_mom_20d"}, {"feature": "missing_feature"}]
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2026-05-13")],
        "ticker": ["AAA"],
        "xs_rank_market_mom_20d": [0.9],
    })

    with pytest.raises(SystemExit) as exc:
        csa._validate_live_feature_inputs(specs, panel)

    assert "missing_feature" in str(exc.value)


def test_write_paper_signal_marks_regime_refresh_failure_not_tradeable(tmp_path, monkeypatch):
    metrics = _base_live_metrics()
    metrics.update({
        "core_preset": "test_regime",
        "regime_mode": "test_regime",
        "current_regime": "risk_on",
        "score_source": "factor_walkforward",
        "live_gate_reasons": [],
    })
    monkeypatch.setitem(csa.REGIME_PRESETS, "test_regime", {
        "risk_on": {"core_weights": {"QQQ": 1.0}, "core_gross": 0.5, "overlay_gross": 0.5},
        "neutral": {"core_weights": {"SPY": 1.0}, "core_gross": 0.5, "overlay_gross": 0.0},
        "risk_off": {"core_weights": {"BIL": 1.0}, "core_gross": 0.5, "overlay_gross": 0.0},
    })
    monkeypatch.setattr(csa, "SIGNAL_DIR", str(tmp_path))
    monkeypatch.setattr(csa, "SENTIMENT_VETO_ENABLED", False)
    monkeypatch.setattr(csa, "_paper_signal_timestamp", lambda: "2026-05-13T22:00+08:00")
    monkeypatch.setattr(csa, "_load_regime_indicators", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("regime source down")))

    out = csa.write_paper_signal(_live_panel(), metrics)
    row = pd.read_csv(out).iloc[0]

    assert bool(row["paper_ready"]) is False
    assert bool(row["gates_all_pass"]) is False
    assert bool(row["live_regime_refresh_failed"]) is True
    assert "regime source down" in str(row["live_regime_refresh_error"])
    assert "regime_refresh_failed" in str(row["reason"])
    assert float(row["gross_exposure"]) == 0.0
