from __future__ import annotations

import json

import pandas as pd

import shadow_paper_journal as spj


def test_shadow_payload_builds_expected_candidate_from_grid():
    payload = spj.build_shadow_live_payload({"approvals": {"core-alpha": {"thresholds": {"min_folds": 3}}}})
    approved = payload["approved_live_configs"]["core-alpha"]

    assert payload["shadow"] is True
    assert approved["approved_exact_config"] == spj.SHADOW_CONFIG_SIGNATURE
    assert approved["config"]["overlay_gross"] == 0.5
    assert approved["config"]["high_vol_mode"] == "percentile"
    assert approved["config"]["score_source"] == "regime_adaptive_riskoff_guard"
    assert approved["source_metrics"]["mean_oos_alpha_vs_qqq_pct"] == 15.82
    assert payload["approvals"]["core-alpha"]["approved"] is True


def test_append_shadow_journal_replaces_same_day_same_config(tmp_path):
    journal = tmp_path / "shadow_paper_journal.csv"
    row = {
        "run_date": "2026-05-25",
        "shadow_config_signature": spj.SHADOW_CONFIG_SIGNATURE,
        "paper_ready": True,
        "overlay_tickers": "AAPL,MSFT",
    }

    spj.append_shadow_journal(row, journal_path=journal)
    newer = {**row, "paper_ready": False, "overlay_tickers": "NVDA"}
    spj.append_shadow_journal(newer, journal_path=journal)

    out = pd.read_csv(journal)
    assert len(out) == 1
    assert bool(out.iloc[0]["paper_ready"]) is False
    assert out.iloc[0]["overlay_tickers"] == "NVDA"


def test_append_shadow_journal_handles_empty_existing_file(tmp_path):
    journal = tmp_path / "shadow_paper_journal.csv"
    journal.touch()
    row = {
        "run_date": "2026-05-25",
        "shadow_config_signature": spj.SHADOW_CONFIG_SIGNATURE,
        "paper_ready": True,
        "overlay_tickers": "AAPL,MSFT",
    }

    spj.append_shadow_journal(row, journal_path=journal)

    out = pd.read_csv(journal)
    assert len(out) == 1
    assert out.iloc[0]["shadow_config_signature"] == spj.SHADOW_CONFIG_SIGNATURE


def test_journal_row_carries_signal_and_validation_metrics():
    signal = {
        "paper_ready": True,
        "gates_all_pass": True,
        "overlay_weights_json": json.dumps({"AAPL": 0.1}),
    }
    metrics = {"sharpe": 1.2, "max_drawdown_pct": -9.5, "total_return_pct": 42.0}

    row = spj._journal_row(signal, metrics)

    assert row["shadow_name"] == spj.SHADOW_NAME
    assert row["paper_ready"] is True
    assert row["source_worst_oos_drawdown_pct"] == -27.56
    assert row["backtest_sharpe"] == 1.2


def test_update_shadow_equity_initializes_first_row(tmp_path):
    equity_path = tmp_path / "shadow_paper_equity.csv"
    row = {
        "journaled_at": "2026-05-25T13:55:00+00:00",
        "run_date": "2026-05-25",
        "latest_factor_date": "2026-05-22",
        "shadow_config_signature": spj.SHADOW_CONFIG_SIGNATURE,
        "target_qqq_weight": 0.6,
        "overlay_weights_json": json.dumps({"MU": 0.2}),
        "paper_ready": True,
        "gates_all_pass": True,
    }

    spj.update_shadow_equity(row, equity_path=equity_path, data_dir=tmp_path, initial_equity=12345.0)

    out = pd.read_csv(equity_path)
    assert len(out) == 1
    assert out.iloc[0]["equity"] == 12345.0
    assert out.iloc[0]["price_status"] == "initialized"
    assert json.loads(out.iloc[0]["target_weights_json"]) == {"MU": 0.2, "QQQ": 0.6}


def test_update_shadow_equity_uses_prior_target_weights(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {"Close": [100.0, 110.0]},
        index=pd.to_datetime(["2026-05-22", "2026-05-26"]),
    ).to_parquet(data_dir / "QQQ.parquet")
    equity_path = tmp_path / "shadow_paper_equity.csv"

    first = {
        "journaled_at": "2026-05-25T13:55:00+00:00",
        "run_date": "2026-05-25",
        "latest_factor_date": "2026-05-22",
        "shadow_config_signature": spj.SHADOW_CONFIG_SIGNATURE,
        "target_qqq_weight": 1.0,
        "overlay_weights_json": "{}",
    }
    second = {
        "journaled_at": "2026-05-26T13:55:00+00:00",
        "run_date": "2026-05-26",
        "latest_factor_date": "2026-05-26",
        "shadow_config_signature": spj.SHADOW_CONFIG_SIGNATURE,
        "target_qqq_weight": 0.5,
        "overlay_weights_json": "{}",
    }

    spj.update_shadow_equity(first, equity_path=equity_path, data_dir=data_dir, initial_equity=100000.0)
    spj.update_shadow_equity(second, equity_path=equity_path, data_dir=data_dir, initial_equity=100000.0)

    out = pd.read_csv(equity_path)
    assert len(out) == 2
    assert out.iloc[-1]["equity"] == 110000.0
    assert out.iloc[-1]["period_return_pct"] == 10.0
    assert json.loads(out.iloc[-1]["applied_weights_json"]) == {"QQQ": 1.0}
    assert json.loads(out.iloc[-1]["target_weights_json"]) == {"QQQ": 0.5}
