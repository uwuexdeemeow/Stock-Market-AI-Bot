from __future__ import annotations

import json

import pandas as pd

import dashboard.data as data


def _clear_dashboard_caches() -> None:
    """Clear Streamlit caches when tests monkeypatch dashboard file paths."""
    for loader in (
        data.refresh_live_alpaca_snapshot,
        data.load_alpaca_status,
        data.load_alpaca_equity_history,
        data.load_slippage_reversal_report,
        data.load_execution_scorecard,
        data.load_etf_data_health,
        data.load_factor_data_health,
        data.load_quant_performance_audit,
        data.load_paper_shadow_compare,
        data.load_broker_truth,
    ):
        try:
            loader.clear()
        except Exception:
            pass


def test_execution_report_falls_back_to_alpaca_paper_log(tmp_path, monkeypatch):
    log_path = tmp_path / "alpaca_paper_log.csv"
    report_path = tmp_path / "missing_slippage_report.json"
    log_path.write_text(
        "\n".join([
            "submitted_at,order_id,ticker,side,quantity,price,trade_value,target_weight,fill_status,filled_qty,filled_avg_price",
            "2026-05-26T13:55:52+00:00,failed,FCX,sell,302,64.665,19528.83,0.0,submission_failed,,",
            "2026-05-26T13:55:52+00:00,partial,NEM,sell,146,110.58,16144.68,0.0,partially_filled,141.0,110.56",
            "2026-05-26T13:55:52+00:00,filled,CAT,buy,3,899.71,2699.13,0.0292,filled,3.0,900.24",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(data, "ALPACA_LOG", log_path)
    monkeypatch.setattr(data, "ALPACA_SLIPPAGE_REPORT", report_path)
    _clear_dashboard_caches()

    report = data.load_slippage_reversal_report()

    assert report is not None
    assert report["source"] == "alpaca_paper_log.csv fallback"
    assert report["summary"]["orders_analyzed"] == 2
    assert report["schema_version"] == 2
    assert report["summary"]["slippage_measured_orders"] == 2
    assert report["summary"]["adverse_15m_measured_orders"] == 0
    assert report["summary"]["avg_slippage_bps"] is not None
    assert report["segments"]["all_orders"]["orders_analyzed"] == 2
    assert {row["symbol"] for row in report["orders"]} == {"NEM", "CAT"}


def test_execution_report_prefers_full_slippage_report(tmp_path, monkeypatch):
    log_path = tmp_path / "alpaca_paper_log.csv"
    report_path = tmp_path / "alpaca_slippage_reversal_report.json"
    log_path.write_text(
        "submitted_at,ticker,side,quantity,price,fill_status,filled_qty,filled_avg_price\n"
        "2026-05-26T13:55:52+00:00,CAT,buy,3,899.71,filled,3.0,900.24\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps({
            "source": "alpaca_api",
            "summary": {"orders_analyzed": 1},
            "orders": [{"symbol": "API"}],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(data, "ALPACA_LOG", log_path)
    monkeypatch.setattr(data, "ALPACA_SLIPPAGE_REPORT", report_path)
    _clear_dashboard_caches()

    report = data.load_slippage_reversal_report()

    assert report is not None
    assert report["source"] == "alpaca_api"
    assert report["orders"][0]["symbol"] == "API"


def test_execution_evidence_states_are_not_conflated():
    assert data.execution_evidence_state(None)["category"] == "operational_failure"
    collecting = data.execution_evidence_state({
        "status": "collecting",
        "decision_eligible": False,
        "decision_blockers": ["adverse_60m_coverage_below_80pct"],
    })
    assert collecting["category"] == "insufficient_evidence"
    stale = data.execution_evidence_state({
        "status": "collecting",
        "decision_eligible": False,
        "decision_blockers": ["scorecard_stale"],
    })
    assert stale["category"] == "stale_evidence"
    failed = data.execution_evidence_state({
        "status": "fail",
        "decision_eligible": True,
        "checks": [{"name": "bad_slippage_rate", "status": "fail"}],
    })
    assert failed["category"] == "measured_failure"
    assert "bad_slippage_rate" in failed["reason"]


def test_account_alignment_missing_evidence_is_collecting_not_false_failure():
    collecting = data.account_alignment_evidence_state({"account_aligned_with_target": False})
    assert collecting["status"] == "collecting"
    assert data.account_alignment_evidence_state({"account_alignment_status": "pass"})["status"] == "pass"
    assert data.account_alignment_evidence_state({"account_alignment_status": "fail"})["status"] == "fail"


def test_account_summary_uses_equity_csv_when_status_missing(tmp_path, monkeypatch):
    status_path = tmp_path / "missing_status.json"
    equity_path = tmp_path / "alpaca_paper_equity.csv"
    health_path = tmp_path / "missing_health.json"
    equity_path.write_text(
        "\n".join([
            "date,timestamp,equity,cash,invested",
            "2026-06-02,2026-06-02T21:30:00+00:00,100000,25000,75000",
            "2026-06-03,2026-06-03T21:30:00+00:00,101500,24000,77500",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(data, "ALPACA_STATUS", status_path)
    monkeypatch.setattr(data, "ALPACA_EQUITY", equity_path)
    monkeypatch.setattr(data, "ALPACA_HEALTH", health_path)
    monkeypatch.setattr(data, "DASHBOARD_LIVE_ALPACA", False)
    _clear_dashboard_caches()

    summary = data.compute_account_summary()

    assert summary["equity"] == 101500
    assert summary["cash"] == 24000
    assert summary["invested"] == 77500
    assert summary["source"] == "alpaca_paper_equity.csv"
    assert summary["change_abs_today"] == 1500
    assert summary["change_pct_today"] == 1.5


def test_dashboard_trade_gate_state_requires_medium_risk_review():
    gate_state = data.signal_trade_gate_state(
        {
            "paper_ready": "True",
            "gates_all_pass": "True",
            "medium_risk_review_pass": "False",
        }
    )

    assert gate_state["paper_ready"] is True
    assert gate_state["gates_all_pass"] is True
    assert gate_state["medium_risk_review_pass"] is False
    assert gate_state["trade_ready"] is False
    assert gate_state["block_reasons"] == ["medium_risk_review_pass=false"]


def test_dashboard_trade_gate_state_ready_only_when_all_submit_gates_pass():
    gate_state = data.signal_trade_gate_state(
        {
            "paper_ready": True,
            "gates_all_pass": True,
            "medium_risk_review_pass": True,
        }
    )

    assert gate_state["trade_ready"] is True
    assert gate_state["block_reasons"] == []


def test_broker_truth_loader_returns_payload_and_issue_table(tmp_path, monkeypatch):
    truth_json = tmp_path / "broker_truth.json"
    truth_csv = tmp_path / "broker_truth.csv"
    truth_json.write_text(
        json.dumps(
            {
                "status": "warning",
                "score": 85.0,
                "summary": {"fail_count": 0, "warning_count": 2},
                "global_issues": [{"severity": "warning", "issue": "paper_log_stale"}],
            }
        ),
        encoding="utf-8",
    )
    truth_csv.write_text(
        "\n".join(
            [
                "ticker,issue_severity,target_weight,broker_weight,broker_qty,open_sell_qty,trailing_stop_qty,issues",
                "QQQ,warning,0.6,0.0,0,0,0,target_position_missing",
                "CAT,pass,0.03,0.03,3,3,3,",
                "MU,fail,0.2,0.0,0,0,0,required_trailing_stop_missing",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(data, "BROKER_TRUTH_JSON", truth_json)
    monkeypatch.setattr(data, "BROKER_TRUTH_CSV", truth_csv)
    _clear_dashboard_caches()

    truth = data.load_broker_truth()
    issue_table = data.broker_truth_issue_table(truth["table"])

    assert truth["payload"]["status"] == "warning"
    assert data.broker_truth_chip_status(truth["payload"]) == "warn"
    assert list(issue_table["ticker"]) == ["MU", "QQQ"]
    assert "CAT" not in set(issue_table["ticker"])


def test_file_status_tracks_broker_truth(tmp_path, monkeypatch):
    truth_json = tmp_path / "broker_truth.json"
    truth_json.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(data, "BROKER_TRUTH_JSON", truth_json)
    _clear_dashboard_caches()

    table = data.file_status_table()

    assert "Broker truth" in set(table["File"])


def test_file_status_tracks_etf_data_health(tmp_path, monkeypatch):
    etf_health = tmp_path / "etf_data_health.json"
    etf_health.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(data, "ETF_DATA_HEALTH", etf_health)
    _clear_dashboard_caches()

    table = data.file_status_table()

    assert "ETF data health" in set(table["File"])


def test_file_status_tracks_factor_data_health(tmp_path, monkeypatch):
    factor_health = tmp_path / "factor_data_health.json"
    factor_health.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(data, "FACTOR_DATA_HEALTH", factor_health)
    _clear_dashboard_caches()

    table = data.file_status_table()

    assert "Factor data health" in set(table["File"])


def test_quant_audit_loader_keeps_json_and_candidate_table_together(tmp_path, monkeypatch):
    audit_json = tmp_path / "quant_performance_audit.json"
    audit_csv = tmp_path / "quant_shadow_experiments.csv"
    audit_json.write_text(json.dumps({"reference_audit_status": "blocked"}), encoding="utf-8")
    audit_csv.write_text("name,gate_pass\nbaseline_frozen_active,true\n", encoding="utf-8")
    monkeypatch.setattr(data, "QUANT_AUDIT_JSON", audit_json)
    monkeypatch.setattr(data, "QUANT_AUDIT_CSV", audit_csv)
    _clear_dashboard_caches()

    loaded = data.load_quant_performance_audit()

    assert loaded["payload"]["reference_audit_status"] == "blocked"
    assert list(loaded["experiments"]["name"]) == ["baseline_frozen_active"]


def test_action_checklist_prioritizes_broker_truth_fail(monkeypatch):
    monkeypatch.setattr(
        data,
        "load_broker_truth",
        lambda: {
            "payload": {
                "status": "fail",
                "score": 70,
                "summary": {"fail_count": 1, "warning_count": 0},
                "global_issues": [{"severity": "fail", "issue": "broker_status_missing"}],
            },
            "table": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        data,
        "load_current_signal",
        lambda: {"paper_ready": True, "gates_all_pass": True, "medium_risk_review_pass": True},
    )
    monkeypatch.setattr(
        data,
        "load_latest_daily_run",
        lambda: {"steps_total": 2, "steps_ok": 2, "steps_failed": 0, "results": []},
    )
    monkeypatch.setattr(data, "load_execution_scorecard", lambda: {"status": "pass", "checks": []})
    monkeypatch.setattr(data, "load_etf_data_health", lambda: {"ok": True, "results": []})
    monkeypatch.setattr(data, "load_factor_data_health", lambda: {"trade_ready": True, "reasons": []})
    monkeypatch.setattr(
        data,
        "file_status_table",
        lambda: pd.DataFrame({"File": ["Broker truth"], "Status": ["green fresh"]}),
    )
    monkeypatch.setattr(
        data,
        "load_workflow_heartbeats",
        lambda: pd.DataFrame({"label": ["Daily paper"], "status": ["success"]}),
    )

    checklist = data.build_action_checklist()

    assert checklist.iloc[0]["severity"] == "fail"
    assert checklist.iloc[0]["area"] == "Broker truth"
    assert "broker_status_missing" in checklist.iloc[0]["why"]


def test_action_checklist_flags_failed_etf_health(monkeypatch):
    monkeypatch.setattr(
        data,
        "load_broker_truth",
        lambda: {
            "payload": {"status": "pass", "summary": {"fail_count": 0, "warning_count": 0}},
            "table": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        data,
        "load_current_signal",
        lambda: {"paper_ready": True, "gates_all_pass": True, "medium_risk_review_pass": True},
    )
    monkeypatch.setattr(
        data,
        "load_latest_daily_run",
        lambda: {"steps_total": 2, "steps_ok": 2, "steps_failed": 0, "results": []},
    )
    monkeypatch.setattr(data, "load_execution_scorecard", lambda: {"status": "pass", "checks": []})
    monkeypatch.setattr(
        data,
        "load_etf_data_health",
        lambda: {
            "ok": False,
            "results": [{"symbol": "QQQ", "ok": False, "issues": ["stale_10_bdays"]}],
        },
    )
    monkeypatch.setattr(data, "load_factor_data_health", lambda: {"trade_ready": True, "reasons": []})
    monkeypatch.setattr(
        data,
        "file_status_table",
        lambda: pd.DataFrame({"File": ["Today's signal"], "Status": ["green fresh"]}),
    )
    monkeypatch.setattr(
        data,
        "load_workflow_heartbeats",
        lambda: pd.DataFrame({"label": ["Daily paper"], "status": ["success"]}),
    )

    checklist = data.build_action_checklist()

    assert checklist.iloc[0]["severity"] == "fail"
    assert checklist.iloc[0]["area"] == "ETF data"
    assert "QQQ:stale_10_bdays" in checklist.iloc[0]["why"]


def test_action_checklist_flags_failed_factor_data_health(monkeypatch):
    monkeypatch.setattr(
        data,
        "load_broker_truth",
        lambda: {
            "payload": {"status": "pass", "summary": {"fail_count": 0, "warning_count": 0}},
            "table": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        data,
        "load_current_signal",
        lambda: {"paper_ready": True, "gates_all_pass": True, "medium_risk_review_pass": True},
    )
    monkeypatch.setattr(
        data,
        "load_latest_daily_run",
        lambda: {"steps_total": 2, "steps_ok": 2, "steps_failed": 0, "results": []},
    )
    monkeypatch.setattr(data, "load_execution_scorecard", lambda: {"status": "pass", "checks": []})
    monkeypatch.setattr(data, "load_etf_data_health", lambda: {"ok": True, "results": []})
    monkeypatch.setattr(
        data,
        "load_factor_data_health",
        lambda: {"trade_ready": False, "reasons": ["factor_data_missing_required_columns"]},
    )
    monkeypatch.setattr(
        data,
        "file_status_table",
        lambda: pd.DataFrame({"File": ["Today's signal"], "Status": ["green fresh"]}),
    )
    monkeypatch.setattr(
        data,
        "load_workflow_heartbeats",
        lambda: pd.DataFrame({"label": ["Daily paper"], "status": ["success"]}),
    )

    checklist = data.build_action_checklist()

    assert checklist.iloc[0]["severity"] == "fail"
    assert checklist.iloc[0]["area"] == "Factor data"
    assert "factor_data_missing_required_columns" in checklist.iloc[0]["why"]


def test_action_checklist_empty_when_core_system_clear(monkeypatch):
    monkeypatch.setattr(
        data,
        "load_broker_truth",
        lambda: {
            "payload": {"status": "pass", "summary": {"fail_count": 0, "warning_count": 0}},
            "table": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        data,
        "load_current_signal",
        lambda: {"paper_ready": True, "gates_all_pass": True, "medium_risk_review_pass": True},
    )
    monkeypatch.setattr(
        data,
        "load_latest_daily_run",
        lambda: {"steps_total": 2, "steps_ok": 2, "steps_failed": 0, "results": []},
    )
    monkeypatch.setattr(data, "load_execution_scorecard", lambda: {"status": "pass", "checks": []})
    monkeypatch.setattr(data, "load_etf_data_health", lambda: {"ok": True, "results": []})
    monkeypatch.setattr(data, "load_factor_data_health", lambda: {"trade_ready": True, "reasons": []})
    monkeypatch.setattr(
        data,
        "file_status_table",
        lambda: pd.DataFrame({"File": ["Today's signal"], "Status": ["green fresh"]}),
    )
    monkeypatch.setattr(
        data,
        "load_workflow_heartbeats",
        lambda: pd.DataFrame({"label": ["Daily paper"], "status": ["success"]}),
    )

    checklist = data.build_action_checklist()

    assert checklist.empty
