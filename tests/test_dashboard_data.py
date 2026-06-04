from __future__ import annotations

import json

import dashboard.data as data


def _clear_dashboard_caches() -> None:
    """Clear Streamlit caches when tests monkeypatch dashboard file paths."""
    for loader in (
        data.refresh_live_alpaca_snapshot,
        data.load_alpaca_status,
        data.load_alpaca_equity_history,
        data.load_slippage_reversal_report,
        data.load_paper_shadow_compare,
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
