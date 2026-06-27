from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import execution_scorecard as esc


def test_build_execution_scorecard_scores_fill_quality_and_throttle(tmp_path):
    log_path = tmp_path / "alpaca_paper_log.csv"
    report_path = tmp_path / "alpaca_slippage_reversal_report.json"
    previous_path = tmp_path / "alpaca_execution_scorecard.json"

    pd.DataFrame(
        [
            {
                "submitted_at": "2026-06-04T14:00:00+00:00",
                "order_id": "buy-mu",
                "ticker": "MU",
                "side": "buy",
                "quantity": 100,
                "price": 100.0,
                "fill_status": "filled",
                "filled_qty": 100,
                "filled_avg_price": 100.05,
                "execution_risk_reason": "high_execution_risk_score_65.00_buy_scale_0.50",
                "execution_risk_buy_scale": 0.5,
                "execution_risk_quantity_before_scale": 200,
                "execution_risk_quantity_after_scale": 100,
            },
            {
                "submitted_at": "2026-06-04T14:00:00+00:00",
                "order_id": "sell-fcx",
                "ticker": "FCX",
                "side": "sell",
                "quantity": 50,
                "price": 50.0,
                "fill_status": "filled",
                "filled_qty": 50,
                "filled_avg_price": 49.98,
            },
            {
                "submitted_at": "2026-06-04T14:00:00+00:00",
                "order_id": "SKIPPED: spread",
                "ticker": "INTC",
                "side": "buy",
                "quantity": 10,
                "price": 30.0,
                "fill_status": "skipped",
                "filled_qty": 0,
            },
        ]
    ).to_csv(log_path, index=False)
    report_path.write_text(
        json.dumps(
            {
                "source": "alpaca_api",
                "summary": {
                    "orders_analyzed": 2,
                    "avg_slippage_bps": 4.0,
                    "slippage_bad_count": 1,
                    "adverse_15m_count": 1,
                    "adverse_60m_count": 0,
                },
                "segments": {
                    "limit_orders": {"avg_slippage_bps": 1.5},
                    "market_orders": {"avg_slippage_bps": 12.0},
                },
            }
        ),
        encoding="utf-8",
    )
    previous_path.write_text(
        json.dumps({"summary": {"avg_slippage_bps": 6.5}}),
        encoding="utf-8",
    )

    payload = esc.build_execution_scorecard(
        paper_log_path=log_path,
        slippage_report_path=report_path,
        previous_scorecard_path=previous_path,
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert payload["status"] == "pass"
    assert payload["score"] == 100.0
    assert payload["summary"]["accepted_orders"] == 2
    assert payload["summary"]["filled_orders"] == 2
    assert payload["summary"]["skipped_orders"] == 1
    assert payload["summary"]["fill_rate"] == 1.0
    assert payload["summary"]["skipped_rate"] == 0.3333
    assert payload["summary"]["bad_slippage_rate"] == 0.5
    assert payload["summary"]["limit_vs_market_delta_bps"] == -10.5
    assert payload["summary"]["slippage_delta_vs_prior_scorecard_bps"] == -2.5
    assert payload["throttle"]["throttled_buy_orders"] == 1
    assert payload["throttle"]["quantity_reduced"] == 100
    assert payload["throttle"]["notional_reduced"] == 10000.0
    assert "limit_orders_are_beating_market_orders" in payload["recommendations"]


def test_build_execution_scorecard_fails_bad_execution(tmp_path):
    log_path = tmp_path / "alpaca_paper_log.csv"
    report_path = tmp_path / "alpaca_slippage_reversal_report.json"

    pd.DataFrame(
        [
            {
                "submitted_at": "2026-06-04T14:00:00+00:00",
                "order_id": "open-a",
                "ticker": "AAPL",
                "side": "buy",
                "quantity": 10,
                "price": 100.0,
                "fill_status": "open",
                "filled_qty": 0,
            },
            {
                "submitted_at": "2026-06-04T14:00:00+00:00",
                "order_id": "SKIPPED: quote",
                "ticker": "MSFT",
                "side": "buy",
                "quantity": 10,
                "price": 100.0,
                "fill_status": "skipped",
                "filled_qty": 0,
            },
        ]
    ).to_csv(log_path, index=False)
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "orders_analyzed": 2,
                    "avg_slippage_bps": 25.0,
                    "slippage_bad_count": 2,
                    "adverse_15m_count": 2,
                    "adverse_60m_count": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    payload = esc.build_execution_scorecard(
        paper_log_path=log_path,
        slippage_report_path=report_path,
        previous_scorecard_path=tmp_path / "missing.json",
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert payload["status"] == "fail"
    failed = {check["name"] for check in payload["checks"] if check["status"] == "fail"}
    assert {"avg_slippage_bps", "bad_slippage_rate", "fill_rate", "adverse_15m_rate", "adverse_60m_rate"} <= failed
    assert any(item.startswith("review_failed_execution_checks:") for item in payload["recommendations"])


def test_bad_slippage_rate_uses_material_threshold_when_order_rows_exist(tmp_path):
    log_path = tmp_path / "alpaca_paper_log.csv"
    report_path = tmp_path / "alpaca_slippage_reversal_report.json"

    pd.DataFrame(
        [
            {
                "submitted_at": "2026-06-04T14:00:00+00:00",
                "order_id": f"order-{idx}",
                "ticker": "QQQ",
                "side": "buy",
                "fill_status": "filled",
                "filled_qty": 1,
            }
            for idx in range(4)
        ]
    ).to_csv(log_path, index=False)
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "orders_analyzed": 4,
                    "avg_slippage_bps": 3.0,
                    "slippage_bad_count": 3,
                    "adverse_15m_count": 1,
                    "adverse_60m_count": 1,
                },
                "orders": [
                    {"slippage_bps": -1.0},
                    {"slippage_bps": 0.5},
                    {"slippage_bps": 2.5},
                    {"slippage_bps": 6.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = esc.build_execution_scorecard(
        paper_log_path=log_path,
        slippage_report_path=report_path,
        previous_scorecard_path=tmp_path / "missing.json",
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert payload["status"] == "pass"
    assert payload["summary"]["bad_slippage_threshold_bps"] == 2.0
    assert payload["summary"]["bad_slippage_count"] == 2
    assert payload["summary"]["bad_slippage_rate"] == 0.5
    assert payload["summary"]["raw_bad_slippage_count"] == 3
    assert payload["summary"]["raw_bad_slippage_rate"] == 0.75
    assert payload["summary"]["minor_bad_slippage_count"] == 1


def test_write_execution_scorecard_writes_latest_and_dated_snapshot(tmp_path, monkeypatch):
    output_path = tmp_path / "signals" / "alpaca_execution_scorecard.json"
    log_dir = tmp_path / "logs"

    monkeypatch.setattr(
        esc,
        "build_execution_scorecard",
        lambda previous_scorecard_path, now: {
            "generated_at": now.isoformat(timespec="seconds"),
            "status": "collecting",
            "score": None,
            "summary": {},
        },
    )

    payload = esc.write_execution_scorecard(
        output_path=output_path,
        log_dir=log_dir,
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert payload["status"] == "collecting"
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "collecting"
    dated = log_dir / "alpaca_execution_scorecard_20260605.json"
    assert json.loads(dated.read_text(encoding="utf-8"))["status"] == "collecting"


def test_daily_workflow_publishes_execution_scorecard():
    workflow = Path(".github/workflows/daily_paper_trading.yml").read_text(encoding="utf-8")

    assert "signals/alpaca_execution_scorecard.json" in workflow
    assert "logs/alpaca_execution_scorecard_*.json" in workflow
