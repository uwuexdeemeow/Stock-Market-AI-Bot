from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import broker_truth


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_broker_truth_flags_failed_order_and_missing_stop(tmp_path):
    signal_path = tmp_path / "signal.csv"
    plan_path = tmp_path / "orders.csv"
    log_path = tmp_path / "alpaca_paper_log.csv"
    status_path = tmp_path / "alpaca_daily_status.json"

    pd.DataFrame(
        [
            {
                "target_qqq_weight": 0.60,
                "overlay_weights_json": json.dumps({"MU": 0.20}),
                "predicted_at": "2026-06-05T13:35:00+00:00",
            }
        ]
    ).to_csv(signal_path, index=False)
    pd.DataFrame(
        [
            {"ticker": "MU", "side": "buy", "quantity": 10, "current_qty": 0, "target_qty": 10, "target_weight": 0.20},
            {"ticker": "QQQ", "side": "buy", "quantity": 6, "current_qty": 0, "target_qty": 6, "target_weight": 0.60},
        ]
    ).to_csv(plan_path, index=False)
    pd.DataFrame(
        [
            {
                "submitted_at": "2026-06-05T14:00:00+00:00",
                "order_id": "ERROR: rejected",
                "ticker": "MU",
                "side": "buy",
                "quantity": 10,
                "fill_status": "submission_failed",
            },
            {
                "submitted_at": "2026-06-05T14:00:00+00:00",
                "order_id": "qqq-order",
                "ticker": "QQQ",
                "side": "buy",
                "quantity": 6,
                "fill_status": "filled",
                "filled_qty": 6,
            },
        ]
    ).to_csv(log_path, index=False)
    _write_json(
        status_path,
        {
            "generated_at": "2026-06-05T14:05:00+00:00",
            "account_equity": 10000,
            "account_cash": 2000,
            "positions": {"QQQ": 6, "MU": 10},
            "position_values": {"QQQ": 6000, "MU": 2000},
        },
    )

    payload = broker_truth.build_broker_truth(
        signal_path=signal_path,
        plan_path=plan_path,
        log_path=log_path,
        status_path=status_path,
        open_orders=[
            {"ticker": "QQQ", "side": "sell", "type": "trailing_stop", "quantity": 6, "order_id": "stop-qqq"}
        ],
        open_orders_meta={"available": True, "count": 1, "error": "", "source": "test"},
        include_live_open_orders=False,
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert payload["status"] == "fail"
    rows = {row["ticker"]: row for row in payload["rows"]}
    assert rows["MU"]["issue_severity"] == "fail"
    assert "latest_logged_order_failed" in rows["MU"]["issues"]
    assert "required_trailing_stop_missing" in rows["MU"]["issues"]
    assert rows["QQQ"]["issue_severity"] == "pass"


def test_write_broker_truth_writes_latest_and_dated_outputs(tmp_path):
    signal_path = tmp_path / "signal.csv"
    plan_path = tmp_path / "orders.csv"
    log_path = tmp_path / "alpaca_paper_log.csv"
    status_path = tmp_path / "alpaca_daily_status.json"
    output_csv = tmp_path / "signals" / "broker_truth.csv"
    output_json = tmp_path / "signals" / "broker_truth.json"
    log_dir = tmp_path / "logs"

    pd.DataFrame([{"target_qqq_weight": 0.0, "overlay_weights_json": "{}"}]).to_csv(signal_path, index=False)
    pd.DataFrame(columns=["ticker", "side", "quantity"]).to_csv(plan_path, index=False)
    pd.DataFrame(columns=["submitted_at", "ticker", "side", "quantity"]).to_csv(log_path, index=False)
    _write_json(status_path, {"account_equity": 10000, "account_cash": 10000, "positions": {}, "position_values": {}})

    payload = broker_truth.write_broker_truth(
        signal_path=signal_path,
        plan_path=plan_path,
        log_path=log_path,
        status_path=status_path,
        output_csv=output_csv,
        output_json=output_json,
        log_dir=log_dir,
        open_orders=[],
        open_orders_meta={"available": True, "count": 0, "error": "", "source": "test"},
        include_live_open_orders=False,
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert payload["status"] == "pass"
    assert output_csv.exists()
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "pass"
    assert json.loads((log_dir / "broker_truth_20260605.json").read_text(encoding="utf-8"))["status"] == "pass"


def test_live_positions_override_saved_status_snapshot(tmp_path):
    signal_path = tmp_path / "signal.csv"
    plan_path = tmp_path / "orders.csv"
    log_path = tmp_path / "alpaca_paper_log.csv"
    status_path = tmp_path / "alpaca_daily_status.json"

    pd.DataFrame([{"target_qqq_weight": 0.60, "overlay_weights_json": "{}"}]).to_csv(signal_path, index=False)
    pd.DataFrame(columns=["ticker", "side", "quantity"]).to_csv(plan_path, index=False)
    pd.DataFrame(columns=["submitted_at", "ticker", "side", "quantity"]).to_csv(log_path, index=False)
    _write_json(
        status_path,
        {
            "generated_at": "2026-06-04T14:05:00+00:00",
            "account_equity": 10000,
            "positions": {"QQQ": 0},
            "position_values": {"QQQ": 0},
        },
    )

    payload = broker_truth.build_broker_truth(
        signal_path=signal_path,
        plan_path=plan_path,
        log_path=log_path,
        status_path=status_path,
        live_positions={"QQQ": {"quantity": 6, "market_value": 6000, "weight": 0.60}},
        live_positions_meta={
            "available": True,
            "attempted": True,
            "source": "alpaca_api",
            "equity": 10000,
            "cash": 4000,
            "position_count": 1,
        },
        open_orders=[
            {"ticker": "QQQ", "side": "sell", "type": "trailing_stop", "quantity": 6, "order_id": "stop-qqq"}
        ],
        open_orders_meta={"available": True, "count": 1, "error": "", "source": "test"},
        include_live_open_orders=False,
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    rows = {row["ticker"]: row for row in payload["rows"]}
    assert rows["QQQ"]["broker_qty"] == 6
    assert rows["QQQ"]["issue_severity"] == "pass"
    assert payload["inputs"]["broker_status"]["source"] == "alpaca_api"


def test_daily_workflow_publishes_broker_truth():
    workflow = Path(".github/workflows/daily_paper_trading.yml").read_text(encoding="utf-8")

    assert "signals/broker_truth.csv" in workflow
    assert "signals/broker_truth.json" in workflow
    assert "logs/broker_truth_*.json" in workflow
