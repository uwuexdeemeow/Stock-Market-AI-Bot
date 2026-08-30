from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

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
        manage_alignment_lifecycle=True,
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert payload["status"] == "pass"
    assert output_csv.exists()
    recovery_path = output_csv.parent / "alignment_recovery_plan.csv"
    incident_path = output_csv.parent / "alignment_incident_ledger.csv"
    assert recovery_path.exists()
    assert incident_path.exists()
    assert list(pd.read_csv(recovery_path).columns) == broker_truth.RECOVERY_COLUMNS
    assert pd.read_csv(recovery_path).empty
    assert list(pd.read_csv(incident_path).columns) == broker_truth.INCIDENT_COLUMNS
    assert pd.read_csv(incident_path).empty
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


def test_live_empty_account_overrides_stale_saved_positions(tmp_path):
    signal_path = tmp_path / "signal.csv"
    plan_path = tmp_path / "orders.csv"
    log_path = tmp_path / "alpaca_paper_log.csv"
    status_path = tmp_path / "alpaca_daily_status.json"
    pd.DataFrame([{"target_qqq_weight": 0.0, "overlay_weights_json": "{}"}]).to_csv(
        signal_path, index=False
    )
    pd.DataFrame(columns=["ticker", "side", "quantity"]).to_csv(plan_path, index=False)
    pd.DataFrame(columns=["submitted_at", "ticker", "side", "quantity"]).to_csv(log_path, index=False)
    _write_json(
        status_path,
        {"account_equity": 10000, "positions": {"QQQ": 10}, "position_values": {"QQQ": 6000}},
    )

    payload = broker_truth.build_broker_truth(
        signal_path=signal_path,
        plan_path=plan_path,
        log_path=log_path,
        status_path=status_path,
        live_positions={},
        live_positions_meta={
            "available": True,
            "attempted": True,
            "source": "alpaca_api",
            "equity": 10000,
            "cash": 10000,
            "position_count": 0,
        },
        open_orders=[],
        open_orders_meta={"available": True, "count": 0, "error": "", "source": "test"},
        include_live_open_orders=False,
    )

    assert payload["inputs"]["broker_status"]["source"] == "alpaca_api"
    assert payload["summary"]["alignment"]["status"] == "pass"
    assert all(row["broker_qty"] == 0 for row in payload["rows"])


def test_empty_signal_targets_do_not_create_false_all_cash_weight_gap(tmp_path):
    """A malformed signal must not make every real holding look 100% wrong."""
    signal_path = tmp_path / "signal.csv"
    plan_path = tmp_path / "orders.csv"
    log_path = tmp_path / "alpaca_paper_log.csv"
    status_path = tmp_path / "alpaca_daily_status.json"

    pd.DataFrame([{"predicted_at": "2026-06-05T13:35:00+00:00"}]).to_csv(signal_path, index=False)
    pd.DataFrame(columns=["ticker", "side", "quantity"]).to_csv(plan_path, index=False)
    pd.DataFrame(columns=["submitted_at", "ticker", "side", "quantity"]).to_csv(log_path, index=False)
    _write_json(
        status_path,
        {
            "account_equity": 10000,
            "positions": {"QQQ": 10},
            "position_values": {"QQQ": 6000},
        },
    )

    payload = broker_truth.build_broker_truth(
        signal_path=signal_path,
        plan_path=plan_path,
        log_path=log_path,
        status_path=status_path,
        open_orders=[],
        open_orders_meta={"available": True, "count": 0, "error": "", "source": "test"},
        include_live_open_orders=False,
        now=datetime(2026, 6, 5, 14, tzinfo=timezone.utc),
    )

    assert payload["summary"]["target_comparison_enabled"] is False
    assert payload["summary"]["maximum_target_weight_gap"] is None
    assert any(item["issue"] == "signal_has_no_target_weights" for item in payload["global_issues"])
    assert "broker_weight_gap" not in payload["rows"][0]["issues"]


def _alignment(rows, *, open_orders_available=True, timed_out=False):
    """Build a canonical live-alignment result with small test inputs."""
    return broker_truth._alignment_result(
        rows=rows,
        target_comparison_enabled=True,
        status_meta={"source": "alpaca_api", "equity": 10000},
        live_positions_meta={"available": True, "source": "alpaca_api", "error": ""},
        open_orders_meta={"available": open_orders_available, "error": "test_unavailable"},
        pending_timed_out=timed_out,
    )


def test_canonical_alignment_passes_and_fails_settled_weight_limits():
    aligned = _alignment(
        [
            {"target_weight": 0.60, "broker_weight": 0.595, "open_rebalance_order_count": 0},
            {"target_weight": 0.40, "broker_weight": 0.399, "open_rebalance_order_count": 0},
        ]
    )
    assert aligned["status"] == "pass"
    assert aligned["maximum_target_weight_gap"] == 0.005

    ticker_fail = _alignment(
        [{"target_weight": 0.60, "broker_weight": 0.50, "open_rebalance_order_count": 0}]
    )
    assert ticker_fail["status"] == "fail"
    assert "max_weight_gap" in ticker_fail["reason"]

    gross_fail = _alignment(
        [
            {"target_weight": 0.50, "broker_weight": 0.47, "open_rebalance_order_count": 0},
            {"target_weight": 0.50, "broker_weight": 0.47, "open_rebalance_order_count": 0},
        ]
    )
    assert gross_fail["status"] == "fail"
    assert "gross_exposure_gap" in gross_fail["reason"]


def test_protective_stops_do_not_hide_drift_or_count_as_rebalance_orders():
    summary = broker_truth.summarize_open_orders(
        [
            {"ticker": "QQQ", "side": "sell", "type": "trailing_stop", "quantity": 10},
            {"ticker": "MU", "side": "sell", "type": "stop", "quantity": 4},
        ]
    )
    assert summary["QQQ"]["open_sell_qty"] == 10
    assert summary["QQQ"]["open_rebalance_sell_qty"] == 0
    assert summary["QQQ"]["open_rebalance_order_count"] == 0
    assert summary["MU"]["open_rebalance_order_count"] == 0

    drift = _alignment(
        [{"target_weight": 0.60, "broker_weight": 0.50, "open_rebalance_order_count": 0}]
    )
    assert drift["status"] == "fail"


def test_ordinary_open_orders_are_pending_then_timeout():
    rows = [
        {
            "target_weight": 0.60,
            "broker_weight": 0.50,
            "open_buy_qty": 2,
            "open_rebalance_sell_qty": 0,
            "open_rebalance_order_count": 1,
        }
    ]
    pending = _alignment(rows)
    assert pending["status"] == "pending"
    assert pending["active_rebalance_order_count"] == 1

    timed_out = _alignment(rows, timed_out=True)
    assert timed_out["status"] == "fail"
    assert timed_out["reason"] == "alignment_pending_timeout"


def test_canonical_alignment_collects_when_live_evidence_is_missing():
    rows = [{"target_weight": 1.0, "broker_weight": 1.0, "open_rebalance_order_count": 0}]
    missing_orders = _alignment(rows, open_orders_available=False)
    assert missing_orders["status"] == "collecting"
    assert "live_open_orders_unavailable" in missing_orders["reason"]

    missing_positions = broker_truth._alignment_result(
        rows=rows,
        target_comparison_enabled=True,
        status_meta={"source": "saved_snapshot", "equity": 10000},
        live_positions_meta={"available": False, "error": "no_credentials"},
        open_orders_meta={"available": True, "error": ""},
    )
    assert missing_positions["status"] == "collecting"
    assert "live_positions_unavailable" in missing_positions["reason"]


def test_required_alignment_wait_polls_until_pass(monkeypatch):
    statuses = iter(["pending", "pass"])
    calls = []

    def fake_write(**kwargs):
        calls.append(kwargs)
        status = next(statuses)
        return {"summary": {"alignment": {"status": status}}}

    clock = iter([0.0, 0.0, 1.0])
    payload = broker_truth.wait_for_required_alignment(
        wait_seconds=10,
        poll_seconds=1,
        write_fn=fake_write,
        sleep_fn=lambda _seconds: None,
        monotonic_fn=lambda: next(clock),
    )

    assert payload["summary"]["alignment"]["status"] == "pass"
    assert len(calls) == 2


def test_required_alignment_wait_marks_pending_timeout():
    calls = []

    def fake_write(**kwargs):
        calls.append(kwargs)
        status = "fail" if kwargs.get("alignment_pending_timed_out") else "pending"
        reason = "alignment_pending_timeout" if status == "fail" else "exposure_changing_orders_open"
        return {"summary": {"alignment": {"status": status, "reason": reason}}}

    clock = iter([0.0, 0.0])
    payload = broker_truth.wait_for_required_alignment(
        wait_seconds=0,
        poll_seconds=1,
        write_fn=fake_write,
        sleep_fn=lambda _seconds: None,
        monotonic_fn=lambda: next(clock),
    )

    assert payload["summary"]["alignment"]["status"] == "fail"
    assert payload["summary"]["alignment"]["reason"] == "alignment_pending_timeout"
    assert calls[-1]["alignment_pending_timed_out"] is True


def test_require_alignment_cli_exits_nonzero_without_live_proof(monkeypatch):
    payload = {
        "status": "warning",
        "score": 90,
        "summary": {
            "alignment": {
                "status": "collecting",
                "reason": "live_positions_unavailable:no_credentials",
            }
        },
        "rows": [],
        "global_issues": [],
    }
    monkeypatch.setattr(broker_truth, "wait_for_required_alignment", lambda **_kwargs: payload)
    monkeypatch.setattr(sys, "argv", ["broker_truth.py", "--require-alignment"])

    with pytest.raises(SystemExit) as exc:
        broker_truth.main()

    assert exc.value.code == 1


def test_safe_alignment_recovery_plan_is_review_only():
    payload = {
        "generated_at": "2026-08-30T12:00:00+00:00",
        "summary": {
            "account_equity": 10000,
            "alignment": {
                "status": "fail",
                "reason": "max_weight_gap_0.100000_above_0.020000",
                "active_rebalance_order_count": 0,
            },
        },
        "inputs": {"signal": {"as_of": "2026-08-30T11:00:00+00:00"}},
        "rows": [
            {
                "ticker": "QQQ",
                "target_weight": 0.60,
                "broker_weight": 0.50,
                "broker_qty": 50,
                "broker_value": 5000,
                "planned_target_qty": 60,
                "planned_target_weight": 0.60,
            },
            {
                "ticker": "OLD",
                "target_weight": 0.0,
                "broker_weight": 0.10,
                "broker_qty": 10,
                "broker_value": 1000,
            },
        ],
    }

    rows = broker_truth.build_alignment_recovery_plan(payload)
    by_ticker = {row["ticker"]: row for row in rows}

    assert by_ticker["QQQ"]["action"] == "buy"
    assert by_ticker["QQQ"]["suggested_quantity"] == 10
    assert by_ticker["QQQ"]["quantity_basis"] == "current_order_plan"
    assert by_ticker["OLD"]["action"] == "sell"
    assert by_ticker["OLD"]["suggested_quantity"] == 10
    assert all(row["review_status"] == "manual_review_required_not_submitted" for row in rows)


def test_recovery_plan_never_duplicates_active_orders_or_non_alignment_failures():
    base = {
        "summary": {
            "account_equity": 10000,
            "alignment": {
                "status": "fail",
                "reason": "alignment_pending_timeout",
                "active_rebalance_order_count": 1,
            },
        },
        "rows": [{"ticker": "QQQ", "target_weight": 0.60, "broker_weight": 0.50}],
    }
    assert broker_truth.build_alignment_recovery_plan(base) == []

    base["summary"]["alignment"] = {
        "status": "fail",
        "reason": "required_trailing_stop_missing",
        "active_rebalance_order_count": 0,
    }
    assert broker_truth.build_alignment_recovery_plan(base) == []


def _incident_payload(*, status, generated_at, max_gap, gross_gap, reason, signal_as_of="signal-1"):
    return {
        "generated_at": generated_at,
        "summary": {
            "alignment": {
                "status": status,
                "reason": reason,
                "maximum_target_weight_gap": max_gap,
                "gross_exposure_gap": gross_gap,
                "active_rebalance_order_count": 0,
            }
        },
        "inputs": {"signal": {"as_of": signal_as_of}},
    }


def test_alignment_incident_ledger_is_idempotent_and_records_resolution(tmp_path):
    ledger_path = tmp_path / "alignment_incident_ledger.csv"
    first = _incident_payload(
        status="fail",
        generated_at="2026-08-30T12:00:00+00:00",
        max_gap=0.08,
        gross_gap=0.06,
        reason="max_weight_gap_0.080000_above_0.020000",
    )
    first_summary = broker_truth.update_alignment_incident_ledger(
        first, ledger_path=ledger_path, recovery_plan_rows=2
    )
    incident_id = first_summary["current_incident_id"]

    worse = _incident_payload(
        status="fail",
        generated_at="2026-08-30T12:01:00+00:00",
        max_gap=0.10,
        gross_gap=0.07,
        reason="max_weight_gap_0.100000_above_0.020000",
    )
    second_summary = broker_truth.update_alignment_incident_ledger(
        worse, ledger_path=ledger_path, recovery_plan_rows=2
    )
    open_ledger = pd.read_csv(ledger_path)

    assert second_summary["current_incident_id"] == incident_id
    assert len(open_ledger) == 1
    assert open_ledger.iloc[0]["status"] == "open"
    assert open_ledger.iloc[0]["maximum_observed_weight_gap"] == 0.10
    assert bool(open_ledger.iloc[0]["orders_submitted"]) is False

    passed = _incident_payload(
        status="pass",
        generated_at="2026-08-30T12:05:00+00:00",
        max_gap=0.005,
        gross_gap=0.01,
        reason="ok",
    )
    resolved_summary = broker_truth.update_alignment_incident_ledger(
        passed, ledger_path=ledger_path, recovery_plan_rows=0
    )
    resolved = pd.read_csv(ledger_path).iloc[0]

    assert resolved_summary["open_incidents"] == 0
    assert resolved["status"] == "resolved"
    assert resolved["resolution"] == "alignment_passed"
    assert resolved["duration_seconds"] == 300
    assert bool(resolved["orders_submitted"]) is False


def test_report_only_write_does_not_open_or_clear_post_trade_incidents(tmp_path):
    signal_path = tmp_path / "signal.csv"
    plan_path = tmp_path / "orders.csv"
    log_path = tmp_path / "alpaca_paper_log.csv"
    status_path = tmp_path / "status.json"
    output_csv = tmp_path / "signals" / "broker_truth.csv"
    pd.DataFrame([{"target_qqq_weight": 0.60, "overlay_weights_json": "{}"}]).to_csv(
        signal_path, index=False
    )
    pd.DataFrame(columns=["ticker", "side", "quantity"]).to_csv(plan_path, index=False)
    pd.DataFrame(columns=["submitted_at", "ticker", "side", "quantity"]).to_csv(log_path, index=False)
    _write_json(
        status_path,
        {"account_equity": 10000, "positions": {"QQQ": 5}, "position_values": {"QQQ": 5000}},
    )

    broker_truth.write_broker_truth(
        signal_path=signal_path,
        plan_path=plan_path,
        log_path=log_path,
        status_path=status_path,
        output_csv=output_csv,
        output_json=tmp_path / "signals" / "broker_truth.json",
        log_dir=tmp_path / "logs",
        live_positions={"QQQ": {"quantity": 5, "market_value": 5000, "weight": 0.50}},
        live_positions_meta={
            "available": True,
            "source": "alpaca_api",
            "equity": 10000,
            "cash": 5000,
        },
        open_orders=[],
        open_orders_meta={"available": True, "count": 0, "error": ""},
        include_live_open_orders=False,
        manage_alignment_lifecycle=False,
    )

    assert not (output_csv.parent / "alignment_recovery_plan.csv").exists()
    assert not (output_csv.parent / "alignment_incident_ledger.csv").exists()


def test_missing_evidence_does_not_close_open_alignment_incident(tmp_path):
    ledger_path = tmp_path / "alignment_incident_ledger.csv"
    failed = _incident_payload(
        status="fail",
        generated_at="2026-08-30T12:00:00+00:00",
        max_gap=0.08,
        gross_gap=0.06,
        reason="max_weight_gap_0.080000_above_0.020000",
    )
    broker_truth.update_alignment_incident_ledger(
        failed, ledger_path=ledger_path, recovery_plan_rows=1
    )
    collecting = _incident_payload(
        status="collecting",
        generated_at="2026-08-30T12:05:00+00:00",
        max_gap=None,
        gross_gap=None,
        reason="live_positions_unavailable",
    )
    summary = broker_truth.update_alignment_incident_ledger(
        collecting, ledger_path=ledger_path, recovery_plan_rows=0
    )

    assert summary["open_incidents"] == 1
    assert pd.read_csv(ledger_path).iloc[0]["status"] == "open"

def test_daily_workflow_publishes_broker_truth():
    workflow = Path(".github/workflows/daily_paper_trading.yml").read_text(encoding="utf-8")

    assert "signals/broker_truth.csv" in workflow
    assert "signals/broker_truth.json" in workflow
    assert "signals/alignment_recovery_plan.csv" in workflow
    assert "signals/alignment_incident_ledger.csv" in workflow
    assert "logs/broker_truth_*.json" in workflow
