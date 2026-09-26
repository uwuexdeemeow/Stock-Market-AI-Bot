"""Monitors must accept the hold days of the tested 20-day calendar (audit fix H1).

PLAIN ENGLISH: the paper account now trades once per 20-day period and then
holds.  While it holds, prices move and the weights drift away from today's
targets.  broker_truth.py's alignment gate is a CRITICAL daily step, so if it
treated that drift as a fault every hold day would turn the daily run red.
These tests pin:

  * hold day   = the period was rebalanced by an EARLIER run -> drift allowed;
  * rebalance day (rebalanced by THIS run) -> weights still enforced;
  * retired stops are no longer "required";
  * the paper-evidence gate accepts a hold-day pass.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import broker_truth
import paper_validation_epoch


def _calendar_signal(path, period="2026-09-16"):
    pd.DataFrame([{
        "target_qqq_weight": 0.60,
        "overlay_weights_json": json.dumps({"MU": 0.20}),
        "rebalance_policy": "tested_calendar_v1",
        "last_scheduled_rebalance_date": period,
    }]).to_csv(path, index=False)


def _period_file(path, period="2026-09-16", run_id="paper-earlier-run"):
    path.write_text(json.dumps({"period": period, "run_id": run_id}), encoding="utf-8")


# ── Hold-day detection ─────────────────────────────────────────────────────

def test_period_rebalanced_by_an_earlier_run_is_a_hold_day(tmp_path):
    _calendar_signal(tmp_path / "s.csv")
    _period_file(tmp_path / "p.json", run_id="paper-earlier-run")
    state = broker_truth.load_rebalance_calendar_state(tmp_path / "s.csv", tmp_path / "p.json", run_id="paper-today")
    assert state["hold_day"] is True


def test_period_rebalanced_by_this_run_is_checked(tmp_path):
    _calendar_signal(tmp_path / "s.csv")
    _period_file(tmp_path / "p.json", run_id="paper-today")
    state = broker_truth.load_rebalance_calendar_state(tmp_path / "s.csv", tmp_path / "p.json", run_id="paper-today")
    assert state == {**state, "hold_day": False, "reason": "rebalanced_in_this_run"}


def test_older_or_missing_period_is_checked(tmp_path):
    _calendar_signal(tmp_path / "s.csv", period="2026-10-14")
    _period_file(tmp_path / "p.json", period="2026-09-16")
    older = broker_truth.load_rebalance_calendar_state(tmp_path / "s.csv", tmp_path / "p.json", run_id="x")
    missing = broker_truth.load_rebalance_calendar_state(tmp_path / "s.csv", tmp_path / "none.json", run_id="x")
    assert older["hold_day"] is False and older["reason"] == "rebalance_due_or_not_completed"
    assert missing["hold_day"] is False and missing["reason"] == "no_rebalance_recorded_yet"


def test_signal_without_calendar_is_checked(tmp_path):
    pd.DataFrame([{"target_qqq_weight": 0.6}]).to_csv(tmp_path / "s.csv", index=False)
    _period_file(tmp_path / "p.json")
    state = broker_truth.load_rebalance_calendar_state(tmp_path / "s.csv", tmp_path / "p.json", run_id="x")
    assert state["hold_day"] is False


# ── The alignment gate, end to end ─────────────────────────────────────────

def _drifted_truth(tmp_path, monkeypatch, *, period_run_id):
    """Account bought MU at 20% on the rebalance day; it has since grown to 30%."""
    monkeypatch.setattr(broker_truth, "CORE_TRAILING_STOP_ENABLED", False)
    monkeypatch.setattr(broker_truth, "OVERLAY_TRAILING_STOP_ENABLED", False)
    monkeypatch.setenv("STOCKBOT_RUN_ID", "paper-today")
    _calendar_signal(tmp_path / "signal.csv")
    _period_file(tmp_path / "period.json", run_id=period_run_id)
    pd.DataFrame(columns=["ticker", "side", "quantity"]).to_csv(tmp_path / "orders.csv", index=False)
    pd.DataFrame(columns=["submitted_at", "ticker", "side", "quantity"]).to_csv(tmp_path / "log.csv", index=False)
    (tmp_path / "status.json").write_text(json.dumps({"account_equity": 10000}), encoding="utf-8")
    return broker_truth.build_broker_truth(
        signal_path=tmp_path / "signal.csv",
        plan_path=tmp_path / "orders.csv",
        log_path=tmp_path / "log.csv",
        status_path=tmp_path / "status.json",
        rebalance_period_path=tmp_path / "period.json",
        live_positions={
            "QQQ": {"quantity": 6, "market_value": 5500, "weight": 0.55},
            "MU": {"quantity": 10, "market_value": 3000, "weight": 0.30},
        },
        live_positions_meta={"available": True, "attempted": True, "source": "alpaca_api",
                             "equity": 10000, "cash": 1500, "position_count": 2},
        open_orders=[],
        open_orders_meta={"available": True, "count": 0, "error": "", "source": "test"},
        include_live_open_orders=False,
        now=datetime(2026, 9, 22, tzinfo=timezone.utc),
    )


def test_hold_day_drift_passes_alignment(tmp_path, monkeypatch):
    payload = _drifted_truth(tmp_path, monkeypatch, period_run_id="paper-earlier-run")
    alignment = payload["summary"]["alignment"]
    assert alignment["status"] == "pass"
    assert alignment["reason"] == "hold_day_weights_drift_by_design"
    assert alignment["weight_gap_enforced"] is False
    # The drift is still reported, just not enforced.
    assert alignment["maximum_target_weight_gap"] > 0.05
    assert payload["status"] == "pass"


def test_same_drift_on_the_rebalance_run_fails(tmp_path, monkeypatch):
    payload = _drifted_truth(tmp_path, monkeypatch, period_run_id="paper-today")
    alignment = payload["summary"]["alignment"]
    assert alignment["status"] == "fail"
    assert alignment["weight_gap_enforced"] is True


def test_retired_stops_are_not_required(monkeypatch):
    monkeypatch.setattr(broker_truth, "CORE_TRAILING_STOP_ENABLED", False)
    monkeypatch.setattr(broker_truth, "OVERLAY_TRAILING_STOP_ENABLED", False)
    assert broker_truth._stop_required("QQQ", 10, 0.6) is False
    assert broker_truth._stop_required("MU", 10, 0.2) is False


def test_stops_switched_on_are_required(monkeypatch):
    monkeypatch.setattr(broker_truth, "CORE_TRAILING_STOP_ENABLED", True)
    monkeypatch.setattr(broker_truth, "OVERLAY_TRAILING_STOP_ENABLED", True)
    assert broker_truth._stop_required("QQQ", 10, 0.6) is True
    assert broker_truth._stop_required("MU", 10, 0.2) is True


def test_stop_switch_code_defaults_are_off():
    # The code defaults (used when no environment/.env value is set).
    source = open(broker_truth.__file__, encoding="utf-8").read()
    assert 'os.environ.get("ALPACA_TRAILING_STOP", "0")' in source
    assert 'os.environ.get("GUARD_CORE_STOP", "0")' in source


# ── The paper-evidence gate ────────────────────────────────────────────────

def test_epoch_accepts_hold_day_alignment_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_validation_epoch, "SIGNAL_DIR", str(tmp_path))
    truth = {"summary": {"fail_count": 0, "alignment": {
        "status": "pass", "maximum_target_weight_gap": 0.10, "gross_exposure_gap": 0.08,
        "weight_gap_enforced": False}}}
    (tmp_path / "broker_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    epoch = {"epoch_id": "e1", "started_at": "2026-09-01T00:00:00+00:00", "requirements": {}}
    result = paper_validation_epoch.evaluate_epoch(epoch)
    checks = result.get("checks", result.get("acceptance_checks", {}))
    assert checks["target_weight_gap"] is True
    assert checks["gross_exposure_gap"] is True


def test_epoch_still_enforces_gap_when_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_validation_epoch, "SIGNAL_DIR", str(tmp_path))
    truth = {"summary": {"fail_count": 0, "alignment": {
        "status": "pass", "maximum_target_weight_gap": 0.10, "gross_exposure_gap": 0.08}}}
    (tmp_path / "broker_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    epoch = {"epoch_id": "e1", "started_at": "2026-09-01T00:00:00+00:00", "requirements": {}}
    result = paper_validation_epoch.evaluate_epoch(epoch)
    checks = result.get("checks", result.get("acceptance_checks", {}))
    assert checks["target_weight_gap"] is False
