"""Replay saved broker evidence; no test contacts or trades with Alpaca."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import execution_scorecard as esc
from order_accounting import classify_logical_orders
import run_evidence


FIXTURE = Path(__file__).parent / "fixtures" / "execution_20260903"


def test_recovered_september_orders_reconcile_to_scorecard(tmp_path):
    # The actual journal records 155+50 INTC shares and 30 FCX shares after
    # cash clamping. Two attempts still mean one intended order per symbol.
    log = pd.read_csv(FIXTURE / "paper_log.csv")
    plan = pd.read_csv(FIXTURE / "order_plan.csv").set_index("ticker")
    journal = log.set_index("ticker")
    assert set(plan.index) == set(journal.index) == {"INTC", "FCX"}
    assert plan.loc["INTC", "quantity"] == journal.loc["INTC", "quantity"] == 205
    # FCX's smaller submitted quantity has an explicit cash-clamp explanation.
    assert plan.loc["FCX", "quantity"] == 43
    assert journal.loc["FCX", "quantity"] == 30
    assert "cash_limited:43->30" in journal.loc["FCX", "cash_clamp_reason"]
    accounting = classify_logical_orders(log)
    assert accounting["accepted_logical_orders"] == 2
    assert accounting["fully_filled_logical_orders"] == 2
    assert accounting["child_attempts"] == 4
    assert accounting["duplicate_child_attempts"] == 0
    assert sorted(row["filled_quantity"] for row in accounting["logical_orders"]) == [30, 205]
    card = esc.build_execution_scorecard(
        paper_log_path=FIXTURE / "paper_log.csv",
        slippage_report_path=FIXTURE / "report.json",
        previous_scorecard_path=tmp_path / "none.json",
        now=datetime(2026, 9, 5, tzinfo=timezone.utc),
    )
    assert card["summary"]["complete_fill_rate"] == 1.0
    assert card["summary"]["avg_slippage_bps"] == 1.03
    assert card["decision_eligible"] is False  # Still only one session.


def test_child_snapshots_cannot_turn_partial_parent_into_complete():
    # Repeated snapshots of the same 155-share fill must not be added twice.
    first = dict(client_order_id="parent-a1", order_id="one", quantity=205,
                 requested_quantity=205, filled_qty=155, status="canceled")
    second = dict(client_order_id="parent-a2", order_id="two", quantity=50,
                  requested_quantity=205, filled_qty=20, status="filled")
    counts = classify_logical_orders(pd.DataFrame([first, first, second]))
    assert counts["logical_orders"][0]["filled_quantity"] == 175
    assert counts["fully_filled_logical_orders"] == 0
    assert counts["partially_filled_logical_orders"] == 1


def test_many_good_fills_cannot_replace_missing_order_journal(tmp_path, monkeypatch):
    # Even enough price observations cannot establish the missing denominator.
    monkeypatch.setattr(esc, "MIN_DECISION_ORDERS", 1)
    card = esc.build_execution_scorecard(
        paper_log_path=tmp_path / "missing.csv",
        slippage_report_path=FIXTURE / "report.json",
        previous_scorecard_path=tmp_path / "missing.json",
        now=datetime(2026, 9, 5, tzinfo=timezone.utc),
        min_rebalance_fills=1, min_rebalance_sessions=1,
    )
    assert card["score"] is None
    assert card["decision_eligible"] is False
    assert "fill_rate_unknown" in card["decision_blockers"]
    assert "skipped_rate_unknown" in card["decision_blockers"]


def test_manifest_requires_order_evidence_even_when_reports_exist(tmp_path, monkeypatch):
    # Complete-looking summary reports must not hide a missing original log.
    monkeypatch.setenv("STOCKBOT_RUN_ID", "test-run")
    monkeypatch.setattr(run_evidence, "OPTIONAL_EVIDENCE_FILES", ())
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"run_id": "test-run"}))
    result = run_evidence.build_evidence_manifest(
        required_files=(summary,), operational_files=(tmp_path / "orders.csv",), write=False,
    )
    assert result["status"] == "incomplete"
    assert "missing:orders.csv" in result["problems"]


def test_partly_restored_journal_exposes_unmatched_broker_fill(tmp_path):
    # Retain INTC but lose FCX: the remaining complete order is not proof
    # that this journal covers all broker activity.
    log = pd.read_csv(FIXTURE / "paper_log.csv")
    log[log.ticker.eq("INTC")].to_csv(tmp_path / "partial.csv", index=False)
    card = esc.build_execution_scorecard(
        paper_log_path=tmp_path / "partial.csv",
        slippage_report_path=FIXTURE / "report.json",
        previous_scorecard_path=tmp_path / "none.json",
        now=datetime(2026, 9, 5, tzinfo=timezone.utc),
    )
    assert card["summary"]["unmatched_broker_fill_count"] == 1
    assert "broker_fills_missing_from_journal_1" in card["decision_blockers"]
    assert card["score"] is None
