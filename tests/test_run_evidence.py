import json
from datetime import datetime, timezone

import run_evidence


def test_run_context_is_shared_and_never_exposes_account_id(monkeypatch):
    monkeypatch.setenv("STOCKBOT_RUN_ID", "paper-test-1")
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "RAW-SECRET-ACCOUNT")
    context = run_evidence.build_run_context(now=datetime(2026, 8, 30, tzinfo=timezone.utc))
    assert context["run_id"] == "paper-test-1"
    assert context["paper_account_hash"] != "RAW-SECRET-ACCOUNT"
    assert "RAW-SECRET-ACCOUNT" not in json.dumps(context)


def test_saved_safe_account_hash_is_reused_without_raw_account_id(monkeypatch, tmp_path):
    expected = run_evidence.paper_account_hash("paper-account-123")
    signals = tmp_path / "signals"
    signals.mkdir()
    (signals / "alpaca_daily_status.json").write_text(
        '{"paper_account_hash":"' + expected + '"}', encoding="utf-8"
    )
    monkeypatch.delenv("ALPACA_ACCOUNT_ID", raising=False)
    monkeypatch.setattr(run_evidence, "SIGNALS", signals)

    assert run_evidence._account_hash() == expected


def test_manifest_rejects_mixed_run_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKBOT_RUN_ID", "run-current")
    first = tmp_path / "broker_truth.json"
    second = tmp_path / "paper_health.json"
    first.write_text(json.dumps({"run_id": "run-current"}), encoding="utf-8")
    second.write_text(json.dumps({"run_id": "run-old"}), encoding="utf-8")
    manifest = run_evidence.build_evidence_manifest(
        required_files=(first, second),
        output_path=tmp_path / "manifest.json",
        write=False,
    )
    assert manifest["status"] == "incomplete"
    assert any(problem.startswith("mixed_run:paper_health.json") for problem in manifest["problems"])


def test_rebalance_state_is_idempotent_and_terminal(tmp_path):
    path = tmp_path / "rebalance_state.json"
    run_evidence.update_rebalance_state("planned", run_id="run-1", path=path)
    run_evidence.update_rebalance_state("planned", run_id="run-1", details={"orders": 2}, path=path)
    final = run_evidence.update_rebalance_state("aligned", run_id="run-1", path=path)
    assert [row["status"] for row in final["history"]] == ["planned", "aligned"]
    assert final["terminal"] is True
    assert final["real_capital_approved"] is False
