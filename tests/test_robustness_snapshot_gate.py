"""Tests for safe workflow handling of a coherent trading rejection."""
from __future__ import annotations

import json

import pandas as pd

import robustness_snapshot_gate as gate


def test_classify_snapshot_separates_identity_from_health(tmp_path, monkeypatch):
    """A failed safety review stays blocked without becoming invalid evidence."""
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"config_fingerprint": "cfg", "config": {"shape": "top3"}}))
    monkeypatch.setattr(gate, "validate_validation_bundle", lambda _bundle: (True, []))
    monkeypatch.setattr(gate, "load_dataset_context", lambda: {"dataset_fingerprint": "data"})
    monkeypatch.setattr(
        gate,
        "current_robustness_evidence",
        lambda **_kwargs: {
            "pass": False,
            "reasons": ["execution_stress_review_failed"],
            "identity_pass": True,
            "identity_reasons": [],
            "health_pass": False,
            "health_reasons": ["execution_stress_review_failed"],
            "medium_risk_review": {"pass": False},
        },
    )

    status = gate.classify_snapshot(bundle)

    assert status["status"] == "trading_blocked"
    assert status["snapshot_usable"] is True
    assert status["trading_allowed"] is False


def test_classify_snapshot_rejects_mismatched_evidence(tmp_path, monkeypatch):
    """A fingerprint mismatch remains an operational workflow failure."""
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"config_fingerprint": "cfg", "config": {}}))
    monkeypatch.setattr(gate, "validate_validation_bundle", lambda _bundle: (True, []))
    monkeypatch.setattr(gate, "load_dataset_context", lambda: {"dataset_fingerprint": "data"})
    monkeypatch.setattr(
        gate,
        "current_robustness_evidence",
        lambda **_kwargs: {
            "pass": False,
            "reasons": ["execution_stress:dataset_fingerprint_mismatch"],
            "identity_pass": False,
            "identity_reasons": ["execution_stress:dataset_fingerprint_mismatch"],
            "health_pass": True,
            "health_reasons": [],
            "medium_risk_review": {"pass": True},
        },
    )

    status = gate.classify_snapshot(bundle)

    assert status["status"] == "invalid"
    assert status["snapshot_usable"] is False
    assert status["trading_allowed"] is False


def test_classify_snapshot_rejects_invalid_approval_bundle(tmp_path, monkeypatch):
    """A damaged approval record cannot produce a usable cache snapshot."""
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"config_fingerprint": "cfg", "config": {}}))
    monkeypatch.setattr(
        gate,
        "validate_validation_bundle",
        lambda _bundle: (False, ["validation_bundle_hash_mismatch"]),
    )

    status = gate.classify_snapshot(bundle)

    assert status["status"] == "invalid"
    assert status["reasons"] == ["validation_bundle_invalid:validation_bundle_hash_mismatch"]


def test_block_signal_zeroes_targets_and_clears_approval(tmp_path):
    """A blocked snapshot cannot leave an old tradable allocation visible."""
    signal = tmp_path / "signal.csv"
    pd.DataFrame([{
        "paper_ready": True,
        "gates_all_pass": True,
        "target_spy_weight": 0.4,
        "target_qqq_weight": 0.3,
        "target_tqqq_weight": 0.1,
        "target_cash_weight": 0.0,
        "gross_exposure": 1.0,
        "overlay_tickers": "AAPL",
        "overlay_weights_json": '{"AAPL": 0.2}',
    }]).to_csv(signal, index=False)

    gate.block_signal({"reasons": ["execution_stress_review_failed"]}, signal)
    row = pd.read_csv(signal).iloc[0]

    assert bool(row["paper_ready"]) is False
    assert bool(row["gates_all_pass"]) is False
    assert float(row["gross_exposure"]) == 0.0
    assert float(row["target_cash_weight"]) == 1.0
    assert pd.isna(row["overlay_tickers"]) or row["overlay_tickers"] == ""
