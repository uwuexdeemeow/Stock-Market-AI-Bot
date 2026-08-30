import hashlib
import json

import disaster_recovery


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovery_verifies_and_restores_without_trading(tmp_path):
    artifact = tmp_path / "artifact"
    source = artifact / "signals"
    source.mkdir(parents=True)
    health = source / "alpaca_paper_health.json"
    health.write_text('{"status":"collecting"}', encoding="utf-8")
    manifest = {
        "status": "complete",
        "run_id": "run-1",
        "files": {
            "alpaca_paper_health.json": {
                "relative_path": "signals/alpaca_paper_health.json",
                "required": True,
                "sha256": _hash(health),
            }
        },
    }
    (source / "paper_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    destination = tmp_path / "restore"
    result = disaster_recovery.restore_artifact(artifact, destination_root=destination)
    assert result["status"] == "restored"
    assert result["orders_submitted"] is False
    assert (destination / "signals" / "alpaca_paper_health.json").exists()


def test_recovery_refuses_checksum_mismatch(tmp_path):
    import pytest

    source = tmp_path / "signals"
    source.mkdir()
    (source / "broker_truth.json").write_text("{}", encoding="utf-8")
    (source / "paper_run_manifest.json").write_text(json.dumps({
        "status": "complete",
        "files": {"broker_truth.json": {
            "relative_path": "signals/broker_truth.json", "required": True, "sha256": "bad"
        }},
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        disaster_recovery.restore_artifact(tmp_path, destination_root=tmp_path / "out")
