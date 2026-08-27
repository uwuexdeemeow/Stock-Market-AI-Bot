from __future__ import annotations

import json

import model_registry


def test_register_records_git_data_metrics_and_artifact_checksum(tmp_path, monkeypatch):
    """A successful run has enough evidence to identify and verify its model."""
    artifact = tmp_path / "AAPL_model.json"
    artifact.write_text('{"trees": 3}', encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_registry, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(model_registry, "_git_dirty", lambda: True)
    monkeypatch.setattr(model_registry, "_source_fingerprint", lambda: "source789")
    monkeypatch.setattr(model_registry, "_dataset_fingerprint", lambda: "data456")

    run = model_registry.register(
        "AAPL",
        [artifact],
        metrics={"test_auc": 0.61},
        registry_path=registry_path,
    )

    saved = json.loads(registry_path.read_text(encoding="utf-8"))
    assert saved["runs"][0]["run_id"] == run["run_id"]
    assert run["git_commit"] == "abc123"
    assert run["git_dirty"] is True
    assert run["source_fingerprint"] == "source789"
    assert run["dataset_fingerprint"] == "data456"
    assert run["metrics"]["test_auc"] == 0.61
    assert run["artifacts"][0]["path"] == "AAPL_model.json"
    assert model_registry.verify_artifacts(run) == (True, [])
    assert model_registry.verify(run) == (True, [])
    assert model_registry.latest("AAPL", registry_path=registry_path)["run_id"] == run["run_id"]


def test_verify_detects_changed_artifact(tmp_path, monkeypatch):
    """Changing a saved model after registration produces a clear mismatch."""
    artifact = tmp_path / "model.json"
    artifact.write_text("first", encoding="utf-8")
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_registry, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(model_registry, "_git_dirty", lambda: False)
    monkeypatch.setattr(model_registry, "_source_fingerprint", lambda: "source789")
    monkeypatch.setattr(model_registry, "_dataset_fingerprint", lambda: "data456")
    run = model_registry.register("pooled", [artifact], registry_path=tmp_path / "registry.json")

    artifact.write_text("changed", encoding="utf-8")

    ok, issues = model_registry.verify(run)
    assert ok is False
    assert issues == ["checksum_mismatch:model.json"]
    assert model_registry.verify_artifacts(run) == (
        False,
        ["checksum_mismatch:model.json"],
    )


def test_verify_detects_source_change(tmp_path, monkeypatch):
    artifact = tmp_path / "model.json"
    artifact.write_text("model", encoding="utf-8")
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_registry, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(model_registry, "_git_dirty", lambda: False)
    monkeypatch.setattr(model_registry, "_dataset_fingerprint", lambda: "data456")
    monkeypatch.setattr(model_registry, "_source_fingerprint", lambda: "before")
    run = model_registry.register("pooled", [artifact], registry_path=tmp_path / "registry.json")
    monkeypatch.setattr(model_registry, "_source_fingerprint", lambda: "after")

    assert model_registry.verify_artifacts(run) == (True, [])
    assert model_registry.verify(run) == (False, ["source_fingerprint_mismatch"])


def test_register_refuses_missing_artifact(tmp_path):
    """A failed/incomplete save must never appear as a successful training run."""
    try:
        model_registry.register(
            "AAPL",
            [tmp_path / "missing.json"],
            registry_path=tmp_path / "registry.json",
        )
    except FileNotFoundError as exc:
        assert "model artifact missing" in str(exc)
    else:
        raise AssertionError("missing model artifact was registered")
