from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research_output_guard import build_manifest, validate_research_outputs


def test_research_output_guard_passes_clean_json_and_csv(tmp_path):
    json_path = tmp_path / "clean.json"
    csv_path = tmp_path / "clean.csv"

    # Beginner note: write one valid JSON file and one valid CSV file, then the
    # guard should report a full pass.
    json_path.write_text(json.dumps({"ok": True, "rows": [1, 2]}), encoding="utf-8")
    pd.DataFrame([{"feature": "ret_5d", "grade": "A"}]).to_csv(csv_path, index=False)

    report = validate_research_outputs([json_path, csv_path])

    assert report["status"] == "pass"
    assert report["failed_count"] == 0


def test_research_output_guard_fails_conflict_markers(tmp_path):
    csv_path = tmp_path / "conflicted.csv"
    csv_path.write_text(
        "feature,grade\n<<<<<<< Updated upstream\nret_5d,A\n=======\nret_10d,B\n>>>>>>> Stashed changes\n",
        encoding="utf-8",
    )

    report = validate_research_outputs([csv_path])

    assert report["status"] == "fail"
    assert report["failed_count"] == 1
    assert "merge_conflict_markers" in report["files"][0]["issues"][0]


def test_research_output_guard_rejects_nonstandard_json_constants(tmp_path):
    json_path = tmp_path / "nan.json"
    json_path.write_text('{"score": NaN}', encoding="utf-8")

    report = validate_research_outputs([json_path])

    assert report["status"] == "fail"
    assert any("json_parse_error" in issue for issue in report["files"][0]["issues"])


def test_research_manifest_records_output_and_data_checksums(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    signals = tmp_path / "signals"
    data = tmp_path / "data"
    signals.mkdir()
    data.mkdir()

    output_path = signals / "feature_health_profile.json"
    output_path.write_text(json.dumps({"summary": {"feature_health_gate_pass": True}}), encoding="utf-8")
    # Beginner note: the guard fingerprints parquet files by bytes only; this
    # test file does not need to be a real parquet table.
    (data / "QQQ.parquet").write_bytes(b"fake parquet bytes")

    validation = validate_research_outputs([output_path])
    manifest = build_manifest(
        validation=validation,
        output_paths=[output_path],
        data_dir=data,
        command="python refresh_local_research_data.py",
        root=tmp_path,
        max_data_files=10,
    )

    assert manifest["validation"]["status"] == "pass"
    assert manifest["outputs"]["files"][0]["path"] == "signals/feature_health_profile.json"
    assert manifest["input_data"]["file_count"] == 1
    assert manifest["input_data"]["files"][0]["path"] == "data/QQQ.parquet"
    assert manifest["input_data"]["combined_sha256"]
