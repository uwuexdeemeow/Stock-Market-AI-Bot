"""The Colab snapshot must carry every file the research manifest fingerprints.

PLAIN ENGLISH: Windows stores text with CRLF line endings, but a Git clone on
Colab gets LF endings. A fingerprinted file that is left out of the snapshot
would be restored from Git with different bytes and fail its check.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import prepare_colab_walkforward as colab


def _write_manifest(root: Path, paths: list[str]) -> None:
    rows = [{"path": path, "sha256": "x"} for path in paths]
    (root / "signals").mkdir(parents=True, exist_ok=True)
    (root / "signals" / "research_run_manifest.json").write_text(
        json.dumps({"validation": {"files": rows}, "outputs": {"files": rows}}),
        encoding="utf-8",
    )


def test_manifest_listed_inputs_only_returns_top_level_signal_and_log_files(tmp_path):
    _write_manifest(tmp_path, [
        "signals/feature_research_summary.csv",
        "logs\\feature_ic_shortlist.csv",
        "signals/nested/other.json",
        "data/AAPL.parquet",
        ".env",
    ])

    listed = colab._manifest_listed_inputs(tmp_path / "signals" / "research_run_manifest.json")

    assert [path.as_posix() for path in listed] == [
        "logs/feature_ic_shortlist.csv",
        "signals/feature_research_summary.csv",
    ]


def test_snapshot_includes_manifest_listed_file_missing_from_fixed_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(colab, "_working_tree_changes", lambda: [])
    monkeypatch.setattr(colab, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(colab, "SAFE_SIGNAL_INPUTS", ("research_run_manifest.json",))
    monkeypatch.setattr(colab, "SAFE_LOG_INPUTS", ())
    _write_manifest(tmp_path, ["signals/extra_fingerprinted_input.csv"])
    (tmp_path / "signals" / "extra_fingerprinted_input.csv").write_bytes(b"a,b\r\n1,2\r\n")

    archive, manifest = colab.build_snapshot(tmp_path / "out")

    with tarfile.open(archive) as handle:
        names = handle.getnames()
        packed = handle.extractfile("signals/extra_fingerprinted_input.csv").read()
    assert "signals/extra_fingerprinted_input.csv" in names
    assert packed == b"a,b\r\n1,2\r\n"
    assert "signals/extra_fingerprinted_input.csv" in json.loads(manifest.read_text())["files"]


def test_snapshot_includes_incumbent_metrics_for_research_scripts(tmp_path, monkeypatch):
    # The phase-luck research scripts load the approved incumbent from these
    # two files, so Colab must get this computer's exact bytes of both.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(colab, "_working_tree_changes", lambda: [])
    monkeypatch.setattr(colab, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(colab, "SAFE_LOG_INPUTS", ())
    (tmp_path / "signals").mkdir()
    for name in ("core_satellite_alpha_metrics.json", "core_satellite_validation_bundle.json"):
        (tmp_path / "signals" / name).write_bytes(b'{"shape": "top3"}\r\n')

    archive, _manifest = colab.build_snapshot(tmp_path / "out")

    with tarfile.open(archive) as handle:
        names = handle.getnames()
    assert "signals/core_satellite_alpha_metrics.json" in names
    assert "signals/core_satellite_validation_bundle.json" in names


def test_notebook_phase_luck_cell_is_opt_in_and_saves_separately():
    # The research cell must be off by default and must not reuse the
    # walk-forward result archive name.
    notebook = json.loads(Path("Colab/stockbot_walkforward.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    settings = next(src for src in sources if "SNAPSHOT_NAME =" in src)
    assert "RUN_PHASE_LUCK = False" in settings
    assert "RUN_WALKFORWARD = True" in settings
    phase_cell = next(src for src in sources if "if RUN_PHASE_LUCK:" in src)
    assert "stockbot_phase_luck_result.tar.gz" in phase_cell
    assert "stockbot_colab_result.tar.gz" not in phase_cell
    for script in ("phase_luck.py", "tranche_preview.py"):
        assert script in phase_cell
        assert Path("research_evidence/phase_luck_20260926", script).exists()
