"""Package a reproducible, secret-free walk-forward snapshot for Google Colab.

PLAIN ENGLISH: Colab needs the same parquet data and research metadata as this
computer. This script packages only those inputs, writes a checksum, and
refuses to include account credentials or broker state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from safe_io import atomic_write_json


DEFAULT_OUTPUT_DIR = Path("colab")
SAFE_SIGNAL_INPUTS = (
    "research_run_manifest.json",
    "feature_quality_report.json",
    "feature_quality_summary.csv",
    "feature_health_profile.json",
    "feature_health_profile.csv",
    "adaptive_factor_weights.json",
    "core_satellite_live_configs.json",
    "core_satellite_validation_bundle.json",
)
SAFE_LOG_INPUTS = (
    "feature_ic_shortlist.csv",
    "core_satellite_survivorship_audit.json",
    "core_satellite_execution_stress.json",
    "factor_decay_monitor.json",
)


def _sha256(path: Path) -> str:
    """Hash the completed archive for upload verification."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    """Record the exact code version Colab must check out."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _working_tree_changes() -> list[str]:
    """List uncommitted files that Colab could not reproduce from Git."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def build_snapshot(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    allow_dirty: bool = False,
) -> tuple[Path, Path]:
    """Create the compressed snapshot and its small checksum manifest."""
    changes = _working_tree_changes()
    if changes and not allow_dirty:
        raise RuntimeError(
            "Commit and push the project changes before preparing Colab; "
            "otherwise the recorded Git commit would not reproduce this run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output_dir / f"stockbot_walkforward_snapshot_{stamp}.tar.gz"
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    inputs = sorted(Path("data").glob("*.parquet"))
    inputs += sorted(Path("data/manifests").glob("*.json"))
    inputs += [Path("signals") / name for name in SAFE_SIGNAL_INPUTS if (Path("signals") / name).exists()]
    inputs += [Path("logs") / name for name in SAFE_LOG_INPUTS if (Path("logs") / name).exists()]
    with tarfile.open(temporary, "w:gz") as handle:
        for path in inputs:
            handle.add(path, arcname=path.as_posix(), recursive=False)
    temporary.replace(archive)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": archive.name,
        "sha256": _sha256(archive),
        "git_commit": _git_commit(),
        "file_count": len(inputs),
        "contains_secrets": False,
        "working_tree_clean": not changes,
        "files": [path.as_posix() for path in inputs],
    }
    manifest_path = archive.with_suffix(".manifest.json")
    atomic_write_json(manifest, manifest_path)
    return archive, manifest_path


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Package data even when code changes are uncommitted (not reproducible).",
    )
    args = parser.parse_args()
    archive, manifest = build_snapshot(args.output_dir, allow_dirty=args.allow_dirty)
    print(f"Colab snapshot -> {archive}")
    print(f"Checksum manifest -> {manifest}")
    print("Upload both files to Google Drive; no Alpaca credentials are included")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
