"""Record reproducible model-training runs and verify their saved artifacts.

PLAIN ENGLISH: model files alone do not explain which code and data created
them.  This registry appends one record after a successful training run.  Each
record stores the Git commit, data fingerprint, command, metrics, and checksums
of every saved model file so a beginner can later prove what was produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from safe_io import atomic_write_json
from settings import MODEL_DIR, SIGNAL_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = Path(MODEL_DIR) / "registry.json"
RESEARCH_MANIFEST_PATH = Path(SIGNAL_DIR) / "research_run_manifest.json"


def _file_sha256(path: Path) -> str:
    """Hash a file in chunks so even a large model does not fill memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    """Return the exact checked-out source revision used for training."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _git_dirty() -> bool:
    """Report whether tracked or untracked project source differs from Git."""
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.returncode != 0 or bool(completed.stdout.strip())


def _source_fingerprint() -> str:
    """Hash trainer source so a dirty local run still has exact identity."""
    candidates = list(PROJECT_ROOT.glob("*.py"))
    for name in ("requirements.txt", "requirements.lock", "pyproject.toml"):
        path = PROJECT_ROOT / name
        if path.is_file():
            candidates.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dataset_fingerprint() -> str:
    """Read the checksum of all research inputs without rehashing every parquet."""
    try:
        payload = json.loads(RESEARCH_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str((payload.get("input_data", {}) or {}).get("combined_sha256", ""))


def _relative(path: Path) -> str:
    """Prefer portable project-relative paths over one laptop's absolute path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _load_registry(path: Path) -> dict:
    """Load existing history, or start an empty versioned registry."""
    if not path.exists():
        return {"schema_version": 1, "runs": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("runs"), list):
        raise ValueError("model registry runs must be a list")
    return payload


def register(
    run_name: str,
    artifacts: Iterable[str | Path],
    *,
    metrics: dict | None = None,
    metadata: dict | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict:
    """Append one successful training run after checking every artifact exists."""
    artifact_rows: list[dict] = []
    for raw_path in artifacts:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"model artifact missing: {path}")
        artifact_rows.append({
            "path": _relative(path),
            "sha256": _file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        })
    if not artifact_rows:
        raise ValueError("at least one model artifact is required")

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    identity_text = json.dumps(
        {"run_name": run_name, "created_at": created_at, "artifacts": artifact_rows},
        sort_keys=True,
    )
    run = {
        "run_id": hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:16],
        "run_name": str(run_name),
        "created_at": created_at,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "source_fingerprint": _source_fingerprint(),
        "dataset_fingerprint": _dataset_fingerprint(),
        "python_version": sys.version.split()[0],
        "command": [str(value) for value in sys.argv],
        "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
        "artifacts": artifact_rows,
    }
    registry = _load_registry(registry_path)
    registry["runs"].append(run)
    registry["updated_at"] = created_at
    atomic_write_json(registry, registry_path)
    return run


def latest(run_name: str | None = None, *, registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict | None:
    """Return the newest registry entry, optionally for one ticker/model name."""
    runs = _load_registry(registry_path).get("runs", [])
    if run_name is not None:
        runs = [row for row in runs if str(row.get("run_name")) == str(run_name)]
    return dict(runs[-1]) if runs else None


def verify_artifacts(run: dict) -> tuple[bool, list[str]]:
    """Rehash saved artifacts without conflating them with source-code drift."""
    issues: list[str] = []
    for artifact in run.get("artifacts", []):
        path = Path(str(artifact.get("path", "")))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            issues.append(f"missing:{artifact.get('path')}")
        elif _file_sha256(path) != str(artifact.get("sha256", "")):
            issues.append(f"checksum_mismatch:{artifact.get('path')}")
    return not issues, issues


def verify(run: dict) -> tuple[bool, list[str]]:
    """Verify both saved artifacts and the current trainer source identity."""
    _artifacts_ok, issues = verify_artifacts(run)
    expected_source = str(run.get("source_fingerprint", ""))
    if expected_source and _source_fingerprint() != expected_source:
        issues.append("source_fingerprint_mismatch")
    return not issues, issues


def main() -> int:
    """Print the newest run and verify that its artifacts still match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", help="Show the latest run for one ticker or pooled model.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args()
    run = latest(args.run_name, registry_path=args.registry)
    if run is None:
        print("No registered model runs found")
        return 1
    artifacts_ok, artifact_issues = verify_artifacts(run)
    ok, issues = verify(run)
    source_matches = "source_fingerprint_mismatch" not in issues
    print(json.dumps({
        **run,
        "artifacts_valid": artifacts_ok,
        "source_matches_current": source_matches,
        "reproducibility_valid": ok,
        "verification_issues": issues,
    }, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
