"""research_output_guard.py - validate generated research artifacts.

PLAIN ENGLISH:
Research scripts write JSON and CSV files that the live bot later reads. If
Git accidentally commits merge-conflict text, or if a file is half-written, the
next machine can pull broken research state. This script checks those files and
optionally writes a small run manifest that explains exactly where the outputs
came from.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from safe_io import atomic_write_json


# PLAIN ENGLISH: These are the generated research files that have caused
# machine-to-machine conflicts before. They are tracked in git as known-good
# baselines, so they must be parseable before we trust a commit or CI artifact.
DEFAULT_RESEARCH_OUTPUTS = [
    Path("signals/adaptive_factor_weights.json"),
    Path("signals/feature_quality_report.json"),
    Path("signals/feature_quality_summary.csv"),
    Path("signals/feature_health_profile.json"),
    Path("signals/feature_health_profile.csv"),
    Path("signals/feature_research_report.json"),
    Path("signals/feature_research_summary.csv"),
]

# PLAIN ENGLISH: Git writes these strings into files when a merge conflict was
# not resolved. They should never appear inside generated JSON/CSV outputs.
CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")

# PLAIN ENGLISH: These packages explain most cross-machine differences in
# research outputs. Missing packages are recorded as "not_installed".
PACKAGE_FINGERPRINTS = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "yfinance",
    "pyarrow",
    "alpaca-trade-api",
]

# PLAIN ENGLISH: Only record non-secret environment settings. Anything that
# smells like a key/token/password is redacted even if its prefix is allowed.
ENV_PREFIXES_TO_RECORD = (
    "ADAPTIVE_",
    "ALPHA_",
    "CORE_SATELLITE_",
    "DATA_",
    "EXECUTION_",
    "FACTOR_",
    "FEATURE_",
    "RESEARCH_",
    "RISK_",
    "TRAIN_",
    "WALKFORWARD_",
    "YFINANCE_",
)
ENV_NAMES_TO_RECORD = {"PYTHONHASHSEED", "TZ", "SENTIMENT_ENGINE_LEVEL"}
SECRET_NAME_PARTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CHAT_ID")


def _repo_relative(path: Path) -> str:
    """Return a stable slash-separated path for reports.

    PLAIN ENGLISH: Windows uses backslashes and Mac/Linux use slashes. Reports
    are easier to compare when paths always use forward slashes.
    """
    return path.as_posix()


def _run_git(args: list[str]) -> str:
    """Run a git command and return stripped stdout, or an empty string."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _strict_json_loads(text: str) -> Any:
    """Parse JSON while rejecting NaN/Infinity values.

    PLAIN ENGLISH: Python's JSON parser accepts NaN by default, but real JSON
    does not. Rejecting those values keeps GitHub, Mac, and Windows behavior
    aligned.
    """

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    return json.loads(text, parse_constant=reject_constant)


def _conflict_marker_lines(text: str) -> list[str]:
    """Return up to ten line references containing merge-conflict markers."""
    hits = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if any(stripped.startswith(marker) for marker in CONFLICT_MARKERS):
            hits.append(f"line {line_no}: {stripped[:80]}")
        if len(hits) >= 10:
            break
    return hits


def validate_research_file(path: Path) -> dict[str, Any]:
    """Validate one JSON/CSV/text research artifact.

    PLAIN ENGLISH: This checks that the file exists, has no Git conflict
    markers, and can be parsed by the tool that normally reads it.
    """
    info: dict[str, Any] = {
        "path": _repo_relative(path),
        "exists": path.exists(),
        "status": "pass",
        "issues": [],
    }
    if not path.exists():
        info["status"] = "fail"
        info["issues"].append("missing_file")
        return info

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        info["status"] = "fail"
        info["issues"].append(f"utf8_decode_error:{exc}")
        return info
    except OSError as exc:
        info["status"] = "fail"
        info["issues"].append(f"read_error:{exc}")
        return info

    info["bytes"] = path.stat().st_size
    marker_hits = _conflict_marker_lines(text)
    if marker_hits:
        info["issues"].append("merge_conflict_markers:" + "; ".join(marker_hits))

    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            parsed = _strict_json_loads(text)
            info["top_level_type"] = type(parsed).__name__
            if isinstance(parsed, dict):
                info["top_level_keys"] = list(parsed.keys())[:12]
        except Exception as exc:
            info["issues"].append(f"json_parse_error:{exc}")
    elif suffix == ".csv":
        try:
            df = pd.read_csv(io.StringIO(text))
            info["rows"] = int(len(df))
            info["columns"] = int(len(df.columns))
            if len(df.columns) == 0:
                info["issues"].append("csv_has_no_columns")
            if len(df) == 0:
                info["issues"].append("csv_has_no_rows")
        except Exception as exc:
            info["issues"].append(f"csv_parse_error:{exc}")

    if info["issues"]:
        info["status"] = "fail"
    return info


def validate_research_outputs(paths: list[Path]) -> dict[str, Any]:
    """Validate all requested files and return a summary report."""
    files = [validate_research_file(path) for path in paths]
    failed = [row for row in files if row.get("status") != "pass"]
    return {
        "status": "pass" if not failed else "fail",
        "checked_count": len(files),
        "failed_count": len(failed),
        "files": files,
    }


def _sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum for a file.

    PLAIN ENGLISH: A checksum is a short fingerprint. If two machines produce
    different checksums for the same path, the file contents are different.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_file(path: Path, *, root: Path) -> dict[str, Any]:
    """Return size, modified time, and checksum metadata for one file."""
    rel = path.resolve().relative_to(root.resolve()) if path.resolve().is_relative_to(root.resolve()) else path
    stat = path.stat()
    return {
        "path": _repo_relative(Path(rel)),
        "bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": _sha256_file(path),
    }


def _combined_sha256(files: list[dict[str, Any]]) -> str:
    """Create one checksum for a whole group of file fingerprints."""
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda row: str(row.get("path", ""))):
        digest.update(str(item.get("path", "")).encode("utf-8"))
        digest.update(str(item.get("bytes", "")).encode("utf-8"))
        digest.update(str(item.get("sha256", "")).encode("utf-8"))
    return digest.hexdigest()


def _fingerprint_outputs(paths: list[Path], *, root: Path) -> list[dict[str, Any]]:
    """Fingerprint generated research outputs that exist on disk."""
    return [_fingerprint_file(path, root=root) for path in paths if path.exists()]


def _fingerprint_data_dir(data_dir: Path, *, root: Path, max_files: int) -> dict[str, Any]:
    """Fingerprint parquet input files used by research.

    PLAIN ENGLISH: The strategy can differ across machines simply because the
    data files differ. Hashing the parquet inputs tells us whether Mac and PC
    were studying the same raw panel.
    """
    if not data_dir.exists():
        return {
            "data_dir": _repo_relative(data_dir),
            "exists": False,
            "file_count": 0,
            "files": [],
            "combined_sha256": "",
        }
    files = sorted(data_dir.glob("*.parquet"))[: max(0, int(max_files))]
    fingerprints = [_fingerprint_file(path, root=root) for path in files]
    return {
        "data_dir": _repo_relative(data_dir),
        "exists": True,
        "file_count": len(list(data_dir.glob("*.parquet"))),
        "fingerprinted_count": len(fingerprints),
        "max_files": int(max_files),
        "combined_sha256": _combined_sha256(fingerprints) if fingerprints else "",
        "files": fingerprints,
    }


def _package_versions() -> dict[str, str]:
    """Return versions for packages that commonly affect research outputs."""
    versions: dict[str, str] = {}
    for package in PACKAGE_FINGERPRINTS:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _selected_env() -> dict[str, str]:
    """Return safe environment variables that influence research behavior."""
    selected: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        allowed = key in ENV_NAMES_TO_RECORD or any(key.startswith(prefix) for prefix in ENV_PREFIXES_TO_RECORD)
        if not allowed:
            continue
        if any(part in key.upper() for part in SECRET_NAME_PARTS):
            selected[key] = "<redacted>"
        else:
            selected[key] = value
    return selected


def build_manifest(
    *,
    validation: dict[str, Any],
    output_paths: list[Path],
    data_dir: Path,
    command: str,
    root: Path,
    max_data_files: int,
) -> dict[str, Any]:
    """Build a reproducibility manifest for the research outputs."""
    now = datetime.now(timezone.utc)
    local_now = datetime.now().astimezone()
    return {
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "generated_at_local": local_now.isoformat(),
        "command": command,
        "cwd": str(root.resolve()),
        "git": {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "branch": _run_git(["branch", "--show-current"]),
            "origin_main": _run_git(["rev-parse", "origin/main"]),
            "dirty": bool(_run_git(["status", "--short"])),
            "status_short": _run_git(["status", "--short"]).splitlines(),
        },
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "packages": _package_versions(),
        "environment": _selected_env(),
        "random_seed": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "NUMPY_RANDOM_SEED": os.environ.get("NUMPY_RANDOM_SEED", ""),
            "RANDOM_SEED": os.environ.get("RANDOM_SEED", ""),
        },
        "validation": validation,
        "outputs": {
            "files": _fingerprint_outputs(output_paths, root=root),
        },
        "input_data": _fingerprint_data_dir(data_dir, root=root, max_files=max_data_files),
    }


def _print_validation_report(report: dict[str, Any]) -> None:
    """Print a short terminal summary."""
    print("\nRESEARCH OUTPUT GUARD")
    print("=" * 72)
    print(f"Status:  {str(report.get('status')).upper()}")
    print(f"Checked: {report.get('checked_count')} files")
    print(f"Failed:  {report.get('failed_count')} files")
    for row in report.get("files", []):
        status = str(row.get("status", "unknown")).upper()
        print(f"  {status:<4} {row.get('path')}")
        for issue in row.get("issues", [])[:5]:
            print(f"       - {issue}")


def main() -> int:
    """CLI entry point for local and GitHub Actions validation."""
    parser = argparse.ArgumentParser(
        description="Validate generated research JSON/CSV outputs and write a reproducibility manifest.",
    )
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="Generated output path to validate. Defaults to the tracked feature/research outputs.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write signals/research_run_manifest.json after validation.",
    )
    parser.add_argument(
        "--manifest-path",
        default="signals/research_run_manifest.json",
        help="Where to write the manifest when --write-manifest is enabled.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing parquet research inputs to fingerprint.",
    )
    parser.add_argument(
        "--max-data-files",
        type=int,
        default=250,
        help="Maximum parquet files to checksum in the manifest.",
    )
    parser.add_argument(
        "--command",
        default=" ".join(sys.argv),
        help="Research command/workflow that produced the outputs.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    output_paths = [Path(p) for p in args.paths] if args.paths else list(DEFAULT_RESEARCH_OUTPUTS)
    validation = validate_research_outputs(output_paths)
    _print_validation_report(validation)

    if args.write_manifest:
        manifest = build_manifest(
            validation=validation,
            output_paths=output_paths,
            data_dir=Path(args.data_dir),
            command=str(args.command),
            root=root,
            max_data_files=max(0, int(args.max_data_files)),
        )
        manifest_path = Path(args.manifest_path)
        atomic_write_json(manifest, manifest_path)
        print(f"\nManifest written: {manifest_path}")

    if validation.get("status") != "pass":
        print("\nGuard failed. Fix or regenerate the listed files before committing/pulling this research state.")
        return 1
    print("\nGuard passed. Research outputs are parseable and conflict-marker free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
