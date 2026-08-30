"""Restore paper evidence from a verified GitHub Actions artifact.

PLAIN ENGLISH: This command downloads or opens a completed evidence bundle,
checks every saved checksum, backs up current local files, and restores only
listed ``signals/`` and ``logs/`` files. It never imports broker code and never
submits, cancels, or modifies an Alpaca order.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from safe_io import atomic_write_text


PROJECT_ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    """Calculate one artifact checksum."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_manifest(root: Path) -> Path:
    """Find exactly one complete paper-run manifest below an artifact root."""
    matches = list(root.rglob("paper_run_manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paper_run_manifest.json, found {len(matches)}")
    return matches[0]


def _latest_successful_github_artifact(destination: Path) -> Path:
    """Use GitHub CLI to download the latest successful daily-run artifacts."""
    query = subprocess.run(
        [
            "gh", "run", "list", "--workflow", "daily_paper_trading.yml",
            "--status", "success", "--limit", "1", "--json", "databaseId",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(query.stdout)
    if not rows:
        raise RuntimeError("No successful Daily Paper Trading workflow run found")
    run_id = str(rows[0]["databaseId"])
    subprocess.run(
        ["gh", "run", "download", run_id, "--dir", str(destination)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return destination


def restore_artifact(
    artifact_root: Path,
    *,
    destination_root: Path = PROJECT_ROOT,
    dry_run: bool = False,
) -> dict:
    """Validate and restore only checksum-listed evidence files."""
    manifest_path = _find_manifest(artifact_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Artifact manifest is not complete")
    source_base = manifest_path.parent.parent if manifest_path.parent.name == "signals" else manifest_path.parent
    verified: list[tuple[Path, Path]] = []
    for entry in (manifest.get("files", {}) or {}).values():
        relative = str(entry.get("relative_path", "")).replace("\\", "/")
        if not relative or not relative.startswith(("signals/", "logs/")):
            continue
        relative_path = Path(relative)
        if ".." in relative_path.parts:
            raise RuntimeError(f"Unsafe artifact path: {relative}")
        source = source_base / relative_path
        if not source.is_file():
            if entry.get("required"):
                raise RuntimeError(f"Required artifact file missing: {relative}")
            continue
        expected = str(entry.get("sha256", ""))
        if not expected or _sha256(source) != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        destination = (destination_root / relative_path).resolve()
        allowed_roots = {
            (destination_root / "signals").resolve(),
            (destination_root / "logs").resolve(),
        }
        if not any(root == destination.parent or root in destination.parents for root in allowed_roots):
            raise RuntimeError(f"Restore target escaped allowed directories: {destination}")
        verified.append((source, destination))

    backup_root = destination_root / "archive" / "recovery_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    restored: list[str] = []
    if not dry_run:
        for source, destination in verified:
            relative = destination.relative_to(destination_root)
            if destination.exists():
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup)
            # Use the project's atomic writer so interruption cannot create a
            # half-restored JSON or CSV.
            atomic_write_text(destination, source.read_text(encoding="utf-8"))
            restored.append(str(relative).replace("\\", "/"))
    return {
        "status": "validated" if dry_run else "restored",
        "run_id": manifest.get("run_id"),
        "verified_files": len(verified),
        "restored_files": restored,
        "backup_path": str(backup_root) if not dry_run else "",
        "orders_submitted": False,
    }


def main() -> int:
    """Parse recovery options and print a concise audit result."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", type=Path, help="Downloaded artifact directory.")
    source.add_argument("--from-github", action="store_true", help="Download the latest successful daily artifact with gh.")
    parser.add_argument("--dry-run", action="store_true", help="Verify only; write no files.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    args = parser.parse_args()

    if args.from_github:
        with tempfile.TemporaryDirectory(prefix="stockbot-recovery-") as temp:
            result = restore_artifact(_latest_successful_github_artifact(Path(temp)), dry_run=args.dry_run)
    else:
        result = restore_artifact(args.artifact.resolve(), dry_run=args.dry_run)
    print(json.dumps(result, indent=2) if args.json else f"Recovery {result['status']}: {result['verified_files']} verified, orders submitted=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
