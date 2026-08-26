"""Build a safe project bundle for running a walk-forward in Google Colab.

The bundle contains the current local code, market data, model files, and
research signals. It deliberately leaves out secrets, logs, Git history,
virtual environments, and Python cache files.
"""

from __future__ import annotations

# argparse turns command-line options into Python values.
import argparse
# fnmatch compares file names with simple patterns such as "*.pyc".
from fnmatch import fnmatch
# Path makes file and folder paths easier to work with on every platform.
from pathlib import Path
# zipfile creates the compressed archive that the user uploads to Google Drive.
from zipfile import ZIP_DEFLATED, ZipFile


# These whole folders are local-only or can be recreated automatically.
EXCLUDED_FOLDERS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "logs",
    "node_modules",
    "venv",
}

# These file patterns may contain secrets or are temporary computer files.
EXCLUDED_FILE_PATTERNS = (
    ".DS_Store",
    ".env",
    ".env.*",
    "*.lock",
    "*.log",
    "*.pyc",
    "*.pyo",
    "*credentials*.json",
    "*secret*.json",
    "access_token.json",
    "oauth_token.json",
    "refresh_token.json",
    "token.json",
    "tokens.json",
)


def should_include(path: Path, project_root: Path, output_path: Path) -> bool:
    """Return True when a project file is safe and useful for Colab.

    PLAIN ENGLISH: The walk-forward needs code and research data. It does not
    need passwords, old logs, Git history, or a local Python environment.
    """
    # Never put the archive inside itself when the output is under the project.
    if path.resolve() == output_path.resolve():
        return False

    relative_path = path.relative_to(project_root)
    if any(part in EXCLUDED_FOLDERS for part in relative_path.parts):
        return False

    return not any(fnmatch(path.name, pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def build_bundle(project_root: Path, output_path: Path) -> tuple[int, int]:
    """Create the zip archive and return its file count and uncompressed size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    source_bytes = 0

    # ZIP_DEFLATED makes large Parquet/model files smaller for the Drive upload.
    with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(project_root.rglob("*")):
            if not path.is_file() or not should_include(path, project_root, output_path):
                continue

            # Keep one top-level project folder so extraction is predictable.
            archive_name = Path(project_root.name) / path.relative_to(project_root)
            archive.write(path, arcname=archive_name)
            file_count += 1
            source_bytes += path.stat().st_size

    return file_count, source_bytes


def main() -> int:
    """Read options, build the archive, and print the exact upload location."""
    project_root = Path(__file__).resolve().parent
    default_output = project_root.parent / "Stock_Market_AI_Bot_Colab.zip"

    parser = argparse.ArgumentParser(
        description="Package the current Stock Market AI Bot for Google Colab."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Archive path (default: {default_output})",
    )
    args = parser.parse_args()
    output_path = args.output.expanduser().resolve()

    file_count, source_bytes = build_bundle(project_root, output_path)
    archive_bytes = output_path.stat().st_size
    mib = 1024 * 1024
    print("Colab bundle ready")
    print(f"  file: {output_path}")
    print(f"  files included: {file_count}")
    print(f"  source size: {source_bytes / mib:.1f} MiB")
    print(f"  bundle size: {archive_bytes / mib:.1f} MiB")
    print("  secrets and local logs were excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
