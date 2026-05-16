"""safe_io.py — Crash-safe file writing utilities.

PLAIN ENGLISH: When you write a file, if the program crashes MID-WRITE (power
outage, kill signal, out-of-memory), the file ends up half-written and corrupt.
This module solves that by writing to a temporary file first, then atomically
renaming it over the target.  On Unix, `os.replace()` is atomic — the file is
either fully old or fully new, never half-and-half.

HOW TO USE:
    from safe_io import atomic_write_text, atomic_write_csv

    atomic_write_text(path, json.dumps(data, indent=2))
    atomic_write_csv(df, path)

KEY CONCEPTS:
  - Atomic operation: happens all-at-once, never partially.  If the machine
    crashes during the rename, the old file is still intact.
  - Temp file: we write to "myfile.csv.tmp" then rename to "myfile.csv".
    If the crash happens during the write, only the .tmp file is corrupt —
    the real file is untouched.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd


def atomic_write_text(path: Union[str, Path], content: str, *, encoding: str = "utf-8") -> None:
    """Write text to file atomically (write .tmp then rename).

    PLAIN ENGLISH: Writes your content to a temporary file next to the target,
    then renames the temp file over the target.  This guarantees the target file
    is never half-written.
    """
    path = Path(path)
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(content, encoding=encoding)
        os.replace(str(tmp_path), str(path))  # atomic on POSIX
    except BaseException:
        # Clean up temp file if something goes wrong
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_csv(df: pd.DataFrame, path: Union[str, Path], *, index: bool = False, **kwargs) -> None:
    """Write a DataFrame to CSV atomically (write .tmp then rename).

    PLAIN ENGLISH: Same idea as atomic_write_text but for pandas DataFrames.
    Writes to a .tmp file first, then renames it over the real file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp_path, index=index, **kwargs)
        os.replace(str(tmp_path), str(path))  # atomic on POSIX
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(data: object, path: Union[str, Path], *, indent: int = 2) -> None:
    """Write JSON to file atomically.

    PLAIN ENGLISH: Convenience wrapper — serializes your data to JSON and
    writes it atomically.
    """
    import json
    content = json.dumps(data, indent=indent, default=str)
    atomic_write_text(path, content)
