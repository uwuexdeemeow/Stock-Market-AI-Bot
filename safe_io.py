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


# ─────────────────────────────────────────────────────────────────────────
# Process-level locking
# ─────────────────────────────────────────────────────────────────────────
class PidLockTaken(RuntimeError):
    """Raised when another process already holds the PID lock."""


class PidLock:
    """Cross-platform PID lock for guarding concurrent script execution.

    PLAIN ENGLISH: When you start a script that shouldn't run twice at the
    same time (e.g. signal generation, broker submission), wrap it with this
    lock.  If a second copy starts while the first is still running, the
    second one raises PidLockTaken and exits instead of stomping on the
    first's output.

    Mechanism: opens a file and acquires an OS-level exclusive lock on its
    file descriptor.  The lock is automatically released when the process
    exits — even if it crashes — because the kernel cleans up the file
    descriptor.  This is safer than checking a "pid in a text file"
    approach (which leaves stale locks behind after a crash).

    Usage as context manager:
        with PidLock("logs/signal_gen.lock"):
            generate_signal()

    Usage manually:
        lock = PidLock("logs/signal_gen.lock")
        lock.acquire()
        try:
            ...
        finally:
            lock.release()
    """

    def __init__(self, lock_path: Union[str, Path]):
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def acquire(self) -> None:
        """Try to grab the lock; raise PidLockTaken if held by another process."""
        # Open in 'w' so the file is truncated each time we acquire — the
        # PID we write inside is purely informational (the actual lock is
        # the kernel-level flock on the file descriptor).
        self._fh = open(self.lock_path, "w")
        try:
            import sys as _sys
            if _sys.platform == "win32":
                # Windows: use msvcrt for file locking (no fcntl available)
                import msvcrt
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                # macOS / Linux: use fcntl for file locking
                import fcntl
                fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Lock held — record our PID for humans reading the lock file.
            self._fh.write(str(os.getpid()))
            self._fh.flush()
        except (IOError, OSError) as exc:
            # Read the existing PID for the error message (best effort).
            try:
                existing_pid = self.lock_path.read_text().strip() or "unknown"
            except OSError:
                existing_pid = "unknown"
            self._fh.close()
            self._fh = None
            raise PidLockTaken(
                f"Lock {self.lock_path} already held by PID {existing_pid}"
            ) from exc

    def release(self) -> None:
        """Release the lock (no-op if never acquired)."""
        if self._fh is None:
            return
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None
        # Don't unlink the lock file — leaves the PID readable after exit
        # for debugging, and the kernel's flock auto-released on close.

    def __enter__(self) -> "PidLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
