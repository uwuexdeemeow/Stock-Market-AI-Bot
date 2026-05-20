"""safe_io.py — Crash-safe file writing utilities.

PLAIN ENGLISH: When you write a file, if the program crashes MID-WRITE (power
outage, kill signal, out-of-memory), the file ends up half-written and corrupt.
This module solves that by writing to a temporary file first, then atomically
renaming it over the target.  On Unix, `os.replace()` is atomic — the file is
either fully old or fully new, never half-and-half.

Also includes a disk-space guard so writes fail-fast with a clear error
instead of silently truncating when the disk fills up (a real failure mode
on GitHub Actions runners and on smaller VPS instances).

HOW TO USE:
    from safe_io import atomic_write_text, atomic_write_csv

    atomic_write_text(path, json.dumps(data, indent=2))
    atomic_write_csv(df, path)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Union

import pandas as pd


# Minimum free disk space (MiB) required before any atomic write proceeds.
# Env-overridable.  100 MiB is enough for a full parquet write + room for
# the .tmp file during rename.  Set to 0 to disable.
SAFE_IO_MIN_FREE_MIB = int(os.environ.get("SAFE_IO_MIN_FREE_MIB", "100"))


class DiskSpaceLow(OSError):
    """Raised when free disk space is below SAFE_IO_MIN_FREE_MIB."""


def _check_free_space(path: Path) -> None:
    """Fail-fast if the disk holding `path` is nearly full.

    PLAIN ENGLISH: Before any write, peek at how much free space is left
    on the partition that holds the target file.  If we're below the
    safety floor, raise an explicit error instead of starting a write
    that may silently truncate when the disk fills.
    """
    if SAFE_IO_MIN_FREE_MIB <= 0:
        return
    try:
        # shutil.disk_usage works for any path that exists; walk up the
        # tree to find the nearest existing ancestor (the target dir may
        # not exist yet for first-time writes).
        probe = path if path.exists() else path.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(str(probe))
    except OSError:
        # If we can't probe (e.g. weird filesystem), don't block the write.
        return
    free_mib = usage.free / (1024 * 1024)
    if free_mib < SAFE_IO_MIN_FREE_MIB:
        raise DiskSpaceLow(
            f"Refusing to write {path}: only {free_mib:.1f} MiB free on "
            f"{probe}, need at least {SAFE_IO_MIN_FREE_MIB} MiB. "
            f"Free up space or lower SAFE_IO_MIN_FREE_MIB."
        )


def atomic_write_text(path: Union[str, Path], content: str, *, encoding: str = "utf-8") -> None:
    """Write text to file atomically (write .tmp then rename).

    PLAIN ENGLISH: Writes your content to a temporary file next to the target,
    then renames the temp file over the target.  This guarantees the target file
    is never half-written.
    """
    path = Path(path)
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    # Disk-space guard — refuse to start the write if the partition is
    # nearly full instead of silently truncating mid-write.
    _check_free_space(path)
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
    _check_free_space(path)
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


def configure_console_output() -> None:
    """Make Python prints survive older Windows code pages.

    PLAIN ENGLISH: Some Windows terminals use cp1252, which cannot print
    common status symbols. Replacing unsupported characters is better than
    crashing a trading run.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


configure_console_output()


def utf8_subprocess_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return environment variables that keep child Python output UTF-8.

    PLAIN ENGLISH: Parent scripts often read child script output through a
    pipe. This tells child Python processes to write UTF-8 so the parent can
    decode the output consistently on Windows, macOS, and Linux.
    """
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env["PYTHONIOENCODING"] = "utf-8"
    return child_env


def run_utf8(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with UTF-8 text decoding.

    PLAIN ENGLISH: Use this instead of `subprocess.run(..., text=True)` when
    capturing output. Bad bytes become replacement characters instead of a
    UnicodeDecodeError.
    """
    env = kwargs.pop("env", None)
    return subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_subprocess_env(env),
        **kwargs,
    )


def popen_utf8(cmd: list[str], **kwargs) -> subprocess.Popen:
    """Start a subprocess whose text output is decoded as UTF-8 safely."""
    env = kwargs.pop("env", None)
    return subprocess.Popen(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_subprocess_env(env),
        **kwargs,
    )


def check_output_utf8(cmd: list[str], **kwargs) -> str:
    """Return subprocess output decoded as UTF-8 safely."""
    env = kwargs.pop("env", None)
    return subprocess.check_output(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_subprocess_env(env),
        **kwargs,
    )


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
                existing_pid = self.lock_path.read_text(encoding="utf-8", errors="replace").strip() or "unknown"
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
