"""
run_walkforward_batched.py — drive the nested walkforward one fold at a time.

PLAIN ENGLISH:
The nested walkforward script (`core_satellite_nested_walkforward.py`)
has a memory leak — the longer it runs, the more RAM it uses, until
eventually the laptop runs out of memory and either freezes or kills
the process.  We can't fix the leak right now (it's spread across
many files and would take a real engineering pass), but we CAN work
around it: every fold of the walkforward is a self-contained piece
of work, and the checkpoint mechanism lets us pick up where we left
off in a fresh Python process.

This script automates that.  It:

  1. Launches `core_satellite_nested_walkforward.py` with
     `--exit-after-folds 1` — that flag makes the script process ONE
     outer fold then exit cleanly, saving its checkpoint.
  2. When the subprocess exits, this script checks the checkpoint to
     see how many folds are done.
  3. If more folds remain, it re-launches the subprocess.  Each
     re-launch starts with a fresh Python interpreter, which means
     the memory leak resets to zero.
  4. When the checkpoint disappears (the walkforward deletes it after
     a successful full-run completion), or when no progress is made
     in a batch, the script stops.

So instead of running ONE Python process that grows to 30 GB and
crashes, we run ~14 small Python processes that each peak at ~10 GB.

Usage:
    # Equivalent of `python core_satellite_nested_walkforward.py --resume`
    # but auto-restarts after each fold:
    python run_walkforward_batched.py --recent-alpha-grid

    # Pass any flag the inner script accepts; --resume and
    # --exit-after-folds are added automatically if missing.
    python run_walkforward_batched.py --recent-alpha-grid --workers 1

    # Process 2 folds per subprocess instead of 1 (faster but uses
    # more memory per subprocess):
    python run_walkforward_batched.py --recent-alpha-grid --batch-size 2

    # Override the safety limit on subprocess restarts:
    python run_walkforward_batched.py --recent-alpha-grid --max-batches 30
"""
from __future__ import annotations

# `sys` gives us access to the Python executable path (`sys.executable`)
# and the command-line arguments the user typed (`sys.argv`).  Using
# sys.executable rather than hard-coding "python" makes the script
# work the same in conda envs, venvs, etc.
import sys
# `argparse` parses command-line flags.  We use it for the wrapper's
# own knobs (--batch-size, --max-batches) and forward everything else
# to the inner walkforward script.
import argparse
# `json` reads the checkpoint file to count completed folds.
import json
# `pathlib.Path` is a cleaner replacement for os.path.* string juggling.
from pathlib import Path
# `time` lets us measure how long each batch takes so the user sees
# progress reporting.
import time

from safe_io import run_utf8


# Where the walkforward writes its per-fold checkpoint file.  This
# matches the path constructed inside the walkforward script
# (see `_ckpt_path()` in core_satellite_nested_walkforward.py).
CHECKPOINT_PATH = Path("signals") / "walkforward_checkpoint_core_alpha.json"

# The inner script we're going to launch in a subprocess.
INNER_SCRIPT = "core_satellite_nested_walkforward.py"

# Safety limit: even if something goes wrong with progress detection
# we don't want to spin in an infinite restart loop.  14 outer folds
# is the maximum for the current panel, so 20 is generous.
DEFAULT_MAX_BATCHES = 20


def _checkpoint_completed_years() -> list[int]:
    """Read the checkpoint and return the list of completed outer years.

    Returns [] if the file is missing (which happens when the
    walkforward has fully completed and cleaned up).  Returns []
    if the file is corrupt / unparseable — the next subprocess will
    rebuild from scratch.
    """
    if not CHECKPOINT_PATH.exists():
        return []
    try:
        payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return list(payload.get("completed_years", []))


def _build_inner_command(forwarded_args: list[str], batch_size: int) -> list[str]:
    """Construct the argv that launches the inner walkforward subprocess.

    `forwarded_args` is whatever the user typed on the wrapper's command
    line (minus the wrapper's own flags).  We always add `--resume` so
    the subprocess picks up the checkpoint, and we always add
    `--exit-after-folds N` so it exits cleanly after N folds rather
    than running until OOM.

    If the user explicitly passed `--resume`/`--no-resume` or
    `--exit-after-folds` themselves, we honor their value instead of
    duplicating the flag (argparse would error on duplicates).
    """
    cmd = [sys.executable, INNER_SCRIPT]
    cmd.extend(forwarded_args)
    # Add --resume unless the user already controlled it.
    if not any(a == "--resume" or a == "--no-resume" for a in forwarded_args):
        cmd.append("--resume")
    # Add --exit-after-folds unless the user already controlled it.
    if not any(a == "--exit-after-folds" or a.startswith("--exit-after-folds=")
               for a in forwarded_args):
        cmd.extend(["--exit-after-folds", str(batch_size)])
    return cmd


def main() -> int:
    # The wrapper has its own flags AND forwards unknown flags to the
    # inner script.  argparse's parse_known_args() gives us both.
    parser = argparse.ArgumentParser(
        description="Run nested walkforward in fresh-process batches to dodge the per-fold memory leak.",
        add_help=False,  # let the user pass --help to the inner script
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="How many folds to process per subprocess (default 1).  "
             "Higher is faster but uses more memory per subprocess.",
    )
    parser.add_argument(
        "--max-batches", type=int, default=DEFAULT_MAX_BATCHES,
        help=f"Safety limit on subprocess restarts (default {DEFAULT_MAX_BATCHES}).",
    )
    parser.add_argument(
        "-h", "--help-wrapper", action="store_true",
        help="Show this wrapper's help; pass --help directly to see "
             "the inner walkforward script's flags.",
    )
    wrapper_args, forwarded_args = parser.parse_known_args()

    if wrapper_args.help_wrapper:
        parser.print_help()
        return 0

    # ─── Main loop ─────────────────────────────────────────────────
    # Each iteration spawns ONE subprocess that runs ONE batch of folds,
    # then exits.  We watch the checkpoint to detect progress and stop
    # the loop when (a) the walkforward fully finished (checkpoint
    # deleted) or (b) a batch made zero progress (something's wrong).
    started_at = time.time()
    last_completed: list[int] = _checkpoint_completed_years()
    print(f"[batched] starting with {len(last_completed)} fold(s) already in checkpoint: {last_completed}")

    for batch_idx in range(1, wrapper_args.max_batches + 1):
        batch_started_at = time.time()
        cmd = _build_inner_command(forwarded_args, wrapper_args.batch_size)
        print(f"\n[batched] batch {batch_idx}/{wrapper_args.max_batches}: launching subprocess")
        print(f"          cmd: {' '.join(cmd)}")
        # `run_utf8` with no `capture_output` lets the inner
        # script's stdout/stderr go straight to the user's terminal —
        # they see the same logging they'd see without the wrapper.
        # `check=False` because we want to inspect the return code
        # ourselves, not raise on non-zero exit.
        result = run_utf8(cmd, check=False)
        batch_elapsed = time.time() - batch_started_at

        # The subprocess may exit non-zero for legitimate reasons (e.g.
        # the inner script's own argparse rejects a flag).  Surface
        # the failure and stop the loop — no point retrying a bad
        # command in a fresh process.
        if result.returncode != 0:
            print(f"[batched] subprocess exited with code {result.returncode}; stopping.")
            return result.returncode

        # If the walkforward fully completed, it deletes its own
        # checkpoint on success and writes the final JSON/CSV outputs.
        # We're done.
        if not CHECKPOINT_PATH.exists():
            elapsed = time.time() - started_at
            print(f"\n[batched] checkpoint gone — walkforward complete! "
                  f"({batch_idx} batches, {elapsed:.0f}s total wall time)")
            return 0

        # Otherwise, check that the batch made forward progress.  If
        # the completed-years list didn't grow, something is wrong
        # (probably the fold itself crashed before the per-fold save).
        # Stop the loop so we don't burn cycles on a broken state.
        completed_now = _checkpoint_completed_years()
        progress = len(completed_now) - len(last_completed)
        print(f"[batched] batch {batch_idx} finished in {batch_elapsed:.0f}s: "
              f"{len(last_completed)} -> {len(completed_now)} folds "
              f"({progress:+d} this batch)")
        if progress <= 0:
            print("[batched] no progress this batch — stopping to avoid an infinite loop.")
            return 1
        last_completed = completed_now

    print(f"[batched] hit --max-batches={wrapper_args.max_batches}; stopping.  "
          f"{len(last_completed)} folds done.  Increase --max-batches or run "
          f"`python {INNER_SCRIPT} --resume` to finish the rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
