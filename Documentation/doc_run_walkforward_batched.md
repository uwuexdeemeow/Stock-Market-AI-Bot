# run_walkforward_batched.py — Auto-restart wrapper for the walkforward

## What it does (plain English)

The nested walkforward script (`core_satellite_nested_walkforward.py`)
processes ~14 yearly "outer folds" inside ONE long-running Python
process.  Each fold leaks 1–2 GB of memory that we can't easily
reclaim — by the time the script reaches fold 12 or so, the main
process is ~24 GB on a 31 GB laptop and starts paging the active
heap to disk, which makes the whole machine unresponsive.

This wrapper sidesteps the leak by running ONE fold per Python
process.  Every time the inner walkforward finishes a fold, the
wrapper restarts it as a fresh subprocess.  Fresh interpreter =
fresh heap = leak reset.  The walkforward's checkpoint mechanism
(`signals/walkforward_checkpoint_core_alpha.json`) carries state
between subprocesses, so each one just resumes from the prior.

## Why it exists

Two real fixes were considered and rejected for now:

1. **Find and fix the leak in `alpha_factor_backtest.py`.**  The
   PerformanceWarning about "DataFrame is highly fragmented" hints
   at where the leak lives (repeated `frame.insert` calls leaving
   stale buffers behind), but a clean fix needs a memory-profiling
   pass through several hundred lines of scoring code.  Not in
   scope tonight.
2. **Increase the machine's RAM.**  Even 64 GB only delays the
   problem — the leak still grows per fold, so a longer walkforward
   would eventually hit the same wall.

A subprocess wrapper is simpler, has no risk of breaking the
strategy logic, and makes the walkforward usable on memory-
constrained boxes today.

## How to run

```bash
# Default: 1 fold per subprocess (most memory-safe)
python run_walkforward_batched.py --recent-alpha-grid

# Forward any inner-script flag exactly as you would to the inner
# script.  --resume and --exit-after-folds are added automatically
# if you don't pass them yourself.
python run_walkforward_batched.py --recent-alpha-grid --workers 1

# Two folds per subprocess (faster overall, more memory per process)
python run_walkforward_batched.py --recent-alpha-grid --batch-size 2

# Safety: by default the wrapper will stop after 20 subprocess
# restarts.  If you have a non-default fold count, override:
python run_walkforward_batched.py --stable-grid --max-batches 30
```

## Inputs

| File | Source | Purpose |
|------|--------|---------|
| `signals/walkforward_checkpoint_core_alpha.json` | inner walkforward | Per-fold progress preserved across restarts |
| The inner script's data inputs (panels, ETF parquets, etc.) | research/refresh scripts | Inner script handles loading |

## Outputs

Same as the inner walkforward:

| File | What's in it |
|------|--------------|
| `signals/core_satellite_nested_walkforward.json` | Final per-fold + aggregate result.  Only written when the whole run finishes — partial runs leave this stale. |
| `signals/core_satellite_nested_walkforward.csv` | Same, in CSV form. |
| `signals/core_satellite_live_configs.json` | Promoted live config (only on approval). |

## Key concepts

- **Subprocess** — a separate operating-system process started from
  another process.  Each subprocess gets its own memory and dies
  independently.  We use one subprocess per fold so that memory
  leaks in any single fold can't carry into the next.
- **Checkpoint** — a small JSON file the walkforward writes after
  each fold.  When the wrapper restarts the inner script, it reads
  this file and skips the folds that are already done.
- **Batch size** — how many folds the wrapper asks the inner script
  to process per subprocess.  Larger batches finish faster (less
  startup overhead) but use more memory per subprocess.  Default 1
  is the safest.
- **Exit code** — a number a process returns when it exits.  0 means
  "success", non-zero means "something went wrong".  The wrapper
  stops the loop and returns the failure code if any subprocess
  exits with a non-zero code.

## How the wrapper stops

The loop ends when ANY of these happens:

1. **Walkforward completed.**  The inner script deletes its
   checkpoint on a successful full-run completion, so when the
   wrapper sees the checkpoint file missing, it knows we're done
   and exits 0.
2. **A subprocess exited with an error.**  The wrapper does NOT
   retry — restarting a broken command in a fresh process won't
   fix it.  Returns the subprocess's exit code.
3. **A batch made no progress.**  If the checkpoint's
   `completed_years` list didn't grow, something stopped the
   walkforward mid-fold.  Loop exits with code 1 instead of
   spinning forever.
4. **`--max-batches` reached.**  Default 20.  Defensive ceiling so
   a bug somewhere can't run subprocesses forever.

## Resume / interrupt behavior

Hit Ctrl+C while the wrapper is running and the wrapper passes that
to the inner subprocess.  The inner script catches it, writes a
final checkpoint update, and exits.  Re-running the wrapper picks
up exactly where you left off — no work is lost.
