"""
memprofile_walkforward.py — Quantify the nested-walkforward memory leak.

PLAIN ENGLISH:
The walkforward main process grows ~1.5-2 GB per outer fold and OOMs
the laptop around fold 12 on 32 GB.  The leak is real but spread
across pandas / numpy / our own scoring code, and we've been working
around it with subprocess restarts instead of finding the source.

This profiler reproduces the leak in minutes instead of hours by
skipping the inner fold/cost-stress loop and just hammering
`evaluate_window` on the same window N times.  Each iteration is
SUPPOSED to be a no-op against memory (identical inputs, identical
allocations, identical frees).  If working-set grows linearly with
iteration count, the leak is in `evaluate_window` itself (or its
callees like `run_core_satellite`).

Output:
  - per-iteration working-set size (RSS)
  - `tracemalloc` top 10 allocators by net allocated bytes
  - a "leaks_per_iter_mb" headline number for triage

Usage:
    python memprofile_walkforward.py                 # 30 iterations, default config
    python memprofile_walkforward.py --iters 50      # more iterations = more signal
    python memprofile_walkforward.py --window 2022   # use a different outer year
    python memprofile_walkforward.py --no-tracemalloc # skip the slow allocator trace
"""
from __future__ import annotations

# tracemalloc is the stdlib memory profiler.  When started, it
# attaches a Python-frame stack to every allocation, so we can see
# which lines are responsible for retained memory growth.
# It's slow (~3x runtime overhead) but precise.
import tracemalloc
# argparse for CLI flags.
import argparse
# gc lets us force a collection between iterations so we measure
# *real* leaks, not just allocations awaiting collection.
import gc
# os/sys/time for housekeeping (RSS lookup, paths, timing).
import os
import sys
import time
# Path for nicer path handling.
from pathlib import Path

import pandas as pd

# We'll import the walkforward module on demand to keep argparse fast.
# These two helpers come from the same module.
from core_satellite_nested_walkforward import (
    evaluate_window,
    iter_candidate_configs,
)
# Panel-loading helpers from alpha_factor_backtest.  Same code the
# real walkforward uses, so the leak (if any) will reproduce.
from alpha_factor_backtest import (
    attach_scores,
    load_factor_panel,
    load_feature_specs,
    load_prediction_scores,
)
import core_satellite_alpha as core


# Optional psutil for accurate RSS sampling on every OS.  If psutil
# isn't installed we fall back to a Windows-specific ctypes path or
# os-level resource info.  This lets the profiler work without adding
# a hard dependency.
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def _rss_mb() -> float:
        return _PROC.memory_info().rss / (1024 ** 2)
except ImportError:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        _psapi = ctypes.WinDLL("psapi.dll")

        def _rss_mb() -> float:
            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            _psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            return counters.WorkingSetSize / (1024 ** 2)
    else:
        import resource
        def _rss_mb() -> float:
            # ru_maxrss is in KB on Linux, bytes on macOS.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return rss / (1024 ** 2)
            return rss / 1024


def _build_panel():
    """Load + attach scores the same way the real walkforward does."""
    print("[setup] loading feature specs...")
    specs = load_feature_specs()
    print("[setup] loading factor panel...")
    panel = load_factor_panel(specs)
    print("[setup] attach_scores (this allocates the fragmented frame)...")
    panel = attach_scores(panel, specs, load_prediction_scores())
    print("[setup] _ensure_robust_score_columns...")
    panel = core._ensure_robust_score_columns(panel)
    panel["_date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.loc[panel["_date"].notna()].sort_values("_date").reset_index(drop=True)
    for column in panel.select_dtypes(include=["float64"]).columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").astype("float32")
    # The defragment .copy() we just landed in fix 1b65c5c — keep it
    # so we measure the leak on top of our existing partial fix.
    print("[setup] defragmenting via panel.copy()...")
    panel = panel.copy()
    gc.collect()
    return panel


def _pick_one_config():
    """Return one canonical config (no grid) for repeated evaluation."""
    configs = list(iter_candidate_configs(
        strategy="core-alpha",
        holding_days=(20,),
        overlay_gross=(0.5,),
        ma_windows=(100,),
        high_vol_values=(0.30,),
        high_vol_modes=("percentile",),
        score_sources=("regime_adaptive",),
        shapes=("top3",),
        weightings=("sticky_score",),
        tqqq_weights=(0.0,),
        risk_control_modes=("off",),
    ))
    return configs[0]


def _format_top_allocators(snapshot_after, snapshot_before, limit: int = 10) -> str:
    """Compare two tracemalloc snapshots and format the top N diffs."""
    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")[:limit]
    lines = []
    for idx, stat in enumerate(top_stats, 1):
        # Stat carries a `traceback` of one frame (since key_type="lineno").
        # Use the first frame so we get filename:lineno.
        frame = stat.traceback[0]
        filename = Path(frame.filename).name
        size_mb = stat.size_diff / (1024 ** 2)
        count = stat.count_diff
        lines.append(
            f"  {idx:>2}. {size_mb:+8.2f} MB  ({count:+d} blocks)  "
            f"{filename}:{frame.lineno}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=30,
                        help="Number of evaluate_window calls to run (default 30).")
    parser.add_argument("--window", type=int, default=2024,
                        help="Outer-year window to evaluate over (default 2024).")
    parser.add_argument("--no-tracemalloc", action="store_true",
                        help="Skip the tracemalloc instrumentation (faster).")
    parser.add_argument("--gc-every", type=int, default=1,
                        help="Force gc.collect() every N iterations (default 1).")
    args = parser.parse_args()

    print("=" * 72)
    print(" memprofile_walkforward — leak reproduction harness")
    print("=" * 72)

    # ── Set up panel + config ─────────────────────────────────────────
    rss_before_panel = _rss_mb()
    print(f"[setup] starting RSS: {rss_before_panel:.0f} MB")

    panel = _build_panel()

    rss_after_panel = _rss_mb()
    print(f"[setup] after panel load: {rss_after_panel:.0f} MB "
          f"(panel cost ~{rss_after_panel - rss_before_panel:.0f} MB)")

    config = _pick_one_config()
    start = pd.Timestamp(f"{args.window}-01-01")
    end = pd.Timestamp(f"{args.window}-12-31")
    # Add the cost_stress field evaluate_window expects.
    config = dict(config)
    config["cost_stress"] = 2.0

    # Warm-up call.  The first evaluate_window pays one-time setup
    # costs we don't want to count as leak (regime indicator caches,
    # date parsing JITs, etc).  Run once, GC, then start measuring.
    print(f"\n[warm-up] one untimed evaluate_window call to settle caches...")
    _ = evaluate_window(panel, config, start, end)
    gc.collect()
    rss_warm = _rss_mb()
    print(f"[warm-up] RSS after warm-up: {rss_warm:.0f} MB "
          f"(grew {rss_warm - rss_after_panel:+.0f} MB during warm-up)")

    # ── tracemalloc snapshot before the loop ─────────────────────────
    snapshot_before = None
    if not args.no_tracemalloc:
        print("[trace] starting tracemalloc (slow but precise)...")
        tracemalloc.start(25)  # keep 25 frames per allocation
        gc.collect()
        snapshot_before = tracemalloc.take_snapshot()

    # ── Hot loop ──────────────────────────────────────────────────────
    print(f"\n[loop] running {args.iters} evaluate_window calls "
          f"(gc.collect every {args.gc_every} iter)...")
    print(f"{'iter':>5} {'rss_mb':>10} {'delta_mb':>10} {'elapsed_s':>10}")

    rss_samples: list[float] = []
    loop_started = time.time()
    last_rss = rss_warm
    for i in range(1, args.iters + 1):
        _ = evaluate_window(panel, config, start, end)
        if i % args.gc_every == 0:
            gc.collect()
        rss = _rss_mb()
        delta = rss - last_rss
        elapsed = time.time() - loop_started
        rss_samples.append(rss)
        print(f"{i:>5} {rss:>10.0f} {delta:>+10.0f} {elapsed:>10.1f}")
        last_rss = rss

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n[summary]")
    print(f"  iterations:    {args.iters}")
    print(f"  start RSS:     {rss_warm:.0f} MB (post warm-up)")
    print(f"  end RSS:       {rss_samples[-1]:.0f} MB")
    print(f"  total growth:  {rss_samples[-1] - rss_warm:+.0f} MB")
    print(f"  leak per iter: {(rss_samples[-1] - rss_warm) / args.iters:+.1f} MB/iter")
    print(f"  loop elapsed:  {time.time() - loop_started:.1f}s")

    # ── tracemalloc diff after the loop ──────────────────────────────
    if snapshot_before is not None:
        gc.collect()
        snapshot_after = tracemalloc.take_snapshot()
        print(f"\n[trace] top 15 allocators by NET retained bytes:")
        print(_format_top_allocators(snapshot_after, snapshot_before, limit=15))
        tracemalloc.stop()

    # ── Verdict ──────────────────────────────────────────────────────
    leak_per_iter_mb = (rss_samples[-1] - rss_warm) / args.iters
    print(f"\n[verdict]")
    if leak_per_iter_mb < 1.0:
        print(f"  ✓ Per-iter growth is below 1 MB.  Leak is likely "
              f"acceptable for short runs.")
        return 0
    if leak_per_iter_mb < 5.0:
        print(f"  ⚠ Per-iter growth is {leak_per_iter_mb:.1f} MB.  "
              f"Significant but tolerable on a 32 GB box for a single fold.")
        return 0
    print(f"  ✗ Per-iter growth is {leak_per_iter_mb:.1f} MB.  This "
          f"matches the OOM-around-fold-12 pattern we see in production.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
