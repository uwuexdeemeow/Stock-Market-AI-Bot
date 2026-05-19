"""
Proper nested walk-forward validation for the core-satellite alpha strategy.

This is research-only. It does not generate broker orders or paper-trading
signals. The outer loop holds out true unseen years. The inner loop selects
parameters only from data available before that outer test year.
"""

from __future__ import annotations

# ── macOS malloc / fork-safety fix ───────────────────────────────────────────
# Problem: when VS Code (or any IDE) runs a Python script it sometimes *forks*
# its own Python process to create ours.  If the parent had PyArrow / numpy /
# XGBoost threads running at fork-time, the child inherits a corrupted malloc
# heap.  Any subsequent memory allocation (even inside Python's own BytesIO) can
# hit small_free_list_remove_ptr → malloc_zone_error → SIGABRT.
#
# Fix: re-exec this very script with the two env-vars that make malloc
# fork-safe, then exit the dirty process.  The fresh exec'd child starts with a
# completely clean heap.  A sentinel variable ("_WF_FORK_SAFE") breaks the loop.
#
# This must be the ABSOLUTE FIRST code in the file — before any import that
# could touch Arrow, numpy, or any C extension with thread pools.
import os as _os, sys as _sys

_WF_ENTRYPOINTS = {"core_satellite_nested_walkforward.py"}
_WF_IS_CLI = _os.path.basename(_sys.argv[0]) in _WF_ENTRYPOINTS
if _sys.platform == "darwin" and _WF_IS_CLI and not _os.environ.get("_WF_FORK_SAFE"):
    # Carry all current env vars forward, add the two safety flags and the
    # sentinel so the re-exec'd process knows not to loop.
    _env = _os.environ.copy()
    _env["_WF_FORK_SAFE"]                      = "1"
    # MallocNanoZone=0  — disables Apple's NanoZone allocator, which is not
    #     fork-safe when threads are running.  Falls back to the ScalableZone
    #     which handles fork correctly.
    _env["MallocNanoZone"]                     = "0"
    # OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  — suppresses the Obj-C runtime
    #     abort triggered when +initialize ran before fork.
    _env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    # os.execve replaces THIS process image entirely — fresh heap, fresh
    # allocator, same PID.  No return; the line below never executes.
    _os.execve(_sys.executable, [_sys.executable] + _sys.argv, _env)

# Already running in the clean re-exec'd image from here onward.
# Set the ObjC flag as a belt-and-suspenders guard for any child processes
# we subsequently fork for worker pools.
_os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
_os.environ.setdefault("MallocNanoZone", "0")
# Suppress the harmless "MallocStackLogging: can't turn off malloc stack
# logging because it was not enabled" warning that macOS spams on every
# fork().  Redirect stderr briefly during the import phase is too fragile —
# instead just pre-set the var so the C runtime sees it before forking.
_os.environ.setdefault("MallocStackLogging", "0")

# Squelch the MallocStackLogging stderr spam from forked workers by
# redirecting fd 2 to /dev/null for the noisy C-level message, then
# restoring it.  Python-level warnings still flow through logging.
import contextlib as _contextlib, io as _io
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*MallocStackLogging.*")
# ─────────────────────────────────────────────────────────────────────────────

import gc
import multiprocessing as mp
import os
import time

import argparse
import itertools
import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ── Parallel workers for config evaluation ──────────────────────────
# Each worker evaluates one config across all inner folds.  On macOS
# we use fork so the large panel DataFrame is shared via copy-on-write
# (no pickling).  Set WALKFORWARD_WORKERS=1 to disable parallelism.
#
# Memory rule of thumb: each worker needs ~1.5-2 GB headroom for
# panel slices, equity curves, benchmarks, and trade logs.  On a
# 16 GB laptop, 3-4 workers is the safe max.  On 32 GB+, use all cores.

def _auto_detect_workers() -> int:
    """Pick a safe worker count based on available RAM.

    Each forked worker inherits the panel via copy-on-write but creates
    its own DataFrames during evaluate_window().  Later folds (bigger
    training windows) use more memory per worker.  This function caps
    workers so total usage stays under ~80% of physical RAM.
    """
    cpu_workers = max(1, (os.cpu_count() or 1) - 1)
    try:
        if _sys.platform == "win32":
            # Windows: use ctypes to query physical RAM
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = stat.ullTotalPhys / (1024 ** 3)
        elif _sys.platform == "darwin":
            # macOS: sysctl returns total physical RAM in bytes.
            import subprocess
            _clean_env = {k: v for k, v in os.environ.items()
                          if not k.startswith("Malloc")}
            raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                           timeout=5, text=True,
                                           env=_clean_env,
                                           stderr=subprocess.DEVNULL).strip()
            total_gb = int(raw) / (1024 ** 3)
        else:
            # Linux: read /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_gb = int(line.split()[1]) / (1024 ** 2)
                        break
                else:
                    total_gb = 16.0
    except Exception:
        total_gb = 16.0

    # Reserve 4 GB for OS + VS Code + main process, allocate ~2 GB per worker
    usable_gb = max(1, total_gb - 4)
    ram_workers = max(1, int(usable_gb / 2))

    chosen = min(cpu_workers, ram_workers)
    return chosen

_DEFAULT_WORKERS = _auto_detect_workers()
_PARALLEL_WORKERS = int(os.getenv("WALKFORWARD_WORKERS", str(_DEFAULT_WORKERS)))

# Module-level references set before forking so workers inherit them
# via copy-on-write.  Only the main process writes these; workers read.
# On Windows (spawn), these are set by _init_pool_worker in each child.
_SHARED_PANEL: pd.DataFrame | None = None
_SHARED_INNER_FOLDS: list | None = None
_SHARED_PRIOR_SIGS: list[str] | None = None


def _init_pool_worker(panel, inner_folds=None, screen_fold=None, prior_sigs=None):
    """Initializer for spawn-based Pool workers (Windows).

    Called once per worker process at startup.  Sets the module globals
    so worker functions can read them the same way they do on macOS/Linux
    where fork gives copy-on-write access.
    """
    global _SHARED_PANEL, _SHARED_INNER_FOLDS, _SHARED_SCREEN_FOLD, _SHARED_PRIOR_SIGS
    _SHARED_PANEL = panel
    _SHARED_INNER_FOLDS = inner_folds
    _SHARED_SCREEN_FOLD = screen_fold
    _SHARED_PRIOR_SIGS = prior_sigs

from alpha_factor_backtest import (
    attach_scores,
    benchmark_equity,
    compare_to_benchmarks,
    load_factor_panel,
    load_feature_specs,
    load_prediction_scores,
    portfolio_stats,
)
from core_satellite_alpha import (
    ADAPTIVE_EXIT_MODES,
    COST_STRESS_MULTIPLIERS,
    EARNINGS_BLACKOUT_DAY_OPTIONS,
    EXIT_RANK_FLOORS,
    MAX_GROSS_EXPOSURE,
    MAX_SINGLE_NAME_WEIGHT,
    REGIME_PRESETS,
    SCORE_SOURCES,
    SHAPES,
    WEIGHTING_MODES,
    _ensure_robust_score_columns,
    evaluate,
)
from settings import LOG_DIR, SIGNAL_DIR
from robustness_scoring import add_cost_stress_approval_columns, robustness_score_components


BASE_REGIME = "qqq_trend_switch_overlay70_core55_cashbuffer"
# ── Default grid dimensions ───────────────────────────────────────────────
# These define the full search space.  On a laptop (16 GB) the full grid
# should finish in ~1-2 hours with successive halving + parallel workers.
#
# Old grid was 768 configs (2×2×1×2×2×3×2×4×2) — way too many for a
# laptop, took 5+ hours.  The trimmed grid below keeps the dimensions
# that actually matter and drops the ones that rarely change the winner:
#   - tqqq_weights: grid search chose 0.0 every time historically.
#     Keep just (0.0, 0.10) so the data can still say "yes" to TQQQ.
#   - risk_control: "defensive" almost never wins; moved to --full flag.
#   - shapes: top5 vs top10 vs top15 all matter — keep all 3.
DEFAULT_HOLDING_DAYS = (10, 20)
# Overlay gross: the fraction of portfolio in stock-picking overlay.
# 0.25 = conservative (75% core ETFs, 25% picks) — too passive in bull markets
# 0.50 = balanced
# 0.70 = aggressive (55% core, 70% picks) — can actually beat QQQ in momentum years
# The old grid only had 0.25/0.50, which structurally couldn't beat QQQ when it rips.
DEFAULT_OVERLAY_GROSS = (0.25, 0.50, 0.70)
DEFAULT_MA_WINDOWS = (100,)
DEFAULT_HIGH_VOL_VALUES = (0.30,)
DEFAULT_HIGH_VOL_MODES = ("fixed", "percentile")
DEFAULT_TQQQ_WEIGHTS = (0.0, 0.10)
MIN_YEAR_DATES = 20
# ── Unified strategy ──────────────────────────────────────────────────────
# TQQQ used to be its own strategy but underperforms on a risk-adjusted basis
# (grid search chose tqqq_weight=0.0 every time).  Now tqqq_weight is just
# another knob in the core-alpha grid — the data decides if any TQQQ helps.
# "tqqq" is kept as a deprecated alias that maps to "core-alpha" internally.
STRATEGIES = ("core-alpha",)
_STRATEGY_ALIASES = {"tqqq": "core-alpha"}  # backward compat
LIVE_CONFIG_PATH = Path(SIGNAL_DIR) / "core_satellite_live_configs.json"
MEDIUM_RISK_SURVIVORSHIP_PATH = Path(LOG_DIR) / "core_satellite_survivorship_audit.json"
MEDIUM_RISK_EXECUTION_PATH = Path(LOG_DIR) / "core_satellite_execution_stress.json"
MEDIUM_RISK_FACTOR_DECAY_PATH = Path(LOG_DIR) / "factor_decay_monitor.json"
DEFAULT_OUTPUT_PREFIX = "core_satellite_nested_walkforward"
DEFAULT_MAX_SPECS = 48
DEFAULT_MIN_TRAIN_YEARS = 3
# Hard cap on mean inner-fold turnover. Configs above this are rejected.
# Raised from 400→600 because ov=0.70 configs naturally churn more and
# the SOFT penalty in robustness_scoring.py already handles the economics.
MAX_INNER_MEAN_TURNOVER_PCT = 600.0
# "defensive" adds drawdown circuit breakers and vol targeting.
# Historical walkforward data shows 5/7 years selected defensive —
# it was incorrectly excluded from the default grid.  Now included.
RISK_CONTROL_MODES = ("off", "defensive")
FULL_RISK_CONTROL_MODES = ("off", "defensive")
FULL_TQQQ_WEIGHTS = (0.0, 0.10, 0.20, 0.30)
STABLE_GRID_TQQQ_WEIGHTS = (0.0, 0.10)
STABLE_GRID_SHAPES = ("top5", "top10", "top15")
STABLE_GRID_HIGH_VOL_MODES = ("fixed", "percentile")
RECENT_ALPHA_GRID_SHAPES = ("top3", "top5", "top15")
RECENT_ALPHA_GRID_OVERLAY_GROSS = (0.50, 0.70)
RECENT_ALPHA_GRID_TQQQ_WEIGHTS = (0.0, 0.10)
RECENT_ALPHA_GRID_WEIGHTINGS = ("sticky_score", "risk_parity")
RECENT_ALPHA_GRID_HIGH_VOL_MODES = ("fixed", "percentile")
# Survivorship gate thresholds — tuned for the limited audit data reality:
# Only 5 of 17 known failed tickers have local parquet data.  With a
# 147-ticker universe, random chance alone would select them ~20 times.
# The strategy already has trailing stops (8%) that cap individual-name
# damage.  These thresholds ask: "does adding known failed names
# catastrophically break the strategy?" — not "does it never pick them?"
SURVIVORSHIP_MIN_ADJUSTED_SCORE = 0.50  # retain at least 50% of return
SURVIVORSHIP_MAX_AUDIT_SELECTIONS = 60  # allow up to 60 selections (realistic with 5 failed names × 411 rebals)
SURVIVORSHIP_MIN_RETURN_DELTA_PCT = -5000.0  # absolute return delta (wide, since returns are compounded %)
SURVIVORSHIP_MIN_DRAWDOWN_DELTA_PCT = -5.0  # drawdown can't get >5% worse
EXECUTION_STRESS_MIN_WORST_DRAWDOWN_PCT = -35.0  # allow up to 35% dd under worst stress (delay+25bps)

# ── Checkpoint helpers ────────────────────────────────────────────────────────
# The walkforward can take hours.  After each outer fold we persist the
# completed rows to a small JSON file so a crash or reboot loses at most one
# fold worth of work.  On the next run we reload completed folds and skip them.
def _ckpt_path(strategy: str) -> Path:
    """Return the checkpoint file path for the given strategy."""
    return Path(SIGNAL_DIR) / f"walkforward_checkpoint_{strategy.replace('-', '_')}.json"


def _ckpt_key(strategy: str, min_train_years: int, configs: list, start_year, end_year) -> str:
    """A short fingerprint that changes when the grid or fold range changes.

    If the fingerprint changes we discard the old checkpoint automatically so
    stale progress from a different grid is never mixed in.
    """
    import hashlib, json as _json
    sigs = sorted(config_signature(c) for c in configs)
    blob = _json.dumps({
        "strategy": strategy,
        "min_train_years": min_train_years,
        "configs": sigs,
        "start_year": start_year,
        "end_year": end_year,
    }, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def _save_checkpoint(strategy: str, key: str, fold_rows: list, selected_configs: list, inner_details: list) -> None:
    """Persist completed fold data to disk.  Overwrites the previous checkpoint."""
    import json as _json
    path = _ckpt_path(strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "strategy": strategy,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "completed_years": [r["outer_year"] for r in fold_rows],
        "fold_rows": fold_rows,
        "selected_configs": selected_configs,
        "inner_details": inner_details,
    }
    # Write atomically: tmp file then rename
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _load_checkpoint(strategy: str, key: str) -> dict | None:
    """Return a checkpoint dict if it exists and its key matches, else None."""
    import json as _json
    path = _ckpt_path(strategy)
    if not path.exists():
        return None
    try:
        payload = _json.loads(path.read_text())
    except Exception:
        return None
    if payload.get("key") != key:
        return None  # Grid or settings changed — start fresh
    return payload
# Legacy constants — kept for backward compat, updated to new base values.
MIN_LIVE_APPROVAL_FOLDS = 3
MIN_LIVE_CONFIG_FREQUENCY = 0.50
MIN_LIVE_MEAN_OOS_SHARPE = 0.50
MIN_LIVE_OOS_ALPHA_HIT_RATE = 0.60
BASE_COST_STRESS = float(COST_STRESS_MULTIPLIERS[0])

# ── Strategy-specific approval thresholds ───────────────────────────────────
# Both strategies share tightened base gates.  TQQQ gets stricter drawdown
# and bias limits because 3x leverage amplifies drawdowns and can fake good
# backtests.  Each key maps to a gate checked in approval_status().
_APPROVAL_THRESHOLDS: dict[str, dict[str, float]] = {
    "core-alpha": {
        "min_folds": 3,                          # minimum outer folds for statistical meaning
        "min_config_frequency": 0.20,            # winning config must be selected ≥20% of folds
                                                 # (was 0.30, but 14-fold walkforward picks more
                                                 # diversity — 20% = 3/14 folds is the floor)
        "min_mean_oos_sharpe": 0.50,             # mean OOS Sharpe across folds
        "min_oos_alpha_hit_rate": 0.60,          # fraction of folds with positive alpha
        "max_mean_oos_drawdown_pct": -25.0,      # mean max drawdown across folds (negative %)
        "max_worst_oos_drawdown_pct": -35.0,     # worst single-fold max drawdown
        "max_worst_oos_turnover_pct": 600.0,     # reject one-off churn blowups
        "max_selection_bias_gap_sharpe": 1.50,   # inner-vs-OOS Sharpe gap (overfitting detector)
    },
    "tqqq": {
        "min_folds": 3,
        "min_config_frequency": 0.30,            # winning config must be selected ≥30% of folds
        "min_mean_oos_sharpe": 0.50,
        "min_oos_alpha_hit_rate": 0.60,
        "max_mean_oos_drawdown_pct": -20.0,      # tighter — leverage amplifies drawdowns
        "max_worst_oos_drawdown_pct": -25.0,     # must prove regime switching protects in crashes
        "max_worst_oos_turnover_pct": 600.0,
        "max_selection_bias_gap_sharpe": 1.00,   # tighter — overfitting on leverage is costly
        "min_worst_oos_return_pct": -30.0,       # TQQQ-only: catch crash-year blowups
    },
}


@dataclass(frozen=True)
class FoldSplit:
    outer_year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    inner_train_end: pd.Timestamp
    inner_validation_year: int
    inner_validation_start: pd.Timestamp
    inner_validation_end: pd.Timestamp
    outer_start: pd.Timestamp
    outer_end: pd.Timestamp


@dataclass(frozen=True)
class InnerFold:
    validation_year: int
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def available_years(panel: pd.DataFrame, *, min_dates: int = MIN_YEAR_DATES) -> list[int]:
    date_series = panel["_date"] if "_date" in panel.columns else pd.to_datetime(panel["date"], errors="coerce")
    counts = date_series.dropna().dt.year.value_counts()
    return sorted(int(year) for year, count in counts.items() if int(count) >= min_dates)


def build_fold_splits(
    panel: pd.DataFrame,
    *,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[FoldSplit]:
    years = available_years(panel)
    if not years:
        return []
    first_year = years[0]
    candidate_years = [year for year in years if year >= first_year + min_train_years]
    if start_year is not None:
        candidate_years = [year for year in candidate_years if year >= int(start_year)]
    if end_year is not None:
        candidate_years = [year for year in candidate_years if year <= int(end_year)]

    splits: list[FoldSplit] = []
    for outer_year in candidate_years:
        train_years = [year for year in years if year < outer_year]
        if len(train_years) < min_train_years:
            continue
        inner_validation_year = train_years[-1]
        inner_train_years = [year for year in train_years if year < inner_validation_year]
        if len(inner_train_years) < max(1, min_train_years - 1):
            continue
        splits.append(
            FoldSplit(
                outer_year=outer_year,
                train_start=pd.Timestamp(f"{train_years[0]}-01-01"),
                train_end=pd.Timestamp(f"{train_years[-1]}-12-31"),
                inner_train_end=pd.Timestamp(f"{inner_train_years[-1]}-12-31"),
                inner_validation_year=inner_validation_year,
                inner_validation_start=pd.Timestamp(f"{inner_validation_year}-01-01"),
                inner_validation_end=pd.Timestamp(f"{inner_validation_year}-12-31"),
                outer_start=pd.Timestamp(f"{outer_year}-01-01"),
                outer_end=pd.Timestamp(f"{outer_year}-12-31"),
            )
        )
    return splits


def build_inner_folds(train_years: list[int], *, min_inner_train_years: int) -> list[InnerFold]:
    """Build yearly validation folds strictly inside an outer train period."""
    folds: list[InnerFold] = []
    for idx in range(max(1, int(min_inner_train_years)), len(train_years)):
        validation_year = int(train_years[idx])
        train_end_year = int(train_years[idx - 1])
        folds.append(
            InnerFold(
                validation_year=validation_year,
                train_end=pd.Timestamp(f"{train_end_year}-12-31"),
                validation_start=pd.Timestamp(f"{validation_year}-01-01"),
                validation_end=pd.Timestamp(f"{validation_year}-12-31"),
            )
        )
    return folds


def build_regime_preset_variant(
    *,
    base_preset: dict,
    ma_window: int,
    high_vol: float,
    high_vol_mode: str,
    tqqq_weight: float,
    overlay_gross: float | None = None,
    risk_on_overlay_gross: float | None = None,
) -> dict:
    """Return a non-mutating regime preset with tuned exposure/regime knobs."""
    preset = deepcopy(base_preset)
    preset["ma_window"] = int(ma_window)
    preset["high_vol"] = float(high_vol)
    preset["high_vol_mode"] = str(high_vol_mode)

    overlay_value = overlay_gross if overlay_gross is not None else risk_on_overlay_gross
    if overlay_value is None:
        raise ValueError("overlay_gross is required")
    overlay = float(overlay_value)
    if overlay <= 0 or overlay >= MAX_GROSS_EXPOSURE:
        raise ValueError(f"overlay_gross must be between 0 and {MAX_GROSS_EXPOSURE}: {overlay}")
    preset["risk_on"]["overlay_gross"] = overlay
    preset["risk_on"]["core_gross"] = round(float(MAX_GROSS_EXPOSURE) - overlay, 4)

    # Keep the defensive states inside the same gross envelope while letting
    # the inner loop choose how aggressive the risk-on sleeve should be.
    preset["neutral"]["overlay_gross"] = min(float(preset["neutral"]["overlay_gross"]), overlay)
    preset["neutral"]["core_gross"] = min(
        float(preset["neutral"]["core_gross"]),
        round(float(MAX_GROSS_EXPOSURE) - float(preset["neutral"]["overlay_gross"]), 4),
    )
    preset["risk_off"]["overlay_gross"] = min(float(preset["risk_off"]["overlay_gross"]), overlay)
    preset["risk_off"]["core_gross"] = min(
        float(preset["risk_off"]["core_gross"]),
        round(float(MAX_GROSS_EXPOSURE) - float(preset["risk_off"]["overlay_gross"]), 4),
    )

    tqqq = float(tqqq_weight)
    if tqqq < 0 or tqqq > 0.50:
        raise ValueError(f"tqqq_weight must be between 0 and 0.50: {tqqq}")
    risk_on_weights = dict(preset["risk_on"]["core_weights"])
    qqq_weight = float(risk_on_weights.get("QQQ", 0.0))
    tqqq = min(tqqq, qqq_weight)
    risk_on_weights["QQQ"] = round(qqq_weight - tqqq, 6)
    risk_on_weights["TQQQ"] = round(tqqq, 6)
    preset["risk_on"]["core_weights"] = risk_on_weights
    for regime in ("neutral", "risk_off"):
        preset[regime]["core_weights"] = {**dict(preset[regime]["core_weights"]), "TQQQ": 0.0}
    return preset


def iter_candidate_configs(
    *,
    strategy: str = "core-alpha",
    holding_days: Iterable[int] = DEFAULT_HOLDING_DAYS,
    overlay_gross: Iterable[float] = DEFAULT_OVERLAY_GROSS,
    ma_windows: Iterable[int] = DEFAULT_MA_WINDOWS,
    high_vol_values: Iterable[float] = DEFAULT_HIGH_VOL_VALUES,
    high_vol_modes: Iterable[str] = DEFAULT_HIGH_VOL_MODES,
    score_sources: Iterable[str] = SCORE_SOURCES,
    shapes: Iterable[str] = SHAPES,
    weightings: Iterable[str] = WEIGHTING_MODES,
    tqqq_weights: Iterable[float] = DEFAULT_TQQQ_WEIGHTS,
    risk_control_modes: Iterable[str] = RISK_CONTROL_MODES,
    max_configs: int | None = None,
) -> list[dict]:
    strategy = str(_STRATEGY_ALIASES.get(strategy, strategy))
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    # tqqq_weight is now a full grid knob for core-alpha — the data decides
    # whether any TQQQ allocation helps on a risk-adjusted basis.
    base = REGIME_PRESETS[BASE_REGIME]
    configs: list[dict] = []
    for hold, overlay, ma_window, high_vol_mode, score_source, shape, weighting, tqqq_weight, risk_control_mode in itertools.product(
        holding_days,
        overlay_gross,
        ma_windows,
        high_vol_modes,
        score_sources,
        shapes,
        weightings,
        tqqq_weights,
        risk_control_modes,
    ):
        hv_options = high_vol_values if str(high_vol_mode) == "fixed" else (base.get("high_vol", 0.30),)
        for high_vol in hv_options:
            preset = build_regime_preset_variant(
                base_preset=base,
                overlay_gross=float(overlay),
                ma_window=int(ma_window),
                high_vol=float(high_vol),
                high_vol_mode=str(high_vol_mode),
                tqqq_weight=float(tqqq_weight),
            )
            risk_on = preset["risk_on"]
            config = {
                "strategy": strategy,
                "core_preset": (
                    f"nested_{BASE_REGIME}_h{hold}_ov{overlay:.2f}_ma{ma_window}"
                    f"_vol{high_vol_mode}{high_vol:.2f}_tqqq{tqqq_weight:.2f}_risk{risk_control_mode}"
                ),
                "regime_mode": BASE_REGIME,
                "regime_preset": preset,
                "regime_ma_window": int(ma_window),
                "regime_high_vol": float(high_vol),
                "high_vol_mode": str(high_vol_mode),
                "score_blend": False,
                "early_rebalance_on_regime_change": False,
                "core_weights": dict(risk_on["core_weights"]),
                "tqqq_preset": "tqqq_enhanced_cashbuffer",
                "score_source": str(score_source),
                "shape": str(shape),
                "weighting": str(weighting),
                "exit_rank_floor": float(EXIT_RANK_FLOORS[0]),
                "adaptive_exit_mode": str(ADAPTIVE_EXIT_MODES[0]),
                "max_per_sector": 2,
                "earnings_blackout_days": int(EARNINGS_BLACKOUT_DAY_OPTIONS[0]),
                "core_gross": float(risk_on["core_gross"]),
                "overlay_gross": float(risk_on["overlay_gross"]),
                "max_gross_exposure": MAX_GROSS_EXPOSURE,
                "max_single_name_weight": MAX_SINGLE_NAME_WEIGHT,
                "holding_days": int(hold),
                "risk_control_mode": str(risk_control_mode),
                "drawdown_circuit_breaker": 0.15 if str(risk_control_mode) == "defensive" else 0.0,
                "vol_target": 0.15 if str(risk_control_mode) == "defensive" else 0.0,
                "nested_params": {
                    "holding_days": int(hold),
                    "overlay_gross": round(float(overlay), 4),
                    "risk_on_overlay_gross": round(float(overlay), 4),
                    "ma_window": int(ma_window),
                    "high_vol": round(float(high_vol), 4),
                    "high_vol_mode": str(high_vol_mode),
                    "score_source": str(score_source),
                    "shape": str(shape),
                    "weighting": str(weighting),
                    "tqqq_weight": round(float(tqqq_weight), 4),
                    "risk_control_mode": str(risk_control_mode),
                },
            }
            configs.append(config)
            if max_configs is not None and len(configs) >= int(max_configs):
                return configs
    return configs


def stable_grid_candidate_configs(
    *,
    strategy: str = "core-alpha",
    max_configs: int | None = None,
) -> list[dict]:
    """Return the consensus-aligned baseline grid for stable approval.

    Pins the 14-fold consensus dimensions (h=20, ma=100, score=regime_adaptive,
    risk=off) AND drops overlay=0.7 — the May 2026 stable-grid run showed it's
    a turnover disaster:
      ov=0.5:  308% mean turnover, 2.03 mean Sharpe, +13.5% mean α vs QQQ
      ov=0.7:  562% mean turnover, 0.95 mean Sharpe,  +2.2% mean α vs QQQ
    The 991% blowup on the 2014 fold was an ov=0.7 fold.

    Also drops tqqq=(0.2, 0.3) which won 0/14 folds and risk=defensive
    which won 0/14 folds.  Keeps the dimensions that actually vary across
    winners: shape, weighting, vol mode, and tqqq=(0.0, 0.1).
    """
    return iter_candidate_configs(
        strategy=strategy,
        holding_days=(20,),
        overlay_gross=(0.50,),
        ma_windows=(100,),
        high_vol_values=(0.30,),
        high_vol_modes=STABLE_GRID_HIGH_VOL_MODES,
        score_sources=("regime_adaptive",),
        shapes=STABLE_GRID_SHAPES,
        weightings=("sticky_score", "risk_parity"),
        tqqq_weights=STABLE_GRID_TQQQ_WEIGHTS,
        risk_control_modes=("off",),
        max_configs=max_configs,
    )


def recent_alpha_grid_candidate_configs(
    *,
    strategy: str = "core-alpha",
    max_configs: int | None = None,
) -> list[dict]:
    """Return the focused post-2020 alpha research grid.

    This grid keeps the dimensions the latest 14-fold run favored while
    leaving only the newer-regime uncertainty open: overlay aggression,
    concentration, weighting, high-vol mode, and a small TQQQ sleeve.
    """
    return iter_candidate_configs(
        strategy=strategy,
        holding_days=(20,),
        overlay_gross=RECENT_ALPHA_GRID_OVERLAY_GROSS,
        ma_windows=(100,),
        high_vol_values=(0.30,),
        high_vol_modes=RECENT_ALPHA_GRID_HIGH_VOL_MODES,
        score_sources=("regime_adaptive",),
        shapes=RECENT_ALPHA_GRID_SHAPES,
        weightings=RECENT_ALPHA_GRID_WEIGHTINGS,
        tqqq_weights=RECENT_ALPHA_GRID_TQQQ_WEIGHTS,
        risk_control_modes=("off",),
        max_configs=max_configs,
    )


# ── Benchmark cache ─────────────────────────────────────────────────────
# benchmark_equity() reads SPY/QQQ parquet from disk and normalises
# them for each equity window.  Within one fold, ALL configs share
# very similar date ranges so the same parquet gets read thousands of
# times.  This module-level cache stores the raw ETF price Series
# (pre-normalisation) so disk I/O only happens once per process.
_BENCH_RAW_CACHE: dict[str, pd.Series] | None = None


def _get_cached_bench_raw(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return benchmark equity using cached raw ETF prices.

    First call loads SPY/QQQ from disk.  Subsequent calls reuse the
    cached Series and just reindex + normalise — no disk I/O.  This
    is ~10x faster than calling benchmark_equity() repeatedly.
    """
    global _BENCH_RAW_CACHE
    if _BENCH_RAW_CACHE is None:
        # First call: load raw prices once and cache them
        from backtest import DATA_DIR
        _BENCH_RAW_CACHE = {}
        for sym in ("SPY", "QQQ"):
            local_path = os.path.join(DATA_DIR, f"{sym}.parquet")
            if os.path.exists(local_path):
                try:
                    df = pd.read_parquet(local_path)
                    df.index = pd.DatetimeIndex(df.index)
                    _BENCH_RAW_CACHE[sym] = df["Close"]
                except Exception:
                    _BENCH_RAW_CACHE[sym] = pd.Series(dtype=float)

    # Reindex and normalise from cache — no disk I/O
    from alpha_factor_backtest import INITIAL_CAPITAL
    out = pd.DataFrame(index=index)
    for sym in ("SPY", "QQQ"):
        raw = _BENCH_RAW_CACHE.get(sym)
        if raw is not None and not raw.empty:
            close = raw.reindex(index).ffill().bfill()
            first = float(close.iloc[0]) if not close.empty else 1.0
            if first != 0.0:
                out[sym] = INITIAL_CAPITAL * close / first
            else:
                out[sym] = INITIAL_CAPITAL
        else:
            out[sym] = INITIAL_CAPITAL
    out["BLEND"] = 0.60 * out["SPY"] + 0.40 * out["QQQ"]
    return out.ffill().bfill()


def evaluate_window(panel: pd.DataFrame, config: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Evaluate a config using only panel rows up to end, then score [start, end].

    This function is called many times by nested CV. It avoids repeated date
    parsing and avoids copying the full panel, which prevents macOS from killing
    long nested runs due to memory pressure.

    Performance notes (optimised for nested walkforward):
    - Uses _get_cached_bench_raw() instead of benchmark_equity() to avoid
      re-reading SPY/QQQ parquet from disk on every call (~10x faster).
    - Calls run_core_satellite() directly instead of evaluate() to skip
      unused work (subperiod_metrics, holdout, gate_metrics, yearly_alpha).
    - Nulls all intermediate DataFrames in finally block to let Python's
      refcount free them immediately without waiting for GC.
    """
    end_ts = pd.Timestamp(end)
    start_ts = pd.Timestamp(start)

    if "_date" in panel.columns:
        eval_panel = panel.loc[panel["_date"] <= end_ts]
    else:
        eval_panel = panel.loc[pd.to_datetime(panel["date"], errors="coerce") <= end_ts]

    metrics = None
    equity = None
    trades = None
    try:
        # ── Routing: use TQQQ backtest engine when tqqq_weight > 0 ────────
        # The unified strategy includes tqqq_weight as a grid knob.  When
        # it's > 0, we need the TQQQ-aware backtest (regime switching with
        # TQQQ in risk_on core).  When it's 0, the standard path is faster.
        params = config.get("nested_params", {})
        tqqq_weight = float(params.get("tqqq_weight", config.get("tqqq_weight", 0.0)))
        if tqqq_weight > 0:
            from core_satellite_tqqq import run_tqqq_backtest

            metrics, equity, trades = run_tqqq_backtest(
                eval_panel,
                tqqq_weight=tqqq_weight,
                cost_stress=float(config.get("cost_stress", BASE_COST_STRESS)),
                holding_days=int(params.get("holding_days", config.get("holding_days", 10))),
                preset_name=str(config.get("tqqq_preset", "tqqq_enhanced_cashbuffer")),
                score_source=str(params.get("score_source", config.get("score_source", "regime_adaptive"))),
                shape=str(config.get("shape", "top5")),
                weighting=str(config.get("weighting", "sticky_score")),
                overlay_gross=float(params.get("overlay_gross", config.get("overlay_gross", 0.70))),
                regime_ma_window=int(params.get("ma_window", config.get("regime_ma_window", 100))),
                regime_high_vol=float(params.get("high_vol", config.get("regime_high_vol", 0.30))),
                high_vol_mode=str(params.get("high_vol_mode", config.get("high_vol_mode", "fixed"))),
                max_per_sector=int(config.get("max_per_sector", 2)),
                earnings_blackout_days=int(config.get("earnings_blackout_days", 0)),
                drawdown_circuit_breaker=float(config.get("drawdown_circuit_breaker", 0.0)),
                quiet=True,
            )
        else:
            # ── Direct call to run_core_satellite instead of evaluate() ──
            # evaluate() computes subperiod_metrics, holdout_comparisons,
            # gate_metrics, yearly_alpha, etc. — NONE of which are used by
            # the nested walkforward.  Calling run_core_satellite directly
            # skips all that work (~30% faster per call).
            from core_satellite_alpha import run_core_satellite
            equity, trades, extra_metrics = run_core_satellite(eval_panel, config)
            metrics = extra_metrics

        eq = equity.loc[equity.index <= end_ts]
        anchor = eq.loc[eq.index < start_ts].tail(1)
        window = eq.loc[(eq.index >= start_ts) & (eq.index <= end_ts)]
        if not anchor.empty:
            window = pd.concat([anchor, window])
        window = window[~window.index.duplicated(keep="last")].sort_index()
        if len(window) < 3:
            raise ValueError(f"Too few equity points in window {start_ts.date()} to {end_ts.date()}: {len(window)}")

        periods_per_year = 252.0 / int(config.get("holding_days", 10))
        stats = portfolio_stats(window, periods_per_year)
        # Use cached benchmark instead of benchmark_equity() — avoids
        # re-reading SPY/QQQ parquet from disk on every call.
        bench = _get_cached_bench_raw(pd.DatetimeIndex(window.index))
        comps = compare_to_benchmarks(window, bench)
        turnover_pct = None
        if not trades.empty and "date" in trades.columns and "turnover" in trades.columns:
            trade_dates = pd.to_datetime(trades["date"], errors="coerce")
            mask = (trade_dates >= start_ts) & (trade_dates <= end_ts)
            turnover_pct = round(float(pd.to_numeric(trades.loc[mask, "turnover"], errors="coerce").fillna(0.0).sum()) * 100.0, 2)

        return {
            **stats,
            "alpha_vs_spy_pct": comps["SPY"]["alpha_pct"],
            "alpha_vs_qqq_pct": comps["QQQ"]["alpha_pct"],
            "alpha_vs_blend_pct": comps["BLEND"]["alpha_pct"],
            "benchmark_blend_return_pct": comps["BLEND"]["benchmark_return_pct"],
            "turnover_pct": turnover_pct if turnover_pct is not None else metrics.get("turnover_pct"),
            "window_start": str(pd.Timestamp(window.index[0]).date()),
            "window_end": str(pd.Timestamp(window.index[-1]).date()),
            "n_equity_points": int(len(window)),
            "full_sample_sharpe_to_window_end": metrics.get("sharpe"),
        }
    finally:
        # ── Aggressively null out every intermediate DataFrame ─────────
        # run_core_satellite() / run_tqqq_backtest() create equity curves,
        # trade logs, benchmarks, etc.  Nulling lets Python's refcount
        # free them immediately instead of waiting for a gc cycle.
        metrics = None
        equity = None
        trades = None
        eval_panel = None  # drop the panel slice reference too


def inner_selection_score(metrics: dict) -> float:
    return float(robustness_score_components(metrics)["robustness_score"])


def config_with_cost_stress(config: dict, cost_stress: float) -> dict:
    """Return a shallow config copy with validation cost attached."""
    out = dict(config)
    out["nested_params"] = dict(config.get("nested_params", {}))
    out["cost_stress"] = float(cost_stress)
    return out


def nested_cost_stress_approval(
    panel: pd.DataFrame,
    config: dict,
    fold: InnerFold,
    eval_cache: dict[str, dict],
    *,
    base_metrics: dict | None = None,
) -> dict:
    """Validate one candidate/fold across required cost-stress levels."""
    rows: list[dict] = []
    for cost in COST_STRESS_MULTIPLIERS:
        cost_config = config_with_cost_stress(config, float(cost))
        cache_key = _eval_cache_key(cost_config, fold.validation_start, fold.validation_end)
        if base_metrics is not None and float(cost) == BASE_COST_STRESS:
            metrics = base_metrics
        elif cache_key in eval_cache:
            metrics = eval_cache[cache_key]
        else:
            metrics = evaluate_window(panel, cost_config, fold.validation_start, fold.validation_end)
            eval_cache[cache_key] = metrics
        rows.append(
            {
                "candidate": "candidate",
                "cost_stress": float(cost),
                "alpha_vs_spy_pct": metrics.get("alpha_vs_spy_pct"),
                "alpha_vs_qqq_pct": metrics.get("alpha_vs_qqq_pct"),
                "alpha_vs_blend_pct": metrics.get("alpha_vs_blend_pct"),
                # Nested validation windows do not have a separate 2023-2026
                # holdout; reuse the same validation-window alpha requirements.
                "holdout_alpha_vs_qqq_pct": metrics.get("alpha_vs_qqq_pct"),
                "holdout_alpha_vs_blend_pct": metrics.get("alpha_vs_blend_pct"),
                "nested_stress_row_gate_pass": bool(float(metrics.get("max_drawdown_pct", -999.0) or -999.0) > -50.0),
            }
        )
    stress = add_cost_stress_approval_columns(
        pd.DataFrame(rows),
        key_cols=["candidate"],
        required_costs=COST_STRESS_MULTIPLIERS,
        row_gate_col="nested_stress_row_gate_pass",
    )
    first = stress.iloc[0]
    return {
        "cost_stress_approval_pass": bool(first.get("robust_cost_stress_pass", False)),
        "cost_stress_summary": {
            "cost_levels": str(first.get("stress_cost_levels", "")),
            "has_required_costs": bool(first.get("stress_has_required_costs", False)),
            "all_gates_pass": bool(first.get("stress_all_gates_pass", False)),
            "min_alpha_vs_spy_pct": float(first.get("stress_min_alpha_vs_spy_pct", 0.0) or 0.0),
            "min_alpha_vs_qqq_pct": float(first.get("stress_min_alpha_vs_qqq_pct", 0.0) or 0.0),
            "min_alpha_vs_blend_pct": float(first.get("stress_min_alpha_vs_blend_pct", 0.0) or 0.0),
        },
    }


def config_signature(config: dict) -> str:
    p = config.get("nested_params", {})
    return (
        f"h={p.get('holding_days')},ov={p.get('overlay_gross')},"
        f"ma={p.get('ma_window')},vol={p.get('high_vol_mode')}:{p.get('high_vol')},"
        f"score={p.get('score_source')},shape={p.get('shape')},"
        f"weighting={p.get('weighting')},tqqq={p.get('tqqq_weight')},"
        f"risk={p.get('risk_control_mode', config.get('risk_control_mode', 'off'))}"
    )


def _stable_family_float(value, default: float = 0.0) -> str:
    """Format small grid floats so family keys match across strings/configs."""
    try:
        number = round(float(value), 4)
    except (TypeError, ValueError):
        number = float(default)
    text = f"{number:.4f}".rstrip("0").rstrip(".")
    return f"{text}.0" if "." not in text else text


def _parse_config_signature(signature: str) -> dict[str, str]:
    """Turn ``k=v,k=v`` config text into a dictionary for family grouping."""
    parsed: dict[str, str] = {}
    for part in str(signature).split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def stable_family_signature_from_config_signature(signature: str) -> str:
    """Group exact configs by the durable choices that should repeat.

    PLAIN ENGLISH: holding days, overlay size, MA window, and volatility mode
    are small tuning knobs.  The family key keeps the bigger behavior choices:
    scoring method, concentration shape, weighting method, risk mode, and
    whether TQQQ is used.  Keeping TQQQ in the key prevents leveraged and
    non-leveraged configs from being counted as the same family.
    """
    parsed = _parse_config_signature(signature)
    return (
        f"score={parsed.get('score')},"
        f"shape={parsed.get('shape')},"
        f"weighting={parsed.get('weighting')},"
        f"risk={parsed.get('risk', 'off')},"
        f"tqqq={_stable_family_float(parsed.get('tqqq', 0.0))}"
    )


def stable_family_signature(config: dict) -> str:
    """Return the stable family key for a candidate config dictionary."""
    return stable_family_signature_from_config_signature(config_signature(config))


def _stable_family_tqqq_weight(family_signature: str) -> float:
    parsed = _parse_config_signature(family_signature)
    try:
        return float(parsed.get("tqqq", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def stable_family_frequency(valid_rows: list[dict]) -> list[dict]:
    """Summarize and rank stable families from valid outer-fold rows."""
    grouped: dict[str, list[dict]] = {}
    for row in valid_rows:
        family = stable_family_signature_from_config_signature(str(row["selected_config"]))
        grouped.setdefault(family, []).append(row)

    summaries: list[dict] = []
    fold_count = max(1, len(valid_rows))
    for family, rows_for_family in grouped.items():
        tqqq_weight = _stable_family_tqqq_weight(family)
        summaries.append(
            {
                "stable_family_signature": family,
                "fold_count": int(len(rows_for_family)),
                "frequency": round(float(len(rows_for_family) / fold_count), 3),
                "uses_tqqq": bool(tqqq_weight > 0.0),
                "tqqq_weight": round(float(tqqq_weight), 4),
                "outer_years": [int(row["outer_year"]) for row in rows_for_family],
                "latest_outer_year": int(max(row["outer_year"] for row in rows_for_family)),
                "mean_oos_cagr_pct": round(float(np.mean([float(row["oos_cagr_pct"]) for row in rows_for_family])), 2),
                "mean_oos_sharpe": round(float(np.mean([float(row["oos_sharpe"]) for row in rows_for_family])), 3),
                "mean_oos_max_drawdown_pct": round(float(np.mean([float(row["oos_max_drawdown_pct"]) for row in rows_for_family])), 2),
                "worst_oos_max_drawdown_pct": round(float(np.min([float(row["oos_max_drawdown_pct"]) for row in rows_for_family])), 2),
                "mean_oos_turnover_pct": round(float(np.mean([float(row["oos_turnover_pct"]) for row in rows_for_family])), 2),
                "worst_oos_turnover_pct": round(float(np.max([float(row["oos_turnover_pct"]) for row in rows_for_family])), 2),
                "mean_oos_alpha_vs_spy_pct": round(float(np.mean([float(row["oos_alpha_vs_spy_pct"]) for row in rows_for_family])), 2),
                "mean_oos_alpha_vs_qqq_pct": round(float(np.mean([float(row["oos_alpha_vs_qqq_pct"]) for row in rows_for_family])), 2),
            }
        )

    # Sort in the exact promotion order: more repeats, no TQQQ, smaller
    # drawdown, better Sharpe, then most recent representative fold.
    return sorted(
        summaries,
        key=lambda item: (
            -int(item["fold_count"]),
            bool(item["uses_tqqq"]),
            -float(item["mean_oos_max_drawdown_pct"]),
            -float(item["mean_oos_sharpe"]),
            -int(item["latest_outer_year"]),
        ),
    )


def live_signal_config(config: dict) -> dict:
    """Return the stable subset live signal generators are allowed to consume.

    The unified strategy may have tqqq_weight > 0 — in that case the live
    signal generator needs the TQQQ-specific fields (preset, regime params)
    alongside the normal core-alpha fields.
    """
    params = dict(config.get("nested_params", {}))
    tqqq_weight = float(params.get("tqqq_weight", 0.0))

    # Base config: strip internal-only keys from core-alpha config
    base = {
        key: value
        for key, value in deepcopy(config).items()
        if key not in {"nested_params", "cost_stress"}
    }
    for key in ("holding_days", "score_source", "shape", "weighting"):
        if key not in base and key in params:
            base[key] = params[key]

    # When tqqq_weight > 0, the live signal generator needs these extra fields
    # to run the TQQQ-aware regime engine (core_satellite_tqqq.py).
    if tqqq_weight > 0:
        base["tqqq_weight"] = tqqq_weight
        base["tqqq_preset"] = str(config.get("tqqq_preset", "tqqq_enhanced_cashbuffer"))
        # Ensure regime params are explicit so the signal generator doesn't
        # have to guess defaults.
        base["regime_ma_window"] = int(params.get("ma_window", config.get("regime_ma_window", 100)))
        base["regime_high_vol"] = float(params.get("high_vol", config.get("regime_high_vol", 0.30)))
        base["high_vol_mode"] = str(params.get("high_vol_mode", config.get("high_vol_mode", "fixed")))

    return base


def approval_status(result: dict) -> dict:
    """Decide whether a nested result is stable enough to feed live signals.

    Uses strategy-specific thresholds from _APPROVAL_THRESHOLDS.  TQQQ faces
    stricter drawdown and overfitting gates because 3x leverage amplifies
    losses and can fake good backtests.  All gates must pass (fail-closed).
    """
    strategy = str(result.get("strategy", "core-alpha"))
    thresholds = dict(_APPROVAL_THRESHOLDS.get(strategy, _APPROVAL_THRESHOLDS["core-alpha"]))

    # ── Unified strategy: if the winning config uses TQQQ, apply the stricter
    # TQQQ drawdown and bias gates on top of core-alpha thresholds.  This
    # ensures leveraged configs don't sneak through with worse risk numbers.
    approved_family = str(result.get("approved_family_signature", result.get("most_common_config", "")))
    if "tqqq=" in approved_family:
        import re as _re
        _tqqq_match = _re.search(r"tqqq=([\d.]+)", approved_family)
        if _tqqq_match and float(_tqqq_match.group(1)) > 0:
            tqqq_thresholds = _APPROVAL_THRESHOLDS.get("tqqq", {})
            # Only tighten — never loosen existing thresholds
            for key in ("max_mean_oos_drawdown_pct", "max_worst_oos_drawdown_pct",
                        "max_selection_bias_gap_sharpe"):
                if key in tqqq_thresholds:
                    thresholds[key] = min(thresholds[key], tqqq_thresholds[key])
            # Add TQQQ-only gates that core-alpha normally doesn't have
            if "min_worst_oos_return_pct" in tqqq_thresholds:
                thresholds["min_worst_oos_return_pct"] = tqqq_thresholds["min_worst_oos_return_pct"]

    if not bool(result.get("valid", False)):
        return {"approved": False, "reasons": ["nested_result_invalid"], "thresholds": thresholds}

    reasons: list[str] = []
    warnings: list[str] = []

    # ── Existing gates (tightened) ──────────────────────────────────────────
    fold_count = int(result.get("fold_count", 0) or 0)
    family_frequency = float(
        result.get(
            "approved_family_frequency",
            result.get("best_config_frequency", result.get("config_stability", 0.0)),
        )
        or 0.0
    )
    # The EXACT config being published must also clear a frequency gate.
    # Otherwise a config that only appeared once can still pass approval
    # because its broader family was picked enough times.  (Publish-safety bug
    # found Oct 2026: an h=20,ov=0.5,vol=percentile config with 7.1% frequency
    # was getting published because its family had 21.4% frequency.)
    exact_frequency = float(result.get("approved_config_frequency", family_frequency) or 0.0)
    # Exact config can clear at half the family threshold (e.g. 10% when
    # family threshold is 20%).  This requires the SPECIFIC tuning to appear
    # at least twice in 14 folds, not just be in a popular family.
    min_exact_config_frequency = thresholds["min_config_frequency"] * 0.5
    mean_oos_sharpe = float(result.get("mean_oos_sharpe", 0.0) or 0.0)
    alpha_hit_rate = float(result.get("oos_positive_alpha_hit_rate", 0.0) or 0.0)

    if fold_count < thresholds["min_folds"]:
        reasons.append(f"fold_count<{thresholds['min_folds']:.0f}")
    if family_frequency < thresholds["min_config_frequency"]:
        reasons.append(f"family_frequency<{thresholds['min_config_frequency']:.2f}")
    # Hard gate on the exact config being published (was a warning before).
    if exact_frequency < min_exact_config_frequency:
        reasons.append(
            f"config_frequency<{min_exact_config_frequency:.2f} (exact={exact_frequency:.3f})"
        )
    if mean_oos_sharpe < thresholds["min_mean_oos_sharpe"]:
        reasons.append(f"mean_oos_sharpe<{thresholds['min_mean_oos_sharpe']:.2f}")
    if alpha_hit_rate < thresholds["min_oos_alpha_hit_rate"]:
        reasons.append(f"oos_alpha_hit_rate<{thresholds['min_oos_alpha_hit_rate']:.2f}")
    if not bool(result.get("cost_stress_approval_pass", False)):
        reasons.append("cost_stress_approval_failed")

    # ── Drawdown gates (new) ────────────────────────────────────────────────
    # Drawdowns are negative percentages (e.g. -20.4%).  Gate fires when the
    # value is MORE negative than the threshold.
    mean_dd = float(result.get("mean_oos_max_drawdown_pct", 0.0) or 0.0)
    worst_dd = float(result.get("worst_oos_max_drawdown_pct", 0.0) or 0.0)
    if mean_dd < thresholds["max_mean_oos_drawdown_pct"]:
        reasons.append(
            f"mean_oos_drawdown={mean_dd:.1f}%<{thresholds['max_mean_oos_drawdown_pct']:.0f}%"
        )
    if worst_dd < thresholds["max_worst_oos_drawdown_pct"]:
        reasons.append(
            f"worst_oos_drawdown={worst_dd:.1f}%<{thresholds['max_worst_oos_drawdown_pct']:.0f}%"
        )
    if "max_worst_oos_turnover_pct" in thresholds:
        # Two-tier turnover check:
        # 1. Run-wide worst fold (HARD reject) — if ANY outer fold churned
        #    above the cap, the strategy is too unstable to deploy.  This
        #    catches one-off blowups like the 2014 fold at 991% turnover.
        # 2. Approved family worst (info/secondary) — what specifically
        #    happened to the published config family.
        run_wide_worst = float(result.get("worst_oos_turnover_pct", 0.0) or 0.0)
        family_worst = float(result.get("approved_family_worst_oos_turnover_pct", 0.0) or 0.0)
        cap = thresholds["max_worst_oos_turnover_pct"]
        if run_wide_worst > cap:
            reasons.append(
                f"worst_oos_turnover={run_wide_worst:.1f}%>{cap:.0f}%"
            )
        elif family_worst > cap:
            # Run-wide passed but approved family exceeded — still reject.
            reasons.append(
                f"approved_family_worst_turnover={family_worst:.1f}%>{cap:.0f}%"
            )

    # ── Selection bias gate (new — overfitting detector) ────────────────────
    # The gap between inner (optimized) Sharpe and OOS Sharpe.  A large gap
    # means the optimizer captured noise, not signal.
    bias_gap = float(result.get("selection_bias_gap_sharpe", 0.0) or 0.0)
    if bias_gap > thresholds["max_selection_bias_gap_sharpe"]:
        reasons.append(
            f"selection_bias_gap={bias_gap:.2f}>{thresholds['max_selection_bias_gap_sharpe']:.2f}"
        )

    # ── TQQQ-only: worst single-fold return ─────────────────────────────────
    # A single fold losing more than the threshold means regime switching
    # failed to protect during a crash year.
    if "min_worst_oos_return_pct" in thresholds:
        worst_ret = float(result.get("worst_oos_return_pct", 0.0) or 0.0)
        if worst_ret < thresholds["min_worst_oos_return_pct"]:
            reasons.append(
                f"worst_oos_return={worst_ret:.1f}%<{thresholds['min_worst_oos_return_pct']:.0f}%"
            )

    return {"approved": not reasons, "reasons": reasons, "thresholds": thresholds, "warnings": warnings}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def medium_risk_review_from_reports(
    *,
    survivorship: dict | None = None,
    execution: dict | None = None,
    factor_decay: dict | None = None,
) -> dict:
    survivorship = survivorship if survivorship is not None else _read_json(MEDIUM_RISK_SURVIVORSHIP_PATH)
    execution = execution if execution is not None else _read_json(MEDIUM_RISK_EXECUTION_PATH)
    factor_decay = factor_decay if factor_decay is not None else _read_json(MEDIUM_RISK_FACTOR_DECAY_PATH)
    reasons: list[str] = []

    rows = survivorship.get("rows", []) if isinstance(survivorship, dict) else []
    by_scenario = {str(row.get("scenario")): row for row in rows if isinstance(row, dict)}
    stressed = by_scenario.get("watchlist_plus_failed_audit_tickers", {})
    delta = by_scenario.get("delta_stressed_minus_base", {})
    surv_score = float(survivorship.get("survivorship_adjusted_score", 0.0) or 0.0) if survivorship else 0.0
    audit_picks = int(float(stressed.get("audit_rebalance_selections", 0) or 0)) if stressed else 0
    return_delta = float(delta.get("total_return_pct", 0.0) or 0.0) if delta else 0.0
    dd_delta = float(delta.get("max_drawdown_pct", 0.0) or 0.0) if delta else 0.0
    survivorship_pass = bool(
        survivorship
        and stressed
        and bool(stressed.get("paper_ready", False))
        and surv_score > SURVIVORSHIP_MIN_ADJUSTED_SCORE
        and audit_picks <= SURVIVORSHIP_MAX_AUDIT_SELECTIONS
        and return_delta >= SURVIVORSHIP_MIN_RETURN_DELTA_PCT
        and dd_delta >= SURVIVORSHIP_MIN_DRAWDOWN_DELTA_PCT
    )
    if not survivorship:
        reasons.append("survivorship_review_missing")
    elif not survivorship_pass:
        reasons.append("survivorship_review_failed")

    exec_rows = [
        row for row in (execution.get("rows", []) if isinstance(execution, dict) else [])
        if isinstance(row, dict) and not str(row.get("scenario", "")).startswith("delta_")
    ]
    exec_failed = [
        row for row in exec_rows
        if not bool(row.get("paper_ready", False))
        or float(row.get("alpha_vs_qqq_pct", -999.0) or -999.0) <= 0.0
        or float(row.get("alpha_vs_blend_pct", -999.0) or -999.0) <= 0.0
    ]
    worst_dd = min((float(row.get("max_drawdown_pct", 0.0) or 0.0) for row in exec_rows), default=0.0)
    execution_pass = bool(exec_rows and not exec_failed and worst_dd >= EXECUTION_STRESS_MIN_WORST_DRAWDOWN_PCT)
    if not execution:
        reasons.append("execution_stress_review_missing")
    elif not execution_pass:
        reasons.append("execution_stress_review_failed")

    edge_status = str((factor_decay or {}).get("edge_health_status", "missing"))
    factor_pass = edge_status in {"pass", "advisory"}
    if not factor_decay:
        reasons.append("factor_decay_review_missing")
    elif not factor_pass:
        reasons.append(f"factor_decay_review_{edge_status}")

    return {
        "pass": not reasons,
        "reasons": reasons,
        "survivorship_review": {
            "pass": survivorship_pass,
            "survivorship_adjusted_score": round(surv_score, 4),
            "audit_rebalance_selections": audit_picks,
            "total_return_delta_pct": round(return_delta, 4),
            "max_drawdown_delta_pct": round(dd_delta, 4),
        },
        "execution_stress_review": {
            "pass": execution_pass,
            "failed_scenarios": len(exec_failed),
            "worst_stressed_drawdown_pct": round(worst_dd, 4),
        },
        "factor_decay_review": {
            "pass": factor_pass,
            "edge_health_status": edge_status,
            "reason": (factor_decay or {}).get("reason"),
        },
    }


def apply_medium_risk_review(summary: dict, review: dict | None = None) -> dict:
    review = review or medium_risk_review_from_reports()
    summary["medium_risk_review"] = review
    approval = dict(summary.get("live_config_approval", {}) or {})
    if not bool(review.get("pass", False)):
        approval["approved"] = False
        reasons = list(approval.get("reasons") or [])
        for reason in review.get("reasons", []) or ["medium_risk_review_failed"]:
            item = f"medium_risk_review_failed:{reason}"
            if item not in reasons:
                reasons.append(item)
        approval["reasons"] = reasons
        summary["live_config_approval"] = approval
        summary.pop("approved_live_config", None)
    elif "approved_live_config" in summary:
        summary["approved_live_config"]["medium_risk_review"] = review
        summary["approved_live_config"].setdefault("source_metrics", {})["medium_risk_review_pass"] = True
    return summary


def _screen_one_config(config: dict, panel: pd.DataFrame, screen_fold,
                        low_memory: bool) -> dict | None:
    """Quick single-fold screen for successive halving.

    Evaluates one config on ONE inner fold (no cost stress — just the
    base robustness score).  Returns {config, screen_score} or None
    if the config errors out.  This is ~4x faster than a full eval
    because it skips cost stress (3 extra evaluate_window calls) and
    only runs 1 fold instead of 5.

    Used as Phase 1 of successive halving: quickly rank all configs
    so we can throw away the bottom 75% before doing expensive full
    evaluation.
    """
    base_config = config_with_cost_stress(config, BASE_COST_STRESS)
    try:
        metrics = evaluate_window(panel, base_config,
                                  screen_fold.validation_start,
                                  screen_fold.validation_end)
        score = inner_selection_score(metrics)
        return {"config": config, "screen_score": float(score)}
    except (ValueError, KeyError, RuntimeError, ZeroDivisionError):
        return None


# Module-level reference for the screening fold — set before forking
# so workers inherit it via copy-on-write (same pattern as _SHARED_PANEL).
_SHARED_SCREEN_FOLD = None


def _screen_worker(config_low_memory: tuple[dict, bool]) -> dict | None:
    """Parallel worker for Phase 1 screening — evaluates one config on
    a single fold.  Much cheaper than _parallel_worker because it skips
    cost stress and only runs 1 fold.

    No gc.collect() here — each screen eval is tiny (~100 KB of
    DataFrames) and the GC overhead (~5ms) is significant relative
    to the eval time (~200ms).  maxtasksperchild handles memory.
    """
    config, low_memory = config_low_memory
    return _screen_one_config(config, _SHARED_PANEL, _SHARED_SCREEN_FOLD, low_memory)


def _evaluate_one_config(config: dict, panel: pd.DataFrame, inner_folds: list,
                          low_memory: bool, best_score_so_far: float = -np.inf,
                          skip_stress_gate: bool = False,
                          prior_selected_sigs: list[str] | None = None) -> dict | None:
    """Evaluate a single config across all inner folds.

    Returns a result dict with {config, score, metrics, fold_metrics,
    failed_evaluations} or None if the config is disqualified (stress
    failure or no valid folds).

    This is the hot inner function — called once per config, either
    sequentially or in a forked worker process.  Each call builds its
    own eval_cache so there is no shared mutable state.

    Early termination: if best_score_so_far > -inf, the function checks
    after each fold whether the running average can still beat the best.
    If the remaining folds would need impossibly high scores, we bail
    out early.  This saves 30-50% of evaluations in later configs.
    """
    # Each config gets its own small cache for cost-stress variants
    # (same config at different cost levels shares cache entries).
    # ── Inner fold stress tolerance ────────────────────────────────
    # Old behavior: if ANY inner fold fails cost stress → reject config.
    # Problem: no strategy beats SPY+QQQ+BLEND at 5x costs in EVERY year.
    # 2016-2026 folds all fail because one bad inner fold kills the config.
    #
    # New behavior: require a MAJORITY of inner folds to pass stress.
    # A config that works in 3/5 market environments is robust enough.
    # The remaining 2/5 failures get scored normally (they just don't
    # contribute to the stress-pass count).
    MIN_STRESS_PASS_RATIO = 0.6   # at least 60% of inner folds must pass stress

    # For early termination: the highest realistic single-fold score.
    # Robustness scores are typically 0-3 range.  We use 5.0 as a
    # generous upper bound so we only skip truly hopeless configs.
    MAX_PLAUSIBLE_FOLD_SCORE = 5.0

    eval_cache: dict[str, dict] = {}
    fold_scores: list[float] = []
    fold_metrics: list[dict] = []
    failed = 0
    stress_passed = 0
    stress_tested = 0
    n_total_folds = len(inner_folds)

    for fold_idx, fold in enumerate(inner_folds):
        # ── Early termination check ───────────────────────────────
        # After completing at least 2 folds, check if this config
        # can still beat the current best even with perfect remaining
        # scores.  If not, bail out early — saves time on weak configs.
        if fold_idx >= 2 and best_score_so_far > -np.inf and fold_scores:
            current_sum = sum(fold_scores)
            remaining = n_total_folds - fold_idx
            # Best possible: remaining folds all score MAX_PLAUSIBLE_FOLD_SCORE
            optimistic_mean = (current_sum + remaining * MAX_PLAUSIBLE_FOLD_SCORE) / n_total_folds
            # Subtract a small stability penalty (optimistic: assume 0 std)
            if optimistic_mean < best_score_so_far:
                # Can't catch up — bail out early
                return None

        # GC every 3 folds — not every fold.  Each fold creates equity
        # curves + trade logs but they're small (~100 KB each).  GC
        # overhead (~5-10ms) adds up when called 5× per config × 48
        # configs = 240 GC cycles per outer fold.  Every-3rd-fold
        # cuts that to 80 while still keeping memory bounded.
        if fold_idx > 0 and fold_idx % 3 == 0:
            gc.collect()
        base_config = config_with_cost_stress(config, BASE_COST_STRESS)
        cache_key = _eval_cache_key(base_config, fold.validation_start, fold.validation_end)
        try:
            if cache_key in eval_cache:
                metrics = eval_cache[cache_key]
            else:
                metrics = evaluate_window(panel, base_config, fold.validation_start, fold.validation_end)
                if not low_memory:
                    eval_cache[cache_key] = metrics
            stress_result = nested_cost_stress_approval(
                panel, config, fold, eval_cache, base_metrics=metrics,
            )
        except (ValueError, KeyError, RuntimeError, ZeroDivisionError):
            failed += 1
            continue

        # Track stress pass/fail but don't reject the config yet.
        # We allow a minority of inner folds to fail stress — no
        # strategy beats all benchmarks at 5x costs in every year.
        stress_tested += 1
        fold_stress_pass = bool(stress_result["cost_stress_approval_pass"])
        if fold_stress_pass:
            stress_passed += 1

        score = inner_selection_score(metrics)
        score_components = robustness_score_components(metrics)
        fold_scores.append(score)
        fold_metrics.append(
            {
                "validation_year": int(fold.validation_year),
                "train_end": str(fold.train_end.date()),
                "score": round(float(score), 4),
                "drawdown_penalty": score_components["drawdown_penalty"],
                "turnover_penalty": score_components["turnover_penalty"],
                "instability_penalty": score_components["instability_penalty"],
                "sharpe": metrics.get("sharpe"),
                "return_pct": metrics.get("total_return_pct"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "turnover_pct": metrics.get("turnover_pct"),
                "alpha_vs_spy_pct": metrics.get("alpha_vs_spy_pct"),
                "alpha_vs_qqq_pct": metrics.get("alpha_vs_qqq_pct"),
                "alpha_vs_blend_pct": metrics.get("alpha_vs_blend_pct"),
                "cost_stress_approval_pass": fold_stress_pass,
                "cost_stress_summary": stress_result["cost_stress_summary"],
            }
        )

    if not fold_scores:
        return None

    # Reject if fewer than 60% of inner folds passed cost stress.
    # This replaces the old "all must pass" rule.
    # skip_stress_gate=True bypasses this (used in fallback mode when
    # no config passes the gate — better to have a config than NaN).
    stress_ratio = stress_passed / stress_tested if stress_tested > 0 else 0.0
    if not skip_stress_gate and stress_ratio < MIN_STRESS_PASS_RATIO:
        return None

    mean_turnover_pct = float(np.mean([
        float(m.get("turnover_pct", 0.0) or 0.0)
        for m in fold_metrics
    ]))
    if mean_turnover_pct > MAX_INNER_MEAN_TURNOVER_PCT:
        return None

    mean_score = float(np.mean(fold_scores))
    score_std = float(np.std(fold_scores, ddof=0)) if len(fold_scores) > 1 else 0.0
    # Stability penalty: mildly prefer configs that score consistently
    # across folds.  Was 0.35 which crushed concentrated configs (top3)
    # because 3-stock portfolios have inherently higher per-year variance.
    # At 0.35, a top3 config with std=1.5 lost 0.525 points — enough to
    # lose to a mediocre top10 config every time.  Reduced to 0.10 so
    # the penalty is a tiebreaker, not a dominant selection force.
    # A config with std=1.5 now loses only 0.15 points — still meaningful
    # but won't override a 0.3+ Sharpe advantage.
    STABILITY_PENALTY_WEIGHT = 0.10
    stable_score = mean_score - STABILITY_PENALTY_WEIGHT * score_std

    # ── QQQ opportunity-cost penalty (v2 — strengthened) ────────────
    # If you can't beat QQQ, why not just hold QQQ?  This penalizes
    # configs that consistently underperform QQQ across inner folds.
    #
    # v2 changes after seeing 6/8 OOS years trail QQQ despite the
    # original 0.03 penalty.  Root cause: ov=0.25 configs get high
    # Sharpe (low vol) but structurally can't beat QQQ in bull markets.
    # The penalty must offset that Sharpe advantage.
    #
    # Base penalty (linear):
    #   mean_alpha_vs_qqq >= 0  → no penalty
    #   mean_alpha_vs_qqq = -5% → penalty = 0.25
    #   mean_alpha_vs_qqq = -10% → penalty = 0.50
    #
    # Consistency multiplier: if EVERY fold trails QQQ, multiply by 1.5x
    # because it's structural, not bad luck.
    QQQ_PENALTY_RATE = 0.05  # per 1% underperformance vs QQQ (was 0.03)
    alpha_vs_qqq_values = [
        float(m.get("alpha_vs_qqq_pct", 0.0) or 0.0)
        for m in fold_metrics
    ]
    mean_alpha_vs_qqq = float(np.mean(alpha_vs_qqq_values)) if alpha_vs_qqq_values else 0.0
    qqq_penalty = max(0.0, -mean_alpha_vs_qqq * QQQ_PENALTY_RATE)

    # Consistency multiplier: ALL folds trailing QQQ = structural problem
    if len(alpha_vs_qqq_values) >= 3 and all(v < 0 for v in alpha_vs_qqq_values):
        qqq_penalty *= 1.5

    stable_score -= qqq_penalty

    # ── Config momentum bonus ──────────────────────────────────────────
    # Prevent random config-hopping between outer folds.  If this config
    # was selected in recent prior outer folds, give it a small bonus.
    # This creates "stickiness" — a config that worked recently gets the
    # benefit of the doubt, which reduces the 14-unique-in-14-folds noise
    # problem.
    #
    # Bonus = +0.15 if this config matches ≥50% of the last 3 selections.
    # 0.15 is large enough to break ties but small enough that a config
    # with genuinely better inner-fold performance will still win.
    CONFIG_MOMENTUM_BONUS = 0.15
    momentum_bonus = 0.0
    if prior_selected_sigs:
        this_sig = config_signature(config)
        recent = prior_selected_sigs[-3:]  # last 3 outer folds
        matches = sum(1 for s in recent if s == this_sig)
        if matches >= max(1, len(recent) // 2):
            momentum_bonus = CONFIG_MOMENTUM_BONUS
            stable_score += momentum_bonus

    return {
        "config": config,
        "score": stable_score,
        "metrics": {
            "inner_mean_score": round(mean_score, 4),
            "inner_score_std": round(score_std, 4),
            "inner_stability_adjusted_score": round(stable_score, 4),
            "inner_fold_count": int(len(fold_scores)),
            "inner_failed_fold_count": int(failed),
            "inner_mean_sharpe": round(float(np.mean([float(m.get("sharpe", 0.0) or 0.0) for m in fold_metrics])), 4),
            "inner_mean_return_pct": round(float(np.mean([float(m.get("return_pct", 0.0) or 0.0) for m in fold_metrics])), 2),
            "inner_mean_alpha_vs_spy_pct": round(float(np.mean([float(m.get("alpha_vs_spy_pct", 0.0) or 0.0) for m in fold_metrics])), 2),
            "inner_mean_alpha_vs_qqq_pct": round(mean_alpha_vs_qqq, 2),
            "inner_qqq_opportunity_cost_penalty": round(qqq_penalty, 4),
            "inner_config_momentum_bonus": round(momentum_bonus, 4),
            "inner_mean_turnover_pct": round(mean_turnover_pct, 2),
            "inner_cost_stress_approval_pass": bool(stress_ratio >= MIN_STRESS_PASS_RATIO),
            "inner_stress_pass_ratio": round(stress_ratio, 2),
            "inner_stress_passed": stress_passed,
            "inner_stress_tested": stress_tested,
        },
        "fold_metrics": fold_metrics,
        "failed_evaluations": failed,
    }


def _parallel_worker(config_low_memory_best: tuple[dict, bool, float, bool]) -> dict | None:
    """Thin wrapper for multiprocessing.Pool — reads panel and folds
    from module-level globals (inherited via fork, no pickling).

    Each worker aggressively GCs after finishing so forked processes
    release dirty pages back to the OS.  Without this, 7 workers on
    a 16 GB laptop can balloon to 70+ GB virtual memory and get
    OOM-killed by macOS before the fold completes.

    The third element in the tuple is the best score seen so far —
    passed to _evaluate_one_config for early termination of hopeless
    configs (configs that can't beat the current best even with
    perfect remaining fold scores).
    """
    config, low_memory, best_score, skip_stress_gate = config_low_memory_best
    try:
        return _evaluate_one_config(config, _SHARED_PANEL, _SHARED_INNER_FOLDS,
                                     low_memory, best_score_so_far=best_score,
                                     skip_stress_gate=skip_stress_gate,
                                     prior_selected_sigs=_SHARED_PRIOR_SIGS)
    finally:
        # Release any DataFrames the worker created (equity curves,
        # trade logs, benchmark series) so macOS can reclaim pages.
        gc.collect()


def _quiet_pyarrow_threads():
    """Drain Arrow's thread pools before forking.

    Arrow / PyArrow maintain CPU + IO thread pools.  If any thread
    is mid-allocation when fork() fires, the child inherits a
    corrupted malloc heap (SIGABRT / small_free_list_* crash).
    Setting counts to 1 idles all but one thread.
    """
    try:
        import pyarrow as _pa
        _pa.set_cpu_count(1)
        _pa.set_io_thread_count(1)
    except Exception:
        pass


def _suppress_malloc_stderr():
    """Suppress the C-level 'MallocStackLogging: can't turn off...' warning.

    macOS prints this directly to file descriptor 2 (stderr) whenever a
    forked child process touches malloc internals.  Python's warnings
    module can't catch it because it bypasses Python entirely.  We
    redirect fd 2 to /dev/null for the brief moment of Pool creation,
    then restore it.
    """
    try:
        _devnull = os.open(os.devnull, os.O_WRONLY)
        _old_stderr = os.dup(2)
        os.dup2(_devnull, 2)
        os.close(_devnull)
        return _old_stderr
    except OSError:
        return None


def _restore_stderr(old_fd):
    """Restore stderr after _suppress_malloc_stderr()."""
    if old_fd is not None:
        try:
            os.dup2(old_fd, 2)
            os.close(old_fd)
        except OSError:
            pass


# ── Successive halving parameters ─────────────────────────────────
# Phase 1 ("screening"): evaluate ALL configs on a single inner fold
#   (no cost stress — just base robustness score).  This is ~4x
#   cheaper per config than a full evaluation.
# Phase 2 ("full eval"): take the top SCREEN_SURVIVE_RATIO of
#   configs from Phase 1 and run full evaluation (all folds + cost
#   stress + early termination).
#
# With 768 configs and 5 inner folds:
#   Without halving:  768 × 5 folds × 4 evals/fold = ~15,360 evaluate_window calls
#   With halving:     768 × 1 screen + 192 × 5 × 4 = ~4,608 calls  → 3.3x speedup
#
# SCREEN_SURVIVE_RATIO controls how aggressive the pruning is.
# 0.25 = keep top 25% (aggressive, good for large grids like 768)
# 0.50 = keep top 50% (conservative, safer for small grids)
SCREEN_SURVIVE_RATIO = 0.25
# Only use successive halving when there are enough configs to
# make the screening overhead worthwhile.
SCREEN_MIN_CONFIGS = 32


def select_config_from_inner_folds(
    panel: pd.DataFrame,
    configs: list[dict],
    inner_folds: list[InnerFold],
    eval_cache: dict[str, dict] | None = None,
    low_memory: bool = False,
    n_workers: int = 1,
    skip_halving: bool = False,
    skip_stress_gate: bool = False,
    prior_selected_sigs: list[str] | None = None,
) -> dict:
    best: dict = {
        "config": None,
        "score": -np.inf,
        "metrics": {},
        "fold_metrics": [],
        "failed_evaluations": 0,
    }
    _n_configs = len(configs)

    # ── Successive halving: screen → prune → full eval ────────────
    # When the grid is large enough, a quick single-fold screen
    # eliminates the bottom 75% of configs before expensive full
    # evaluation.  This cuts total runtime by ~3x.
    # Disabled in --full mode: the whole point of --full is exhaustive
    # evaluation — screening on 1 fold risks dropping the true best
    # config if it happens to score poorly on that specific fold.
    use_halving = (not skip_halving
                   and _n_configs >= SCREEN_MIN_CONFIGS
                   and len(inner_folds) >= 2)

    if use_halving:
        # Pick the middle inner fold for screening — it's a balanced
        # market environment (not the oldest/smallest, not the most
        # recent which might be an outlier).
        screen_fold = inner_folds[len(inner_folds) // 2]
        survivors = _run_screening_phase(
            panel, configs, screen_fold, low_memory, n_workers
        )
        if survivors:
            configs = survivors
            _n_configs = len(configs)
            print(f"    Phase 2: full evaluation on {_n_configs} survivors "
                  f"across {len(inner_folds)} inner folds", flush=True)

    # ── Phase 2: full evaluation on survivors (or all if no halving) ──
    if n_workers > 1 and _n_configs > 1:
        best = _run_parallel_full_eval(panel, configs, inner_folds,
                                        low_memory, n_workers,
                                        prior_selected_sigs=prior_selected_sigs,
                                        skip_stress_gate=skip_stress_gate)
    else:
        best = _run_sequential_full_eval(panel, configs, inner_folds,
                                          low_memory,
                                          prior_selected_sigs=prior_selected_sigs,
                                          skip_stress_gate=skip_stress_gate)
    return best


def _run_screening_phase(
    panel: pd.DataFrame,
    configs: list[dict],
    screen_fold,
    low_memory: bool,
    n_workers: int,
) -> list[dict]:
    """Phase 1 of successive halving: quick single-fold screen.

    Evaluates every config on ONE inner fold (no cost stress).
    Returns the top SCREEN_SURVIVE_RATIO configs sorted by score.
    """
    global _SHARED_PANEL, _SHARED_SCREEN_FOLD
    _n_configs = len(configs)
    n_survive = max(4, int(_n_configs * SCREEN_SURVIVE_RATIO))
    print(f"    Phase 1: screening {_n_configs} configs on fold "
          f"{screen_fold.validation_year} → keeping top {n_survive}", flush=True)

    scored: list[tuple[float, dict]] = []
    _t0 = time.time()

    if n_workers > 1:
        _SHARED_PANEL = panel
        _SHARED_SCREEN_FOLD = screen_fold
        try:
            _quiet_pyarrow_threads()
            _use_spawn = _sys.platform == "win32"
            ctx = mp.get_context("spawn" if _use_spawn else "fork")
            work = [(c, low_memory) for c in configs]
            completed = 0
            _old_fd = _suppress_malloc_stderr()
            _pool_kw = dict(processes=n_workers, maxtasksperchild=8)
            if _use_spawn:
                _pool_kw["initializer"] = _init_pool_worker
                _pool_kw["initargs"] = (panel, None, screen_fold)
            with ctx.Pool(**_pool_kw) as pool:
                _restore_stderr(_old_fd)
                for result in pool.imap_unordered(_screen_worker, work, chunksize=2):
                    completed += 1
                    if completed % 20 == 0 or completed == _n_configs:
                        elapsed = time.time() - _t0
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (_n_configs - completed) / rate if rate > 0 else 0
                        print(f"      [screen {completed}/{_n_configs}, "
                              f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]", flush=True)
                    if result is not None:
                        scored.append((result["screen_score"], result["config"]))
        finally:
            _SHARED_PANEL = None
            _SHARED_SCREEN_FOLD = None
            gc.collect()
    else:
        for idx, config in enumerate(configs):
            result = _screen_one_config(config, panel, screen_fold, low_memory)
            if idx > 0 and idx % 20 == 0:
                elapsed = time.time() - _t0
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (_n_configs - idx) / rate if rate > 0 else 0
                print(f"      [screen {idx}/{_n_configs}, "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]", flush=True)
            if result is not None:
                scored.append((result["screen_score"], result["config"]))
            if (idx + 1) % 8 == 0:
                gc.collect()

    if not scored:
        return configs  # screening failed — fall back to all configs

    # Sort by score descending, keep top N
    scored.sort(key=lambda x: x[0], reverse=True)
    survivors = [cfg for _, cfg in scored[:n_survive]]
    elapsed = time.time() - _t0
    print(f"    Phase 1 done: {len(scored)} scored in {elapsed:.0f}s, "
          f"best screen score={scored[0][0]:.3f}, "
          f"cutoff={scored[min(n_survive, len(scored))-1][0]:.3f}", flush=True)
    return survivors


def _phase2_workers(n_workers: int, n_inner_folds: int) -> int:
    """Adaptive worker count for Phase 2.

    Phase 2 is memory-heavy: each worker does n_inner_folds × 4 cost
    levels = many evaluate_window calls, dirtying COW pages.  On 16 GB
    laptops, 6 workers causes swap thrashing in later folds.

    Rule: cap workers so (workers × inner_folds) doesn't exceed ~20.
    This keeps total active evaluations bounded and prevents memory
    pressure from climbing as folds get bigger.
    """
    # Each worker processes ~(inner_folds × 4_cost_levels) evals
    # before completing one config.  More inner folds = more memory.
    max_safe = max(2, 20 // max(1, n_inner_folds))
    return min(n_workers, max_safe)


def _run_parallel_full_eval(
    panel: pd.DataFrame,
    configs: list[dict],
    inner_folds: list,
    low_memory: bool,
    n_workers: int,
    prior_selected_sigs: list[str] | None = None,
    skip_stress_gate: bool = False,
) -> dict:
    """Phase 2 parallel: full evaluation with early termination."""
    global _SHARED_PANEL, _SHARED_INNER_FOLDS, _SHARED_PRIOR_SIGS
    best: dict = {
        "config": None,
        "score": -np.inf,
        "metrics": {},
        "fold_metrics": [],
        "failed_evaluations": 0,
    }
    _n_configs = len(configs)
    # Reduce workers for Phase 2 to prevent swap thrashing.
    # Phase 2 is much heavier per-worker than Phase 1 screening.
    actual_workers = _phase2_workers(n_workers, len(inner_folds))
    if actual_workers < n_workers:
        print(f"    (reducing workers {n_workers}→{actual_workers} for Phase 2 "
              f"with {len(inner_folds)} inner folds to prevent memory pressure)", flush=True)

    _SHARED_PANEL = panel
    _SHARED_INNER_FOLDS = inner_folds
    _SHARED_PRIOR_SIGS = prior_selected_sigs
    try:
        _quiet_pyarrow_threads()
        _use_spawn = _sys.platform == "win32"
        ctx = mp.get_context("spawn" if _use_spawn else "fork")
        best_score = float(best["score"])
        work = [(config, low_memory, best_score, skip_stress_gate) for config in configs]
        _t0 = time.time()
        completed = 0
        max_tasks = 8 if _n_configs <= 64 else 4
        _old_fd = _suppress_malloc_stderr()
        _pool_kw = dict(processes=actual_workers, maxtasksperchild=max_tasks)
        if _use_spawn:
            _pool_kw["initializer"] = _init_pool_worker
            _pool_kw["initargs"] = (panel, inner_folds, None, prior_selected_sigs)
        with ctx.Pool(**_pool_kw) as pool:
            _restore_stderr(_old_fd)
            for result in pool.imap_unordered(_parallel_worker, work, chunksize=1):
                completed += 1
                if completed % 5 == 0 or completed == _n_configs:
                    elapsed = time.time() - _t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (_n_configs - completed) / rate if rate > 0 else 0
                    print(f"    [{completed}/{_n_configs} configs, "
                          f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                          f"{actual_workers} workers]", flush=True)
                if result is None:
                    continue
                if result["score"] > float(best["score"]):
                    best = result
                else:
                    best["failed_evaluations"] += result.get("failed_evaluations", 0)
    finally:
        _SHARED_PANEL = None
        _SHARED_INNER_FOLDS = None
        _SHARED_PRIOR_SIGS = None
        gc.collect()
    return best


def _run_sequential_full_eval_relaxed(
    panel: pd.DataFrame,
    configs: list[dict],
    inner_folds: list,
    low_memory: bool,
    prior_selected_sigs: list[str] | None = None,
) -> dict:
    """Fallback evaluation with relaxed gates.

    Same as _run_sequential_full_eval but calls _evaluate_one_config
    with skip_stress_gate=True.  Used when the primary pass finds no
    valid config — better to have a non-stress-approved config than a
    blank year (NaN).
    """
    best: dict = {
        "config": None,
        "score": -np.inf,
        "metrics": {},
        "fold_metrics": [],
        "failed_evaluations": 0,
    }
    _n_configs = len(configs)
    _t0 = time.time()
    for cfg_idx, config in enumerate(configs):
        if cfg_idx > 0 and (cfg_idx % 5 == 0 or cfg_idx == _n_configs - 1):
            elapsed = time.time() - _t0
            rate = cfg_idx / elapsed if elapsed > 0 else 0
            eta = (_n_configs - cfg_idx) / rate if rate > 0 else 0
            print(f"    [relaxed {cfg_idx}/{_n_configs} configs, "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]", flush=True)
        result = _evaluate_one_config(config, panel, inner_folds, low_memory,
                                       best_score_so_far=float(best["score"]),
                                       skip_stress_gate=True,
                                       prior_selected_sigs=prior_selected_sigs)
        if result is None:
            continue
        if result["score"] > float(best["score"]):
            best = result
    return best


def _run_sequential_full_eval(
    panel: pd.DataFrame,
    configs: list[dict],
    inner_folds: list,
    low_memory: bool,
    prior_selected_sigs: list[str] | None = None,
    skip_stress_gate: bool = False,
) -> dict:
    """Phase 2 sequential: full evaluation with early termination."""
    best: dict = {
        "config": None,
        "score": -np.inf,
        "metrics": {},
        "fold_metrics": [],
        "failed_evaluations": 0,
    }
    _n_configs = len(configs)
    _GC_EVERY_N_CONFIGS = 1 if low_memory else 8
    _t0 = time.time()
    for cfg_idx, config in enumerate(configs):
        if cfg_idx > 0 and (cfg_idx % 10 == 0 or cfg_idx == _n_configs - 1):
            elapsed = time.time() - _t0
            rate = cfg_idx / elapsed if elapsed > 0 else 0
            eta = (_n_configs - cfg_idx) / rate if rate > 0 else 0
            print(f"    [{cfg_idx}/{_n_configs} configs, {elapsed:.0f}s elapsed, "
                  f"~{eta:.0f}s remaining]", flush=True)

        # Pass current best score for early termination — if this
        # config can't beat it after 2 folds, skip the rest.
        result = _evaluate_one_config(config, panel, inner_folds, low_memory,
                                       best_score_so_far=float(best["score"]),
                                       skip_stress_gate=skip_stress_gate,
                                       prior_selected_sigs=prior_selected_sigs)

        if (cfg_idx + 1) % _GC_EVERY_N_CONFIGS == 0:
            gc.collect()

        if result is None:
            continue
        if result["score"] > float(best["score"]):
            best = result
        else:
            best["failed_evaluations"] += result.get("failed_evaluations", 0)
    return best


def _eval_cache_key(config: dict, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Small stable key used to avoid repeating identical nested evaluations."""
    return (
        f"{config.get('strategy', 'core-alpha')}|"
        f"name={config.get('name', config.get('core_preset', ''))}|"
        f"{config_signature(config)}|"
        f"cost={float(config.get('cost_stress', BASE_COST_STRESS))}|"
        f"{pd.Timestamp(start).date()}|"
        f"{pd.Timestamp(end).date()}"
    )


def run_nested_walkforward(
    panel: pd.DataFrame,
    *,
    strategy: str = "core-alpha",
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    start_year: int | None = None,
    end_year: int | None = None,
    max_configs: int | None = None,
    max_folds: int | None = None,
    min_inner_train_years: int | None = None,
    fast: bool = False,
    full: bool = False,
    stable_grid: bool = False,
    recent_alpha_grid: bool = False,
    low_memory: bool = False,
    n_workers: int = 1,
    resume: bool = True,
) -> dict:
    strategy = str(_STRATEGY_ALIASES.get(strategy, strategy))
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    splits = build_fold_splits(panel, min_train_years=min_train_years, start_year=start_year, end_year=end_year)
    if max_folds is not None:
        splits = splits[: int(max_folds)]
    if not splits:
        return {"valid": False, "reason": "no_valid_yearly_folds", "folds": []}

    if fast:
        # ── Fast mode: collapse the grid to ~48 configs ──────────────────
        # Fast mode keeps the shape/weighting dimensions but pins holding
        # days to 10 and uses only two TQQQ weights (0% and 10%) to keep
        # smoke runs bounded while still testing whether TQQQ helps.
        # With halving: ~48 screen + ~12 full eval = done in ~15 min.
        configs = iter_candidate_configs(
            strategy=strategy,
            holding_days=(10,),
            overlay_gross=DEFAULT_OVERLAY_GROSS,
            ma_windows=(100,),
            high_vol_values=(0.30,),
            high_vol_modes=("fixed", "percentile"),
            shapes=SHAPES,
            weightings=WEIGHTING_MODES,
            tqqq_weights=(0.0, 0.10),
            risk_control_modes=RISK_CONTROL_MODES,
            max_configs=max_configs,
        )
    elif full:
        # ── Full mode: exhaustive grid (768 configs) ─────────────────────
        # Adds defensive risk control and full TQQQ sweep.  Use for
        # overnight runs on beefy machines.  On a 16 GB laptop this takes
        # ~3-5 hours with halving + parallel workers.
        configs = iter_candidate_configs(
            strategy=strategy,
            tqqq_weights=FULL_TQQQ_WEIGHTS,
            risk_control_modes=FULL_RISK_CONTROL_MODES,
            max_configs=max_configs,
        )
    elif stable_grid:
        # ── Stable grid: pinned alpha-decay baseline (~24 configs) ───────
        # Pins the dimensions that were repeatedly selected in completed
        # folds and leaves only shape, high-vol mode, and TQQQ weight open.
        configs = stable_grid_candidate_configs(
            strategy=strategy,
            max_configs=max_configs,
        )
    elif recent_alpha_grid:
        # ── Recent-alpha grid: focused post-2020 research grid (~48 configs)
        # Keeps the dimensions favored by the latest 14-fold result, but
        # tests whether overlay aggression, concentration, weighting, vol
        # mode, and a small TQQQ sleeve are truly robust.
        configs = recent_alpha_grid_candidate_configs(
            strategy=strategy,
            max_configs=max_configs,
        )
    else:
        # ── Default mode: focused grid (~72 configs) ─────────────────────
        # Pins dimensions that 7 years of walkforward unanimously/strongly
        # selected, freeing only the dimensions that actually vary:
        #   Pinned (consensus): weighting=sticky_score (7/7), ma=100 (7/7),
        #                       score=regime_adaptive (5/7)
        #   Free: shape(4), hold(2), overlay(3), vol_mode(2), tqqq(2),
        #          weighting(2), risk(2)
        # Grid: 4×2×3×2×2×2×2 = 384 configs → with halving ~96 full eval.
        # Added: top3 shape (dominated 2021-2025 in full run),
        #        risk_parity weighting (paired with top3 for best results),
        #        ov=0.70 (aggressive overlay for bull markets).
        configs = iter_candidate_configs(
            strategy=strategy,
            holding_days=DEFAULT_HOLDING_DAYS,
            overlay_gross=DEFAULT_OVERLAY_GROSS,
            ma_windows=(100,),
            high_vol_values=(0.30,),
            high_vol_modes=DEFAULT_HIGH_VOL_MODES,
            score_sources=("regime_adaptive",),
            shapes=SHAPES,
            weightings=("sticky_score", "risk_parity"),
            tqqq_weights=DEFAULT_TQQQ_WEIGHTS,
            risk_control_modes=RISK_CONTROL_MODES,
            max_configs=max_configs,
        )
    if not configs:
        return {"valid": False, "reason": "no_candidate_configs", "folds": []}

    fold_rows: list[dict] = []
    selected_configs: list[dict] = []
    inner_details: list[dict] = []
    eval_cache: dict[str, dict] = {}

    # ── Resume from checkpoint ────────────────────────────────────────────
    # Build a fingerprint of this exact run (strategy + grid + fold range).
    # If a checkpoint exists with a matching fingerprint we restore completed
    # folds from disk and skip them in the loop below — saving hours of work.
    ckpt_key = _ckpt_key(strategy, min_train_years, configs, start_year, end_year)
    ckpt = _load_checkpoint(strategy, ckpt_key) if resume else None
    completed_years: set[int] = set()
    if ckpt:
        fold_rows       = ckpt.get("fold_rows", [])
        selected_configs = ckpt.get("selected_configs", [])
        inner_details   = ckpt.get("inner_details", [])
        completed_years = {int(r["outer_year"]) for r in fold_rows}
        skipped = sorted(completed_years)
        print(
            f"Nested walk-forward ({strategy}): {len(splits)} folds, "
            f"{len(configs)} candidate configs per fold  "
            f"[resuming — {len(skipped)} fold(s) already done: {skipped}]"
        )
    else:
        print(f"Nested walk-forward ({strategy}): {len(splits)} folds, {len(configs)} candidate configs per fold")
    # ─────────────────────────────────────────────────────────────────────

    for split in splits:
        # ── Skip folds completed in a previous run ────────────────────────
        if split.outer_year in completed_years:
            print(f"  {split.outer_year}: already done (loaded from checkpoint) ✓")
            continue
        # ─────────────────────────────────────────────────────────────────
        outer_train_years = list(range(split.train_start.year, split.train_end.year + 1))
        inner_train_years_required = (
            int(min_inner_train_years)
            if min_inner_train_years is not None
            else max(2, int(min_train_years) - 1)
        )
        inner_folds = build_inner_folds(outer_train_years, min_inner_train_years=inner_train_years_required)
        # ── Cap inner folds to prevent runtime explosion ───────────────
        # Without a cap, the 2026 fold has 13 inner folds and TQQQ alone
        # needs 14,976 evaluations (~5 hours).  Cap at 5 most-recent inner
        # folds — earlier folds add diminishing information because the
        # market regime was too different.
        MAX_INNER_FOLDS = 5
        if len(inner_folds) > MAX_INNER_FOLDS:
            inner_folds = inner_folds[-MAX_INNER_FOLDS:]
        if not inner_folds:
            fold_rows.append({"fold_year": split.outer_year, "outer_year": split.outer_year, "valid": False, "reason": "no_inner_folds"})
            continue

        workers_label = f" ({n_workers} workers)" if n_workers > 1 else ""
        print(f"  {split.outer_year}: selecting from {len(configs)} configs × {len(inner_folds)} inner folds...{workers_label}")
        skip_inner_stress_gate = bool(recent_alpha_grid)
        if skip_inner_stress_gate:
            print("    recent-alpha grid: cost-stress gate is diagnostic-only during selection", flush=True)
        # Build prior config signatures for momentum bonus — configs
        # selected in earlier outer folds get a small score boost to
        # encourage consistency across folds (reduces config-hopping).
        prior_sigs = [
            config_signature(sc["config"])
            for sc in selected_configs
        ] if selected_configs else None

        selected = select_config_from_inner_folds(
            panel, configs, inner_folds, eval_cache=eval_cache,
            low_memory=low_memory, n_workers=n_workers,
            skip_halving=full,
            skip_stress_gate=skip_inner_stress_gate,
            prior_selected_sigs=prior_sigs,
        )
        print()  # newline after progress \r
        best_config = selected.get("config")

        # ── Fallback: if no config passed the 60% stress gate, relax ──
        # A missing year (NaN) is worse than a suboptimal config.  If the
        # primary pass found nothing, re-run with stress gate disabled.
        # The config won't be "stress-approved" but at least we get an
        # OOS result to evaluate whether the strategy works at all.
        if best_config is None:
            print(f"    ⚠ No config passed strict inner gates — retrying with relaxed stress gate...", flush=True)
            selected = _run_sequential_full_eval_relaxed(
                panel, configs, inner_folds, low_memory,
                prior_selected_sigs=prior_sigs,
            )
            best_config = selected.get("config")
            if best_config is not None:
                print(f"    ✓ Fallback found config (stress gate relaxed)", flush=True)

        if best_config is None:
            fold_rows.append(
                {
                    "fold_year": split.outer_year,
                    "outer_year": split.outer_year,
                    "valid": False,
                    "reason": "no_valid_inner_config",
                    "failed_evaluations": int(selected.get("failed_evaluations", 0)),
                    "inner_fold_count": int(len(inner_folds)),
                }
            )
            continue

        outer_cache_key = _eval_cache_key(best_config, split.outer_start, split.outer_end)
        try:
            if outer_cache_key in eval_cache:
                outer_metrics = eval_cache[outer_cache_key]
            else:
                outer_metrics = evaluate_window(panel, best_config, split.outer_start, split.outer_end)
                eval_cache[outer_cache_key] = outer_metrics
        except (ValueError, KeyError, RuntimeError, ZeroDivisionError) as exc:
            fold_rows.append(
                {
                    "fold_year": split.outer_year,
                    "outer_year": split.outer_year,
                    "valid": False,
                    "reason": str(exc),
                    "failed_evaluations": int(selected.get("failed_evaluations", 0)),
                    "selected_config": config_signature(best_config),
                }
            )
            continue

        params = best_config["nested_params"]
        inner_metrics = selected["metrics"]
        row = {
            "valid": True,
            "fold_year": int(split.outer_year),
            "outer_year": int(split.outer_year),
            "strategy": strategy,
            "inner_validation_years": ",".join(str(f.validation_year) for f in inner_folds),
            "inner_fold_count": int(inner_metrics["inner_fold_count"]),
            "train_start": str(split.train_start.date()),
            "train_end": str(split.train_end.date()),
            "inner_score": round(float(selected["score"]), 4),
            "inner_mean_score": inner_metrics["inner_mean_score"],
            "inner_score_std": inner_metrics["inner_score_std"],
            "failed_evaluations": int(selected.get("failed_evaluations", 0)),
            "candidate_configs": int(len(configs)),
            "selected_config": config_signature(best_config),
            "holding_days": params["holding_days"],
            "overlay_gross": params["overlay_gross"],
            "ma_window": params["ma_window"],
            "high_vol": params["high_vol"],
            "high_vol_mode": params["high_vol_mode"],
            "score_source": params["score_source"],
            "tqqq_weight": params["tqqq_weight"],
            "oos_return_pct": outer_metrics.get("total_return_pct"),
            "oos_sharpe": outer_metrics.get("sharpe"),
            "oos_drawdown_pct": outer_metrics.get("max_drawdown_pct"),
            "oos_turnover_pct": outer_metrics.get("turnover_pct"),
            "oos_alpha_vs_spy_pct": outer_metrics.get("alpha_vs_spy_pct"),
            "oos_alpha_vs_qqq_pct": outer_metrics.get("alpha_vs_qqq_pct"),
        }
        row.update(inner_metrics)
        for prefix, metrics in (("oos", outer_metrics),):
            for key in (
                "total_return_pct",
                "cagr_pct",
                "sharpe",
                "max_drawdown_pct",
                "turnover_pct",
                "alpha_vs_spy_pct",
                "alpha_vs_blend_pct",
                "alpha_vs_qqq_pct",
                "benchmark_blend_return_pct",
                "n_equity_points",
            ):
                row[f"{prefix}_{key}"] = metrics.get(key)
        fold_rows.append(row)
        selected_configs.append({"outer_year": int(split.outer_year), "config": best_config})
        inner_details.append(
            {
                "outer_year": int(split.outer_year),
                "selected_config": config_signature(best_config),
                "inner_folds": selected.get("fold_metrics", []),
            }
        )
        print(
            f"  {split.outer_year}: inner folds {row['inner_fold_count']}, "
            f"OOS Sharpe {row['oos_sharpe']:.2f}, "
            f"OOS alpha SPY/QQQ {row['oos_alpha_vs_spy_pct']:.1f}%/{row['oos_alpha_vs_qqq_pct']:.1f}% | "
            f"{row['selected_config']}"
        )
        # ── Memory management: always clear eval cache between outer folds ──
        # Inner-fold cache keys embed date ranges specific to each outer fold,
        # so cached entries from fold N are never reused in fold N+1.
        # Holding onto them just balloons RAM — clearing after each fold
        # keeps peak memory at one-fold-worth rather than all-folds-worth.
        eval_cache.clear()
        gc.collect()

        # ── Checkpoint: persist this fold so a crash loses at most one fold ──
        if resume:
            try:
                _save_checkpoint(strategy, ckpt_key, fold_rows, selected_configs, inner_details)
            except Exception as _ckpt_err:
                print(f"  [checkpoint save failed: {_ckpt_err}]")
        # ────────────────────────────────────────────────────────────────────

    valid_rows = [row for row in fold_rows if row.get("valid")]
    if not valid_rows:
        return {"valid": False, "reason": "all_folds_failed", "folds": fold_rows}

    oos_returns = np.array([float(row["oos_total_return_pct"]) / 100.0 for row in valid_rows], dtype=float)
    oos_cagrs = np.array([float(row["oos_cagr_pct"]) for row in valid_rows], dtype=float)
    inner_sharpes = np.array([float(row["inner_mean_sharpe"]) for row in valid_rows], dtype=float)
    oos_sharpes = np.array([float(row["oos_sharpe"]) for row in valid_rows], dtype=float)
    oos_drawdowns = np.array([float(row["oos_max_drawdown_pct"]) for row in valid_rows], dtype=float)
    oos_turnovers = np.array([float(row["oos_turnover_pct"]) for row in valid_rows], dtype=float)
    inner_alpha = np.array([float(row["inner_mean_alpha_vs_spy_pct"]) for row in valid_rows], dtype=float)
    oos_alpha = np.array([float(row["oos_alpha_vs_blend_pct"]) for row in valid_rows], dtype=float)
    oos_spy_alpha = np.array([float(row["oos_alpha_vs_spy_pct"]) for row in valid_rows], dtype=float)
    oos_qqq_alpha = np.array([float(row["oos_alpha_vs_qqq_pct"]) for row in valid_rows], dtype=float)
    signatures = Counter(str(row["selected_config"]) for row in valid_rows)
    most_common_sig, most_common_count = signatures.most_common(1)[0]
    config_stability = round(float(most_common_count / len(valid_rows)), 3)
    family_frequency = stable_family_frequency(valid_rows)
    approved_family = family_frequency[0]
    approved_family_sig = str(approved_family["stable_family_signature"])
    approved_family_rows = [
        row
        for row in valid_rows
        if stable_family_signature_from_config_signature(str(row["selected_config"])) == approved_family_sig
    ]
    # The live representative comes from the newest fold inside the stable
    # family.  That lets the big behavior stay repeatable while smaller knobs
    # still reflect the freshest market window.
    approved_row = max(approved_family_rows, key=lambda row: int(row["outer_year"]))
    approved_exact_sig = str(approved_row["selected_config"])
    approved_exact_count = int(signatures[approved_exact_sig])
    approved_exact_frequency = round(float(approved_exact_count / len(valid_rows)), 3)
    config_frequency = []
    for signature, count in signatures.most_common():
        rows_for_config = [row for row in valid_rows if str(row["selected_config"]) == signature]
        config_frequency.append(
            {
                "selected_config": signature,
                "fold_count": int(count),
                "frequency": round(float(count / len(valid_rows)), 3),
                "outer_years": [int(row["outer_year"]) for row in rows_for_config],
                "mean_oos_cagr_pct": round(float(np.mean([float(row["oos_cagr_pct"]) for row in rows_for_config])), 2),
                "mean_oos_sharpe": round(float(np.mean([float(row["oos_sharpe"]) for row in rows_for_config])), 3),
                "mean_oos_max_drawdown_pct": round(float(np.mean([float(row["oos_max_drawdown_pct"]) for row in rows_for_config])), 2),
                "mean_oos_turnover_pct": round(float(np.mean([float(row["oos_turnover_pct"]) for row in rows_for_config])), 2),
                "mean_oos_alpha_vs_spy_pct": round(float(np.mean([float(row["oos_alpha_vs_spy_pct"]) for row in rows_for_config])), 2),
                "mean_oos_alpha_vs_qqq_pct": round(float(np.mean([float(row["oos_alpha_vs_qqq_pct"]) for row in rows_for_config])), 2),
            }
        )
    for row in fold_rows:
        if row.get("valid"):
            family_sig = stable_family_signature_from_config_signature(str(row["selected_config"]))
            matching_family = next(
                item for item in family_frequency
                if str(item["stable_family_signature"]) == family_sig
            )
            row["config_stability"] = config_stability
            row["selected_config_fold_count"] = int(signatures[str(row["selected_config"])])
            row["selected_config_frequency"] = round(float(signatures[str(row["selected_config"])] / len(valid_rows)), 3)
            row["config_is_most_common"] = bool(str(row["selected_config"]) == most_common_sig)
            row["stable_family_signature"] = family_sig
            row["stable_family_fold_count"] = int(matching_family["fold_count"])
            row["stable_family_frequency"] = float(matching_family["frequency"])
            row["stable_family_is_approved"] = bool(family_sig == approved_family_sig)

    approved_config = None
    for item in reversed(selected_configs):
        config = item.get("config", {})
        if int(item.get("outer_year", -1)) == int(approved_row["outer_year"]) and config_signature(config) == approved_exact_sig:
            approved_config = config
            break
    if approved_config is None:
        for item in reversed(selected_configs):
            config = item.get("config", {})
            if stable_family_signature(config) == approved_family_sig:
                approved_config = config
                break

    summary = {
        "valid": True,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "nested_walk_forward_yearly_outer_multi_inner_validation",
        "strategy": strategy,
        "candidate_config_count": int(len(configs)),
        "fold_count": int(len(valid_rows)),
        "failed_fold_count": int(len(fold_rows) - len(valid_rows)),
        "outer_years": [int(row["outer_year"]) for row in valid_rows],
        "mean_inner_fold_count": round(float(np.mean([int(row["inner_fold_count"]) for row in valid_rows])), 2),
        "compound_oos_return_pct": round(float(np.prod(1.0 + oos_returns) - 1.0) * 100.0, 2),
        "mean_oos_return_pct": round(float(np.mean(oos_returns)) * 100.0, 2),
        "mean_oos_cagr_pct": round(float(np.mean(oos_cagrs)), 2),
        "mean_oos_sharpe": round(float(np.mean(oos_sharpes)), 3),
        "median_oos_sharpe": round(float(np.median(oos_sharpes)), 3),
        "mean_oos_max_drawdown_pct": round(float(np.mean(oos_drawdowns)), 2),
        "worst_oos_max_drawdown_pct": round(float(np.min(oos_drawdowns)), 2),
        "mean_oos_turnover_pct": round(float(np.mean(oos_turnovers)), 2),
        "worst_oos_turnover_pct": round(float(np.max(oos_turnovers)), 2),
        "worst_oos_return_pct": round(float(np.min(oos_returns)) * 100.0, 2),
        "mean_oos_alpha_vs_blend_pct": round(float(np.mean(oos_alpha)), 2),
        "mean_oos_alpha_vs_spy_pct": round(float(np.mean(oos_spy_alpha)), 2),
        "mean_oos_alpha_vs_qqq_pct": round(float(np.mean(oos_qqq_alpha)), 2),
        "oos_positive_alpha_hit_rate": round(float(np.mean(oos_alpha > 0.0)), 3),
        "mean_inner_sharpe": round(float(np.mean(inner_sharpes)), 3),
        "mean_inner_alpha_vs_spy_pct": round(float(np.mean(inner_alpha)), 2),
        "selection_bias_gap_sharpe": round(float(np.mean(inner_sharpes) - np.mean(oos_sharpes)), 3),
        "selection_bias_gap_alpha_vs_spy_pct": round(float(np.mean(inner_alpha) - np.mean(oos_spy_alpha)), 2),
        "config_stability": config_stability,
        "best_config_frequency": round(float(most_common_count / len(valid_rows)), 3),
        "approved_config_fold_count": approved_exact_count,
        "approved_config_frequency": approved_exact_frequency,
        "approved_exact_config": approved_exact_sig,
        "approved_family_signature": approved_family_sig,
        "approved_family_fold_count": int(approved_family["fold_count"]),
        "approved_family_frequency": float(approved_family["frequency"]),
        "approved_family_uses_tqqq": bool(approved_family["uses_tqqq"]),
        "approved_family_worst_oos_turnover_pct": float(approved_family["worst_oos_turnover_pct"]),
        "approved_family_mean_oos_max_drawdown_pct": float(approved_family["mean_oos_max_drawdown_pct"]),
        "approved_family_mean_oos_sharpe": float(approved_family["mean_oos_sharpe"]),
        "most_common_config": most_common_sig,
        "config_frequency": config_frequency,
        "stable_family_frequency": family_frequency,
        "cost_stress_approval_pass": bool(all(row.get("inner_cost_stress_approval_pass", False) for row in valid_rows)),
        "required_cost_stresses": [float(v) for v in COST_STRESS_MULTIPLIERS],
        "folds": fold_rows,
        "inner_fold_details": inner_details,
        "selected_configs": selected_configs,
    }
    approval = approval_status(summary)
    summary["live_config_approval"] = {
        **approval,
        "strategy": strategy,
        "approved_config_family": approved_family_sig,
        "approved_family_signature": approved_family_sig,
        "approved_family_fold_count": int(approved_family["fold_count"]),
        "approved_family_frequency": float(approved_family["frequency"]),
        "approved_exact_config": approved_exact_sig,
        "approved_config_fold_count": approved_exact_count,
        "approved_config_frequency": approved_exact_frequency,
        "source": "nested_walkforward",
        "created_at": summary["created_at"],
    }
    if approval["approved"] and approved_config is not None:
        summary["approved_live_config"] = {
            "strategy": strategy,
            "approved_config_family": approved_family_sig,
            "approved_family_signature": approved_family_sig,
            "approved_exact_config": approved_exact_sig,
            "config": live_signal_config(approved_config),
            "source_metrics": {
                "fold_count": summary["fold_count"],
                "best_config_frequency": summary["best_config_frequency"],
                "approved_family_fold_count": summary["approved_family_fold_count"],
                "approved_family_frequency": summary["approved_family_frequency"],
                "approved_family_worst_oos_turnover_pct": summary["approved_family_worst_oos_turnover_pct"],
                "approved_family_mean_oos_max_drawdown_pct": summary["approved_family_mean_oos_max_drawdown_pct"],
                "approved_family_mean_oos_sharpe": summary["approved_family_mean_oos_sharpe"],
                "mean_oos_sharpe": summary["mean_oos_sharpe"],
                "mean_oos_cagr_pct": summary["mean_oos_cagr_pct"],
                "mean_oos_alpha_vs_spy_pct": summary["mean_oos_alpha_vs_spy_pct"],
                "mean_oos_alpha_vs_qqq_pct": summary["mean_oos_alpha_vs_qqq_pct"],
                "oos_positive_alpha_hit_rate": summary["oos_positive_alpha_hit_rate"],
                "cost_stress_approval_pass": summary["cost_stress_approval_pass"],
                "required_cost_stresses": summary["required_cost_stresses"],
                # New gate metrics (drawdown, bias, worst return)
                "mean_oos_max_drawdown_pct": summary["mean_oos_max_drawdown_pct"],
                "worst_oos_max_drawdown_pct": summary["worst_oos_max_drawdown_pct"],
                "worst_oos_turnover_pct": summary["worst_oos_turnover_pct"],
                "worst_oos_return_pct": summary["worst_oos_return_pct"],
                "selection_bias_gap_sharpe": summary["selection_bias_gap_sharpe"],
                "approved_config_fold_count": summary["approved_config_fold_count"],
                "approved_config_frequency": summary["approved_config_frequency"],
            },
        }
    return apply_medium_risk_review(summary)


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_outputs(
    result: dict,
    *,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    publish_live_config: bool = False,
) -> tuple[Path, Path]:
    signal_dir = Path(SIGNAL_DIR)
    signal_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(ch for ch in output_prefix if ch.isalnum() or ch in {"_", "-"}).strip("_-")
    if not safe_prefix:
        safe_prefix = "core_satellite_nested_walkforward"
    json_path = signal_dir / f"{safe_prefix}.json"
    csv_path = signal_dir / f"{safe_prefix}.csv"
    json_path.write_text(json.dumps(result, indent=2, default=_json_default))
    pd.DataFrame(result.get("folds", [])).to_csv(csv_path, index=False)
    if not publish_live_config:
        return json_path, csv_path

    if result.get("strategy") == "both":
        approvals = dict(result.get("live_config_approvals", {}))
        approved_configs = dict(result.get("approved_live_configs", {}))
        medium_reviews = {
            strategy: strategy_result.get("medium_risk_review", {})
            for strategy, strategy_result in result.get("strategy_results", {}).items()
        }
    else:
        strategy = str(result.get("strategy", "core-alpha"))
        approvals = {strategy: result.get("live_config_approval", {"approved": False, "reasons": ["missing_approval"]})}
        approved = result.get("approved_live_config")
        approved_configs = {strategy: approved} if isinstance(approved, dict) else {}
        medium_reviews = {strategy: result.get("medium_risk_review", {})}
    current_strategies = set(approvals)
    existing_payload: dict = {}
    if LIVE_CONFIG_PATH.exists():
        try:
            existing_payload = json.loads(LIVE_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            existing_payload = {}
    merged_approvals = dict(existing_payload.get("approvals", {}))
    merged_configs = dict(existing_payload.get("approved_live_configs", {}))
    for deprecated in ("tqqq", "both"):
        merged_approvals.pop(deprecated, None)
        merged_configs.pop(deprecated, None)
    merged_approvals.update(approvals)
    for strategy in current_strategies:
        if strategy in approved_configs:
            merged_configs[strategy] = approved_configs[strategy]
        else:
            merged_configs.pop(strategy, None)
    live_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_json": str(json_path),
        "method": result.get("method"),
        "approvals": merged_approvals,
        "approved_live_configs": merged_configs,
        "medium_risk_reviews": medium_reviews,
    }
    LIVE_CONFIG_PATH.write_text(json.dumps(live_payload, indent=2, default=_json_default))
    return json_path, csv_path


def live_config_publish_decision(args: argparse.Namespace) -> tuple[bool, str]:
    """Return whether this nested run should publish live approval state."""
    explicit = getattr(args, "publish_live_config", None)
    if explicit is True:
        return True, "forced_by_--publish-live-config"
    if explicit is False:
        return False, "disabled_by_--no-publish-live-config"

    debug_reasons: list[str] = []
    if bool(getattr(args, "fast", False)):
        debug_reasons.append("--fast")
    if bool(getattr(args, "stable_grid", False)):
        debug_reasons.append("--stable-grid")
    if bool(getattr(args, "recent_alpha_grid", False)):
        debug_reasons.append("--recent-alpha-grid")
    if getattr(args, "max_folds", None) is not None:
        debug_reasons.append("--max-folds")
    if getattr(args, "max_configs", None) is not None:
        debug_reasons.append("--max-configs")
    if getattr(args, "start_year", None) is not None:
        debug_reasons.append("--start-year")
    if getattr(args, "end_year", None) is not None:
        debug_reasons.append("--end-year")
    if int(getattr(args, "max_specs", DEFAULT_MAX_SPECS)) != DEFAULT_MAX_SPECS:
        debug_reasons.append("--max-specs")
    if str(getattr(args, "output_prefix", DEFAULT_OUTPUT_PREFIX)) != DEFAULT_OUTPUT_PREFIX:
        debug_reasons.append("--output-prefix")

    if debug_reasons:
        return False, "auto_disabled_for_debug_run:" + ",".join(debug_reasons)
    return True, "auto_full_nested_default"


def _combine_strategy_results(results: dict[str, dict]) -> dict:
    folds: list[dict] = []
    approved_live_configs: dict[str, dict] = {}
    live_config_approvals: dict[str, dict] = {}
    for strategy, result in results.items():
        for row in result.get("folds", []):
            folds.append({"strategy": strategy, **row})
        live_config_approvals[strategy] = result.get("live_config_approval", {"approved": False, "reasons": ["missing_approval"]})
        if bool(result.get("approved_live_config", {}).get("config")):
            approved_live_configs[strategy] = result["approved_live_config"]
    return {
        "valid": all(bool(result.get("valid")) for result in results.values()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "nested_walk_forward_both_signal_generators",
        "strategy": "both",
        "strategies": list(results.keys()),
        "strategy_results": results,
        "live_config_approvals": live_config_approvals,
        "approved_live_configs": approved_live_configs,
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Proper nested walk-forward validation for core-satellite signal generators")
    # "tqqq" and "both" are deprecated — TQQQ is now a grid knob inside core-alpha.
    # Kept for backward compat: "tqqq" maps to "core-alpha", "both" maps to "core-alpha".
    parser.add_argument("--strategy", choices=("core-alpha", "tqqq", "both"), default="core-alpha")
    parser.add_argument("--min-train-years", type=int, default=DEFAULT_MIN_TRAIN_YEARS)
    parser.add_argument("--min-inner-train-years", type=int, default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--max-configs", type=int, default=None, help="Debug/smoke limit for candidate configs")
    parser.add_argument("--max-folds", type=int, default=None, help="Debug/smoke limit for outer folds")
    parser.add_argument("--max-specs", type=int, default=DEFAULT_MAX_SPECS)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--fast", action="store_true",
                        help="Use a smaller grid (~48 configs) for quick smoke tests (~15 min)")
    parser.add_argument("--full", action="store_true",
                        help="Use the exhaustive grid (768 configs) for overnight runs. "
                             "Default is a balanced 192-config grid (~1 hour on laptop)")
    parser.add_argument("--stable-grid", action="store_true",
                        help="Use the consensus baseline grid (24 configs): pins "
                             "14-fold winners (h=20, ma=100, regime_adaptive, "
                             "risk=off, overlay=0.50) and drops ov=0.7 which had "
                             "562% turnover and 0.95 Sharpe; tunes shape, weighting, "
                             "vol mode, and tqqq=(0.0, 0.1).")
    parser.add_argument("--recent-alpha-grid", action="store_true",
                        help="Use the focused recent-alpha research grid (~48 configs): "
                             "h=20, risk off, ma=100, regime_adaptive; tunes "
                             "overlay aggression, shape, weighting, vol mode, and TQQQ.")
    publish_group = parser.add_mutually_exclusive_group()
    publish_group.add_argument(
        "--publish-live-config",
        dest="publish_live_config",
        action="store_true",
        help=(
            "Force writing approvals to signals/core_satellite_live_configs.json, "
            "even for bounded/debug runs."
        ),
    )
    publish_group.add_argument(
        "--no-publish-live-config",
        dest="publish_live_config",
        action="store_false",
        help=(
            "Do not write approvals to signals/core_satellite_live_configs.json. "
            "Useful for dry-run full validations."
        ),
    )
    parser.set_defaults(publish_live_config=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from the last checkpoint if available.  ON by default — if "
            "the run crashes or is interrupted, re-running automatically picks "
            "up from the last completed fold.  Use --no-resume to force a clean "
            "start and ignore any existing checkpoint."
        ),
    )
    parser.add_argument(
        "--low-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Aggressive memory management — disables the eval cache and runs "
            "gc.collect() after every config.  ON by default to prevent macOS "
            "OOM kills on laptops.  Use --no-low-memory on machines with 32+ GB "
            "RAM to get ~2x faster runs."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_PARALLEL_WORKERS,
        help=(
            f"Number of parallel worker processes for config evaluation.  "
            f"Default: {_PARALLEL_WORKERS} (auto-detected from CPU count and "
            f"available RAM — each worker needs ~2 GB headroom).  Uses "
            f"fork-based multiprocessing so the panel is shared via "
            f"copy-on-write.  Set to 1 to disable parallelism."
        ),
    )
    args = parser.parse_args()
    if sum(bool(flag) for flag in (args.fast, args.full, args.stable_grid, args.recent_alpha_grid)) > 1:
        parser.error("Choose at most one grid mode: --fast, --full, --stable-grid, or --recent-alpha-grid")

    # Force the 'spawn' start method so child workers do NOT inherit PyArrow's
    # background ThreadPool.  With 'fork' (the macOS default before Python 3.12)
    # the child side has corrupted malloc state and crashes with EXC_BREAKPOINT.
    # 'spawn' starts a fresh interpreter in each worker — slightly slower to
    # boot but completely safe with Arrow/Parquet threads.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set — fine, ignore

    specs = load_feature_specs(max_specs=int(args.max_specs))
    panel = load_factor_panel(specs)
    panel = attach_scores(panel, specs, load_prediction_scores())
    panel = _ensure_robust_score_columns(panel)
    panel["_date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.loc[panel["_date"].notna()].sort_values("_date").reset_index(drop=True)
    for column in panel.select_dtypes(include=["float64"]).columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").astype("float32")
    # Resolve deprecated strategy aliases: "tqqq" → "core-alpha", "both" → just "core-alpha"
    # (TQQQ is now a grid knob inside core-alpha, not a separate strategy)
    resolved_strategy = _STRATEGY_ALIASES.get(args.strategy, args.strategy)
    if args.strategy == "both":
        resolved_strategy = "core-alpha"
        print("Note: --strategy=both is deprecated. TQQQ is now a grid knob inside core-alpha.")
    elif args.strategy == "tqqq":
        print("Note: --strategy=tqqq is deprecated. TQQQ is now a grid knob inside core-alpha.")
    strategies = [resolved_strategy]
    strategy_results: dict[str, dict] = {}
    for strategy in strategies:
        strategy_results[strategy] = run_nested_walkforward(
            panel,
            strategy=strategy,
            min_train_years=int(args.min_train_years),
            start_year=args.start_year,
            end_year=args.end_year,
            max_configs=args.max_configs,
            max_folds=args.max_folds,
            min_inner_train_years=args.min_inner_train_years,
            fast=bool(args.fast),
            full=bool(args.full),
            stable_grid=bool(args.stable_grid),
            recent_alpha_grid=bool(args.recent_alpha_grid),
            low_memory=bool(args.low_memory),
            n_workers=int(args.workers),
            resume=bool(args.resume),
        )
        # ── Free memory between strategies ─────────────────────────────
        # Each strategy run creates thousands of temporary DataFrames.
        # Force a full GC pass before starting the next strategy so macOS
        # does not OOM-kill us on 16 GB machines.
        gc.collect()
        # ── Delete checkpoint on successful completion ──────────────────
        # A finished run is its own record (the JSON/CSV outputs).  Remove
        # the checkpoint so the NEXT run starts fresh instead of replaying
        # a stale partial state.
        if args.resume:
            ckpt_file = _ckpt_path(strategy)
            try:
                ckpt_file.unlink(missing_ok=True)
            except Exception:
                pass
    result = _combine_strategy_results(strategy_results) if len(strategy_results) > 1 else next(iter(strategy_results.values()))
    publish_live_config, publish_reason = live_config_publish_decision(args)
    json_path, csv_path = write_outputs(
        result,
        output_prefix=str(args.output_prefix),
        publish_live_config=publish_live_config,
    )
    print("\nNested walk-forward complete")
    print(f"  valid: {result.get('valid')}")
    if result.get("strategy") == "both":
        for strategy, strategy_result in strategy_results.items():
            approval = strategy_result.get("live_config_approval", {})
            print(
                f"  {strategy}: valid={strategy_result.get('valid')} "
                f"folds={strategy_result.get('fold_count')} "
                f"mean OOS Sharpe={strategy_result.get('mean_oos_sharpe')} "
                f"mean OOS alpha vs SPY/QQQ={strategy_result.get('mean_oos_alpha_vs_spy_pct')}%/"
                f"{strategy_result.get('mean_oos_alpha_vs_qqq_pct')}%"
            )
            print(f"    live config approved: {approval.get('approved')} {approval.get('reasons', [])}")
    elif result.get("valid"):
        approval = result.get("live_config_approval", {})
        print(f"  folds: {result.get('fold_count')}")
        print(f"  mean OOS CAGR: {result.get('mean_oos_cagr_pct')}%")
        print(f"  mean OOS Sharpe: {result.get('mean_oos_sharpe')}")
        print(f"  mean OOS max drawdown: {result.get('mean_oos_max_drawdown_pct')}%")
        print(f"  mean OOS turnover: {result.get('mean_oos_turnover_pct')}%")
        print(f"  mean OOS alpha vs SPY/QQQ: {result.get('mean_oos_alpha_vs_spy_pct')}%/{result.get('mean_oos_alpha_vs_qqq_pct')}%")
        print(f"  mean OOS alpha vs BLEND: {result.get('mean_oos_alpha_vs_blend_pct')}%")
        print(f"  selection-bias Sharpe gap: {result.get('selection_bias_gap_sharpe')}")
        print(f"  best config frequency: {result.get('best_config_frequency')}")
        print(f"  approved family frequency: {result.get('approved_family_frequency')}")
        print(f"  live config approved: {approval.get('approved')} {approval.get('reasons', [])}")
        for row in result.get("stable_family_frequency", [])[:5]:
            print(f"    family {row['fold_count']} folds ({row['frequency']:.0%}): {row['stable_family_signature']}")
        for row in result.get("config_frequency", [])[:5]:
            print(f"    {row['fold_count']} folds ({row['frequency']:.0%}): {row['selected_config']}")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    if publish_live_config:
        print(f"  live configs: {LIVE_CONFIG_PATH} ({publish_reason})")
    else:
        print(
            f"  live configs: not published ({publish_reason}; "
            f"use --publish-live-config to force update {LIVE_CONFIG_PATH})"
        )

    # ── Notify on completion (always, not just on publish) ────────────
    # PLAIN ENGLISH: When ANY walkforward run finishes, send a Telegram/email
    # alert so you know whether the config was approved or rejected.  This
    # fires regardless of whether --publish-live-config was set, because you
    # always want to know the result of a 1-hour run.
    try:
        from notifications import send_alert as _notify
        approval = result.get("live_config_approval", {})
        approved = bool(approval.get("approved", False))
        config_family = result.get("most_common_config", "unknown")
        sharpe = result.get("mean_oos_sharpe", "?")
        folds = result.get("fold_count", "?")
        published_str = "✓ published" if publish_live_config else "✗ not published"
        _notify(
            f"Walkforward finished ({folds} folds)\n"
            f"Approved: {approved}\n"
            f"Config: {config_family}\n"
            f"Mean OOS Sharpe: {sharpe}\n"
            f"Live config: {published_str}",
            title="Walkforward Complete",
            priority="info" if approved else "warning",
        )
    except Exception:
        pass  # don't let notification failure crash the run


if __name__ == "__main__":
    main()
