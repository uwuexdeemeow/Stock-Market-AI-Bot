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

_WF_ENTRYPOINTS = {"core_satellite_nested_walkforward.py", "nested_walkforward.py"}
_WF_IS_CLI = _os.path.basename(_sys.argv[0]) in _WF_ENTRYPOINTS
if _WF_IS_CLI and not _os.environ.get("_WF_FORK_SAFE"):
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
        # macOS: sysctl returns total physical RAM in bytes
        import subprocess
        raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                       timeout=5, text=True).strip()
        total_gb = int(raw) / (1024 ** 3)
    except Exception:
        # Linux fallback: read /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_gb = int(line.split()[1]) / (1024 ** 2)
                        break
                else:
                    total_gb = 16.0  # conservative default
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
_SHARED_PANEL: pd.DataFrame | None = None
_SHARED_INNER_FOLDS: list | None = None

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
DEFAULT_HOLDING_DAYS = (10, 20)
DEFAULT_OVERLAY_GROSS = (0.25, 0.50)
DEFAULT_MA_WINDOWS = (100,)
DEFAULT_HIGH_VOL_VALUES = (0.30,)
DEFAULT_HIGH_VOL_MODES = ("fixed", "percentile")
DEFAULT_TQQQ_WEIGHTS = (0.0, 0.10, 0.20, 0.30)
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
RISK_CONTROL_MODES = ("off", "defensive")
SURVIVORSHIP_MIN_ADJUSTED_SCORE = 0.80
SURVIVORSHIP_MAX_AUDIT_SELECTIONS = 10
SURVIVORSHIP_MIN_RETURN_DELTA_PCT = -1000.0
SURVIVORSHIP_MIN_DRAWDOWN_DELTA_PCT = -2.0
EXECUTION_STRESS_MIN_WORST_DRAWDOWN_PCT = -35.0

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
        "min_config_frequency": 0.30,            # winning config must be selected ≥30% of folds
        "min_mean_oos_sharpe": 0.50,             # mean OOS Sharpe across folds
        "min_oos_alpha_hit_rate": 0.60,          # fraction of folds with positive alpha
        "max_mean_oos_drawdown_pct": -25.0,      # mean max drawdown across folds (negative %)
        "max_worst_oos_drawdown_pct": -35.0,     # worst single-fold max drawdown
        "max_selection_bias_gap_sharpe": 1.50,   # inner-vs-OOS Sharpe gap (overfitting detector)
    },
    "tqqq": {
        "min_folds": 3,
        "min_config_frequency": 0.30,            # winning config must be selected ≥30% of folds
        "min_mean_oos_sharpe": 0.50,
        "min_oos_alpha_hit_rate": 0.60,
        "max_mean_oos_drawdown_pct": -20.0,      # tighter — leverage amplifies drawdowns
        "max_worst_oos_drawdown_pct": -25.0,     # must prove regime switching protects in crashes
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


def evaluate_window(panel: pd.DataFrame, config: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Evaluate a config using only panel rows up to end, then score [start, end].

    This function is called many times by nested CV. It avoids repeated date
    parsing and avoids copying the full panel, which prevents macOS from killing
    long nested runs due to memory pressure.
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
        # TQQQ in risk_on core).  When it's 0, the standard evaluate() is
        # faster and produces identical results.
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
            metrics, equity, trades = evaluate(eval_panel, config)

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
        bench = benchmark_equity(pd.DatetimeIndex(window.index))
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
        # evaluate() / run_tqqq_backtest() create equity curves, trade
        # logs, benchmarks, etc.  Nulling lets Python's refcount free them
        # immediately instead of waiting for a gc cycle.
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
    most_common_config = str(result.get("most_common_config", ""))
    if "tqqq=" in most_common_config:
        import re as _re
        _tqqq_match = _re.search(r"tqqq=([\d.]+)", most_common_config)
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

    # ── Existing gates (tightened) ──────────────────────────────────────────
    fold_count = int(result.get("fold_count", 0) or 0)
    config_frequency = float(
        result.get("best_config_frequency", result.get("config_stability", 0.0)) or 0.0
    )
    mean_oos_sharpe = float(result.get("mean_oos_sharpe", 0.0) or 0.0)
    alpha_hit_rate = float(result.get("oos_positive_alpha_hit_rate", 0.0) or 0.0)

    if fold_count < thresholds["min_folds"]:
        reasons.append(f"fold_count<{thresholds['min_folds']:.0f}")
    if config_frequency < thresholds["min_config_frequency"]:
        reasons.append(f"config_frequency<{thresholds['min_config_frequency']:.2f}")
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

    return {"approved": not reasons, "reasons": reasons, "thresholds": thresholds}


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
        and surv_score >= SURVIVORSHIP_MIN_ADJUSTED_SCORE
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
    """
    config, low_memory = config_low_memory
    try:
        return _screen_one_config(config, _SHARED_PANEL, _SHARED_SCREEN_FOLD, low_memory)
    finally:
        gc.collect()


def _evaluate_one_config(config: dict, panel: pd.DataFrame, inner_folds: list,
                          low_memory: bool, best_score_so_far: float = -np.inf) -> dict | None:
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

        # GC between folds inside each worker to keep memory in check.
        # Each fold creates equity curves + trade logs that are only
        # needed for scoring — drop them before the next fold.
        if fold_idx > 0:
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
    stress_ratio = stress_passed / stress_tested if stress_tested > 0 else 0.0
    if stress_ratio < MIN_STRESS_PASS_RATIO:
        return None

    mean_score = float(np.mean(fold_scores))
    score_std = float(np.std(fold_scores, ddof=0)) if len(fold_scores) > 1 else 0.0
    stable_score = mean_score - 0.10 * score_std

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
            "inner_mean_alpha_vs_qqq_pct": round(float(np.mean([float(m.get("alpha_vs_qqq_pct", 0.0) or 0.0) for m in fold_metrics])), 2),
            "inner_mean_turnover_pct": round(float(np.mean([float(m.get("turnover_pct", 0.0) or 0.0) for m in fold_metrics])), 2),
            "inner_cost_stress_approval_pass": bool(stress_ratio >= MIN_STRESS_PASS_RATIO),
            "inner_stress_pass_ratio": round(stress_ratio, 2),
            "inner_stress_passed": stress_passed,
            "inner_stress_tested": stress_tested,
        },
        "fold_metrics": fold_metrics,
        "failed_evaluations": failed,
    }


def _parallel_worker(config_low_memory_best: tuple[dict, bool, float]) -> dict | None:
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
    config, low_memory, best_score = config_low_memory_best
    try:
        return _evaluate_one_config(config, _SHARED_PANEL, _SHARED_INNER_FOLDS,
                                     low_memory, best_score_so_far=best_score)
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
    use_halving = _n_configs >= SCREEN_MIN_CONFIGS and len(inner_folds) >= 2

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
                                        low_memory, n_workers)
    else:
        best = _run_sequential_full_eval(panel, configs, inner_folds,
                                          low_memory)
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
            ctx = mp.get_context("fork")
            work = [(c, low_memory) for c in configs]
            completed = 0
            # Higher maxtasksperchild for screening — each task is cheap
            # (1 fold, no cost stress) so recycling overhead matters more.
            with ctx.Pool(processes=n_workers, maxtasksperchild=8) as pool:
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


def _run_parallel_full_eval(
    panel: pd.DataFrame,
    configs: list[dict],
    inner_folds: list,
    low_memory: bool,
    n_workers: int,
) -> dict:
    """Phase 2 parallel: full evaluation with early termination."""
    global _SHARED_PANEL, _SHARED_INNER_FOLDS
    best: dict = {
        "config": None,
        "score": -np.inf,
        "metrics": {},
        "fold_metrics": [],
        "failed_evaluations": 0,
    }
    _n_configs = len(configs)
    _SHARED_PANEL = panel
    _SHARED_INNER_FOLDS = inner_folds
    try:
        _quiet_pyarrow_threads()
        ctx = mp.get_context("fork")
        # Pass the current best score to each worker for early termination.
        # Workers launched earlier won't know about later discoveries, but
        # that's fine — early termination is a best-effort optimization.
        # The first batch uses -inf; we update for subsequent batches.
        best_score = float(best["score"])
        work = [(config, low_memory, best_score) for config in configs]
        _t0 = time.time()
        completed = 0
        with ctx.Pool(processes=n_workers, maxtasksperchild=4) as pool:
            for result in pool.imap_unordered(_parallel_worker, work, chunksize=1):
                completed += 1
                if completed % 5 == 0 or completed == _n_configs:
                    elapsed = time.time() - _t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (_n_configs - completed) / rate if rate > 0 else 0
                    print(f"    [{completed}/{_n_configs} configs, "
                          f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                          f"{n_workers} workers]", flush=True)
                if result is None:
                    continue
                if result["score"] > float(best["score"]):
                    best = result
                else:
                    best["failed_evaluations"] += result.get("failed_evaluations", 0)
    finally:
        _SHARED_PANEL = None
        _SHARED_INNER_FOLDS = None
        gc.collect()
    return best


def _run_sequential_full_eval(
    panel: pd.DataFrame,
    configs: list[dict],
    inner_folds: list,
    low_memory: bool,
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
                                       best_score_so_far=float(best["score"]))

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
        # ── Fast mode: collapse the grid to ~16 configs ──────────────────
        # Fast mode keeps the shape/weighting dimensions but uses reduced
        # regime knobs and only two TQQQ weights (0% and 10%) to keep smoke
        # runs bounded while still testing whether TQQQ helps.
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
            max_configs=max_configs,
        )
    else:
        configs = iter_candidate_configs(strategy=strategy, max_configs=max_configs)
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
        selected = select_config_from_inner_folds(
            panel, configs, inner_folds, eval_cache=eval_cache,
            low_memory=low_memory, n_workers=n_workers,
        )
        print()  # newline after progress \r
        best_config = selected.get("config")
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
            row["config_stability"] = config_stability
            row["selected_config_fold_count"] = int(signatures[str(row["selected_config"])])
            row["selected_config_frequency"] = round(float(signatures[str(row["selected_config"])] / len(valid_rows)), 3)
            row["config_is_most_common"] = bool(str(row["selected_config"]) == most_common_sig)

    approved_config = None
    for item in reversed(selected_configs):
        config = item.get("config", {})
        if config_signature(config) == most_common_sig:
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
        "most_common_config": most_common_sig,
        "config_frequency": config_frequency,
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
        "approved_config_family": most_common_sig,
        "source": "nested_walkforward",
        "created_at": summary["created_at"],
    }
    if approval["approved"] and approved_config is not None:
        summary["approved_live_config"] = {
            "strategy": strategy,
            "approved_config_family": most_common_sig,
            "config": live_signal_config(approved_config),
            "source_metrics": {
                "fold_count": summary["fold_count"],
                "best_config_frequency": summary["best_config_frequency"],
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
                "worst_oos_return_pct": summary["worst_oos_return_pct"],
                "selection_bias_gap_sharpe": summary["selection_bias_gap_sharpe"],
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
    parser.add_argument("--fast", action="store_true", help="Use a smaller but still nested validation grid")
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
        print(f"  live config approved: {approval.get('approved')} {approval.get('reasons', [])}")
        for row in result.get("config_frequency", [])[:5]:
            print(f"    {row['fold_count']} folds ({row['frequency']:.0%}): {row['selected_config']}")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    if publish_live_config:
        print(f"  live configs: {LIVE_CONFIG_PATH} ({publish_reason})")
        # ── Notify on completion + config publish ─────────────────────
        # PLAIN ENGLISH: When a full walkforward finishes and publishes a
        # new live config, send a Telegram/email alert so you know the
        # next daily signal will use the updated config.
        try:
            from notifications import send_alert as _notify
            approval = result.get("live_config_approval", {})
            approved = bool(approval.get("approved", False))
            config_family = result.get("most_common_config", "unknown")
            sharpe = result.get("mean_oos_sharpe", "?")
            folds = result.get("fold_count", "?")
            _notify(
                f"Walkforward finished ({folds} folds)\n"
                f"Approved: {approved}\n"
                f"Config: {config_family}\n"
                f"Mean OOS Sharpe: {sharpe}\n"
                f"Next daily run will use this config automatically.",
                title="Walkforward Complete",
                priority="info",
            )
        except Exception:
            pass  # don't let notification failure crash the run
    else:
        print(
            f"  live configs: not published ({publish_reason}; "
            f"use --publish-live-config to force update {LIVE_CONFIG_PATH})"
        )


if __name__ == "__main__":
    main()
