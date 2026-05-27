"""
walkforward_selector_diagnostics.py - research-only selector diagnostics.

PLAIN ENGLISH:
The nested walkforward chooses one strategy configuration for each test year.
This script checks two simpler questions without changing the live config:

1. Would one fixed configuration have worked better than yearly selection?
2. When many candidates are replayed, does their inner score rank their later
   out-of-sample result in the same order?

Every output is a separate research JSON/CSV pair under ``signals/``.  Nothing
in this script publishes a config to paper trading.
"""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

import core_satellite_nested_walkforward as nested
from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
from robustness_scoring import DEFAULT_OBJECTIVE, robustness_score_components
from safe_io import configure_console_output
from settings import SIGNAL_DIR


configure_console_output()

DEFAULT_BASELINE_PREFIX = "wf_fixed_config_baseline"
DEFAULT_REPLAY_PREFIX = "wf_candidate_selector_replay"
MAX_REPLAY_INNER_FOLDS = 5


def _apply_concentration_overlay_signature(config: dict, conc_ov: str | None) -> dict:
    """Attach concentration-overlay fields parsed from a config signature."""
    if not conc_ov:
        return config
    if ":" not in conc_ov or "-" not in conc_ov:
        raise ValueError(f"conc_ov must look like mode:low-high, got {conc_ov!r}")
    mode, gross_range = str(conc_ov).split(":", 1)
    low, high = gross_range.split("-", 1)
    out = deepcopy(config)
    overlay = {
        "concentration_overlay_mode": str(mode),
        "concentration_overlay_low_gross": float(low),
        "concentration_overlay_high_gross": float(high),
        "concentration_overlay_threshold": float(
            nested.STABLE_GRID_CONCENTRATION_OVERLAY.get("concentration_overlay_threshold", 0.05)
        ),
        "concentration_overlay_span": float(
            nested.STABLE_GRID_CONCENTRATION_OVERLAY.get("concentration_overlay_span", 0.05)
        ),
    }
    out.update(overlay)
    params = out.setdefault("nested_params", {})
    params.update(overlay)
    return out


def config_from_signature(signature: str, *, strategy: str = "core-alpha") -> dict:
    """Build one candidate config from the compact walkforward signature text."""
    parsed = nested._parse_config_signature(signature)
    required = {"h", "ov", "ma", "vol", "score", "shape", "weighting", "tqqq", "risk"}
    missing = sorted(required - set(parsed))
    if missing:
        raise ValueError(f"Config signature missing fields {missing}: {signature}")
    if ":" not in parsed["vol"]:
        raise ValueError(f"Config signature vol must look like mode:value: {signature}")
    high_vol_mode, high_vol = parsed["vol"].split(":", 1)

    # The normal candidate builder wires in the regime preset and all safety
    # defaults.  Feeding it one value per dimension recreates one real config.
    configs = nested.iter_candidate_configs(
        strategy=strategy,
        holding_days=(int(float(parsed["h"])),),
        overlay_gross=(float(parsed["ov"]),),
        ma_windows=(int(float(parsed["ma"])),),
        high_vol_values=(float(high_vol),),
        high_vol_modes=(str(high_vol_mode),),
        score_sources=(str(parsed["score"]),),
        shapes=(str(parsed["shape"]),),
        weightings=(str(parsed["weighting"]),),
        tqqq_weights=(float(parsed["tqqq"]),),
        risk_control_modes=(str(parsed["risk"]),),
    )
    for config in configs:
        config = _apply_concentration_overlay_signature(config, parsed.get("conc_ov"))
        if nested.config_signature(config) == str(signature):
            return config
    raise ValueError(f"Signature did not recreate an exact candidate: {signature}")


def _oos_objective(metrics: dict, objective: str) -> float:
    """Score an OOS metric row with the same robustness objective."""
    mapped = {
        "sharpe": metrics.get("sharpe"),
        "alpha_vs_qqq_pct": metrics.get("alpha_vs_qqq_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "turnover_pct": metrics.get("turnover_pct"),
    }
    return float(robustness_score_components(mapped, objective=objective)["robustness_score"])


def _fold_row(split: nested.FoldSplit, signature: str, metrics: dict, *, label: str) -> dict:
    """Format one outer-year result as a CSV-friendly record."""
    return {
        "valid": True,
        "label": label,
        "fold_year": int(split.outer_year),
        "outer_year": int(split.outer_year),
        "selected_config": signature,
        "oos_total_return_pct": metrics.get("total_return_pct"),
        "oos_cagr_pct": metrics.get("cagr_pct"),
        "oos_sharpe": metrics.get("sharpe"),
        "oos_max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "oos_turnover_pct": metrics.get("turnover_pct"),
        "oos_alpha_vs_spy_pct": metrics.get("alpha_vs_spy_pct"),
        "oos_alpha_vs_qqq_pct": metrics.get("alpha_vs_qqq_pct"),
        "oos_alpha_vs_blend_pct": metrics.get("alpha_vs_blend_pct"),
        "oos_benchmark_blend_return_pct": metrics.get("benchmark_blend_return_pct"),
        "oos_n_equity_points": metrics.get("n_equity_points"),
    }


def _summary_for_rows(rows: list[dict], *, label: str, signature: str) -> dict:
    """Summarize one fixed config across all valid outer folds."""
    valid = [row for row in rows if row.get("valid")]
    if not valid:
        return {"label": label, "selected_config": signature, "valid": False}
    returns = np.array([float(row["oos_total_return_pct"]) / 100.0 for row in valid])
    return {
        "label": label,
        "selected_config": signature,
        "valid": True,
        "fold_count": int(len(valid)),
        "compound_oos_return_pct": round(float(np.prod(1.0 + returns) - 1.0) * 100.0, 2),
        "mean_oos_cagr_pct": round(float(np.mean([float(row["oos_cagr_pct"]) for row in valid])), 2),
        "mean_oos_sharpe": round(float(np.mean([float(row["oos_sharpe"]) for row in valid])), 3),
        "mean_oos_alpha_vs_qqq_pct": round(float(np.mean([float(row["oos_alpha_vs_qqq_pct"]) for row in valid])), 2),
        "beat_qqq_folds": int(sum(float(row["oos_alpha_vs_qqq_pct"]) > 0.0 for row in valid)),
        "mean_oos_turnover_pct": round(float(np.mean([float(row["oos_turnover_pct"]) for row in valid])), 2),
        "worst_oos_turnover_pct": round(float(np.max([float(row["oos_turnover_pct"]) for row in valid])), 2),
        "worst_oos_max_drawdown_pct": round(float(np.min([float(row["oos_max_drawdown_pct"]) for row in valid])), 2),
    }


def run_fixed_baseline(
    panel: pd.DataFrame,
    signatures: list[str],
    *,
    labels: list[str] | None,
    min_train_years: int,
    start_year: int | None,
    end_year: int | None,
) -> dict:
    """Evaluate each fixed config on every outer walkforward year."""
    splits = nested.build_fold_splits(
        panel,
        min_train_years=min_train_years,
        start_year=start_year,
        end_year=end_year,
    )
    labels = labels or []
    rows: list[dict] = []
    summaries: list[dict] = []
    for idx, signature in enumerate(signatures):
        label = labels[idx] if idx < len(labels) else f"fixed_{idx + 1}"
        config = config_from_signature(signature)
        config_rows: list[dict] = []
        print(f"\n[fixed] {label}: {signature}")
        for split in splits:
            try:
                metrics = nested.evaluate_window(panel, config, split.outer_start, split.outer_end)
                row = _fold_row(split, signature, metrics, label=label)
                print(
                    f"  {split.outer_year}: Sharpe {float(row['oos_sharpe']):.2f}, "
                    f"QQQ alpha {float(row['oos_alpha_vs_qqq_pct']):.1f}%, "
                    f"turnover {float(row['oos_turnover_pct']):.1f}%"
                )
            except (KeyError, RuntimeError, ValueError, ZeroDivisionError) as exc:
                row = {
                    "valid": False,
                    "label": label,
                    "fold_year": int(split.outer_year),
                    "outer_year": int(split.outer_year),
                    "selected_config": signature,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                print(f"  {split.outer_year}: failed {row['reason']}")
            rows.append(row)
            config_rows.append(row)
            gc.collect()
        summaries.append(_summary_for_rows(config_rows, label=label, signature=signature))
    return {
        "valid": bool(summaries) and all(summary.get("valid") for summary in summaries),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "fixed_config_outer_walkforward_baseline",
        "notes": [
            "Research only: fixed configs are evaluated on each outer test year.",
            "No inner selector is used, so this compares selection against boring fixed baselines.",
        ],
        "summaries": summaries,
        "folds": rows,
    }


def _score_sources(include_riskoff_guard_score: bool) -> tuple[str, ...]:
    """Return score routes for replay, optionally adding the risk-off guard."""
    sources = ["regime_adaptive"]
    if include_riskoff_guard_score:
        sources.append("regime_adaptive_riskoff_guard")
    return tuple(sources)


def _without_score_route_key(config: dict) -> tuple[Any, ...]:
    """Group configs that differ only by score route for bounded A/B replay."""
    params = dict(config.get("nested_params") or {})
    return (
        params.get("holding_days"),
        params.get("overlay_gross"),
        params.get("ma_window"),
        params.get("high_vol"),
        params.get("high_vol_mode"),
        params.get("shape"),
        params.get("weighting"),
        params.get("tqqq_weight"),
        params.get("risk_control_mode"),
    )


def _interleave_score_routes(configs: list[dict], score_sources: tuple[str, ...]) -> list[dict]:
    """Keep paired score-route configs adjacent before applying --max-configs."""
    grouped: dict[tuple[Any, ...], dict[str, dict]] = {}
    order: list[tuple[Any, ...]] = []
    for config in configs:
        key = _without_score_route_key(config)
        if key not in grouped:
            grouped[key] = {}
            order.append(key)
        source = str((config.get("nested_params") or {}).get("score_source"))
        grouped[key][source] = config

    interleaved: list[dict] = []
    for key in order:
        by_source = grouped[key]
        for source in score_sources:
            if source in by_source:
                interleaved.append(by_source[source])
    return interleaved


def _bounded_configs(configs: list[dict], max_configs: int | None) -> list[dict]:
    """Apply the optional CLI cap after any research-only reordering."""
    if max_configs is None:
        return configs
    return configs[: max(0, int(max_configs))]


def _grid_configs(
    name: str,
    *,
    max_configs: int | None,
    include_riskoff_guard_score: bool = False,
) -> list[dict]:
    """Load the candidate grid requested by the replay audit."""
    score_sources = _score_sources(include_riskoff_guard_score)
    if name == "low-turnover":
        configs = nested.iter_candidate_configs(
            holding_days=(20,),
            overlay_gross=nested.LOW_TURNOVER_GRID_OVERLAY_GROSS,
            ma_windows=(100,),
            high_vol_values=(0.30,),
            high_vol_modes=nested.RECENT_ALPHA_GRID_HIGH_VOL_MODES,
            score_sources=score_sources,
            shapes=nested.LOW_TURNOVER_GRID_SHAPES,
            weightings=nested.RECENT_ALPHA_GRID_WEIGHTINGS,
            tqqq_weights=nested.RECENT_ALPHA_GRID_TQQQ_WEIGHTS,
            risk_control_modes=("off",),
        )
    elif name == "recent-alpha":
        configs = nested.iter_candidate_configs(
            holding_days=(20,),
            overlay_gross=nested.RECENT_ALPHA_GRID_OVERLAY_GROSS,
            ma_windows=(100,),
            high_vol_values=(0.30,),
            high_vol_modes=nested.RECENT_ALPHA_GRID_HIGH_VOL_MODES,
            score_sources=score_sources,
            shapes=nested.RECENT_ALPHA_GRID_SHAPES,
            weightings=nested.RECENT_ALPHA_GRID_WEIGHTINGS,
            tqqq_weights=nested.RECENT_ALPHA_GRID_TQQQ_WEIGHTS,
            risk_control_modes=("off",),
        )
    elif name == "stable":
        configs = nested.iter_candidate_configs(
            holding_days=(20,),
            overlay_gross=(0.50,),
            ma_windows=(100,),
            high_vol_values=(0.30,),
            high_vol_modes=nested.STABLE_GRID_HIGH_VOL_MODES,
            score_sources=score_sources,
            shapes=nested.STABLE_GRID_SHAPES,
            weightings=("sticky_score", "risk_parity"),
            tqqq_weights=nested.STABLE_GRID_TQQQ_WEIGHTS,
            risk_control_modes=("off",),
        )
    else:
        configs = nested.iter_candidate_configs(score_sources=score_sources)
    if include_riskoff_guard_score:
        configs = _interleave_score_routes(configs, score_sources)
    return _bounded_configs(configs, max_configs)


def _recent_inner_folds(split: nested.FoldSplit, *, min_train_years: int, min_inner_train_years: int | None) -> list[nested.InnerFold]:
    """Build the same capped recent inner-fold list as nested selection."""
    outer_train_years = list(range(split.train_start.year, split.train_end.year + 1))
    required = int(min_inner_train_years) if min_inner_train_years is not None else max(2, int(min_train_years) - 1)
    folds = nested.build_inner_folds(outer_train_years, min_inner_train_years=required)
    return folds[-MAX_REPLAY_INNER_FOLDS:] if len(folds) > MAX_REPLAY_INNER_FOLDS else folds


def _corr(rows: pd.DataFrame, left: str, right: str, *, method: str) -> float | None:
    """Return a rounded correlation for one replay fold."""
    if rows.empty or left not in rows.columns or right not in rows.columns:
        return None
    pair = rows[[left, right]].dropna()
    if len(pair) < 4 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return None
    value = pair[left].corr(pair[right], method=method)
    return round(float(value), 4) if pd.notna(value) else None


def _evaluate_inner_score_only(config: dict, panel: pd.DataFrame, inner_folds: list[nested.InnerFold]) -> dict | None:
    """Cheap replay path: score inner folds without running cost-stress variants."""
    fold_scores: list[float] = []
    fold_metrics: list[dict[str, Any]] = []
    failed = 0
    for fold in inner_folds:
        try:
            metrics = nested.evaluate_window(
                panel,
                nested.config_with_cost_stress(config, nested.BASE_COST_STRESS),
                fold.validation_start,
                fold.validation_end,
            )
        except (KeyError, RuntimeError, ValueError, ZeroDivisionError):
            failed += 1
            continue

        score = nested.inner_selection_score(metrics)
        fold_scores.append(float(score))
        fold_metrics.append(
            {
                "validation_year": int(fold.validation_year),
                "score": round(float(score), 4),
                "sharpe": metrics.get("sharpe"),
                "return_pct": metrics.get("total_return_pct"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "turnover_pct": metrics.get("turnover_pct"),
                "alpha_vs_spy_pct": metrics.get("alpha_vs_spy_pct"),
                "alpha_vs_qqq_pct": metrics.get("alpha_vs_qqq_pct"),
                "alpha_vs_blend_pct": metrics.get("alpha_vs_blend_pct"),
            }
        )

    if not fold_scores:
        return nested._rejected_inner_config(
            "no_valid_inner_fold_scores",
            failed_evaluations=failed,
            rejection_metrics={"inner_fold_count": int(len(inner_folds))},
        )

    turnover_values = [float(row.get("turnover_pct", 0.0) or 0.0) for row in fold_metrics]
    mean_turnover_pct = float(np.mean(turnover_values))
    worst_turnover_pct = float(np.max(turnover_values)) if turnover_values else 0.0
    if mean_turnover_pct > nested.MAX_INNER_MEAN_TURNOVER_PCT:
        return nested._rejected_inner_config(
            "mean_turnover_cap",
            failed_evaluations=failed,
            rejection_metrics={
                "inner_mean_turnover_pct": round(mean_turnover_pct, 2),
                "cap_pct": float(nested.MAX_INNER_MEAN_TURNOVER_PCT),
            },
        )
    if worst_turnover_pct > nested.MAX_INNER_WORST_TURNOVER_PCT:
        return nested._rejected_inner_config(
            "worst_turnover_cap",
            failed_evaluations=failed,
            rejection_metrics={
                "inner_worst_turnover_pct": round(worst_turnover_pct, 2),
                "cap_pct": float(nested.MAX_INNER_WORST_TURNOVER_PCT),
            },
        )

    mean_score = float(np.mean(fold_scores))
    median_score = float(np.median(fold_scores))
    aggregation = nested.inner_score_aggregation_from_env()
    aggregate_score = median_score if aggregation == "median" else mean_score
    score_std = float(np.std(fold_scores, ddof=0)) if len(fold_scores) > 1 else 0.0
    alpha_vs_qqq_values = [
        float(row.get("alpha_vs_qqq_pct", 0.0) or 0.0)
        for row in fold_metrics
    ]
    mean_alpha_vs_qqq = float(np.mean(alpha_vs_qqq_values)) if alpha_vs_qqq_values else 0.0
    qqq_penalty = max(0.0, -mean_alpha_vs_qqq * 0.05)
    if len(alpha_vs_qqq_values) >= 3 and all(value < 0 for value in alpha_vs_qqq_values):
        qqq_penalty *= 1.5
    stable_score = aggregate_score - 0.10 * score_std - qqq_penalty

    return {
        "config": config,
        "score": stable_score,
        "metrics": {
            "inner_mean_score": round(mean_score, 4),
            "inner_median_score": round(median_score, 4),
            "inner_score_aggregation": aggregation,
            "inner_score_std": round(score_std, 4),
            "inner_stability_adjusted_score": round(stable_score, 4),
            "inner_fold_count": int(len(fold_scores)),
            "inner_failed_fold_count": int(failed),
            "inner_mean_sharpe": round(float(np.mean([float(row.get("sharpe", 0.0) or 0.0) for row in fold_metrics])), 4),
            "inner_mean_return_pct": round(float(np.mean([float(row.get("return_pct", 0.0) or 0.0) for row in fold_metrics])), 2),
            "inner_mean_alpha_vs_spy_pct": round(float(np.mean([float(row.get("alpha_vs_spy_pct", 0.0) or 0.0) for row in fold_metrics])), 2),
            "inner_mean_alpha_vs_qqq_pct": round(mean_alpha_vs_qqq, 2),
            "inner_qqq_opportunity_cost_penalty": round(qqq_penalty, 4),
            "inner_mean_turnover_pct": round(mean_turnover_pct, 2),
            "inner_cost_stress_approval_pass": None,
            "inner_stress_pass_ratio": None,
        },
        "fold_metrics": fold_metrics,
        "failed_evaluations": failed,
    }


def run_candidate_replay(
    panel: pd.DataFrame,
    *,
    grid: str,
    max_configs: int | None,
    min_train_years: int,
    min_inner_train_years: int | None,
    start_year: int | None,
    end_year: int | None,
    skip_stress_gate: bool,
    objective: str,
    include_riskoff_guard_score: bool = False,
    fast_inner_score_only: bool = False,
) -> dict:
    """Replay candidates to test whether inner ranks match OOS ranks."""
    configs = _grid_configs(
        grid,
        max_configs=max_configs,
        include_riskoff_guard_score=include_riskoff_guard_score,
    )
    splits = nested.build_fold_splits(
        panel,
        min_train_years=min_train_years,
        start_year=start_year,
        end_year=end_year,
    )
    rows: list[dict[str, Any]] = []
    yearly: list[dict[str, Any]] = []
    print(f"[replay] {len(splits)} outer folds x {len(configs)} configs from {grid!r}")
    for split in splits:
        inner_folds = _recent_inner_folds(
            split,
            min_train_years=min_train_years,
            min_inner_train_years=min_inner_train_years,
        )
        print(f"\n[replay] {split.outer_year}: {len(configs)} configs x {len(inner_folds)} inner folds")
        year_rows: list[dict[str, Any]] = []
        for idx, config in enumerate(configs, start=1):
            signature = nested.config_signature(config)
            params = dict(config.get("nested_params") or {})
            if fast_inner_score_only:
                inner = _evaluate_inner_score_only(config, panel, inner_folds)
            else:
                inner = nested._evaluate_one_config(
                    config,
                    panel,
                    inner_folds,
                    low_memory=True,
                    best_score_so_far=-np.inf,
                    skip_stress_gate=skip_stress_gate,
                    prior_selected_sigs=None,
                )
            row: dict[str, Any] = {
                "valid": False,
                "fold_year": int(split.outer_year),
                "outer_year": int(split.outer_year),
                "candidate_index": int(idx),
                "selected_config": signature,
                "holding_days": params.get("holding_days"),
                "overlay_gross": params.get("overlay_gross"),
                "ma_window": params.get("ma_window"),
                "high_vol": params.get("high_vol"),
                "high_vol_mode": params.get("high_vol_mode"),
                "score_source": params.get("score_source"),
                "shape": params.get("shape"),
                "weighting": params.get("weighting"),
                "tqqq_weight": params.get("tqqq_weight"),
                "risk_control_mode": params.get("risk_control_mode"),
            }
            if not inner or inner.get("config") is None:
                row["rejection_reason"] = (inner or {}).get("rejection_reason", "missing_inner_result")
                row["rejection_metrics"] = json.dumps((inner or {}).get("rejection_metrics", {}), sort_keys=True)
            else:
                inner_metrics = dict(inner.get("metrics") or {})
                try:
                    outer = nested.evaluate_window(panel, config, split.outer_start, split.outer_end)
                    row.update(
                        {
                            "valid": True,
                            "inner_score": round(float(inner["score"]), 6),
                            "inner_mean_score": inner_metrics.get("inner_mean_score"),
                            "inner_mean_alpha_vs_qqq_pct": inner_metrics.get("inner_mean_alpha_vs_qqq_pct"),
                            "inner_mean_sharpe": inner_metrics.get("inner_mean_sharpe"),
                            "inner_mean_turnover_pct": inner_metrics.get("inner_mean_turnover_pct"),
                            "inner_stress_pass_ratio": inner_metrics.get("inner_stress_pass_ratio"),
                            "oos_objective_score": round(_oos_objective(outer, objective), 6),
                            "oos_sharpe": outer.get("sharpe"),
                            "oos_alpha_vs_qqq_pct": outer.get("alpha_vs_qqq_pct"),
                            "oos_total_return_pct": outer.get("total_return_pct"),
                            "oos_max_drawdown_pct": outer.get("max_drawdown_pct"),
                            "oos_turnover_pct": outer.get("turnover_pct"),
                        }
                    )
                except (KeyError, RuntimeError, ValueError, ZeroDivisionError) as exc:
                    row["rejection_reason"] = f"outer_{type(exc).__name__}"
                    row["rejection_metrics"] = str(exc)
            rows.append(row)
            year_rows.append(row)
            if idx == 1 or idx % 5 == 0 or idx == len(configs):
                print(f"  {idx}/{len(configs)} candidates done", flush=True)
            gc.collect()
        frame = pd.DataFrame(year_rows)
        valid = frame[frame["valid"].eq(True)].copy()
        yearly.append(
            {
                "fold_year": int(split.outer_year),
                "candidate_count": int(len(frame)),
                "valid_candidate_count": int(len(valid)),
                "rejection_counts": {
                    str(reason): int(count)
                    for reason, count in frame["rejection_reason"].dropna().value_counts().items()
                } if "rejection_reason" in frame else {},
                "pearson_inner_vs_oos_objective": _corr(valid, "inner_score", "oos_objective_score", method="pearson"),
                "spearman_inner_vs_oos_objective": _corr(valid, "inner_score", "oos_objective_score", method="spearman"),
                "spearman_inner_vs_oos_qqq_alpha": _corr(valid, "inner_score", "oos_alpha_vs_qqq_pct", method="spearman"),
            }
        )
    valid_frame = pd.DataFrame(rows)
    valid_frame = valid_frame[valid_frame["valid"].eq(True)].copy() if not valid_frame.empty else valid_frame
    return {
        "valid": not valid_frame.empty,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "candidate_inner_to_outer_selector_replay",
        "objective": objective,
        "grid": grid,
        "include_riskoff_guard_score": bool(include_riskoff_guard_score),
        "fast_inner_score_only": bool(fast_inner_score_only),
        "score_sources": list(_score_sources(include_riskoff_guard_score)),
        "candidate_config_count": int(len(configs)),
        "yearly_rank_correlations": yearly,
        "pooled_pearson_inner_vs_oos_objective": _corr(valid_frame, "inner_score", "oos_objective_score", method="pearson"),
        "pooled_spearman_inner_vs_oos_objective": _corr(valid_frame, "inner_score", "oos_objective_score", method="spearman"),
        "notes": [
            "Research only: each valid candidate gets an inner score and the matching OOS score.",
            "Within-year Spearman correlation is the cleanest check of selector ranking quality.",
        ],
        "folds": rows,
    }


def _load_panel(max_specs: int) -> pd.DataFrame:
    """Load one scored panel without rewriting feature-health reports."""
    specs = load_feature_specs(max_specs=int(max_specs), write_health_outputs=False)
    panel = attach_scores(load_factor_panel(specs), specs, load_prediction_scores())
    panel = nested._ensure_robust_score_columns(panel)
    panel["_date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.loc[panel["_date"].notna()].sort_values("_date").reset_index(drop=True)
    for column in panel.select_dtypes(include=["float64"]).columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").astype("float32")
    return panel.copy()


def _write_research(result: dict, output_prefix: str) -> tuple[Path, Path]:
    """Write unique JSON/CSV research outputs using the nested writer."""
    return nested.write_outputs(result, output_prefix=output_prefix, publish_live_config=False)


def _print_baseline(result: dict) -> None:
    """Print the fixed-config summary table."""
    print("\nFIXED CONFIG BASELINE")
    print("=" * 72)
    if not result.get("summaries"):
        print("No summaries.")
        return
    table = pd.DataFrame(result["summaries"])
    columns = [
        "label",
        "fold_count",
        "compound_oos_return_pct",
        "mean_oos_sharpe",
        "mean_oos_alpha_vs_qqq_pct",
        "mean_oos_turnover_pct",
        "worst_oos_turnover_pct",
        "worst_oos_max_drawdown_pct",
        "beat_qqq_folds",
    ]
    print(table.reindex(columns=columns).to_string(index=False))


def _print_replay(result: dict) -> None:
    """Print the candidate replay selector table."""
    print("\nCANDIDATE SELECTOR REPLAY")
    print("=" * 72)
    table = pd.DataFrame(result.get("yearly_rank_correlations", []))
    print(table.to_string(index=False) if not table.empty else "No replay rows.")
    for row in result.get("yearly_rank_correlations", []):
        counts = row.get("rejection_counts") or {}
        if counts:
            compact = ", ".join(f"{reason}={count}" for reason, count in counts.items())
            print(f"  {row.get('fold_year')} rejections: {compact}")
    print(f"\nPooled Pearson inner score vs OOS objective:  {result.get('pooled_pearson_inner_vs_oos_objective')}")
    print(f"Pooled Spearman inner score vs OOS objective: {result.get('pooled_spearman_inner_vs_oos_objective')}")


def main() -> None:
    """Parse research CLI args and run the requested diagnostic."""
    parser = argparse.ArgumentParser(description="Research-only walkforward selector diagnostics.")
    parser.add_argument("--max-specs", type=int, default=nested.DEFAULT_MAX_SPECS)
    parser.add_argument("--min-train-years", type=int, default=nested.DEFAULT_MIN_TRAIN_YEARS)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    fixed = subparsers.add_parser("fixed", help="Evaluate fixed config signatures across outer years.")
    fixed.add_argument("--config", action="append", required=True, help="Walkforward config signature. Repeat for A/B.")
    fixed.add_argument("--label", action="append", default=[], help="Optional label paired with each --config.")
    fixed.add_argument("--output-prefix", default=DEFAULT_BASELINE_PREFIX)

    replay = subparsers.add_parser("replay", help="Replay candidate inner ranks against OOS ranks.")
    replay.add_argument("--grid", choices=("low-turnover", "recent-alpha", "stable", "default"), default="low-turnover")
    replay.add_argument("--max-configs", type=int, default=16)
    replay.add_argument("--min-inner-train-years", type=int, default=None)
    replay.add_argument("--skip-stress-gate", action="store_true")
    replay.add_argument(
        "--fast-inner-score-only",
        action="store_true",
        help=(
            "Replay inner ranks without evaluating cost-stress variants. "
            "Use this for quick score-predictiveness probes; exact nested "
            "selection still requires leaving this off."
        ),
    )
    replay.add_argument("--objective", default=DEFAULT_OBJECTIVE, choices=("alpha_vs_qqq", "sharpe", "hybrid"))
    replay.add_argument(
        "--include-riskoff-guard-score",
        action="store_true",
        help=(
            "Also replay the regime_adaptive_riskoff_guard score route. "
            "This is research-only and does not alter the nested walkforward grid."
        ),
    )
    replay.add_argument("--output-prefix", default=DEFAULT_REPLAY_PREFIX)

    args = parser.parse_args()
    panel = _load_panel(args.max_specs)
    if args.mode == "fixed":
        result = run_fixed_baseline(
            panel,
            list(args.config),
            labels=list(args.label),
            min_train_years=int(args.min_train_years),
            start_year=args.start_year,
            end_year=args.end_year,
        )
        _print_baseline(result)
    else:
        result = run_candidate_replay(
            panel,
            grid=str(args.grid),
            max_configs=args.max_configs,
            min_train_years=int(args.min_train_years),
            min_inner_train_years=args.min_inner_train_years,
            start_year=args.start_year,
            end_year=args.end_year,
            skip_stress_gate=bool(args.skip_stress_gate),
            objective=str(args.objective),
            include_riskoff_guard_score=bool(args.include_riskoff_guard_score),
            fast_inner_score_only=bool(args.fast_inner_score_only),
        )
        _print_replay(result)
    json_path, csv_path = _write_research(result, str(args.output_prefix))
    print("\nWrote research outputs:")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    print("  live configs: not published")


if __name__ == "__main__":
    main()
