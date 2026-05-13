"""
core_satellite_robust_mode.py — select a calmer core-satellite paper candidate.

Production can stay on the highest-alpha row, but this writes a lower-overlay
candidate for paper comparison. It is meant to answer: how much return do we
give up for slightly better Sharpe/drawdown and less single-name dependence?
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_backtest import MAX_GROSS_EXPOSURE, attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
import core_satellite_alpha as core
from robustness_scoring import robustness_score_components
from settings import SIGNAL_DIR


GRID_PATH = Path(SIGNAL_DIR) / "core_satellite_alpha_grid.csv"
METRICS_PATH = Path(SIGNAL_DIR) / "core_satellite_robust_mode_metrics.json"
SIGNAL_PATH = Path(SIGNAL_DIR) / "core_satellite_robust_mode_signal.csv"


def _selected_row(grid: pd.DataFrame) -> pd.Series:
    candidates = grid[
        grid["paper_ready"].astype(bool)
        & grid["robust_cost_stress_pass"].astype(bool)
        & (pd.to_numeric(grid["cost_stress"], errors="coerce") == 2.0)
        & (pd.to_numeric(grid["overlay_gross"], errors="coerce") <= 0.50)
        & (pd.to_numeric(grid["core_gross"], errors="coerce") >= 0.75)
    ].copy()
    if candidates.empty:
        raise SystemExit("No robust-mode candidates found. Run core_satellite_alpha.py first.")
    if "robustness_score" not in candidates.columns:
        robustness_cols = candidates.apply(
            lambda row: pd.Series(robustness_score_components(row)),
            axis=1,
        )
        candidates = pd.concat([candidates, robustness_cols], axis=1)
    candidates = candidates.sort_values(
        ["robustness_score", "sharpe", "max_drawdown_pct", "alpha_vs_blend_pct"],
        ascending=[False, False, False, False],
    )
    return candidates.iloc[0]


def _config_from_row(row: pd.Series) -> dict:
    config = {
        "core_preset": str(row["core_preset"]),
        "regime_mode": str(row.get("regime_mode", "static")),
        "core_weights": {"SPY": float(row["core_spy_weight"]), "QQQ": float(row["core_qqq_weight"])},
        "score_source": str(row["score_source"]),
        "shape": str(row["shape"]),
        "weighting": str(row["weighting"]),
        "exit_rank_floor": float(row["exit_rank_floor"]),
        "adaptive_exit_mode": str(row.get("adaptive_exit_mode", "fixed")),
        "max_per_sector": int(row["max_per_sector"]),
        "earnings_blackout_days": int(row.get("earnings_blackout_days", 0)),
        "core_gross": float(row["core_gross"]),
        "overlay_gross": float(row["overlay_gross"]),
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "max_single_name_weight": float(row.get("max_single_name_weight", core.MAX_SINGLE_NAME_WEIGHT)),
        "cost_stress": float(row["cost_stress"]),
        "holding_days": int(row["holding_days"]),
    }
    if str(config["regime_mode"]) in core.REGIME_PRESETS:
        preset = core.REGIME_PRESETS[str(config["regime_mode"])]
        config["regime_ma_window"] = int(preset["ma_window"])
        config["regime_high_vol"] = float(preset["high_vol"])
    return config


def _write_signal(panel: pd.DataFrame, metrics: dict) -> Path:
    latest_date = pd.Timestamp(panel["date"].max())
    regime_indicators = None
    if str(metrics.get("regime_mode", "static")) in core.REGIME_PRESETS:
        regime_indicators = core._load_regime_indicators(
            pd.DatetimeIndex([latest_date]),
            pd.DatetimeIndex([latest_date]),
            metrics,
        )
    current_regime, core_weights, core_gross, overlay_gross = core._resolve_allocation(latest_date, metrics, regime_indicators)
    score_col = core._score_col_for_regime(str(metrics["score_source"]), current_regime)
    day = panel[panel["date"] == latest_date]
    selected = core._select_sticky_holdings(
        day,
        set(),
        score_col=score_col,
        return_col=None,
        shape=str(metrics["shape"]),
        exit_rank_floor=float(metrics["exit_rank_floor"]),
        max_per_sector=int(metrics["max_per_sector"]),
        earnings_blackout_days=int(metrics.get("earnings_blackout_days", 0)),
    )
    overlay = core._sticky_overlay_weights(
        selected,
        overlay_gross,
        str(metrics["weighting"]),
        pd.Series(dtype=float),
        max_single_name_weight=float(metrics.get("max_single_name_weight", core.MAX_SINGLE_NAME_WEIGHT)),
    )
    gross = core_gross + float(overlay.abs().sum())
    row = {
        "paper_signal_type": "core_satellite_robust_mode",
        "paper_ready": bool(metrics.get("paper_ready", False)),
        "current_regime": current_regime,
        "score_source": metrics["score_source"],
        "target_spy_weight": round(core_gross * float(core_weights["SPY"]), 4),
        "target_qqq_weight": round(core_gross * float(core_weights["QQQ"]), 4),
        "target_cash_weight": round(1.0 - gross, 4),
        "gross_exposure": round(gross, 4),
        "core_gross": round(core_gross, 4),
        "overlay_gross": round(float(overlay.abs().sum()), 4),
        "earnings_blackout_days": int(metrics.get("earnings_blackout_days", 0)),
        "overlay_tickers": ",".join(overlay.index.astype(str).tolist()),
        "overlay_weights_json": json.dumps({str(k): round(float(v), 6) for k, v in overlay.items()}, sort_keys=True),
        "single_name_stock_picker_enabled": False,
        "factor_overlay_enabled": True,
        "latest_factor_date": str(latest_date.date()),
        "gates_all_pass": bool(metrics.get("core_satellite_gate_results", {}).get("all_pass", False)),
        "reason": "robust-mode comparison candidate; not primary production signal",
        "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    pd.DataFrame([row]).to_csv(SIGNAL_PATH, index=False)
    return SIGNAL_PATH


def main() -> None:
    if not GRID_PATH.exists():
        raise SystemExit("Missing core_satellite_alpha_grid.csv. Run core_satellite_alpha.py first.")
    grid = pd.read_csv(GRID_PATH)
    row = _selected_row(grid)
    config = _config_from_row(row)
    specs = load_feature_specs()
    panel = core._ensure_robust_score_columns(attach_scores(load_factor_panel(specs), specs, load_prediction_scores()))
    metrics, equity, trades = core.evaluate(panel, config)
    metrics["paper_ready"] = bool(metrics["core_satellite_gate_results"]["all_pass"])
    metrics["robustness_score"] = float(row.get("robustness_score", 0))
    metrics["drawdown_penalty"] = float(row.get("drawdown_penalty", 0))
    metrics["turnover_penalty"] = float(row.get("turnover_penalty", 0))
    metrics["instability_penalty"] = float(row.get("instability_penalty", 0))
    metrics["selection_reason"] = "best passing <=0.50 overlay candidate sorted by robustness score"
    metrics["primary_strategy_metrics_path"] = str(Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"equity": equity}).to_csv(Path(SIGNAL_DIR) / "core_satellite_robust_mode_equity.csv")
    trades.to_csv(Path(SIGNAL_DIR) / "core_satellite_robust_mode_trades.csv", index=False)
    signal_path = _write_signal(panel, metrics)

    comps = metrics["benchmark_comparisons"]
    print("Robust-mode candidate written")
    print(f"  metrics: {METRICS_PATH}")
    print(f"  signal:  {signal_path}")
    print(
        f"  return {metrics['total_return_pct']:,.2f}% | Sharpe {metrics['sharpe']:.3f} | "
        f"max DD {metrics['max_drawdown_pct']:,.2f}% | alpha vs QQQ {comps['QQQ']['alpha_pct']:,.2f}%"
    )


if __name__ == "__main__":
    main()
