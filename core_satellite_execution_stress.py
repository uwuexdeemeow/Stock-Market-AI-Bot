"""
core_satellite_execution_stress.py — delayed-fill and extra-slippage stress
checks for the selected core-satellite strategy.

This does not change production allocation. It answers: if fills are one
trading day late or turnover costs are harsher, does the selected strategy still
beat SPY, QQQ, and the 60/40 blend?
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
import core_satellite_alpha as core
from settings import LOG_DIR, SIGNAL_DIR


OUT_CSV = Path(SIGNAL_DIR) / "core_satellite_execution_stress.csv"
OUT_JSON = Path(LOG_DIR) / "core_satellite_execution_stress.json"

CONFIG_KEYS = (
    "core_preset",
    "regime_mode",
    "core_weights",
    "score_source",
    "shape",
    "weighting",
    "exit_rank_floor",
    "adaptive_exit_mode",
    "max_per_sector",
    "earnings_blackout_days",
    "core_gross",
    "overlay_gross",
    "max_gross_exposure",
    "max_single_name_weight",
    "cost_stress",
    "holding_days",
    "regime_ma_window",
    "regime_high_vol",
)


def _selected_config() -> dict:
    metrics_path = Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json"
    if not metrics_path.exists():
        raise SystemExit("Missing signals/core_satellite_alpha_metrics.json. Run core_satellite_alpha.py first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8", errors="replace"))
    return {key: metrics[key] for key in CONFIG_KEYS if key in metrics}


def _row(name: str, metrics: dict, config: dict) -> dict:
    comps = metrics["benchmark_comparisons"]
    gates = metrics["core_satellite_gate_results"]
    holdout = metrics.get("holdout_2023_2026", {})
    return {
        "scenario": name,
        "entry_delay_days": int(config.get("entry_delay_days", 0)),
        "extra_turnover_cost_bps": float(config.get("extra_turnover_cost_bps", 0.0)),
        "total_return_pct": metrics["total_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "sharpe": metrics["sharpe"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "alpha_vs_spy_pct": comps["SPY"]["alpha_pct"],
        "alpha_vs_qqq_pct": comps["QQQ"]["alpha_pct"],
        "alpha_vs_blend_pct": comps["BLEND"]["alpha_pct"],
        "holdout_alpha_vs_qqq_pct": holdout.get("alpha_vs_qqq_pct", np.nan),
        "holdout_alpha_vs_blend_pct": holdout.get("alpha_vs_blend_pct", np.nan),
        "turnover_pct": metrics["turnover_pct"],
        "estimated_cost_pct": metrics["estimated_cost_pct"],
        "paper_ready": bool(gates["all_pass"]),
    }


def _delta_row(base: dict, stressed: dict) -> dict:
    row = {
        "scenario": f"delta_{stressed['scenario']}_minus_base",
        "entry_delay_days": stressed.get("entry_delay_days"),
        "extra_turnover_cost_bps": stressed.get("extra_turnover_cost_bps"),
    }
    for key in (
        "total_return_pct",
        "cagr_pct",
        "sharpe",
        "max_drawdown_pct",
        "alpha_vs_spy_pct",
        "alpha_vs_qqq_pct",
        "alpha_vs_blend_pct",
        "holdout_alpha_vs_qqq_pct",
        "holdout_alpha_vs_blend_pct",
        "turnover_pct",
        "estimated_cost_pct",
    ):
        b = pd.to_numeric(pd.Series([base.get(key)]), errors="coerce").iloc[0]
        s = pd.to_numeric(pd.Series([stressed.get(key)]), errors="coerce").iloc[0]
        row[key] = round(float(s - b), 4) if np.isfinite(b) and np.isfinite(s) else np.nan
    row["paper_ready"] = stressed.get("paper_ready")
    return row


def main() -> None:
    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    specs = load_feature_specs()
    panel = core._ensure_robust_score_columns(attach_scores(load_factor_panel(specs), specs, load_prediction_scores()))
    base_config = _selected_config()
    scenarios = [
        ("base_selected", {}),
        ("delay_1d", {"entry_delay_days": 1}),
        ("extra_10bps", {"extra_turnover_cost_bps": 10.0}),
        ("delay_1d_extra_10bps", {"entry_delay_days": 1, "extra_turnover_cost_bps": 10.0}),
        ("delay_1d_extra_25bps", {"entry_delay_days": 1, "extra_turnover_cost_bps": 25.0}),
    ]

    rows: list[dict] = []
    for name, overrides in scenarios:
        config = {**base_config, **overrides}
        metrics, _equity, _trades = core.evaluate(panel, config)
        rows.append(_row(name, metrics, config))
    base = rows[0]
    rows.extend(_delta_row(base, row) for row in rows[1:].copy())

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "purpose": "core_satellite_execution_stress",
        "selected_config": base_config,
        "rows": rows,
        "pass_condition": "stressed scenarios should keep positive alpha vs SPY, QQQ, and BLEND and retain strategy gates",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    print(f"Execution stress written -> {OUT_CSV}")
    print(f"Detailed report -> {OUT_JSON}")
    display_cols = [
        "scenario",
        "entry_delay_days",
        "extra_turnover_cost_bps",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "alpha_vs_qqq_pct",
        "alpha_vs_blend_pct",
        "paper_ready",
    ]
    print(out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
