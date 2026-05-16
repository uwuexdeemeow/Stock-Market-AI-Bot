from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core_satellite_alpha as core
from alpha_factor_backtest import MAX_GROSS_EXPOSURE, attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
from robustness_scoring import robustness_score_components
from settings import SIGNAL_DIR


OUT_PATH = Path(SIGNAL_DIR) / "core_satellite_robust_research.csv"


def _preset(name: str, *, overlay_gross: float, ma_window: int, high_vol: float) -> dict:
    core_gross = round(MAX_GROSS_EXPOSURE - overlay_gross, 4)
    return {
        "ma_window": int(ma_window),
        "high_vol": float(high_vol),
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": core_gross,
            "overlay_gross": float(overlay_gross),
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": core_gross,
            "overlay_gross": float(overlay_gross),
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": min(0.65, core_gross),
            "overlay_gross": min(0.35, float(overlay_gross)),
        },
    }


def _iter_base_configs(*, full: bool) -> list[dict]:
    overlay_options = (0.50, 0.60, 0.70)
    shape_options = ("top3", "top5")
    weighting_options = ("score", "equal", "vol_score", "sticky_score")
    floor_options = (0.70, 0.80)
    sector_options = (1, 2)
    earnings_blackout_options = (0, 5)
    ma_options = (75, 100, 125) if full else (100,)
    vol_options = (0.25, 0.30, 0.35) if full else (0.30,)
    holding_options = (10, 20) if full else (10,)
    score_sources = ("regime_adaptive", "regime_adaptive_low_vol", "regime_adaptive_consensus")

    configs: list[dict] = []
    for overlay_gross in overlay_options:
        for ma_window in ma_options:
            for high_vol in vol_options:
                preset_name = f"robust_overlay{int(overlay_gross * 100):02d}_ma{ma_window}_vol{int(high_vol * 100):02d}"
                preset = _preset(preset_name, overlay_gross=overlay_gross, ma_window=ma_window, high_vol=high_vol)
                for score_source in score_sources:
                    for shape in shape_options:
                        for weighting in weighting_options:
                            for exit_floor in floor_options:
                                for max_sector in sector_options:
                                    for earnings_blackout_days in earnings_blackout_options:
                                        for holding_days in holding_options:
                                            configs.append({
                                                "core_preset": preset_name,
                                                "regime_mode": preset_name,
                                                "regime_preset": preset,
                                                "regime_ma_window": int(ma_window),
                                                "regime_high_vol": float(high_vol),
                                                "core_weights": dict(preset["risk_on"]["core_weights"]),
                                                "score_source": score_source,
                                                "shape": shape,
                                                "weighting": weighting,
                                                "exit_rank_floor": float(exit_floor),
                                                "adaptive_exit_mode": "fixed",
                                                "max_per_sector": int(max_sector),
                                                "earnings_blackout_days": int(earnings_blackout_days),
                                                "core_gross": float(preset["risk_on"]["core_gross"]),
                                                "overlay_gross": float(preset["risk_on"]["overlay_gross"]),
                                                "max_gross_exposure": MAX_GROSS_EXPOSURE,
                                                "max_single_name_weight": core.MAX_SINGLE_NAME_WEIGHT,
                                                "cost_stress": 2.0,
                                                "holding_days": int(holding_days),
                                            })
    return configs


def _row(metrics: dict, config: dict) -> dict:
    comps = metrics["benchmark_comparisons"]
    gates = metrics["core_satellite_gate_results"]
    holdout = metrics.get("holdout_2023_2026", {})
    return {
        "core_preset": config["core_preset"],
        "regime_ma_window": config["regime_ma_window"],
        "regime_high_vol": config["regime_high_vol"],
        "score_source": config["score_source"],
        "shape": config["shape"],
        "weighting": config["weighting"],
        "exit_rank_floor": config["exit_rank_floor"],
        "max_per_sector": config["max_per_sector"],
        "earnings_blackout_days": config.get("earnings_blackout_days", 0),
        "core_gross": config["core_gross"],
        "overlay_gross": config["overlay_gross"],
        "cost_stress": config["cost_stress"],
        "holding_days": config["holding_days"],
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
        "max_single_name_weight": metrics["max_single_name_weight"],
        "avg_effective_overlay_names": metrics["avg_effective_overlay_names"],
        "max_sector_overlay_weight": metrics["max_sector_overlay_weight"],
        "top_ticker_overlay_contribution_share": metrics["top_ticker_overlay_contribution_share"],
        "top_ticker_overlay_contributor": metrics["top_ticker_overlay_contributor"],
        "paper_ready": gates["all_pass"],
        "year_alpha_concentration_pass": gates["year_alpha_concentration_pass"],
        "single_name_weight_cap_pass": gates["single_name_weight_cap_pass"],
        "top_ticker_contribution_pass": gates["top_ticker_contribution_pass"],
        "holdout_2023_2026_vs_qqq_pass": gates["holdout_2023_2026_vs_qqq_pass"],
        "holdout_2023_2026_vs_blend_pass": gates["holdout_2023_2026_vs_blend_pass"],
        **robustness_score_components({**metrics, **gates}),
    }


def _evaluate_with_temp_preset(panel: pd.DataFrame, config: dict) -> dict:
    preset_name = str(config["regime_mode"])
    old = core.REGIME_PRESETS.get(preset_name)
    core.REGIME_PRESETS[preset_name] = config["regime_preset"]
    eval_config = {k: v for k, v in config.items() if k != "regime_preset"}
    try:
        metrics, _equity, _trades = core.evaluate(panel, eval_config)
        return _row(metrics, eval_config)
    finally:
        if old is None:
            core.REGIME_PRESETS.pop(preset_name, None)
        else:
            core.REGIME_PRESETS[preset_name] = old


def run_research(*, full: bool, stress_top_n: int) -> pd.DataFrame:
    specs = load_feature_specs()
    panel = core._ensure_robust_score_columns(attach_scores(load_factor_panel(specs), specs, load_prediction_scores()))
    rows: list[dict] = []
    configs = _iter_base_configs(full=full)

    for config in configs:
        rows.append(_evaluate_with_temp_preset(panel, config))

    base = pd.DataFrame(rows).sort_values(
        ["paper_ready", "robustness_score", "sharpe", "max_drawdown_pct"],
        ascending=[False, False, False, False],
    )
    stress_configs: list[dict] = []
    lookup = {
        (
            c["core_preset"],
            c["score_source"],
            c["shape"],
            c["weighting"],
            c["exit_rank_floor"],
            c["max_per_sector"],
            c.get("earnings_blackout_days", 0),
            c["holding_days"],
        ): c
        for c in configs
    }
    for row in base[base["paper_ready"]].head(int(stress_top_n)).to_dict("records"):
        key = (
            row["core_preset"],
            row["score_source"],
            row["shape"],
            row["weighting"],
            float(row["exit_rank_floor"]),
            int(row["max_per_sector"]),
            int(row.get("earnings_blackout_days", 0)),
            int(row["holding_days"]),
        )
        original = lookup.get(key)
        if original is None:
            continue
        for cost in (3.0, 5.0):
            stressed = dict(original)
            stressed["cost_stress"] = cost
            stress_configs.append(stressed)
    for config in stress_configs:
        rows.append(_evaluate_with_temp_preset(panel, config))

    out = pd.DataFrame(rows)
    group_cols = [
        "core_preset",
        "regime_ma_window",
        "regime_high_vol",
        "score_source",
        "shape",
        "weighting",
        "exit_rank_floor",
        "max_per_sector",
        "earnings_blackout_days",
        "core_gross",
        "overlay_gross",
        "holding_days",
    ]
    stress = out.groupby(group_cols, dropna=False).agg(
        stress_levels=("cost_stress", lambda s: ",".join(str(float(v)) for v in sorted(set(s)))),
        stress_min_alpha_vs_spy_pct=("alpha_vs_spy_pct", "min"),
        stress_min_alpha_vs_qqq_pct=("alpha_vs_qqq_pct", "min"),
        stress_min_alpha_vs_blend_pct=("alpha_vs_blend_pct", "min"),
        stress_all_gates_pass=("paper_ready", "all"),
    ).reset_index()
    required = {2.0, 3.0, 5.0}
    stress["robust_promotion_candidate"] = (
        stress["stress_levels"].apply(lambda x: required.issubset({float(v) for v in str(x).split(",") if v}))
        & stress["stress_all_gates_pass"]
        & (stress["stress_min_alpha_vs_spy_pct"] > 0)
        & (stress["stress_min_alpha_vs_qqq_pct"] > 0)
        & (stress["stress_min_alpha_vs_blend_pct"] > 0)
    )
    out = out.merge(stress, on=group_cols, how="left")
    out = out.sort_values(
        ["robust_promotion_candidate", "paper_ready", "robustness_score", "sharpe", "max_drawdown_pct"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust core-satellite alpha research grid")
    parser.add_argument("--full", action="store_true", help="Run the full predeclared grid. Default is a focused nearby scan.")
    parser.add_argument("--stress-top-n", type=int, default=12, help="Number of passing 2x-cost rows to stress at 3x/5x.")
    args = parser.parse_args()

    out = run_research(full=bool(args.full), stress_top_n=int(args.stress_top_n))
    print(f"Robust research written -> {OUT_PATH}")
    cols = [
        "robust_promotion_candidate",
        "core_preset",
        "score_source",
        "shape",
        "weighting",
        "exit_rank_floor",
        "max_per_sector",
        "earnings_blackout_days",
        "cost_stress",
        "total_return_pct",
        "robustness_score",
        "drawdown_penalty",
        "turnover_penalty",
        "instability_penalty",
        "sharpe",
        "max_drawdown_pct",
        "alpha_vs_qqq_pct",
        "alpha_vs_blend_pct",
        "stress_levels",
    ]
    print(out[[c for c in cols if c in out.columns]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
