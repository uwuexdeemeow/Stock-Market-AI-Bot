"""
core_satellite_survivorship_audit.py — stress the selected core-satellite
strategy with available failed/delisted audit tickers.

This is not a true point-in-time S&P 500 constituent database. It is a practical
guardrail using the failed-name histories that exist locally, so we can see
whether the current factor overlay avoids obvious survivor-only bias when those
names are allowed into the historical ranking pool.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import alpha_factor_backtest as alpha
import core_satellite_alpha as core
from settings import LOG_DIR, SIGNAL_DIR, SURVIVORSHIP_AUDIT_TICKERS, WATCHLIST
from safe_io import atomic_write_csv, atomic_write_json
from survivorship_audit import available_audit_tickers, existing_audit_profiles
from validation_bundle import add_validation_context
from universe_membership import membership_status


OUT_JSON = Path(LOG_DIR) / "core_satellite_survivorship_audit.json"
OUT_CSV = Path(SIGNAL_DIR) / "core_satellite_survivorship_audit.csv"


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
    "deployment_max_gross_exposure",
    "max_single_name_weight",
    "cost_stress",
    "holding_days",
    "regime_ma_window",
    "regime_high_vol",
    # Keep report identity aligned with the approved live configuration.
    "high_vol_mode",
    "tqqq_weight",
    "risk_control_mode",
)


def _load_selected_config() -> dict:
    metrics_path = Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json"
    if not metrics_path.exists():
        raise SystemExit("Missing signals/core_satellite_alpha_metrics.json. Run core_satellite_alpha.py first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8", errors="replace"))
    return {key: metrics[key] for key in CONFIG_KEYS if key in metrics}


def _with_alpha_watchlist(tickers: Iterable[str], fn):
    old = list(alpha.WATCHLIST)
    alpha.WATCHLIST = list(dict.fromkeys(str(t).upper() for t in tickers))
    try:
        return fn()
    finally:
        alpha.WATCHLIST = old


def _build_panel(tickers: list[str]) -> pd.DataFrame:
    def _load() -> pd.DataFrame:
        specs = alpha.load_feature_specs()
        panel = alpha.attach_scores(alpha.load_factor_panel(specs, require_forward_returns=False), specs, alpha.load_prediction_scores())
        return core._ensure_robust_score_columns(panel)

    return _with_alpha_watchlist(tickers, _load)


def _evaluate(tickers: list[str], config: dict) -> tuple[dict, pd.DataFrame]:
    panel = _build_panel(tickers)
    metrics, _equity, trades = core.evaluate(panel, config)
    return metrics, trades


def _metric_row(name: str, metrics: dict, tickers: list[str], audit_tickers: list[str], trades: pd.DataFrame) -> dict:
    comps = metrics.get("benchmark_comparisons", {})
    overlay_series = trades.get("overlay_tickers", pd.Series(dtype=str)).fillna("").astype(str)
    selected_audit_counts = {ticker: int(overlay_series.str.split(",").apply(lambda xs, t=ticker: t in xs).sum()) for ticker in audit_tickers}
    audit_rebalance_count = int(sum(selected_audit_counts.values()))
    return {
        "scenario": name,
        "universe_size": int(len(tickers)),
        "audit_tickers": ",".join(audit_tickers),
        "audit_rebalance_selections": audit_rebalance_count,
        "audit_selection_counts_json": json.dumps(selected_audit_counts, sort_keys=True),
        "total_return_pct": metrics.get("total_return_pct"),
        "cagr_pct": metrics.get("cagr_pct"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "alpha_vs_spy_pct": comps.get("SPY", {}).get("alpha_pct"),
        "alpha_vs_qqq_pct": comps.get("QQQ", {}).get("alpha_pct"),
        "alpha_vs_blend_pct": comps.get("BLEND", {}).get("alpha_pct"),
        "turnover_pct": metrics.get("turnover_pct"),
        "estimated_cost_pct": metrics.get("estimated_cost_pct"),
        "paper_ready": bool(metrics.get("core_satellite_gate_results", {}).get("all_pass", False)),
    }


def _delta_row(base: dict, stressed: dict) -> dict:
    row = {"scenario": "delta_stressed_minus_base"}
    for key in (
        "total_return_pct",
        "cagr_pct",
        "sharpe",
        "max_drawdown_pct",
        "alpha_vs_spy_pct",
        "alpha_vs_qqq_pct",
        "alpha_vs_blend_pct",
        "turnover_pct",
        "estimated_cost_pct",
    ):
        b = pd.to_numeric(pd.Series([base.get(key)]), errors="coerce").iloc[0]
        s = pd.to_numeric(pd.Series([stressed.get(key)]), errors="coerce").iloc[0]
        row[key] = round(float(s - b), 4) if np.isfinite(b) and np.isfinite(s) else np.nan
    return row


def main() -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    config = _load_selected_config()
    profiles = existing_audit_profiles()
    audit_tickers = available_audit_tickers(profiles)
    known_audit_tickers = sorted(SURVIVORSHIP_AUDIT_TICKERS)
    universe_status = membership_status()

    if not audit_tickers:
        raise SystemExit("No available survivorship audit tickers. Run survivorship_audit.py --build --report first.")

    base_tickers = list(WATCHLIST)
    stressed_tickers = list(dict.fromkeys(base_tickers + [t for t in audit_tickers if t not in base_tickers]))

    base_metrics, base_trades = _evaluate(base_tickers, config)
    stressed_metrics, stressed_trades = _evaluate(stressed_tickers, config)

    base_row = _metric_row("base_watchlist", base_metrics, base_tickers, audit_tickers, base_trades)
    stressed_row = _metric_row("watchlist_plus_failed_audit_tickers", stressed_metrics, stressed_tickers, audit_tickers, stressed_trades)
    delta = _delta_row(base_row, stressed_row)
    rows = [base_row, stressed_row, delta]
    base_return = float(base_row.get("total_return_pct", 0.0) or 0.0)
    stressed_return = float(stressed_row.get("total_return_pct", 0.0) or 0.0)
    survivorship_adjusted_score = stressed_return / max(base_return, 1e-9)
    out = pd.DataFrame(rows)
    atomic_write_csv(out, OUT_CSV, index=False)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "purpose": "core_satellite_survivorship_bias_stress",
        "selected_config": config,
        "base_universe_size": len(base_tickers),
        "stressed_universe_size": len(stressed_tickers),
        "known_audit_tickers": known_audit_tickers,
        "available_audit_tickers": audit_tickers,
        "missing_audit_tickers": [t for t in known_audit_tickers if t not in audit_tickers],
        # PLAIN ENGLISH: A good result on five failed companies cannot prove
        # that the other twelve, or omitted historical constituents, were safe.
        # Keep these completeness facts explicit so a future capital gate can
        # never mistake this useful partial stress test for complete evidence.
        "failed_name_coverage_count": len(audit_tickers),
        "failed_name_required_count": len(known_audit_tickers),
        "failed_name_coverage_rate": round(len(audit_tickers) / max(len(known_audit_tickers), 1), 4),
        "failed_name_coverage_complete": set(audit_tickers) == set(known_audit_tickers),
        "point_in_time_universe_complete": bool(universe_status.get("complete", False)),
        "point_in_time_universe_status": universe_status,
        "survivorship_adjusted_score": round(float(survivorship_adjusted_score), 4),
        "primary_return_pct": base_return,
        "failed_name_stressed_return_pct": stressed_return,
        "stressed_return_delta_pct": float(delta.get("total_return_pct", 0.0) or 0.0),
        "limitations": [
            "This is a failed-name stress test, not a complete point-in-time constituent database.",
            "Only locally available audit tickers can be included.",
            "The production strategy is unchanged; this script only evaluates research scenarios.",
        ],
        "rows": rows,
        "profile_summary": profiles,
    }
    atomic_write_json(add_validation_context(payload, config=config), OUT_JSON)

    print(f"Core-satellite survivorship audit written -> {OUT_CSV}")
    print(f"Detailed report -> {OUT_JSON}")
    display_cols = [
        "scenario",
        "universe_size",
        "audit_rebalance_selections",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "alpha_vs_qqq_pct",
        "alpha_vs_blend_pct",
        "paper_ready",
    ]
    print(out[[c for c in display_cols if c in out.columns]].to_string(index=False))
    print(f"\nSurvivorship-adjusted score: {survivorship_adjusted_score:.4f}")
    print("\nAvailable failed-name audit tickers:", ", ".join(audit_tickers))
    print("Missing failed-name audit tickers:", ", ".join(payload["missing_audit_tickers"]) or "none")


if __name__ == "__main__":
    main()
