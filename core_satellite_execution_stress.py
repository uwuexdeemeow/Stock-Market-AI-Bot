"""
core_satellite_execution_stress.py — delayed-fill and extra-slippage stress
checks for the selected core-satellite strategy.

This does not change production allocation. It answers: if fills are one
trading day late or turnover costs are harsher, does the selected strategy still
beat SPY, QQQ, and the 60/40 blend?
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
import core_satellite_alpha as core
from settings import LOG_DIR, SIGNAL_DIR
from safe_io import atomic_write_csv, atomic_write_json
from validation_bundle import add_validation_context, load_approved_research_config


OUT_CSV = Path(SIGNAL_DIR) / "core_satellite_execution_stress.csv"
OUT_JSON = Path(LOG_DIR) / "core_satellite_execution_stress.json"

# PLAIN ENGLISH: research candidates (for example a Colab walk-forward
# winner) get their own file name prefix in logs/. They must never overwrite
# the official stress reports above, because the daily paper-trading gate
# reads those official files.
CANDIDATE_PREFIX = "research_candidate_execution_stress"


def load_candidate_config(path: str | Path) -> dict:
    """Read a research candidate's strategy settings from a JSON file.

    PLAIN ENGLISH: two file shapes are accepted:
    1. a walk-forward result, where the settings sit under
       ``approved_live_config -> config``;
    2. a plain settings file, where the whole JSON object *is* the config.
    Nothing is published or approved here; we only read the settings so a
    stress test can evaluate them.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Candidate JSON must hold an object: {path}")
    approved = payload.get("approved_live_config")
    if isinstance(approved, dict) and "config" in approved:
        config = approved.get("config")
    elif "approved_live_config" in payload or "live_config_approval" in payload:
        # A walk-forward result WITHOUT a selected config: there is nothing
        # to stress, so stop instead of guessing.
        raise SystemExit(f"Walk-forward result has no approved_live_config.config: {path}")
    else:
        config = payload
    if not isinstance(config, dict) or not config:
        raise SystemExit(f"Candidate config is empty or not an object: {path}")
    return dict(config)


def candidate_name(path: str | Path, name: str | None = None) -> str:
    """Turn a file name (or a chosen name) into a safe file-name piece."""
    raw = name or Path(path).stem
    # Keep only letters, digits, dot, dash and underscore.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw)).strip("._")
    return safe or "candidate"


def mark_research_candidate(payload: dict, *, candidate_path: str | Path, name: str) -> dict:
    """Stamp a report so no gate can mistake it for approval evidence."""
    return {
        **payload,
        "research_candidate": True,
        "approves_trading": False,
        "candidate_source": str(candidate_path),
        "candidate_name": name,
    }


def _selected_config() -> dict:
    metrics_path = Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json"
    if not metrics_path.exists():
        raise SystemExit("Missing signals/core_satellite_alpha_metrics.json. Run core_satellite_alpha.py first.")
    return load_approved_research_config(metrics_path)


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
        # PLAIN ENGLISH: list the exact safety checks that rejected a stress
        # case, so a failed workflow does not hide the reason behind False.
        "failed_gates": [
            key for key, passed in gates.items()
            if key != "all_pass" and key.endswith("_pass") and not bool(passed)
        ],
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--candidate-json",
        default=None,
        help="Stress a research candidate config from this JSON file instead of the approved live config. "
             "Writes only to logs/research_candidate_execution_stress_<name>.* and never approves trading.",
    )
    parser.add_argument(
        "--candidate-name",
        default=None,
        help="Optional short name for the candidate output files (default: the JSON file name).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_csv, out_json = OUT_CSV, OUT_JSON
    cand_name = None
    if args.candidate_json:
        # PLAIN ENGLISH: research mode. Load the candidate first, so a bad
        # file stops the run before any slow data loading happens.
        base_config = load_candidate_config(args.candidate_json)
        cand_name = candidate_name(args.candidate_json, args.candidate_name)
        out_csv = Path(LOG_DIR) / f"{CANDIDATE_PREFIX}_{cand_name}.csv"
        out_json = Path(LOG_DIR) / f"{CANDIDATE_PREFIX}_{cand_name}.json"
    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    specs = load_feature_specs()
    panel = core._ensure_robust_score_columns(attach_scores(load_factor_panel(specs, require_forward_returns=False), specs, load_prediction_scores()))
    if not args.candidate_json:
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
    atomic_write_csv(out, out_csv, index=False)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "purpose": "core_satellite_execution_stress",
        "selected_config": base_config,
        "rows": rows,
        "pass_condition": "stressed scenarios should keep positive alpha vs SPY, QQQ, and BLEND and retain strategy gates",
    }
    if cand_name is not None:
        payload = mark_research_candidate(payload, candidate_path=args.candidate_json, name=cand_name)
    atomic_write_json(add_validation_context(payload, config=base_config), out_json)

    print(f"Execution stress written -> {out_csv}")
    print(f"Detailed report -> {out_json}")
    if cand_name is not None:
        print("Research candidate only: this report does not approve trading.")
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
        "failed_gates",
    ]
    print(out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
