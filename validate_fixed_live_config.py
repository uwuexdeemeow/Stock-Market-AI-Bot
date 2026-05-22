"""
validate_fixed_live_config.py - validate one fixed live candidate safely.

PLAIN ENGLISH:
Nested walkforward selection can hop between configs year by year.  This script
tests a simpler proposal: use one fixed config on every out-of-sample year and
gate that fixed candidate before it is ever allowed near paper trading.

By default this is research-only.  It writes a unique JSON/CSV pair under
``signals/`` and never touches ``core_satellite_live_configs.json`` unless the
user explicitly passes ``--publish-live-config``.
"""
from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import core_satellite_nested_walkforward as nested
from safe_io import configure_console_output
from walkforward_selector_diagnostics import _fold_row, _load_panel, config_from_signature


configure_console_output()

DEFAULT_OUTPUT_PREFIX = "wf_fixed_live_validation"
FIXED_COST_STRESS_MIN_PASS_RATIO = 0.60


def _mean(rows: list[dict], key: str) -> float:
    """Return the arithmetic mean for a numeric result field."""
    return float(np.mean([float(row[key]) for row in rows]))


def _fixed_summary(
    rows: list[dict],
    *,
    signature: str,
    config: dict,
    review: dict | None = None,
) -> dict[str, Any]:
    """Build the fixed-config metrics consumed by the existing live gates."""
    valid = [row for row in rows if row.get("valid")]
    if not valid:
        return {"valid": False, "reason": "all_fixed_outer_folds_failed", "folds": rows}

    returns = np.array([float(row["oos_total_return_pct"]) / 100.0 for row in valid], dtype=float)
    stress_passes = sum(bool(row.get("fixed_cost_stress_approval_pass", False)) for row in valid)
    stress_ratio = float(stress_passes / len(valid))
    family = nested.stable_family_signature(config)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "valid": True,
        "created_at": created_at,
        "method": "fixed_config_outer_walkforward_validation",
        "strategy": str(config.get("strategy", "core-alpha")),
        "fixed_config_signature": signature,
        "fold_count": int(len(valid)),
        "failed_fold_count": int(len(rows) - len(valid)),
        "outer_years": [int(row["outer_year"]) for row in valid],
        "compound_oos_return_pct": round(float(np.prod(1.0 + returns) - 1.0) * 100.0, 2),
        "mean_oos_return_pct": round(float(np.mean(returns)) * 100.0, 2),
        "mean_oos_cagr_pct": round(_mean(valid, "oos_cagr_pct"), 2),
        "mean_oos_sharpe": round(_mean(valid, "oos_sharpe"), 3),
        "median_oos_sharpe": round(float(np.median([float(row["oos_sharpe"]) for row in valid])), 3),
        "mean_oos_max_drawdown_pct": round(_mean(valid, "oos_max_drawdown_pct"), 2),
        "worst_oos_max_drawdown_pct": round(float(np.min([float(row["oos_max_drawdown_pct"]) for row in valid])), 2),
        "mean_oos_turnover_pct": round(_mean(valid, "oos_turnover_pct"), 2),
        "worst_oos_turnover_pct": round(float(np.max([float(row["oos_turnover_pct"]) for row in valid])), 2),
        "worst_oos_return_pct": round(float(np.min(returns)) * 100.0, 2),
        "mean_oos_alpha_vs_blend_pct": round(_mean(valid, "oos_alpha_vs_blend_pct"), 2),
        "mean_oos_alpha_vs_spy_pct": round(_mean(valid, "oos_alpha_vs_spy_pct"), 2),
        "mean_oos_alpha_vs_qqq_pct": round(_mean(valid, "oos_alpha_vs_qqq_pct"), 2),
        # Keep the gate definition aligned with nested walkforward: alpha hit
        # rate is the strategy edge versus the blended benchmark.
        "oos_positive_alpha_hit_rate": round(
            float(np.mean([float(row["oos_alpha_vs_blend_pct"]) > 0.0 for row in valid])),
            3,
        ),
        "best_config_frequency": 1.0,
        "approved_config_fold_count": int(len(valid)),
        "approved_config_frequency": 1.0,
        "approved_exact_config": signature,
        "approved_family_signature": family,
        "approved_family_fold_count": int(len(valid)),
        "approved_family_frequency": 1.0,
        "approved_family_worst_oos_turnover_pct": round(float(np.max([float(row["oos_turnover_pct"]) for row in valid])), 2),
        "approved_family_mean_oos_max_drawdown_pct": round(_mean(valid, "oos_max_drawdown_pct"), 2),
        "approved_family_mean_oos_sharpe": round(_mean(valid, "oos_sharpe"), 3),
        "most_common_config": signature,
        # Fixed validation does not optimize on inner folds, so there is no
        # inner-vs-OOS winner's-curse gap to charge to this gate.
        "selection_bias_gap_sharpe": 0.0,
        "fixed_cost_stress_passed_folds": int(stress_passes),
        "fixed_cost_stress_tested_folds": int(len(valid)),
        "fixed_cost_stress_pass_ratio": round(stress_ratio, 3),
        "cost_stress_approval_pass": bool(stress_ratio >= FIXED_COST_STRESS_MIN_PASS_RATIO),
        "required_cost_stresses": [float(value) for value in nested.COST_STRESS_MULTIPLIERS],
        "folds": rows,
        "notes": [
            "Fixed candidate only: no yearly config selector is used.",
            f"Cost stress must pass on at least {FIXED_COST_STRESS_MIN_PASS_RATIO:.0%} of outer folds.",
        ],
    }
    approval = nested.approval_status(summary)
    summary["live_config_approval"] = {
        **approval,
        "strategy": summary["strategy"],
        "approved_config_family": family,
        "approved_family_signature": family,
        "approved_family_fold_count": int(len(valid)),
        "approved_family_frequency": 1.0,
        "approved_exact_config": signature,
        "approved_config_fold_count": int(len(valid)),
        "approved_config_frequency": 1.0,
        "source": "fixed_config_validation",
        "created_at": created_at,
    }
    if approval["approved"]:
        summary["approved_live_config"] = {
            "strategy": summary["strategy"],
            "approved_config_family": family,
            "approved_family_signature": family,
            "approved_exact_config": signature,
            "config": nested.live_signal_config(config),
            "source_metrics": {
                "fold_count": summary["fold_count"],
                "best_config_frequency": 1.0,
                "approved_family_fold_count": summary["approved_family_fold_count"],
                "approved_family_frequency": 1.0,
                "approved_family_worst_oos_turnover_pct": summary["approved_family_worst_oos_turnover_pct"],
                "approved_family_mean_oos_max_drawdown_pct": summary["approved_family_mean_oos_max_drawdown_pct"],
                "approved_family_mean_oos_sharpe": summary["approved_family_mean_oos_sharpe"],
                "mean_oos_sharpe": summary["mean_oos_sharpe"],
                "mean_oos_cagr_pct": summary["mean_oos_cagr_pct"],
                "mean_oos_alpha_vs_spy_pct": summary["mean_oos_alpha_vs_spy_pct"],
                "mean_oos_alpha_vs_qqq_pct": summary["mean_oos_alpha_vs_qqq_pct"],
                "oos_positive_alpha_hit_rate": summary["oos_positive_alpha_hit_rate"],
                "cost_stress_approval_pass": summary["cost_stress_approval_pass"],
                "fixed_cost_stress_pass_ratio": summary["fixed_cost_stress_pass_ratio"],
                "required_cost_stresses": summary["required_cost_stresses"],
                "mean_oos_max_drawdown_pct": summary["mean_oos_max_drawdown_pct"],
                "worst_oos_max_drawdown_pct": summary["worst_oos_max_drawdown_pct"],
                "worst_oos_turnover_pct": summary["worst_oos_turnover_pct"],
                "worst_oos_return_pct": summary["worst_oos_return_pct"],
                "selection_bias_gap_sharpe": 0.0,
                "approved_config_fold_count": summary["approved_config_fold_count"],
                "approved_config_frequency": 1.0,
            },
        }
    return nested.apply_medium_risk_review(summary, review=review)


def validate_fixed_config(
    panel: pd.DataFrame,
    signature: str,
    *,
    min_train_years: int,
    start_year: int | None,
    end_year: int | None,
) -> dict[str, Any]:
    """Evaluate one fixed candidate and run outer-fold cost-stress checks."""
    config = config_from_signature(signature)
    splits = nested.build_fold_splits(
        panel,
        min_train_years=min_train_years,
        start_year=start_year,
        end_year=end_year,
    )
    rows: list[dict[str, Any]] = []
    eval_cache: dict[str, dict] = {}
    print(f"[fixed-live] {len(splits)} outer folds for {signature}")
    for split in splits:
        try:
            metrics = nested.evaluate_window(panel, config, split.outer_start, split.outer_end)
            stress_fold = nested.InnerFold(
                validation_year=int(split.outer_year),
                train_end=split.train_end,
                validation_start=split.outer_start,
                validation_end=split.outer_end,
            )
            stress = nested.nested_cost_stress_approval(
                panel,
                config,
                stress_fold,
                eval_cache,
                base_metrics=metrics,
            )
            row = _fold_row(split, signature, metrics, label="fixed_live")
            row["fixed_cost_stress_approval_pass"] = bool(stress["cost_stress_approval_pass"])
            row["fixed_cost_stress_summary"] = stress["cost_stress_summary"]
            print(
                f"  {split.outer_year}: Sharpe {float(row['oos_sharpe']):.2f}, "
                f"QQQ alpha {float(row['oos_alpha_vs_qqq_pct']):.1f}%, "
                f"turnover {float(row['oos_turnover_pct']):.1f}%, "
                f"stress {row['fixed_cost_stress_approval_pass']}"
            )
        except (KeyError, RuntimeError, ValueError, ZeroDivisionError) as exc:
            row = {
                "valid": False,
                "label": "fixed_live",
                "fold_year": int(split.outer_year),
                "outer_year": int(split.outer_year),
                "selected_config": signature,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"  {split.outer_year}: failed {row['reason']}")
        rows.append(row)
        gc.collect()
    return _fixed_summary(rows, signature=signature, config=config)


def _print_summary(result: dict[str, Any]) -> None:
    """Print the decision and main metrics after a fixed validation run."""
    print("\nFIXED LIVE CONFIG VALIDATION")
    print("=" * 72)
    print(f"valid:                 {result.get('valid')}")
    print(f"folds:                 {result.get('fold_count')}/{len(result.get('folds', []))}")
    print(f"compound OOS return:   {result.get('compound_oos_return_pct')}%")
    print(f"mean OOS Sharpe:       {result.get('mean_oos_sharpe')}")
    print(f"mean alpha vs QQQ:     {result.get('mean_oos_alpha_vs_qqq_pct')}%")
    print(f"mean/worst turnover:   {result.get('mean_oos_turnover_pct')}%/{result.get('worst_oos_turnover_pct')}%")
    print(f"cost stress folds:     {result.get('fixed_cost_stress_passed_folds')}/{result.get('fixed_cost_stress_tested_folds')}")
    approval = result.get("live_config_approval", {})
    print(f"live approval:         {approval.get('approved')} {approval.get('reasons', [])}")


def main() -> None:
    """Run fixed validation and write research or explicit live outputs."""
    parser = argparse.ArgumentParser(description="Validate one fixed live config without yearly selector hopping.")
    parser.add_argument("--config", required=True, help="Exact walkforward config signature to validate.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--max-specs", type=int, default=nested.DEFAULT_MAX_SPECS)
    parser.add_argument("--min-train-years", type=int, default=nested.DEFAULT_MIN_TRAIN_YEARS)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--publish-live-config",
        action="store_true",
        help="Explicitly update signals/core_satellite_live_configs.json if approval passes.",
    )
    args = parser.parse_args()
    panel = _load_panel(int(args.max_specs))
    result = validate_fixed_config(
        panel,
        str(args.config),
        min_train_years=int(args.min_train_years),
        start_year=args.start_year,
        end_year=args.end_year,
    )
    _print_summary(result)
    publish = bool(args.publish_live_config and result.get("live_config_approval", {}).get("approved"))
    json_path, csv_path = nested.write_outputs(
        result,
        output_prefix=str(args.output_prefix),
        publish_live_config=publish,
    )
    print("\nWrote:")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    if publish:
        print(f"  live configs: {nested.LIVE_CONFIG_PATH}")
    else:
        print("  live configs: not published")
        if args.publish_live_config:
            print("  publish request blocked because approval failed")


if __name__ == "__main__":
    main()
