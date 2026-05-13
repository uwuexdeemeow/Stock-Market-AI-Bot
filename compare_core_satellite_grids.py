"""
Compare the tracked old core-satellite grid with the current reduced grid.

By default this reads the old grid from git HEAD and the new grid from
signals/core_satellite_alpha_grid.csv. It compares the best row as written in
each grid, which preserves the objective used when that grid was generated.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from settings import SIGNAL_DIR


DEFAULT_GRID = Path(SIGNAL_DIR) / "core_satellite_alpha_grid.csv"
DEFAULT_JSON = Path(SIGNAL_DIR) / "core_satellite_grid_comparison.json"
DEFAULT_CSV = Path(SIGNAL_DIR) / "core_satellite_grid_comparison.csv"


def _read_grid(path: str | Path | None, *, git_ref: str | None, git_path: str) -> pd.DataFrame:
    if path:
        return pd.read_csv(path)
    if not git_ref:
        raise ValueError("Either path or git_ref is required")
    raw = subprocess.check_output(["git", "show", f"{git_ref}:{git_path}"], text=True)
    return pd.read_csv(StringIO(raw))


def _get(row: pd.Series, *names: str, default=np.nan):
    for name in names:
        if name in row.index:
            return row[name]
    return default


def _best_row_as_written(grid: pd.DataFrame) -> pd.Series:
    if grid.empty:
        raise ValueError("Grid is empty")
    return grid.iloc[0]


def _config_signature(row: pd.Series) -> str:
    parts = [
        f"preset={_get(row, 'core_preset', 'preset', default='')}",
        f"score={_get(row, 'score_source', default='')}",
        f"shape={_get(row, 'shape', default='')}",
        f"weighting={_get(row, 'weighting', default='')}",
        f"sector={_get(row, 'max_per_sector', default='')}",
        f"hold={_get(row, 'holding_days', default='')}",
        f"overlay={_get(row, 'overlay_gross', default='')}",
        f"cost_gate={_get(row, 'robust_cost_stress_pass', default=False)}",
    ]
    return ",".join(parts)


def _summary(label: str, grid: pd.DataFrame) -> dict:
    row = _best_row_as_written(grid)
    return {
        "label": label,
        "row_count": int(len(grid)),
        "best_config": _config_signature(row),
        "cagr_pct": float(_get(row, "cagr_pct", default=np.nan)),
        "sharpe": float(_get(row, "sharpe", default=np.nan)),
        "max_drawdown_pct": float(_get(row, "max_drawdown_pct", default=np.nan)),
        "turnover_pct": float(_get(row, "turnover_pct", default=np.nan)),
        "alpha_vs_spy_pct": float(_get(row, "alpha_vs_spy_pct", "alpha_vs_spy", default=np.nan)),
        "alpha_vs_qqq_pct": float(_get(row, "alpha_vs_qqq_pct", "alpha_vs_qqq", default=np.nan)),
        "robustness_score": float(_get(row, "robustness_score", default=np.nan)),
        "drawdown_penalty": float(_get(row, "drawdown_penalty", default=np.nan)),
        "turnover_penalty": float(_get(row, "turnover_penalty", default=np.nan)),
        "instability_penalty": float(_get(row, "instability_penalty", default=np.nan)),
        "paper_ready": bool(_get(row, "paper_ready", default=False)),
        "robust_cost_stress_pass": bool(_get(row, "robust_cost_stress_pass", default=False)),
        "stress_cost_levels": str(_get(row, "stress_cost_levels", default="")),
    }


def compare_grids(old_grid: pd.DataFrame, new_grid: pd.DataFrame) -> dict:
    old = _summary("old_huge_grid", old_grid)
    new = _summary("new_reduced_grid", new_grid)
    metric_names = [
        "cagr_pct",
        "sharpe",
        "max_drawdown_pct",
        "turnover_pct",
        "alpha_vs_spy_pct",
        "alpha_vs_qqq_pct",
    ]
    deltas = {}
    for name in metric_names:
        old_value = old[name]
        new_value = new[name]
        deltas[name] = round(float(new_value - old_value), 4)
        if np.isfinite(old_value) and abs(old_value) > 1e-9:
            deltas[f"{name}_ratio_new_to_old"] = round(float(new_value / old_value), 4)
    return {
        "old": old,
        "new": new,
        "delta_new_minus_old": deltas,
        "performance_collapse_flags": {
            "cagr_less_than_half_old": bool(np.isfinite(old["cagr_pct"]) and new["cagr_pct"] < 0.5 * old["cagr_pct"]),
            "sharpe_less_than_half_old": bool(np.isfinite(old["sharpe"]) and new["sharpe"] < 0.5 * old["sharpe"]),
            "alpha_vs_spy_less_than_half_old": bool(
                np.isfinite(old["alpha_vs_spy_pct"]) and new["alpha_vs_spy_pct"] < 0.5 * old["alpha_vs_spy_pct"]
            ),
            "alpha_vs_qqq_less_than_half_old": bool(
                np.isfinite(old["alpha_vs_qqq_pct"]) and new["alpha_vs_qqq_pct"] < 0.5 * old["alpha_vs_qqq_pct"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old huge grid against current reduced grid")
    parser.add_argument("--old-grid", default=None, help="Optional old grid CSV path. Defaults to git show HEAD:signals/core_satellite_alpha_grid.csv")
    parser.add_argument("--old-git-ref", default="HEAD")
    parser.add_argument("--old-git-path", default="signals/core_satellite_alpha_grid.csv")
    parser.add_argument("--new-grid", default=str(DEFAULT_GRID))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--csv-out", default=str(DEFAULT_CSV))
    args = parser.parse_args()

    old_grid = _read_grid(args.old_grid, git_ref=args.old_git_ref, git_path=args.old_git_path)
    new_grid = pd.read_csv(args.new_grid)
    result = compare_grids(old_grid, new_grid)

    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2))
    pd.DataFrame([result["old"], result["new"]]).to_csv(csv_path, index=False)

    print(f"Comparison written -> {json_path}")
    print(f"Summary CSV written -> {csv_path}")
    print(pd.DataFrame([result["old"], result["new"]]).to_string(index=False))
    print("Delta new minus old:")
    print(json.dumps(result["delta_new_minus_old"], indent=2))
    print("Collapse flags:")
    print(json.dumps(result["performance_collapse_flags"], indent=2))


if __name__ == "__main__":
    main()
