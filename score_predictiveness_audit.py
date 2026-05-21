"""
score_predictiveness_audit.py - audit why walkforward inner scores fail.

PLAIN ENGLISH:
The walkforward picks one strategy config per held-out year.  This script checks
whether the numbers used to pick that config actually line up with the later
out-of-sample result.  It ranks inner validation fields from most helpful to
most harmful so we can improve the selector with evidence instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from robustness_scoring import DEFAULT_OBJECTIVE, robustness_score_components
from settings import SIGNAL_DIR


DEFAULT_CSV = Path(SIGNAL_DIR) / "core_satellite_nested_walkforward_alphaqqq.csv"
TARGET_COLUMNS = (
    "oos_objective_score",
    "oos_sharpe",
    "oos_alpha_vs_qqq_pct",
    "oos_return_pct",
    "oos_cagr_pct",
    "oos_max_drawdown_pct",
    "oos_turnover_pct",
)
INNER_DETAIL_METRICS = (
    "score",
    "sharpe",
    "return_pct",
    "alpha_vs_spy_pct",
    "alpha_vs_qqq_pct",
    "alpha_vs_blend_pct",
    "turnover_pct",
    "max_drawdown_pct",
)


def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None for blanks/NaN."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pearson(left: pd.Series, right: pd.Series) -> float | None:
    """Return Pearson correlation when enough usable values exist."""
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 4 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson")
    return round(float(corr), 4) if pd.notna(corr) else None


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    """Return rank correlation when enough usable values exist."""
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 4 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return round(float(corr), 4) if pd.notna(corr) else None


def _detrended_pearson(metric: pd.Series, target: pd.Series, year: pd.Series) -> float | None:
    """Correlate metric and target after removing a linear fold-year trend."""
    pair = pd.concat([metric, target, year], axis=1).dropna()
    if len(pair) < 6 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None

    x = pair.iloc[:, 2].astype(float)
    x_centered = x - x.mean()
    denom = float((x_centered * x_centered).sum())
    if denom <= 1e-12:
        return None

    residuals = []
    for col_idx in (0, 1):
        y = pair.iloc[:, col_idx].astype(float)
        y_centered = y - y.mean()
        beta = float((x_centered * y_centered).sum() / denom)
        residuals.append(y_centered - beta * x_centered)

    corr = residuals[0].corr(residuals[1], method="pearson")
    return round(float(corr), 4) if pd.notna(corr) else None


def _oos_objective_score(row: pd.Series, objective: str) -> float:
    """Compute the OOS score using the same objective formula as selection."""
    mapped = {
        "sharpe": row.get("oos_sharpe"),
        "alpha_vs_qqq_pct": row.get("oos_alpha_vs_qqq_pct"),
        "max_drawdown_pct": row.get("oos_max_drawdown_pct"),
        "turnover_pct": row.get("oos_turnover_pct"),
    }
    return float(robustness_score_components(mapped, objective=objective)["robustness_score"])


def _load_inner_details(json_path: Path) -> dict[int, list[dict]]:
    """Load selected-config inner fold details keyed by outer year."""
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    details: dict[int, list[dict]] = {}
    for row in data.get("inner_fold_details", []) or []:
        year = _safe_float(row.get("outer_year"))
        if year is None:
            continue
        folds = row.get("inner_folds") or []
        if isinstance(folds, list):
            details[int(year)] = [f for f in folds if isinstance(f, dict)]
    return details


def _series_stats(values: list[float]) -> dict[str, float | None]:
    """Return simple summary stats for one inner-fold metric."""
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"mean": None, "median": None, "min": None, "max": None, "last": None, "std": None}
    s = pd.Series(clean, dtype=float)
    return {
        "mean": round(float(s.mean()), 6),
        "median": round(float(s.median()), 6),
        "min": round(float(s.min()), 6),
        "max": round(float(s.max()), 6),
        "last": round(float(clean[-1]), 6),
        "std": round(float(s.std(ddof=0)), 6) if len(clean) > 1 else 0.0,
    }


def add_inner_detail_features(df: pd.DataFrame, json_path: Path) -> pd.DataFrame:
    """Add derived inner-fold fields from the walkforward JSON, if present."""
    details_by_year = _load_inner_details(json_path)
    if not details_by_year:
        return df

    additions: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        year = int(row.get("fold_year") or row.get("outer_year"))
        folds = details_by_year.get(year, [])
        out = {"fold_year": year}
        for metric in INNER_DETAIL_METRICS:
            values = [_safe_float(fold.get(metric)) for fold in folds]
            stats = _series_stats([v for v in values if v is not None])
            for stat_name, value in stats.items():
                out[f"inner_detail_{stat_name}_{metric}"] = value
        additions.append(out)

    detail_df = pd.DataFrame(additions)
    return df.merge(detail_df, on="fold_year", how="left")


def candidate_metric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric fields worth testing against OOS results."""
    prefixes = (
        "inner_",
        "config_",
        "selected_config_",
        "stable_family_",
    )
    exact = {
        "holding_days",
        "overlay_gross",
        "ma_window",
        "high_vol",
        "tqqq_weight",
    }
    skip = set(TARGET_COLUMNS) | {
        "fold_year",
        "outer_year",
        "valid",
        "oos_n_equity_points",
        "candidate_configs",
    }

    cols: list[str] = []
    for col in df.columns:
        if col in skip:
            continue
        if col in exact or col.startswith(prefixes):
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() >= 4 and numeric.nunique(dropna=True) >= 2:
                cols.append(col)
    return cols


def build_audit(csv_path: Path, *, objective: str, json_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Build the metric-correlation audit table and summary payload."""
    df = pd.read_csv(csv_path)
    if json_path is None:
        json_path = csv_path.with_suffix(".json")
    df = add_inner_detail_features(df, json_path)

    required = ["oos_sharpe", "oos_alpha_vs_qqq_pct", "oos_max_drawdown_pct", "oos_turnover_pct"]
    valid = df.dropna(subset=required).copy()
    valid["oos_objective_score"] = valid.apply(_oos_objective_score, axis=1, objective=objective)

    rows: list[dict[str, Any]] = []
    for metric in candidate_metric_columns(valid):
        metric_values = pd.to_numeric(valid[metric], errors="coerce")
        row: dict[str, Any] = {
            "metric": metric,
            "n": int(metric_values.notna().sum()),
            "unique_values": int(metric_values.nunique(dropna=True)),
            "pearson_oos_objective": _pearson(metric_values, valid["oos_objective_score"]),
            "spearman_oos_objective": _spearman(metric_values, valid["oos_objective_score"]),
            "detrended_pearson_oos_objective": _detrended_pearson(
                metric_values,
                valid["oos_objective_score"],
                pd.to_numeric(valid["fold_year"], errors="coerce"),
            ),
        }
        for target in TARGET_COLUMNS[1:]:
            row[f"pearson_{target}"] = _pearson(metric_values, pd.to_numeric(valid[target], errors="coerce"))
        rows.append(row)

    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit["_sort"] = audit["pearson_oos_objective"].fillna(-999)
        audit = audit.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)

    score_corr = None
    if "inner_score" in valid.columns:
        score_corr = _pearson(pd.to_numeric(valid["inner_score"], errors="coerce"), valid["oos_objective_score"])

    harmful = audit[audit["pearson_oos_objective"].fillna(0.0) <= -0.10].head(10).to_dict("records")
    helpful = audit[audit["pearson_oos_objective"].fillna(0.0) >= 0.10].head(10).to_dict("records")
    summary = {
        "csv": str(csv_path),
        "json": str(json_path),
        "objective": objective,
        "folds": int(len(valid)),
        "inner_score_vs_oos_objective": score_corr,
        "top_helpful": helpful,
        "top_harmful": harmful,
        "notes": [
            "This is a selected-config audit, not a full rejected-candidate replay.",
            "Positive correlation means larger inner metric tended to line up with better OOS result.",
            "Negative correlation means the metric tended to point the selector the wrong way.",
        ],
    }
    return audit, summary


def _print_rows(title: str, rows: pd.DataFrame, *, limit: int) -> None:
    """Print a compact metric table."""
    print(f"\n{title}")
    if rows.empty:
        print("  none")
        return
    cols = [
        "metric",
        "pearson_oos_objective",
        "spearman_oos_objective",
        "detrended_pearson_oos_objective",
        "pearson_oos_sharpe",
        "pearson_oos_alpha_vs_qqq_pct",
    ]
    print(rows[cols].head(limit).to_string(index=False))


def print_audit(audit: pd.DataFrame, summary: dict, *, limit: int) -> None:
    """Print the audit summary."""
    print("\nSCORE PREDICTIVENESS AUDIT")
    print("=" * 70)
    print(f"Source:    {summary['csv']}")
    print(f"Objective: {summary['objective']}")
    print(f"Folds:     {summary['folds']}")
    print(f"inner_score vs OOS objective: {summary['inner_score_vs_oos_objective']}")

    helpful = audit[audit["pearson_oos_objective"].fillna(0.0) >= 0.10]
    harmful = audit[audit["pearson_oos_objective"].fillna(0.0) <= -0.10].sort_values(
        "pearson_oos_objective",
        ascending=True,
    )
    _print_rows("Most helpful inner/config fields", helpful, limit=limit)
    _print_rows("Most harmful inner/config fields", harmful, limit=limit)

    print("\nInterpretation")
    if summary["inner_score_vs_oos_objective"] is not None and summary["inner_score_vs_oos_objective"] < 0:
        print("  FAIL: current inner_score is anti-predictive on selected folds.")
    if "inner_config_momentum_bonus" in set(audit["metric"]):
        row = audit[audit["metric"].eq("inner_config_momentum_bonus")].iloc[0]
        corr = row.get("pearson_oos_objective")
        if corr is not None and corr < 0:
            print(
                "  Momentum bonus screens as suspicious on selected folds, "
                "but no-momentum A/B was worse; don't remove it without a better replacement."
            )
    if "inner_mean_turnover_pct" in set(audit["metric"]):
        row = audit[audit["metric"].eq("inner_mean_turnover_pct")].iloc[0]
        corr = row.get("pearson_oos_objective")
        if corr is not None and corr > 0:
            print("  Higher inner turnover was not harmful here; a hard cap may be better than a score penalty.")
    print("  Next best test: rerun walkforward after changing only the clearly harmful score component.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit walkforward score predictiveness fields.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Walkforward CSV to audit.")
    parser.add_argument("--json", default=None, help="Matching walkforward JSON with inner fold details.")
    parser.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        choices=("sharpe", "alpha_vs_qqq", "hybrid"),
        help=f"OOS objective target to audit against (default: {DEFAULT_OBJECTIVE}).",
    )
    parser.add_argument("--limit", type=int, default=10, help="Rows to print in each section.")
    parser.add_argument("--out-csv", default=str(Path(SIGNAL_DIR) / "score_predictiveness_audit.csv"))
    parser.add_argument("--out-json", default=str(Path(SIGNAL_DIR) / "score_predictiveness_audit.json"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    json_path = Path(args.json) if args.json else None
    audit, summary = build_audit(csv_path, objective=args.objective, json_path=json_path)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print_audit(audit, summary, limit=max(1, int(args.limit)))
    print("\nWrote:")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
