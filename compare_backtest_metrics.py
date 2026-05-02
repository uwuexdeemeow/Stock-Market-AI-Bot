"""
compare_backtest_metrics.py — compare current backtest metrics to a saved baseline.

Usage:
    python3 compare_backtest_metrics.py --baseline-dir logs/baselines/iterative_backtest_YYYYMMDD_HHMMSS

The script is intentionally read-only. It prints a compact before/current/delta
table for the timing sleeve, raw stock sleeve, and combined portfolio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SIGNALS = ROOT / "signals"


METRIC_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SPY/QQQ timing", (
        "total_return_pct",
        "cagr_pct",
        "sharpe",
        "sortino",
        "max_drawdown_pct",
        "benchmark_return_pct",
        "alpha_pct",
        "in_market_pct",
        "avg_allocation_pct",
        "turnover_pct",
        "estimated_cost_pct",
    )),
    ("Combined portfolio", (
        "total_return_pct",
        "annual_return_pct",
        "sharpe_ratio_daily",
        "nw_tstat_vs_cash",
        "nw_tstat_vs_primary_benchmark",
        "max_drawdown_pct",
        "alpha_pct",
        "cash_overlay_avg_alloc",
        "overlay_contribution_pct",
    )),
    ("Raw stock sleeve", (
        "raw_strategy_total_return_pct",
        "raw_strategy_sharpe_ratio_daily",
        "raw_stock_gates_pass",
    )),
    ("Rank diagnostics", (
        "rank_performance.rank_spearman",
        "daily_rank_ic.mean_ic",
        "daily_rank_ic.t_stat",
    )),
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _get_nested(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{f:.4g}"


def _delta(before: Any, after: Any) -> str:
    try:
        b = float(before)
        a = float(after)
    except (TypeError, ValueError):
        return "changed" if before != after else ""
    d = a - b
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.4g}"


def _print_section(title: str, metrics: tuple[str, ...], before: dict[str, Any], after: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * 78)
    print(f"{'metric':<38} {'baseline':>12} {'current':>12} {'delta':>12}")
    print("-" * 78)
    for metric in metrics:
        b = _get_nested(before, metric)
        a = _get_nested(after, metric)
        if b is None and a is None:
            continue
        print(f"{metric:<38} {_fmt(b):>12} {_fmt(a):>12} {_delta(b, a):>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved baseline metrics to current backtest outputs.")
    parser.add_argument("--baseline-dir", required=True, help="Directory containing baseline metrics JSON files.")
    parser.add_argument("--current-dir", default=str(SIGNALS), help="Directory containing current metrics JSON files.")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    current_dir = Path(args.current_dir)
    timing_before = _load_json(baseline_dir / "spy_timing_metrics.json")
    timing_after = _load_json(current_dir / "spy_timing_metrics.json")
    portfolio_before = _load_json(baseline_dir / "walkforward_long_only_147tickers_metrics.json")
    portfolio_after = _load_json(current_dir / "walkforward_long_only_147tickers_metrics.json")

    if not timing_before and not portfolio_before:
        raise SystemExit(f"No baseline metrics found in {baseline_dir}")

    print(f"Baseline: {baseline_dir}")
    print(f"Current : {current_dir}")
    _print_section(METRIC_SPECS[0][0], METRIC_SPECS[0][1], timing_before, timing_after)
    for title, metrics in METRIC_SPECS[1:]:
        _print_section(title, metrics, portfolio_before, portfolio_after)


if __name__ == "__main__":
    main()
