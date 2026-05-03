"""
spy_timing_grid.py — predeclared SPY/QQQ timing rule sweep.

This reads saved walk-forward prediction CSVs from signals/ and reruns only the
index timing overlay. It does not retrain models and does not alter stock-pick
ranking. Results are written to signals/spy_timing_grid.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import run_spy_timing_backtest
from settings import SIGNAL_DIR, WATCHLIST


THRESHOLDS = (0.50, 0.55, 0.60)
INVEST_PCTS = (0.75, 0.85, 0.95)
INSTRUMENT_PRESETS: tuple[tuple[str, dict[str, float]], ...] = (
    ("spy_only", {"SPY": 1.0}),
    ("spy_qqq_60_40", {"SPY": 0.60, "QQQ": 0.40}),
)


def load_saved_predictions() -> dict[str, pd.DataFrame]:
    signal_dir = Path(SIGNAL_DIR)
    out: dict[str, pd.DataFrame] = {}
    for ticker in WATCHLIST:
        path = signal_dir / f"{ticker}_walkforward_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "date" not in df.columns or "direction_vote" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date", drop=False).sort_index()
        if not df.empty:
            out[ticker] = df
    return out


def main() -> None:
    predictions = load_saved_predictions()
    if not predictions:
        raise SystemExit("No saved walk-forward predictions found. Run backtest.py first.")

    rows: list[dict] = []
    for preset_name, instruments in INSTRUMENT_PRESETS:
        for threshold in THRESHOLDS:
            for invest_pct in INVEST_PCTS:
                metrics = run_spy_timing_backtest(
                    predictions,
                    threshold=threshold,
                    invest_pct=invest_pct,
                    instruments=instruments,
                    write_outputs=False,
                    quiet=True,
                )
                if not metrics:
                    continue
                rows.append({
                    "preset": preset_name,
                    "threshold": threshold,
                    "invest_pct": invest_pct,
                    "total_return_pct": metrics.get("total_return_pct"),
                    "cagr_pct": metrics.get("cagr_pct"),
                    "sharpe": metrics.get("sharpe"),
                    "sortino": metrics.get("sortino"),
                    "nw_tstat_vs_cash": metrics.get("nw_tstat_vs_cash"),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "benchmark_return_pct": metrics.get("benchmark_return_pct"),
                    "alpha_pct": metrics.get("alpha_pct"),
                    "in_market_pct": metrics.get("in_market_pct"),
                    "avg_allocation_pct": metrics.get("avg_allocation_pct"),
                    "turnover_pct": metrics.get("turnover_pct"),
                    "estimated_cost_pct": metrics.get("estimated_cost_pct"),
                    "paper_ready": metrics.get("paper_ready"),
                })

    if not rows:
        raise SystemExit("Timing grid produced no rows.")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["paper_ready", "sharpe", "max_drawdown_pct", "total_return_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    out_path = Path(SIGNAL_DIR) / "spy_timing_grid.csv"
    summary.to_csv(out_path, index=False)

    print(f"Saved -> {out_path}")
    print("\nTop timing grid rows")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
