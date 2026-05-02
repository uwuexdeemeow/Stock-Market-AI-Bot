"""
Fast ranker top-N portfolio sweep.

This reuses the latest saved per-ticker walk-forward prediction CSVs from
backtest.py and reruns only cross-sectional ranking + portfolio construction.
It does not retrain XGBoost. Use after a full ranker backtest has written
signals/*_walkforward_predictions.csv with ranker_score populated.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("POOLED_RANKER_EXPERIMENT_ENABLED", "1")

import pandas as pd

from settings import EDGE_AUDIT_ENABLED, SIGNAL_DIR, WATCHLIST
from backtest import (
    apply_cross_sectional_ranking,
    build_edge_audit_report,
    build_trade_attribution_report,
    compute_daily_rank_ic,
    run_portfolio_backtest,
    summarize_ticker_breakdown,
)


def _load_saved_predictions(tickers: list[str]) -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    signal_dir = Path(SIGNAL_DIR)
    for ticker in tickers:
        path = signal_dir / f"{ticker}_walkforward_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "ranker_score" not in df.columns or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date", drop=False).sort_index()
        if df["ranker_score"].notna().sum() == 0:
            continue
        predictions[ticker] = df
    return predictions


def _write_variant_outputs(
    out_dir: Path,
    suffix: str,
    predictions: dict[str, pd.DataFrame],
    trades_df: pd.DataFrame,
    equity: pd.Series,
    benchmarks: pd.DataFrame,
    metrics: dict,
) -> None:
    daily_rank_ic, daily_rank_ic_df = compute_daily_rank_ic(predictions)
    daily_rank_ic["source"] = "saved_walkforward_predictions"
    metrics["daily_rank_ic"] = daily_rank_ic

    trades_df.to_csv(out_dir / f"{suffix}_trades.csv", index=False)
    equity_frame = pd.DataFrame({"equity": equity})
    for symbol in benchmarks.columns:
        equity_frame[f"benchmark_{symbol.lower()}_norm"] = benchmarks[symbol].reindex(equity.index)
    equity_frame.to_csv(out_dir / f"{suffix}_equity.csv")
    summarize_ticker_breakdown(trades_df).to_csv(out_dir / f"{suffix}_ticker_breakdown.csv", index=False)
    build_trade_attribution_report(trades_df).to_csv(out_dir / f"{suffix}_trade_attribution.csv", index=False)
    daily_rank_ic_df.to_csv(out_dir / f"{suffix}_daily_rank_ic.csv", index=False)
    metrics["daily_rank_ic_path"] = str(out_dir / f"{suffix}_daily_rank_ic.csv")
    if EDGE_AUDIT_ENABLED:
        edge_path = out_dir / f"{suffix}_edge_audit.csv"
        build_edge_audit_report(trades_df).to_csv(edge_path, index=False)
        metrics["edge_audit_path"] = str(edge_path)
    with open(out_dir / f"{suffix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast top-N sweep using saved ranker predictions.")
    parser.add_argument("--top-n", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--mode", default="long_only", choices=["long_only", "long_short", "long_only_bear_cash"])
    parser.add_argument("--stress", type=float, default=1.0)
    parser.add_argument("--no-eligibility-filter", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    top_ns = sorted({max(1, int(n)) for n in args.top_n})
    predictions = _load_saved_predictions([t.upper() for t in WATCHLIST])
    if not predictions:
        raise SystemExit("No saved ranker predictions found. Run ranker backtest first.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or Path(SIGNAL_DIR) / f"ranker_topn_sweep_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for top_n in top_ns:
        ranked = apply_cross_sectional_ranking(
            {t: df.copy() for t, df in predictions.items()},
            top_n=top_n,
            mode=args.mode,
            eligible_tickers=None,
            use_eligibility_filter=not args.no_eligibility_filter,
        )
        ranked = {t: df for t, df in ranked.items() if df is not None and not df.empty}
        trades_df, equity, benchmarks, metrics = run_portfolio_backtest(
            ranked,
            args.mode,
            args.stress,
            use_trade_rules=False,
        )
        metrics["ranker_top_n"] = int(top_n)
        suffix = f"ranker_top{top_n}_{args.mode}_{len(ranked)}tickers"
        _write_variant_outputs(out_dir, suffix, predictions, trades_df, equity, benchmarks, metrics)
        ic = metrics.get("daily_rank_ic", {}) or {}
        rows.append({
            "top_n": top_n,
            "trades": metrics.get("trades"),
            "raw_sharpe": metrics.get("raw_strategy_sharpe_ratio_daily"),
            "raw_return_pct": metrics.get("raw_strategy_total_return_pct"),
            "total_return_pct": metrics.get("total_return_pct"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "nw_tstat_vs_cash": metrics.get("nw_tstat_vs_cash"),
            "daily_ic": ic.get("mean_ic"),
            "daily_ic_t": ic.get("t_stat"),
        })

    summary = pd.DataFrame(rows).sort_values("top_n")
    summary.to_csv(out_dir / "topn_sweep_summary.csv", index=False)
    print(f"Saved sweep -> {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
