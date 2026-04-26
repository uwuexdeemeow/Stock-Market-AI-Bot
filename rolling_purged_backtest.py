from __future__ import annotations

"""
rolling_purged_backtest.py — rolling OOS validation harness.

This does not choose live tickers directly. It slices the existing strict
walk-forward predictions into calendar folds, runs the portfolio simulator on
each fold, and writes every fold to the experiment ledger. The underlying
prediction generator in backtest.py already purges the forward-return horizon
from training via RETURN_HORIZON_DAYS.
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd

from backtest import (
    build_trade_attribution_report,
    default_backtest_tickers,
    run_portfolio_backtest,
    summarize_ticker_breakdown,
    walk_forward_predictions_for_ticker,
)
from experiment_ledger import append_experiment
from settings import RETURN_HORIZON_DAYS, SIGNAL_DIR


def _year_folds(start_year: int, end_year: int, fold_years: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    folds = []
    year = start_year
    while year <= end_year:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{min(year + fold_years - 1, end_year)}-12-31")
        folds.append((start, end))
        year += fold_years
    return folds


def _filter_predictions(
    predictions: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    out = {}
    for ticker, df in predictions.items():
        if df.empty:
            continue
        idx = pd.DatetimeIndex(df.index)
        fold_df = df.loc[(idx >= start) & (idx <= end)].copy()
        if not fold_df.empty:
            out[ticker] = fold_df
    return out


def _write_fold_outputs(
    label: str,
    trades: pd.DataFrame,
    equity: pd.Series,
    benchmarks: pd.DataFrame,
    metrics: dict,
) -> dict[str, str]:
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    base = os.path.join(SIGNAL_DIR, label)
    trades_path = f"{base}_trades.csv"
    equity_path = f"{base}_equity.csv"
    metrics_path = f"{base}_metrics.json"
    breakdown_path = f"{base}_ticker_breakdown.csv"
    attribution_path = f"{base}_trade_attribution.csv"

    trades.to_csv(trades_path, index=False)
    equity_frame = pd.DataFrame({"equity": equity})
    for symbol in benchmarks.columns:
        equity_frame[f"benchmark_{symbol.lower()}_norm"] = benchmarks[symbol].reindex(equity.index)
    equity_frame.to_csv(equity_path)
    summarize_ticker_breakdown(trades).to_csv(breakdown_path, index=False)
    build_trade_attribution_report(trades).to_csv(attribution_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "trades": trades_path,
        "equity": equity_path,
        "metrics": metrics_path,
        "breakdown": breakdown_path,
        "attribution": attribution_path,
    }


def run_rolling_purged_backtest(
    tickers: list[str],
    start_year: int,
    end_year: int,
    fold_years: int,
    mode: str,
    stress: float,
    use_trade_rules: bool,
) -> pd.DataFrame:
    predictions: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            pdf = walk_forward_predictions_for_ticker(ticker)
            if not pdf.empty:
                predictions[ticker] = pdf
        except Exception as e:
            print(f"ERROR - {ticker}: {e}")

    if not predictions:
        raise SystemExit("No walk-forward predictions generated")

    rows = []
    for fold_idx, (start, end) in enumerate(_year_folds(start_year, end_year, fold_years), start=1):
        fold_predictions = _filter_predictions(predictions, start, end)
        if not fold_predictions:
            rows.append({
                "fold": fold_idx,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "tickers": 0,
                "trades": 0,
                "status": "no_predictions",
            })
            continue

        trades, equity, benchmarks, metrics = run_portfolio_backtest(
            fold_predictions,
            mode=mode,
            stress=stress,
            use_trade_rules=use_trade_rules,
            date_start=start,
            date_end=end,
        )
        label = f"rolling_purged_{start.year}_{end.year}_{mode}_{len(fold_predictions)}tickers"
        artifacts = _write_fold_outputs(label, trades, equity, benchmarks, metrics)
        row = {
            "fold": fold_idx,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "tickers": len(fold_predictions),
            "trades": int(len(trades)),
            "total_return_pct": metrics.get("total_return_pct"),
            "alpha_pct": metrics.get("alpha_pct"),
            "sharpe_ratio_daily": metrics.get("sharpe_ratio_daily"),
            "nw_tstat_vs_cash": metrics.get("nw_tstat_vs_cash"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "kill_all_pass": bool(metrics.get("kill_criteria", {}).get("results", {}).get("all_pass", False)),
            "status": "complete",
        }
        append_experiment(
            name="rolling_purged_backtest",
            params={
                "fold": fold_idx,
                "start": row["start"],
                "end": row["end"],
                "tickers": list(fold_predictions.keys()),
                "mode": mode,
                "stress": stress,
                "use_trade_rules": use_trade_rules,
                "horizon_days": RETURN_HORIZON_DAYS,
            },
            metrics={k: v for k, v in row.items() if k not in {"status", "start", "end"}},
            artifacts=artifacts,
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    out_path = os.path.join(SIGNAL_DIR, f"rolling_purged_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary.to_csv(out_path, index=False)
    print("Saved ->", out_path)
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling purged walk-forward backtest")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--fold-years", type=int, default=1)
    parser.add_argument("--mode", type=str, default="long_only", choices=["long_short", "long_only", "long_only_bear_cash"])
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument("--approved-only", action="store_true", help="Backtest only tickers currently approved for live trading.")
    parser.add_argument("--ignore-trade-rules", action="store_true")
    parser.add_argument("--stress", type=float, default=1.0)
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = default_backtest_tickers(approved_only=args.approved_only)

    if not tickers:
        raise SystemExit("No backtest tickers selected")

    mode = args.mode if args.allow_shorts else "long_only"
    run_rolling_purged_backtest(
        tickers=tickers,
        start_year=args.start_year,
        end_year=args.end_year,
        fold_years=max(1, args.fold_years),
        mode=mode,
        stress=args.stress,
        use_trade_rules=not args.ignore_trade_rules,
    )


if __name__ == "__main__":
    main()
