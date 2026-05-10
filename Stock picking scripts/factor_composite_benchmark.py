"""
Direct literature-factor composite benchmark.

This answers a narrow question: does a simple hand-built factor score beat the
XGBRanker when run through the same top-N ranking, exits, sizing, and portfolio
logic? It reuses saved walk-forward prediction CSVs for prices/exits and joins
factor columns from data/*.parquet by (ticker, date), so no model retraining is
performed.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("POOLED_RANKER_EXPERIMENT_ENABLED", "1")

import numpy as np
import pandas as pd

from settings import DATA_DIR, EDGE_AUDIT_ENABLED, SIGNAL_DIR, WATCHLIST
from backtest import (
    apply_cross_sectional_ranking,
    build_edge_audit_report,
    build_trade_attribution_report,
    compute_daily_rank_ic,
    run_portfolio_backtest,
    summarize_ticker_breakdown,
)


RANK_COMPONENTS: tuple[tuple[str, float], ...] = (
    ("xs_rank_sector_factor_mom_12_1", 1.0),
    ("xs_rank_sector_factor_resid_mom_sector_12_1", 1.0),
    ("xs_rank_sector_factor_illiquidity_amihud_20d", 1.0),
    ("xs_rank_sector_factor_liquidity_dollar_vol_20d", -1.0),
    ("xs_rank_market_factor_idio_vol_252_spy", 1.0),
    ("xs_rank_sector_ret_5d", -1.0),
)

SENTIMENT_COMPONENTS: tuple[str, ...] = (
    "sent_z_sent_pos_decay",
    "sent_z_headline_vol_spike",
    "sent_z_headline_volume",
    "sent_z_sentiment_disagreement",
)


def _load_prediction_panel(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    signal_dir = Path(SIGNAL_DIR)
    for ticker in tickers:
        path = signal_dir / f"{ticker}_walkforward_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date", drop=False).sort_index()
        out[ticker] = df
    return out


def _load_factor_frame(ticker: str) -> pd.DataFrame | None:
    path = Path(DATA_DIR) / f"{ticker}.parquet"
    if not path.exists():
        return None
    cols = [c for c, _ in RANK_COMPONENTS] + list(SENTIMENT_COMPONENTS)
    df = pd.read_parquet(path)
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return None
    out = df[keep].copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    return out


def _attach_factor_scores(predictions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    joined: dict[str, pd.DataFrame] = {}
    panel_parts: list[pd.DataFrame] = []
    for ticker, pred in predictions.items():
        factors = _load_factor_frame(ticker)
        if factors is None:
            continue
        df = pred.join(factors, how="left", rsuffix="_factor")
        df["ticker"] = ticker
        joined[ticker] = df
        panel_parts.append(df[["date", "ticker", *[c for c, _ in RANK_COMPONENTS if c in df.columns], *[c for c in SENTIMENT_COMPONENTS if c in df.columns]]])
    if not panel_parts:
        return {}

    panel = pd.concat(panel_parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    sent_cols = [c for c in SENTIMENT_COMPONENTS if c in panel.columns]
    if sent_cols:
        sent_raw = panel[sent_cols].replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0).mean(axis=1)
        panel["_factor_sentiment_rank"] = sent_raw.groupby(panel["date"]).rank(pct=True)
    else:
        panel["_factor_sentiment_rank"] = np.nan

    score_cols: list[str] = []
    for col, direction in RANK_COMPONENTS:
        if col not in panel.columns:
            continue
        score_col = f"_score_{col}"
        val = pd.to_numeric(panel[col], errors="coerce")
        panel[score_col] = val if direction > 0 else 1.0 - val
        score_cols.append(score_col)
    score_cols.append("_factor_sentiment_rank")
    panel["factor_score"] = panel[score_cols].mean(axis=1, skipna=True)

    score_lookup = panel.set_index(["ticker", "date"])["factor_score"]
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in joined.items():
        keys = pd.MultiIndex.from_arrays([[ticker] * len(df), pd.to_datetime(df["date"])])
        scored = df.copy()
        scored["model_ranker_score"] = scored.get("ranker_score", np.nan)
        scored["ranker_score"] = score_lookup.reindex(keys).to_numpy()
        scored["factor_score"] = scored["ranker_score"]
        scored = scored.dropna(subset=["ranker_score"])
        if not scored.empty:
            out[ticker] = scored
    return out


def _write_outputs(
    out_dir: Path,
    suffix: str,
    predictions: dict[str, pd.DataFrame],
    trades_df: pd.DataFrame,
    equity: pd.Series,
    benchmarks: pd.DataFrame,
    metrics: dict,
) -> None:
    daily_rank_ic, daily_rank_ic_df = compute_daily_rank_ic(predictions)
    daily_rank_ic["source"] = "factor_composite_score"
    metrics["daily_rank_ic"] = daily_rank_ic

    trades_df.to_csv(out_dir / f"{suffix}_trades.csv", index=False)
    equity_frame = pd.DataFrame({"equity": equity})
    for symbol in benchmarks.columns:
        equity_frame[f"benchmark_{symbol.lower()}_norm"] = benchmarks[symbol].reindex(equity.index)
    equity_frame.to_csv(out_dir / f"{suffix}_equity.csv")
    summarize_ticker_breakdown(trades_df).to_csv(out_dir / f"{suffix}_ticker_breakdown.csv", index=False)
    build_trade_attribution_report(trades_df).to_csv(out_dir / f"{suffix}_trade_attribution.csv", index=False)
    daily_rank_ic_df.to_csv(out_dir / f"{suffix}_daily_rank_ic.csv", index=False)
    if EDGE_AUDIT_ENABLED:
        edge_path = out_dir / f"{suffix}_edge_audit.csv"
        build_edge_audit_report(trades_df).to_csv(edge_path, index=False)
        metrics["edge_audit_path"] = str(edge_path)
    with open(out_dir / f"{suffix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct factor composite against saved ranker predictions.")
    parser.add_argument("--top-n", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--mode", default="long_only", choices=["long_only", "long_short", "long_only_bear_cash"])
    parser.add_argument("--stress", type=float, default=1.0)
    parser.add_argument("--no-eligibility-filter", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    predictions = _attach_factor_scores(_load_prediction_panel([t.upper() for t in WATCHLIST]))
    if not predictions:
        raise SystemExit("No factor-scored predictions available. Run ranker backtest and rebuild parquets first.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or Path(SIGNAL_DIR) / f"factor_composite_benchmark_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for top_n in sorted({max(1, int(n)) for n in args.top_n}):
        ranked = apply_cross_sectional_ranking(
            {t: df.copy() for t, df in predictions.items()},
            top_n=top_n,
            mode=args.mode,
            eligible_tickers=None,
            use_eligibility_filter=not args.no_eligibility_filter,
        )
        trades_df, equity, benchmarks, metrics = run_portfolio_backtest(
            ranked,
            args.mode,
            args.stress,
            use_trade_rules=False,
        )
        metrics["factor_composite_top_n"] = int(top_n)
        suffix = f"factor_top{top_n}_{args.mode}_{len(ranked)}tickers"
        _write_outputs(out_dir, suffix, predictions, trades_df, equity, benchmarks, metrics)
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
    summary.to_csv(out_dir / "factor_composite_summary.csv", index=False)
    print(f"Saved benchmark -> {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
