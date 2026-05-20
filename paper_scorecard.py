"""
paper_scorecard.py — Compare live Alpaca paper trading results vs walkforward predictions.

PLAIN ENGLISH: The walkforward says "this config should produce Sharpe X, alpha Y%
over QQQ, drawdown no worse than Z%."  This script checks if your actual paper
trading results match those predictions.  If they diverge significantly, something
is leaking alpha between prediction and execution (stale signals, bad fills,
timing mismatch, etc.).

Usage:
    python3 paper_scorecard.py              # show scorecard
    python3 paper_scorecard.py --reset      # clear paper equity history and start fresh
    python3 paper_scorecard.py --json       # output as JSON for programmatic use
    python3 paper_scorecard.py --verbose    # show daily breakdown

The scorecard answers three questions:
    1. Is the strategy performing as the walkforward predicted?
    2. Are we beating the benchmark (QQQ)?
    3. Where is alpha being lost (fills, timing, signal staleness)?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from settings import SIGNAL_DIR, LOG_DIR

SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)

# ── Data sources ──────────────────────────────────────────────────────────────
# Paper equity: daily snapshots of Alpaca account value
ALPACA_EQUITY_FILE = SIGNALS / "alpaca_paper_equity.csv"
# Moomoo equity (if running both brokers)
MOOMOO_EQUITY_FILE = SIGNALS / "paper_equity.csv"
# Walkforward predictions
LIVE_CONFIG_FILE = SIGNALS / "core_satellite_live_configs.json"
# Trade log
ALPACA_TRADE_LOG = SIGNALS / "alpaca_paper_log.csv"
# ETF price data for benchmark comparison
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


def load_paper_equity(broker: str = "alpaca") -> pd.DataFrame:
    """Load paper trading equity curve.

    Returns DataFrame with columns: date, equity
    """
    path = ALPACA_EQUITY_FILE if broker == "alpaca" else MOOMOO_EQUITY_FILE
    if not path.exists():
        return pd.DataFrame(columns=["date", "equity"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    # Keep one row per day (latest snapshot)
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df[["date", "equity"]].reset_index(drop=True)


def load_benchmark(ticker: str = "QQQ", dates: pd.DatetimeIndex | None = None) -> pd.Series:
    """Load benchmark ETF close prices for the given dates.

    Uses reindex with ffill, then drops NaN tails (dates beyond
    the parquet's range).  Returns only dates with valid prices.
    """
    path = DATA_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    close = df["Close"]
    if dates is not None:
        # Normalize dates to midnight for matching
        norm_dates = dates.normalize()
        close = close.reindex(close.index.union(norm_dates)).ffill()
        close = close.reindex(norm_dates).dropna()
    return close


def load_walkforward_predictions() -> dict:
    """Load the approved live config's predicted metrics."""
    if not LIVE_CONFIG_FILE.exists():
        return {}
    try:
        payload = json.loads(LIVE_CONFIG_FILE.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    live = payload.get("approved_live_configs", {}).get("core-alpha", {})
    return live.get("source_metrics", {})


def load_trade_log() -> pd.DataFrame:
    """Load Alpaca trade log for slippage analysis."""
    if not ALPACA_TRADE_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(ALPACA_TRADE_LOG)
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
    return df


def compute_scorecard(broker: str = "alpaca") -> dict:
    """Compute the full paper trading scorecard.

    Returns a dict with:
        - live_metrics: actual performance from paper trading
        - predicted_metrics: what the walkforward said would happen
        - divergence: the gap between predicted and actual
        - health_flags: warnings if something looks wrong
        - slippage: execution quality metrics
    """
    equity_df = load_paper_equity(broker)
    predictions = load_walkforward_predictions()

    if equity_df.empty or len(equity_df) < 2:
        return {
            "status": "insufficient_data",
            "days_tracked": len(equity_df),
            "message": f"Need at least 2 equity snapshots. Currently have {len(equity_df)}. "
                       "Run daily_run.py for a few days to accumulate data.",
            "predicted_metrics": predictions,
        }

    # ── Live performance metrics ─────────────────────────────────────────
    equities = equity_df["equity"].values.astype(float)
    dates = equity_df["date"]
    n_days = len(equities)
    start_equity = float(equities[0])
    end_equity = float(equities[-1])
    total_return_pct = (end_equity / start_equity - 1.0) * 100.0

    # Annualize
    calendar_days = (dates.iloc[-1] - dates.iloc[0]).days
    trading_days = n_days - 1
    if calendar_days > 0 and trading_days > 0:
        years = calendar_days / 365.25
        cagr_pct = ((end_equity / start_equity) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    else:
        cagr_pct = 0.0

    # Daily returns for Sharpe
    daily_returns = np.diff(equities) / equities[:-1]
    mean_daily = float(np.mean(daily_returns)) if len(daily_returns) > 0 else 0.0
    std_daily = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 1e-9
    sharpe = (mean_daily / max(std_daily, 1e-9)) * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(equities)
    drawdown = (equities - peak) / peak
    max_drawdown_pct = float(np.min(drawdown)) * 100.0

    # ── Benchmark comparison ─────────────────────────────────────────────
    bench_dates = pd.DatetimeIndex(dates)
    qqq = load_benchmark("QQQ", bench_dates)
    spy = load_benchmark("SPY", bench_dates)

    alpha_vs_qqq_pct = 0.0
    alpha_vs_spy_pct = 0.0
    qqq_return_pct = 0.0
    spy_return_pct = 0.0

    if not qqq.empty and len(qqq) >= 2:
        qqq_vals = qqq.values
        qqq_return_pct = (float(qqq_vals[-1]) / float(qqq_vals[0]) - 1.0) * 100.0
        alpha_vs_qqq_pct = total_return_pct - qqq_return_pct

    if not spy.empty and len(spy) >= 2:
        spy_vals = spy.values
        spy_return_pct = (float(spy_vals[-1]) / float(spy_vals[0]) - 1.0) * 100.0
        alpha_vs_spy_pct = total_return_pct - spy_return_pct

    live_metrics = {
        "start_date": str(dates.iloc[0].date()),
        "end_date": str(dates.iloc[-1].date()),
        "trading_days": trading_days,
        "calendar_days": calendar_days,
        "start_equity": round(start_equity, 2),
        "end_equity": round(end_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "alpha_vs_qqq_pct": round(alpha_vs_qqq_pct, 2),
        "alpha_vs_spy_pct": round(alpha_vs_spy_pct, 2),
        "qqq_return_pct": round(qqq_return_pct, 2),
        "spy_return_pct": round(spy_return_pct, 2),
    }

    # ── Execution quality analysis ───────────────────────────────────────
    # NOTE: The "price" column in the trade log is the price at SIGNAL
    # generation time, NOT the limit price.  The gap between signal price
    # and fill price includes market movement (hours/overnight) plus true
    # execution slippage.  We report it as "signal-to-fill gap" — a large
    # gap means either (a) signal is stale by execution time, or (b) bad
    # fills.  Either way it's alpha leakage.
    trades = load_trade_log()
    slippage_metrics = {}
    if not trades.empty and "filled_avg_price" in trades.columns and "price" in trades.columns:
        trades["signal_price"] = pd.to_numeric(trades["price"], errors="coerce")
        trades["fill_price"] = pd.to_numeric(trades["filled_avg_price"], errors="coerce")
        valid = trades.dropna(subset=["signal_price", "fill_price"])
        if not valid.empty:
            # Gap in basis points (positive = paid more than signal price)
            gap_bps = ((valid["fill_price"] - valid["signal_price"]) / valid["signal_price"]) * 10000
            # For sells, flip sign (selling lower than signal = bad)
            if "side" in valid.columns:
                sell_mask = valid["side"].str.lower() == "sell"
                gap_bps.loc[sell_mask] = -gap_bps.loc[sell_mask]

            slippage_metrics = {
                "mean_signal_to_fill_gap_bps": round(float(gap_bps.mean()), 2),
                "median_signal_to_fill_gap_bps": round(float(gap_bps.median()), 2),
                "worst_signal_to_fill_gap_bps": round(float(gap_bps.max()), 2),
                "total_trades": len(valid),
                "trades_with_adverse_gap": int((gap_bps > 0).sum()),
                "note": "Includes market movement + execution slippage",
            }

    # ── Divergence from predictions ──────────────────────────────────────
    divergence = {}
    health_flags = []

    if predictions:
        pred_sharpe = predictions.get("mean_oos_sharpe", 0.0)
        pred_alpha_qqq = predictions.get("mean_oos_alpha_vs_qqq_pct", 0.0)
        pred_dd = predictions.get("mean_oos_max_drawdown_pct", 0.0)
        pred_cagr = predictions.get("mean_oos_cagr_pct", 0.0)

        divergence = {
            "sharpe_gap": round(sharpe - pred_sharpe, 3),
            "alpha_vs_qqq_gap_pct": round(alpha_vs_qqq_pct - pred_alpha_qqq, 2),
            "drawdown_gap_pct": round(max_drawdown_pct - pred_dd, 2),
            "cagr_gap_pct": round(cagr_pct - pred_cagr, 2),
        }

        # Health flags
        if trading_days >= 5:
            # Annualized alpha check (only meaningful after a week+)
            annualized_alpha_qqq = alpha_vs_qqq_pct * (252.0 / max(trading_days, 1))
            if annualized_alpha_qqq < -10.0:
                health_flags.append(
                    f"ALPHA_LEAK: annualized alpha vs QQQ is {annualized_alpha_qqq:.1f}% "
                    f"(predicted {pred_alpha_qqq:.1f}%)"
                )
            if sharpe < 0 and pred_sharpe > 0.5:
                health_flags.append(
                    f"SHARPE_COLLAPSE: live Sharpe={sharpe:.2f} vs predicted={pred_sharpe:.2f}"
                )
            if max_drawdown_pct < pred_dd * 1.5:
                health_flags.append(
                    f"DRAWDOWN_EXCESS: live DD={max_drawdown_pct:.1f}% vs predicted={pred_dd:.1f}%"
                )

        if slippage_metrics.get("mean_signal_to_fill_gap_bps", 0) > 50:
            health_flags.append(
                f"HIGH_SIGNAL_TO_FILL_GAP: mean={slippage_metrics['mean_signal_to_fill_gap_bps']:.1f}bps "
                f"(target <50bps — includes market movement between signal & fill)"
            )

        # Stale signal check
        if calendar_days > 5 and trading_days < calendar_days * 0.4:
            health_flags.append(
                f"STALE_DATA: only {trading_days} equity snapshots over {calendar_days} "
                f"calendar days — are daily runs working?"
            )

    return {
        "status": "ok" if not health_flags else "warning",
        "generated_at": datetime.now().isoformat(),
        "broker": broker,
        "live_metrics": live_metrics,
        "predicted_metrics": {
            "mean_oos_sharpe": predictions.get("mean_oos_sharpe"),
            "mean_oos_cagr_pct": predictions.get("mean_oos_cagr_pct"),
            "mean_oos_alpha_vs_qqq_pct": predictions.get("mean_oos_alpha_vs_qqq_pct"),
            "mean_oos_alpha_vs_spy_pct": predictions.get("mean_oos_alpha_vs_spy_pct"),
            "mean_oos_max_drawdown_pct": predictions.get("mean_oos_max_drawdown_pct"),
            "worst_oos_max_drawdown_pct": predictions.get("worst_oos_max_drawdown_pct"),
        },
        "divergence": divergence,
        "slippage": slippage_metrics,
        "health_flags": health_flags,
    }


def reset_scorecard(broker: str = "alpaca") -> None:
    """Clear equity history to start fresh.

    Backs up existing file with timestamp suffix before deleting.
    """
    path = ALPACA_EQUITY_FILE if broker == "alpaca" else MOOMOO_EQUITY_FILE
    if path.exists():
        backup = path.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        path.rename(backup)
        print(f"Backed up {path.name} → {backup.name}")
        print("Equity history cleared. Next daily run will start fresh.")
    else:
        print(f"No equity file found at {path}. Nothing to reset.")


def print_scorecard(scorecard: dict) -> None:
    """Pretty-print the scorecard to terminal."""
    status = scorecard.get("status", "unknown")

    if status == "insufficient_data":
        print(f"\n{'═' * 60}")
        print(f"  PAPER TRADING SCORECARD")
        print(f"  {scorecard.get('message', 'No data')}")
        print(f"{'═' * 60}\n")
        return

    live = scorecard.get("live_metrics", {})
    pred = scorecard.get("predicted_metrics", {})
    div = scorecard.get("divergence", {})
    slip = scorecard.get("slippage", {})
    flags = scorecard.get("health_flags", [])

    print(f"\n{'═' * 60}")
    print(f"  PAPER TRADING SCORECARD — {scorecard.get('broker', 'alpaca').upper()}")
    print(f"  {live.get('start_date', '?')} → {live.get('end_date', '?')} "
          f"({live.get('trading_days', 0)} trading days)")
    print(f"{'═' * 60}")

    # Performance
    print(f"\n  {'METRIC':<25} {'LIVE':>10} {'PREDICTED':>10} {'GAP':>10}")
    print(f"  {'─' * 55}")
    rows = [
        ("Sharpe", f"{live.get('sharpe', 0):.2f}", f"{pred.get('mean_oos_sharpe', '?')}", f"{div.get('sharpe_gap', '?')}"),
        ("CAGR %", f"{live.get('cagr_pct', 0):.1f}%", f"{pred.get('mean_oos_cagr_pct', '?')}%", f"{div.get('cagr_gap_pct', '?')}%"),
        ("Alpha vs QQQ %", f"{live.get('alpha_vs_qqq_pct', 0):.1f}%", f"{pred.get('mean_oos_alpha_vs_qqq_pct', '?')}%", f"{div.get('alpha_vs_qqq_gap_pct', '?')}%"),
        ("Alpha vs SPY %", f"{live.get('alpha_vs_spy_pct', 0):.1f}%", f"{pred.get('mean_oos_alpha_vs_spy_pct', '?')}%", ""),
        ("Max Drawdown %", f"{live.get('max_drawdown_pct', 0):.1f}%", f"{pred.get('mean_oos_max_drawdown_pct', '?')}%", f"{div.get('drawdown_gap_pct', '?')}%"),
    ]
    for label, live_val, pred_val, gap_val in rows:
        print(f"  {label:<25} {live_val:>10} {pred_val:>10} {gap_val:>10}")

    # Benchmark
    print(f"\n  {'BENCHMARK':<25} {'RETURN':>10}")
    print(f"  {'─' * 35}")
    print(f"  {'Strategy':<25} {live.get('total_return_pct', 0):>+10.2f}%")
    print(f"  {'QQQ':<25} {live.get('qqq_return_pct', 0):>+10.2f}%")
    print(f"  {'SPY':<25} {live.get('spy_return_pct', 0):>+10.2f}%")

    # Execution quality (signal-to-fill gap)
    if slip:
        print(f"\n  EXECUTION QUALITY (signal → fill gap)")
        print(f"  {'─' * 45}")
        print(f"  Mean gap:         {slip.get('mean_signal_to_fill_gap_bps', 0):.1f} bps")
        print(f"  Median gap:       {slip.get('median_signal_to_fill_gap_bps', 0):.1f} bps")
        print(f"  Worst gap:        {slip.get('worst_signal_to_fill_gap_bps', 0):.1f} bps")
        print(f"  Total trades:     {slip.get('total_trades', 0)}")
        print(f"  Adverse fills:    {slip.get('trades_with_adverse_gap', 0)}")
        print(f"  Note: gap includes market movement + execution slippage")

    # Health flags
    if flags:
        print(f"\n  ⚠ HEALTH WARNINGS")
        print(f"  {'─' * 35}")
        for flag in flags:
            print(f"  • {flag}")
    elif status == "ok":
        print(f"\n  ✓ No health warnings")

    print(f"\n{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Paper trading scorecard")
    parser.add_argument("--reset", action="store_true",
                        help="Clear equity history and start fresh")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--broker", default="alpaca", choices=["alpaca", "moomoo"],
                        help="Which broker to check (default: alpaca)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show daily equity breakdown")
    args = parser.parse_args()

    if args.reset:
        reset_scorecard(args.broker)
        return

    scorecard = compute_scorecard(args.broker)

    if args.json:
        print(json.dumps(scorecard, indent=2, default=str))
    else:
        print_scorecard(scorecard)

        if args.verbose and scorecard.get("status") != "insufficient_data":
            equity_df = load_paper_equity(args.broker)
            if not equity_df.empty:
                print("  DAILY EQUITY")
                print(f"  {'─' * 45}")
                for _, row in equity_df.iterrows():
                    print(f"  {row['date'].strftime('%Y-%m-%d')}  ${row['equity']:>12,.2f}")
                print()

    # Save scorecard to logs
    LOGS.mkdir(parents=True, exist_ok=True)
    out_path = LOGS / "paper_scorecard.json"
    out_path.write_text(json.dumps(scorecard, indent=2, default=str))


if __name__ == "__main__":
    main()
