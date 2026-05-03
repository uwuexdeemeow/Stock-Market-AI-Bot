#!/usr/bin/env python3
"""
Quick grid search over SPY timing parameters.

Loads cached walk-forward predictions from signals/ directory (saved by
backtest.py) and re-runs ONLY the SPY timing simulation with different
parameter combos.  Each combo takes ~2 seconds instead of ~30 minutes.

Usage:
    python3 spy_timing_grid_search.py
"""

import os, sys, itertools, time
import pandas as pd

# ── Load project settings & the SPY timing function ─────────────────────────
# We import backtest.py's run_spy_timing_backtest and override the settings
# via environment variables before each call.
from settings import SIGNAL_DIR

# Suppress verbose output during grid search
os.environ["BACKTEST_QUIET"] = "1"

from backtest import run_spy_timing_backtest


def load_cached_predictions() -> dict:
    """
    Read the per-ticker *_walkforward_predictions.csv files that backtest.py
    already saved.  Returns a dict  {ticker: DataFrame}  in the same format
    that run_spy_timing_backtest expects.
    """
    preds = {}
    for fn in os.listdir(SIGNAL_DIR):
        if fn.endswith("_walkforward_predictions.csv"):
            ticker = fn.replace("_walkforward_predictions.csv", "")
            path = os.path.join(SIGNAL_DIR, fn)
            df = pd.read_csv(path, parse_dates=["date"])
            if "date" in df.columns:
                df = df.set_index("date")
            preds[ticker] = df
    return preds


def main():
    print("Loading cached predictions from signals/ ...")
    predictions = load_cached_predictions()
    print(f"  Loaded {len(predictions)} tickers\n")

    if not predictions:
        print("ERROR: No cached predictions found.  Run backtest.py first.")
        sys.exit(1)

    # ── Parameter grid ──────────────────────────────────────────────────────
    # enter_thr: bull_fraction needed to go from cash → invested
    # exit_thr:  bull_fraction to go from invested → cash
    # min_hold:  minimum days between rebalance checks
    enter_values  = [0.50, 0.52, 0.55, 0.60]
    exit_values   = [0.40, 0.45, 0.48, 0.50]
    hold_values   = [1, 3, 5, 10, 15, 20]
    smooth_values = [1]  # no smoothing (already tested, hurts)

    results = []
    combos = list(itertools.product(enter_values, exit_values, hold_values, smooth_values))
    # Filter out invalid combos where exit >= enter (no hysteresis band)
    combos = [(e, x, h, s) for e, x, h, s in combos if x <= e]

    print(f"Testing {len(combos)} parameter combinations ...\n")
    print(f"{'Enter':>6} {'Exit':>6} {'Hold':>5} {'Smooth':>6} | "
          f"{'Return%':>8} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'NW-t':>6} {'MaxDD%':>7} {'InMkt%':>7} {'Flips':>6} {'Cost%':>7}")
    print("-" * 110)

    import backtest as _bt

    for enter_thr, exit_thr, min_hold, smooth in combos:
        # Monkey-patch the module-level variables that run_spy_timing_backtest
        # reads directly.  importlib.reload doesn't work because backtest.py
        # copies them into its own namespace at import time.
        _bt.SPY_TIMING_ENTER_THRESHOLD = enter_thr
        _bt.SPY_TIMING_EXIT_THRESHOLD  = exit_thr
        _bt.SPY_TIMING_MIN_HOLD_DAYS   = min_hold
        _bt.SPY_TIMING_SMOOTH_WINDOW   = smooth

        t0 = time.time()
        try:
            metrics = run_spy_timing_backtest(predictions, quiet=True)
        except Exception as exc:
            print(f"{enter_thr:>6.2f} {exit_thr:>6.2f} {min_hold:>5d} {smooth:>6d} | ERROR: {exc}")
            continue
        elapsed = time.time() - t0

        if not metrics:
            continue

        # Count signal flips from the signals file
        try:
            sig_df = pd.read_csv(os.path.join(SIGNAL_DIR, "spy_timing_signals.csv"))
            n_flips = (sig_df["signal"] != sig_df["signal"].shift()).sum()
        except Exception:
            n_flips = -1

        row = {
            "enter": enter_thr, "exit": exit_thr, "hold": min_hold, "smooth": smooth,
            "return_pct": metrics.get("total_return_pct", 0),
            "cagr_pct": metrics.get("cagr_pct", 0),
            "sharpe": metrics.get("sharpe", 0),
            "sortino": metrics.get("sortino", 0),
            "nw_tstat": metrics.get("nw_tstat_vs_cash", 0),
            "max_dd": metrics.get("max_drawdown_pct", 0),
            "in_market": metrics.get("in_market_pct", 0),
            "n_flips": n_flips,
            "cost_pct": metrics.get("estimated_cost_pct", 0),
            "elapsed": elapsed,
        }
        results.append(row)

        print(f"{enter_thr:>6.2f} {exit_thr:>6.2f} {min_hold:>5d} {smooth:>6d} | "
              f"{row['return_pct']:>8.1f} {row['cagr_pct']:>7.2f} {row['sharpe']:>7.3f} "
              f"{row['sortino']:>8.3f} {row['nw_tstat']:>6.2f} {row['max_dd']:>7.2f} "
              f"{row['in_market']:>7.1f} {row['n_flips']:>6d} {row['cost_pct']:>7.2f}")

    # ── Summary ─────────────────────────────────────────────────────────────
    if results:
        rdf = pd.DataFrame(results)
        rdf.to_csv(os.path.join(SIGNAL_DIR, "spy_timing_grid_results.csv"), index=False)

        print("\n" + "=" * 110)
        print("TOP 10 by Sharpe:")
        top = rdf.nlargest(10, "sharpe")
        for _, r in top.iterrows():
            print(f"  enter={r['enter']:.2f} exit={r['exit']:.2f} hold={int(r['hold']):>2d} | "
                  f"Sharpe={r['sharpe']:.3f}  Return={r['return_pct']:.1f}%  "
                  f"CAGR={r['cagr_pct']:.2f}%  NW-t={r['nw_tstat']:.2f}  "
                  f"MaxDD={r['max_dd']:.1f}%  Flips={int(r['n_flips'])}  Cost={r['cost_pct']:.2f}%")

        print(f"\nResults saved to {SIGNAL_DIR}/spy_timing_grid_results.csv")


if __name__ == "__main__":
    main()
