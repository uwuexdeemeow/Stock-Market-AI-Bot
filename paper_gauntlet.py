"""
paper_gauntlet.py — Track paper-trading performance against model predictions.

PLAIN ENGLISH:
Before live capital, we run a "gauntlet" — at least 3 months of paper
trading where every day we log:
  * what the model predicted (direction, confidence, expected return)
  * what actually happened (realized return, fill price, fees)
  * slippage realized vs slippage modeled

If, after ~60+ trading days:
  * realized Sharpe net of costs ≥ 1.0
  * realized alpha vs SPY statistically positive (Newey-West t > 2)
  * mean realized slippage within 1.5× modeled slippage
  * max drawdown ≤ 10%
…then the system earns the right to go live. Otherwise: iterate.

This script reads signals/paper_trades.csv + signals/paper_equity.csv
(written by moomoo_paper_trading.py) and produces a scorecard to
logs/paper_gauntlet_<YYYYMMDD>.json plus a human-readable markdown summary.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from settings import LOG_DIR, SIGNAL_DIR

TRADES_CSV = os.path.join(SIGNAL_DIR, "paper_trades.csv")
EQUITY_CSV = os.path.join(SIGNAL_DIR, "paper_equity.csv")

GO_LIVE_CRITERIA = {
    "min_days": 60,
    "min_sharpe_net": 1.0,
    "max_drawdown": 0.10,
    "max_slippage_ratio": 1.5,
    "min_alpha_tstat": 2.0,
}


def _sharpe(returns: pd.Series) -> float:
    if returns.std(ddof=0) == 0 or len(returns) < 2:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def _newey_west_tstat(excess: pd.Series, lag: int = 5) -> float:
    """Simple Newey-West t-stat on the mean of `excess` returns."""
    n = len(excess)
    if n < 30:
        return 0.0
    r = excess.to_numpy() - excess.mean()
    gamma0 = float(np.mean(r * r))
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        gk = float(np.mean(r[k:] * r[:-k]))
        lrv += 2 * w * gk
    se = float(np.sqrt(lrv / n))
    if se == 0:
        return 0.0
    return float(excess.mean() / se)


def score() -> dict:
    if not (os.path.exists(TRADES_CSV) and os.path.exists(EQUITY_CSV)):
        return {"status": "no_data",
                "hint": "Run paper trading first; no paper_trades.csv or paper_equity.csv yet."}

    eq = pd.read_csv(EQUITY_CSV, parse_dates=["date"]).set_index("date").sort_index()
    trades = pd.read_csv(TRADES_CSV, parse_dates=["date"]).sort_values("date")

    rets = eq["equity"].pct_change().dropna()
    sharpe = _sharpe(rets)
    dd = _max_drawdown(eq["equity"])

    # SPY benchmark
    spy = yf.download("SPY", start=eq.index[0], end=eq.index[-1] + pd.Timedelta(days=1),
                      progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy_rets = spy["Close"].pct_change().reindex(rets.index).fillna(0)
    excess = rets - spy_rets
    tstat = _newey_west_tstat(excess)

    # slippage: compare modeled vs realized if columns are present
    slip_ratio = None
    if {"modeled_slippage", "realized_slippage"}.issubset(trades.columns):
        modeled = trades["modeled_slippage"].abs().mean()
        realized = trades["realized_slippage"].abs().mean()
        slip_ratio = float(realized / modeled) if modeled else None

    n_days = len(rets)
    gates = {
        "enough_days": n_days >= GO_LIVE_CRITERIA["min_days"],
        "sharpe_ok": sharpe >= GO_LIVE_CRITERIA["min_sharpe_net"],
        "drawdown_ok": dd >= -GO_LIVE_CRITERIA["max_drawdown"],
        "alpha_ok": tstat >= GO_LIVE_CRITERIA["min_alpha_tstat"],
        "slippage_ok": slip_ratio is None or slip_ratio <= GO_LIVE_CRITERIA["max_slippage_ratio"],
    }
    report = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "days_of_data": int(n_days),
        "sharpe_net": round(sharpe, 3),
        "max_drawdown": round(dd, 4),
        "alpha_tstat_vs_spy": round(tstat, 3),
        "slippage_ratio_realized_over_modeled": slip_ratio,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }

    out = os.path.join(LOG_DIR, f"paper_gauntlet_{datetime.utcnow():%Y%m%d}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    print(json.dumps(score(), indent=2, default=str))
