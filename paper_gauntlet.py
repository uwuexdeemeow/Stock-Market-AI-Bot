from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from settings import LOG_DIR, SIGNAL_DIR


SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)
PAPER_TRADES = SIGNALS / "paper_trades.csv"
PAPER_EQUITY = SIGNALS / "paper_equity.csv"

MIN_TRADES = 20
MIN_SHARPE = 0.5
MAX_DRAWDOWN_PCT = -10.0
MIN_WIN_RATE = 0.52


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(dd.min() * 100.0)


def _sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2 or float(returns.std()) == 0.0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def evaluate_paper() -> dict:
    trades = _read_csv(PAPER_TRADES)
    equity_df = _read_csv(PAPER_EQUITY)

    if trades.empty and equity_df.empty:
        return {
            "status": "no_data",
            "approved_for_real_capital": False,
            "reason": "No paper_trades.csv or paper_equity.csv found.",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    pnl = _numeric_series(trades, ["net_pnl", "pnl", "realized_pnl"])
    trade_returns = _numeric_series(trades, ["return_pct", "net_return_pct", "return"])
    if trade_returns.empty and not pnl.empty:
        trade_returns = pnl / 10_000.0

    equity = _numeric_series(equity_df, ["equity", "portfolio_value", "account_value"])
    if equity.empty and not pnl.empty:
        equity = 10_000.0 + pnl.cumsum()

    daily_returns = equity.pct_change().dropna() if not equity.empty else trade_returns
    total_pnl = float(pnl.sum()) if not pnl.empty else float(equity.iloc[-1] - equity.iloc[0]) if len(equity) > 1 else 0.0
    total_return_pct = float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0) if len(equity) > 1 else 0.0
    win_rate = float((pnl > 0).mean()) if not pnl.empty else 0.0
    sharpe = _sharpe(daily_returns)
    max_dd = _max_drawdown(equity)

    reasons: list[str] = []
    if len(trades) < MIN_TRADES:
        reasons.append(f"paper_trades {len(trades)} < {MIN_TRADES}")
    if sharpe < MIN_SHARPE:
        reasons.append(f"paper_sharpe {sharpe:.3f} < {MIN_SHARPE:.3f}")
    if max_dd < MAX_DRAWDOWN_PCT:
        reasons.append(f"paper_max_drawdown {max_dd:.2f}% < {MAX_DRAWDOWN_PCT:.2f}%")
    if win_rate < MIN_WIN_RATE:
        reasons.append(f"paper_win_rate {win_rate:.3f} < {MIN_WIN_RATE:.3f}")

    return {
        "status": "passed" if not reasons else "failed",
        "approved_for_real_capital": not reasons,
        "reason": "paper gauntlet passed" if not reasons else "; ".join(reasons),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper_trades": int(len(trades)),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
    }


def main() -> None:
    result = evaluate_paper()
    LOGS.mkdir(parents=True, exist_ok=True)
    out_path = LOGS / f"paper_gauntlet_{datetime.now().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Paper gauntlet written -> {out_path}")
    print(f"Status: {result['status']} | approved_for_real_capital={result['approved_for_real_capital']}")
    print(f"Reason: {result['reason']}")


if __name__ == "__main__":
    main()
