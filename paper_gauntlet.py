from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from settings import LOG_DIR, PAPER_MODE_STRATEGY, SIGNAL_DIR


SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)
PAPER_TRADES = SIGNALS / "paper_trades.csv"
PAPER_EQUITY = SIGNALS / "paper_equity.csv"
PAPER_STATUS = SIGNALS / "paper_daily_status.json"

MIN_TRADES = 20
MIN_SHARPE = 0.5
MAX_DRAWDOWN_PCT = -10.0
MIN_WIN_RATE = 0.52
MIN_PORTFOLIO_EQUITY_DAYS = int(os.environ.get("PAPER_GAUNTLET_MIN_EQUITY_DAYS", "20"))
MIN_PORTFOLIO_FILL_RATE = float(os.environ.get("PAPER_GAUNTLET_MIN_FILL_RATE", "0.95"))
MAX_PORTFOLIO_CANCEL_RATE = float(os.environ.get("PAPER_GAUNTLET_MAX_CANCEL_RATE", "0.05"))
MAX_PORTFOLIO_DRIFT = float(os.environ.get("PAPER_GAUNTLET_MAX_DRIFT", "0.15"))


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


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def _paper_days(equity_df: pd.DataFrame) -> int:
    if equity_df.empty:
        return 0
    if "date" in equity_df.columns:
        return int(pd.to_datetime(equity_df["date"], errors="coerce").dt.date.nunique())
    if "timestamp" in equity_df.columns:
        return int(pd.to_datetime(equity_df["timestamp"], errors="coerce").dt.date.nunique())
    return int(len(equity_df))


def _order_fill_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "submitted_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "partial_orders": 0,
            "fill_rate": 0.0,
            "cancel_rate": 0.0,
        }

    submitted = trades
    if "submitted" in trades.columns:
        submitted = trades[trades["submitted"].astype(str).str.lower().isin(["true", "1", "yes"])]
    statuses = (
        submitted.get("fill_status", pd.Series(index=submitted.index, dtype=str))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    filled = int(statuses.eq("filled").sum())
    cancelled = int(statuses.eq("cancelled").sum())
    partial = int(statuses.eq("partial").sum() + statuses.eq("partially_filled").sum())
    total = int(len(submitted))
    return {
        "submitted_orders": total,
        "filled_orders": filled,
        "cancelled_orders": cancelled,
        "partial_orders": partial,
        "fill_rate": float((filled + 0.5 * partial) / total) if total else 0.0,
        "cancel_rate": float(cancelled / total) if total else 0.0,
    }


def _filter_current_signal_trades(trades: pd.DataFrame, status: dict) -> pd.DataFrame:
    if trades.empty or "submitted_at" not in trades.columns:
        return trades
    signal_ts = pd.to_datetime(status.get("signal_predicted_at", ""), errors="coerce", utc=True)
    if pd.isna(signal_ts):
        return trades
    submitted_ts = pd.to_datetime(trades["submitted_at"], errors="coerce", utc=True)
    return trades[submitted_ts >= signal_ts].copy()


def _is_portfolio_strategy(status: dict) -> bool:
    strategy = str(status.get("strategy") or PAPER_MODE_STRATEGY).strip().lower()
    return strategy in {"core_satellite_alpha", "etf_rotation"}


def _evaluate_portfolio_paper(
    trades: pd.DataFrame,
    equity_df: pd.DataFrame,
    status: dict,
    equity: pd.Series,
    daily_returns: pd.Series,
    total_return_pct: float,
    sharpe: float,
    max_dd: float,
) -> dict:
    current_signal_trades = _filter_current_signal_trades(trades, status)
    fill_stats = _order_fill_stats(current_signal_trades)
    paper_days = _paper_days(equity_df)
    strategy = str(status.get("strategy") or PAPER_MODE_STRATEGY)
    max_drift_abs = float(status.get("max_drift_abs", np.nan))

    reasons: list[str] = []
    if not bool(status.get("paper_ready", False)):
        reasons.append("strategy paper_ready is false")
    if not bool(status.get("gates_all_pass", False)):
        reasons.append("strategy backtest gates_all_pass is false")
    if not bool(status.get("freshness_ok", True)):
        reasons.append("paper signal freshness check failed")
    if paper_days < MIN_PORTFOLIO_EQUITY_DAYS:
        reasons.append(f"paper_equity_days {paper_days} < {MIN_PORTFOLIO_EQUITY_DAYS}")
    if fill_stats["submitted_orders"] <= 0:
        reasons.append("no submitted paper orders found")
    elif fill_stats["fill_rate"] < MIN_PORTFOLIO_FILL_RATE:
        reasons.append(f"fill_rate {fill_stats['fill_rate']:.3f} < {MIN_PORTFOLIO_FILL_RATE:.3f}")
    if fill_stats["cancel_rate"] > MAX_PORTFOLIO_CANCEL_RATE:
        reasons.append(f"cancel_rate {fill_stats['cancel_rate']:.3f} > {MAX_PORTFOLIO_CANCEL_RATE:.3f}")
    if np.isfinite(max_drift_abs) and max_drift_abs > MAX_PORTFOLIO_DRIFT:
        reasons.append(f"max_drift_abs {max_drift_abs:.3f} > {MAX_PORTFOLIO_DRIFT:.3f}")
    if max_dd < MAX_DRAWDOWN_PCT:
        reasons.append(f"paper_max_drawdown {max_dd:.2f}% < {MAX_DRAWDOWN_PCT:.2f}%")
    if paper_days >= MIN_PORTFOLIO_EQUITY_DAYS and sharpe < MIN_SHARPE:
        reasons.append(f"paper_sharpe {sharpe:.3f} < {MIN_SHARPE:.3f}")

    return {
        "status": "passed" if not reasons else "failed",
        "approved_for_real_capital": not reasons,
        "reason": "portfolio paper gauntlet passed" if not reasons else "; ".join(reasons),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": strategy,
        "portfolio_gauntlet": True,
        "paper_equity_days": paper_days,
        "paper_equity_rows": int(len(equity_df)),
        "paper_trades": int(len(trades)),
        "current_signal_paper_trades": int(len(current_signal_trades)),
        **fill_stats,
        "total_return_pct": round(total_return_pct, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "current_regime": status.get("current_regime"),
        "paper_ready": bool(status.get("paper_ready", False)),
        "gates_all_pass": bool(status.get("gates_all_pass", False)),
        "freshness_ok": bool(status.get("freshness_ok", True)),
        "current_gross_exposure": status.get("current_gross_exposure"),
        "target_gross_exposure": status.get("target_gross_exposure"),
        "max_drift_abs": None if not np.isfinite(max_drift_abs) else round(max_drift_abs, 4),
        "recent_equity": float(equity.iloc[-1]) if not equity.empty else None,
        "daily_return_observations": int(len(daily_returns)),
    }


def evaluate_paper() -> dict:
    trades = _read_csv(PAPER_TRADES)
    equity_df = _read_csv(PAPER_EQUITY)
    status = _read_json(PAPER_STATUS)

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

    if _is_portfolio_strategy(status):
        return _evaluate_portfolio_paper(
            trades=trades,
            equity_df=equity_df,
            status=status,
            equity=equity,
            daily_returns=daily_returns,
            total_return_pct=total_return_pct,
            sharpe=sharpe,
            max_dd=max_dd,
        )

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
