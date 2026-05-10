"""
paper_report.py — Side-by-side performance report for paper trading.

PLAIN ENGLISH: You're running two strategies on two brokers — this script
pulls data from both and shows you which one is winning.  Think of it as
a report card comparing Moomoo (core-satellite) vs Alpaca (TQQQ-enhanced).

Usage:
    python3 paper_report.py           # full comparison report
    python3 paper_report.py --json    # raw JSON output
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from settings import LOG_DIR, SIGNAL_DIR


SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# DATA FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Moomoo (core-satellite)
MOOMOO_EQUITY = SIGNALS / "paper_equity.csv"
MOOMOO_TRADES = SIGNALS / "paper_trades.csv"
MOOMOO_STATUS = SIGNALS / "paper_daily_status.json"

# Alpaca (TQQQ-enhanced)
ALPACA_EQUITY = SIGNALS / "alpaca_paper_equity.csv"
ALPACA_TRADES = SIGNALS / "alpaca_paper_log.csv"
ALPACA_SIGNAL = SIGNALS / "core_satellite_tqqq_signal.csv"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    """Safely read a CSV, return empty DataFrame if missing."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    """Safely read a JSON file."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_equity_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Extract equity series and daily returns from an equity CSV.
    Returns (equity_series_indexed_by_date, daily_returns).
    """
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Find the equity column
    eq_col = None
    for col in ("equity", "portfolio_value", "account_value", "total_equity"):
        if col in df.columns:
            eq_col = col
            break
    if eq_col is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Find the date column
    date_col = None
    for col in ("date", "timestamp"):
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    df = df.copy()
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df[eq_col] = pd.to_numeric(df[eq_col], errors="coerce")
    df = df.dropna(subset=["_date", eq_col])
    df = df.drop_duplicates("_date", keep="last").sort_values("_date")

    equity = pd.Series(df[eq_col].values, index=pd.DatetimeIndex(df["_date"].values))
    returns = equity.pct_change().dropna()
    return equity, returns


def _sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2 or float(returns.std()) == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    """Max drawdown as a percentage (negative number)."""
    if equity.empty or len(equity) < 2:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(dd.min() * 100.0)


def _total_return(equity: pd.Series) -> float:
    """Total return as a percentage."""
    if len(equity) < 2:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)


def _fill_stats(trades: pd.DataFrame) -> dict:
    """Compute fill rate from trade log."""
    if trades.empty:
        return {"total": 0, "filled": 0, "cancelled": 0, "fill_rate": 0.0}

    total = len(trades)
    statuses = pd.Series(dtype=str)
    for col in ("fill_status", "status"):
        if col in trades.columns:
            statuses = trades[col].fillna("").astype(str).str.lower()
            break

    filled = int(statuses.eq("filled").sum() + statuses.eq("filled_all").sum())
    cancelled = int(statuses.eq("cancelled").sum())

    # If no status column, assume all filled (Alpaca paper fills instantly)
    if filled == 0 and cancelled == 0 and total > 0:
        filled = total

    return {
        "total": total,
        "filled": filled,
        "cancelled": cancelled,
        "fill_rate": float(filled / total) if total else 0.0,
    }


def _trading_days(trades: pd.DataFrame) -> int:
    """Count unique trading days in the log."""
    if trades.empty:
        return 0
    for col in ("submitted_at", "timestamp", "date"):
        if col in trades.columns:
            dates = pd.to_datetime(trades[col], errors="coerce").dt.date.dropna()
            return int(dates.nunique())
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_strategy_data(
    name: str,
    equity_path: Path,
    trades_path: Path,
    status_path: Path | None = None,
) -> dict:
    """
    Load all data for one strategy and compute metrics.

    PLAIN ENGLISH: Reads the equity CSV, trade log, and status file for
    one strategy, then computes return, Sharpe, drawdown, and fill stats.
    """
    equity_df = _read_csv(equity_path)
    trades = _read_csv(trades_path)
    status = _read_json(status_path) if status_path else {}

    equity, returns = _get_equity_series(equity_df)
    fills = _fill_stats(trades)
    days = _trading_days(trades)

    # Starting and current equity
    start_eq = float(equity.iloc[0]) if not equity.empty else 0.0
    current_eq = float(equity.iloc[-1]) if not equity.empty else 0.0

    return {
        "name": name,
        "has_data": not equity.empty or not trades.empty,
        "equity_rows": len(equity),
        "trading_days": days,
        "start_equity": round(start_eq, 2),
        "current_equity": round(current_eq, 2),
        "total_return_pct": round(_total_return(equity), 2),
        "sharpe": round(_sharpe(returns), 3),
        "max_drawdown_pct": round(_max_drawdown(equity), 2),
        "total_trades": fills["total"],
        "filled_trades": fills["filled"],
        "cancelled_trades": fills["cancelled"],
        "fill_rate": round(fills["fill_rate"], 4),
        "current_regime": status.get("current_regime", "unknown"),
        "gross_exposure": status.get("current_gross_exposure", 0.0),
        "first_date": str(equity.index[0].date()) if not equity.empty else "n/a",
        "last_date": str(equity.index[-1].date()) if not equity.empty else "n/a",
    }


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def _load_benchmark_returns(start_date: str, end_date: str) -> dict:
    """
    Load SPY and QQQ returns over the paper trading period.

    PLAIN ENGLISH: To know if your strategy is actually good, you need
    to compare it against just buying SPY or QQQ.  This fetches their
    returns over the same dates your paper trading covers.
    """
    try:
        import yfinance as yf
        benchmarks = {}
        for sym in ("SPY", "QQQ"):
            data = yf.download(sym, start=start_date, end=end_date, progress=False)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0]
            else:
                close = data["Close"]
            if len(close) >= 2:
                ret = float(close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
                benchmarks[sym] = round(ret, 2)
        return benchmarks
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report() -> dict:
    """
    Generate a full comparison report for both strategies.

    Returns a dict with all metrics for both strategies and benchmarks.
    """
    moomoo = _load_strategy_data(
        "Moomoo (Core-Satellite)",
        MOOMOO_EQUITY, MOOMOO_TRADES, MOOMOO_STATUS,
    )

    # For Alpaca, also read signal for regime info
    alpaca_status = {}
    if ALPACA_SIGNAL.exists():
        try:
            sig = pd.read_csv(ALPACA_SIGNAL).iloc[0]
            alpaca_status = {
                "current_regime": str(sig.get("current_regime", "unknown")),
                "current_gross_exposure": float(sig.get("gross_exposure", 0.0) or 0.0),
            }
        except Exception:
            pass

    alpaca = _load_strategy_data(
        "Alpaca (TQQQ-Enhanced)",
        ALPACA_EQUITY, ALPACA_TRADES,
    )
    alpaca["current_regime"] = alpaca_status.get("current_regime", alpaca["current_regime"])
    alpaca["gross_exposure"] = alpaca_status.get("current_gross_exposure", alpaca["gross_exposure"])

    # Load benchmarks over the overlapping period
    dates = []
    for s in (moomoo, alpaca):
        if s["first_date"] != "n/a":
            dates.append(s["first_date"])
        if s["last_date"] != "n/a":
            dates.append(s["last_date"])

    benchmarks = {}
    if dates:
        benchmarks = _load_benchmark_returns(min(dates), max(dates))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "moomoo": moomoo,
        "alpaca": alpaca,
        "benchmarks": benchmarks,
    }


def print_report(report: dict) -> None:
    """Print a formatted side-by-side comparison."""
    m = report["moomoo"]
    a = report["alpaca"]
    b = report.get("benchmarks", {})

    print(f"\n{'═'*65}")
    print(f"  PAPER TRADING PERFORMANCE REPORT")
    print(f"  {report['generated_at']}")
    print(f"{'═'*65}")

    # Header row
    print(f"\n  {'Metric':<25s} {'Moomoo':>18s} {'Alpaca':>18s}")
    print(f"  {'─'*61}")

    # Strategy info
    print(f"  {'Strategy':<25s} {'Core-Satellite':>18s} {'TQQQ-Enhanced':>18s}")
    print(f"  {'Regime':<25s} {str(m['current_regime']):>18s} {str(a['current_regime']):>18s}")

    # Data coverage
    print(f"\n  {'── Data Coverage ──'}")
    print(f"  {'Equity snapshots':<25s} {m['equity_rows']:>18d} {a['equity_rows']:>18d}")
    print(f"  {'Trading days':<25s} {m['trading_days']:>18d} {a['trading_days']:>18d}")
    print(f"  {'First date':<25s} {m['first_date']:>18s} {a['first_date']:>18s}")
    print(f"  {'Last date':<25s} {m['last_date']:>18s} {a['last_date']:>18s}")

    # Account
    print(f"\n  {'── Account ──'}")
    if m["start_equity"] > 0:
        print(f"  {'Start equity':<25s} {'${:>16,.2f}'.format(m['start_equity'])} {'${:>16,.2f}'.format(a['start_equity'])}")
    if m["current_equity"] > 0 or a["current_equity"] > 0:
        print(f"  {'Current equity':<25s} {'${:>16,.2f}'.format(m['current_equity'])} {'${:>16,.2f}'.format(a['current_equity'])}")
    print(f"  {'Gross exposure':<25s} {m['gross_exposure']:>17.1%} {a['gross_exposure']:>17.1%}")

    # Performance
    print(f"\n  {'── Performance ──'}")
    m_ret = m["total_return_pct"]
    a_ret = a["total_return_pct"]
    m_ret_str = f"{m_ret:>17.2f}%" if m["has_data"] and m["equity_rows"] >= 2 else "          n/a"
    a_ret_str = f"{a_ret:>17.2f}%" if a["has_data"] and a["equity_rows"] >= 2 else "          n/a"
    print(f"  {'Total return':<25s} {m_ret_str} {a_ret_str}")

    m_sh = m["sharpe"]
    a_sh = a["sharpe"]
    m_sh_str = f"{m_sh:>18.3f}" if m["equity_rows"] >= 3 else "               n/a"
    a_sh_str = f"{a_sh:>18.3f}" if a["equity_rows"] >= 3 else "               n/a"
    print(f"  {'Sharpe ratio':<25s} {m_sh_str} {a_sh_str}")

    m_dd = m["max_drawdown_pct"]
    a_dd = a["max_drawdown_pct"]
    m_dd_str = f"{m_dd:>17.2f}%" if m["equity_rows"] >= 2 else "          n/a"
    a_dd_str = f"{a_dd:>17.2f}%" if a["equity_rows"] >= 2 else "          n/a"
    print(f"  {'Max drawdown':<25s} {m_dd_str} {a_dd_str}")

    # Order execution
    print(f"\n  {'── Order Execution ──'}")
    print(f"  {'Total orders':<25s} {m['total_trades']:>18d} {a['total_trades']:>18d}")
    print(f"  {'Filled':<25s} {m['filled_trades']:>18d} {a['filled_trades']:>18d}")
    print(f"  {'Cancelled':<25s} {m['cancelled_trades']:>18d} {a['cancelled_trades']:>18d}")
    m_fr = f"{m['fill_rate']:>17.1%}" if m['total_trades'] > 0 else "               n/a"
    a_fr = f"{a['fill_rate']:>17.1%}" if a['total_trades'] > 0 else "               n/a"
    print(f"  {'Fill rate':<25s} {m_fr} {a_fr}")

    # Benchmarks
    if b:
        print(f"\n  {'── Benchmarks (same period) ──'}")
        for sym, ret in b.items():
            print(f"  {sym + ' return':<25s} {ret:>17.2f}%")
        if m["has_data"] and m["equity_rows"] >= 2:
            for sym, ret in b.items():
                alpha = m_ret - ret
                print(f"  {'Moomoo α vs ' + sym:<25s} {alpha:>+17.2f}%")
        if a["has_data"] and a["equity_rows"] >= 2:
            for sym, ret in b.items():
                alpha = a_ret - ret
                print(f"  {'Alpaca α vs ' + sym:<25s} {alpha:>+17.2f}%")

    # Winner
    print(f"\n  {'── Verdict ──'}")
    if m["equity_rows"] >= 2 and a["equity_rows"] >= 2:
        if m_ret > a_ret:
            print(f"  🏆 Moomoo (Core-Satellite) leads by {m_ret - a_ret:+.2f}%")
        elif a_ret > m_ret:
            print(f"  🏆 Alpaca (TQQQ-Enhanced) leads by {a_ret - m_ret:+.2f}%")
        else:
            print(f"  🤝 Tied")
    else:
        days_needed = max(0, 20 - max(m["trading_days"], a["trading_days"]))
        print(f"  ⏳ Need more data — ~{days_needed} more trading days for meaningful comparison")

    print(f"\n{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side paper trading performance report"
    )
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted report")
    args = parser.parse_args()

    report = generate_report()

    # Save report
    LOGS.mkdir(parents=True, exist_ok=True)
    out_path = LOGS / f"paper_report_{datetime.now().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
        print(f"  Report saved → {out_path}")


if __name__ == "__main__":
    main()
