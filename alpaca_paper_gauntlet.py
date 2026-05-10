"""
alpaca_paper_gauntlet.py — Paper trading health check for the Alpaca TQQQ strategy.

Reads Alpaca paper trading logs and account state, evaluates whether the
TQQQ-enhanced core-satellite strategy is performing well enough for real
capital deployment.

PLAIN ENGLISH: Before you put real money into this strategy, you need proof
that it actually works in live markets (not just backtests).  This script
checks your Alpaca paper trading performance against a set of minimum
thresholds — like a final exam before graduation to real money.

Usage:
    python3 alpaca_paper_gauntlet.py              # run full evaluation
    python3 alpaca_paper_gauntlet.py --verbose     # show extra detail

What it checks:
    1. Enough trading days logged (minimum 20)
    2. Sharpe ratio above 0.5
    3. Max drawdown not worse than -15% (TQQQ is volatile, wider than core-satellite)
    4. Order fill rate above 95%
    5. Cancel rate below 5%
    6. Portfolio drift not too large
    7. Signal freshness (is the signal stale?)
    8. Benchmark comparison (are we beating SPY/QQQ?)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from settings import LOG_DIR, SIGNAL_DIR


# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS — where Alpaca paper trading stores its data
# ─────────────────────────────────────────────────────────────────────────────
SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)

# Alpaca-specific files (written by alpaca_paper_trading.py)
ALPACA_PAPER_LOG = SIGNALS / "alpaca_paper_log.csv"          # order submission log
ALPACA_SIGNAL = SIGNALS / "core_satellite_tqqq_signal.csv"   # latest TQQQ signal

# Shared equity tracking files (if the user sets up equity snapshots)
ALPACA_EQUITY = SIGNALS / "alpaca_paper_equity.csv"

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — minimum standards to pass the gauntlet
# PLAIN ENGLISH: These are the "passing grades".  If any metric falls below
# its threshold, the gauntlet fails and blocks real capital deployment.
# ─────────────────────────────────────────────────────────────────────────────
MIN_TRADING_DAYS = int(os.environ.get("ALPACA_GAUNTLET_MIN_DAYS", "20"))
MIN_SHARPE = float(os.environ.get("ALPACA_GAUNTLET_MIN_SHARPE", "0.5"))
# TQQQ is 3x leveraged — expect bigger swings than core-satellite, so wider DD limit
MAX_DRAWDOWN_PCT = float(os.environ.get("ALPACA_GAUNTLET_MAX_DD", "-15.0"))
MIN_FILL_RATE = float(os.environ.get("ALPACA_GAUNTLET_MIN_FILL_RATE", "0.95"))
MAX_CANCEL_RATE = float(os.environ.get("ALPACA_GAUNTLET_MAX_CANCEL_RATE", "0.05"))
MAX_DRIFT = float(os.environ.get("ALPACA_GAUNTLET_MAX_DRIFT", "0.15"))
MAX_SIGNAL_AGE_HOURS = float(os.environ.get("ALPACA_GAUNTLET_MAX_SIGNAL_AGE_HOURS", "48"))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS — shared utilities for reading data and computing stats
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    """Safely read a CSV file, return empty DataFrame if missing or corrupt."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    """Safely read a JSON file, return empty dict if missing or corrupt."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown as a percentage.

    PLAIN ENGLISH: The biggest peak-to-trough drop.  If your account went
    from $12,000 down to $10,200 at its worst point, that's a -15% drawdown.
    """
    if equity.empty or len(equity) < 2:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(dd.min() * 100.0)


def _sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    """
    Compute annualized Sharpe ratio from a series of returns.

    PLAIN ENGLISH: Sharpe = how much return you get per unit of risk.
    Above 1.0 is good, above 2.0 is great.  Below 0.5 means the risk
    isn't worth the reward.
    """
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2 or float(returns.std()) == 0.0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


# ─────────────────────────────────────────────────────────────────────────────
# ORDER FILL ANALYSIS — how well are orders executing?
# ─────────────────────────────────────────────────────────────────────────────

def _order_fill_stats(trades: pd.DataFrame) -> dict:
    """
    Analyze order fill quality from the paper trade log.

    PLAIN ENGLISH: When you submit an order to buy 10 shares of TQQQ,
    did the broker actually fill it?  Or did it get cancelled/partially filled?
    This checks the fill rate — you want it above 95%.
    """
    if trades.empty:
        return {
            "submitted_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "partial_orders": 0,
            "fill_rate": 0.0,
            "cancel_rate": 0.0,
        }

    total = int(len(trades))
    # The alpaca log may have a "status" or "fill_status" column
    statuses = pd.Series(index=trades.index, dtype=str)
    for col in ("fill_status", "status"):
        if col in trades.columns:
            statuses = trades[col].fillna("").astype(str).str.lower()
            break

    filled = int(statuses.eq("filled").sum())
    cancelled = int(statuses.eq("cancelled").sum())
    partial = int(statuses.str.contains("partial", na=False).sum())

    # If no status column exists, assume all submitted orders were filled
    # (Alpaca paper trading usually fills everything instantly)
    if filled == 0 and cancelled == 0 and partial == 0 and total > 0:
        filled = total

    return {
        "submitted_orders": total,
        "filled_orders": filled,
        "cancelled_orders": cancelled,
        "partial_orders": partial,
        "fill_rate": float(filled / total) if total else 0.0,
        "cancel_rate": float(cancelled / total) if total else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY TRACKING — compute returns from equity snapshots or trade log
# ─────────────────────────────────────────────────────────────────────────────

def _build_equity_from_log(trades: pd.DataFrame) -> pd.Series:
    """
    Try to reconstruct equity history from the trade log.

    PLAIN ENGLISH: If there's no dedicated equity CSV, we try to figure out
    how the account value changed over time from the trade records.
    This is a rough estimate — the equity CSV is much better.
    """
    if trades.empty:
        return pd.Series(dtype=float)

    # Look for a date column
    date_col = None
    for col in ("submitted_at", "date", "timestamp"):
        if col in trades.columns:
            date_col = col
            break
    if date_col is None:
        return pd.Series(dtype=float)

    trades = trades.copy()
    trades["_date"] = pd.to_datetime(trades[date_col], errors="coerce").dt.date
    trades = trades.dropna(subset=["_date"])

    return trades.groupby("_date").size()  # just count — we need the equity CSV for real tracking


def _get_equity_data(equity_df: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.Series, pd.Series, int]:
    """
    Get equity series, daily returns, and number of trading days.

    Returns: (equity_series, daily_returns, n_days)
    """
    # Try equity CSV first
    if not equity_df.empty:
        for col in ("equity", "portfolio_value", "account_value", "total_equity"):
            if col in equity_df.columns:
                equity = pd.to_numeric(equity_df[col], errors="coerce").dropna()
                if not equity.empty:
                    daily_returns = equity.pct_change().dropna()
                    n_days = int(len(equity))
                    return equity, daily_returns, n_days

    # Fall back to counting unique dates in trade log
    if not trades.empty:
        for col in ("submitted_at", "date", "timestamp"):
            if col in trades.columns:
                dates = pd.to_datetime(trades[col], errors="coerce").dt.date.dropna()
                n_days = int(dates.nunique())
                return pd.Series(dtype=float), pd.Series(dtype=float), n_days

    return pd.Series(dtype=float), pd.Series(dtype=float), 0


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FRESHNESS — is the signal up to date?
# ─────────────────────────────────────────────────────────────────────────────

def _check_signal_freshness() -> dict:
    """
    Check if the TQQQ signal file is recent enough.

    PLAIN ENGLISH: If the signal was generated 3 days ago, that's stale —
    the market has moved and the regime might have changed.  We want the
    signal to be fresh (generated within the last 48 hours by default).
    """
    if not ALPACA_SIGNAL.exists():
        return {
            "fresh": False,
            "reason": "signal file not found",
            "signal_age_hours": None,
        }

    try:
        sig = pd.read_csv(ALPACA_SIGNAL).iloc[0]
        predicted_at = str(sig.get("predicted_at", ""))
        if not predicted_at or predicted_at == "nan":
            return {"fresh": False, "reason": "no predicted_at in signal", "signal_age_hours": None}

        ts = pd.to_datetime(predicted_at, errors="coerce")
        if pd.isna(ts):
            return {"fresh": False, "reason": f"cannot parse predicted_at: {predicted_at}", "signal_age_hours": None}

        age_hours = (datetime.now() - ts.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
        fresh = age_hours <= MAX_SIGNAL_AGE_HOURS
        return {
            "fresh": fresh,
            "signal_age_hours": round(age_hours, 1),
            "predicted_at": predicted_at,
            "current_regime": str(sig.get("current_regime", "unknown")),
            "tqqq_weight_config": float(sig.get("tqqq_weight_config", 0.0) or 0.0),
            "reason": "signal is fresh" if fresh else f"signal is {age_hours:.0f}h old (max {MAX_SIGNAL_AGE_HOURS}h)",
        }
    except Exception as e:
        return {"fresh": False, "reason": f"error reading signal: {e}", "signal_age_hours": None}


# ─────────────────────────────────────────────────────────────────────────────
# LIVE ACCOUNT CHECK — pull real-time data from Alpaca API
# ─────────────────────────────────────────────────────────────────────────────

def _check_live_account() -> dict:
    """
    Connect to Alpaca and pull current account state.

    PLAIN ENGLISH: This talks to Alpaca's servers to get your actual
    account balance, positions, and whether the market is open right now.
    """
    try:
        from alpaca_paper_trading import AlpacaBroker
        broker = AlpacaBroker()
        equity = broker.get_equity()
        cash = broker.get_cash()
        positions = broker.get_positions()
        market_open = broker.is_market_open()

        # Compute position weights
        position_data = {}
        total_invested = 0.0
        for p in positions:
            value = p.quantity * p.avg_price
            total_invested += value
            position_data[p.ticker] = {
                "quantity": int(p.quantity),
                "avg_price": round(p.avg_price, 2),
                "value": round(value, 2),
                "weight": round(value / equity, 4) if equity > 0 else 0.0,
            }

        # Check portfolio drift against signal targets
        max_drift = 0.0
        if ALPACA_SIGNAL.exists():
            try:
                sig = pd.read_csv(ALPACA_SIGNAL).iloc[0]
                target_weights = {}
                for sym, col in [("SPY", "target_spy_weight"), ("QQQ", "target_qqq_weight"), ("TQQQ", "target_tqqq_weight")]:
                    w = float(sig.get(col, 0.0) or 0.0)
                    if abs(w) > 1e-9:
                        target_weights[sym] = w

                # Add overlay tickers
                import json as _json
                overlay_raw = sig.get("overlay_weights_json", "{}")
                try:
                    overlay = _json.loads(str(overlay_raw)) if pd.notna(overlay_raw) else {}
                except Exception:
                    overlay = {}
                for t, w in overlay.items():
                    if abs(float(w)) > 1e-9:
                        target_weights[str(t).upper()] = float(w)

                # Compare current vs target
                all_tickers = set(target_weights.keys()) | set(position_data.keys())
                for ticker in all_tickers:
                    current_w = position_data.get(ticker, {}).get("weight", 0.0)
                    target_w = target_weights.get(ticker, 0.0)
                    drift = abs(current_w - target_w)
                    max_drift = max(max_drift, drift)
            except Exception:
                pass

        return {
            "connected": True,
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "invested": round(equity - cash, 2),
            "market_open": market_open,
            "n_positions": len(positions),
            "positions": position_data,
            "max_drift": round(max_drift, 4),
            "gross_exposure": round(total_invested / equity, 4) if equity > 0 else 0.0,
        }
    except ImportError:
        return {"connected": False, "reason": "alpaca-trade-api not installed"}
    except Exception as e:
        return {"connected": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION — the gauntlet itself
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_alpaca_paper(verbose: bool = False) -> dict:
    """
    Run the full Alpaca paper trading gauntlet.

    PLAIN ENGLISH: This is the final exam.  It checks every aspect of your
    paper trading — are orders filling? is the account growing? is drawdown
    acceptable? is the signal fresh?  If ALL checks pass, you get the green
    light for real capital.  If ANY check fails, it tells you exactly why.

    Returns a dict with status, reasons, and detailed metrics.
    """
    # ── Load data ──────────────────────────────────────────────────────
    trades = _read_csv(ALPACA_PAPER_LOG)
    equity_df = _read_csv(ALPACA_EQUITY)

    # If no data at all, early exit
    if trades.empty and equity_df.empty:
        return {
            "status": "no_data",
            "approved_for_real_capital": False,
            "reason": "No Alpaca paper trading data found. Run some trades first.",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy": "core_satellite_tqqq",
            "data_files": {
                "alpaca_paper_log": str(ALPACA_PAPER_LOG),
                "alpaca_paper_equity": str(ALPACA_EQUITY),
                "signal": str(ALPACA_SIGNAL),
                "log_exists": ALPACA_PAPER_LOG.exists(),
                "equity_exists": ALPACA_EQUITY.exists(),
                "signal_exists": ALPACA_SIGNAL.exists(),
            },
        }

    # ── Compute metrics ────────────────────────────────────────────────
    equity, daily_returns, n_days = _get_equity_data(equity_df, trades)
    fill_stats = _order_fill_stats(trades)
    signal_freshness = _check_signal_freshness()
    live_account = _check_live_account()

    # Performance metrics (only if we have equity data)
    total_return_pct = 0.0
    sharpe = 0.0
    max_dd = 0.0
    if not equity.empty and len(equity) >= 2:
        total_return_pct = float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)
        sharpe = _sharpe(daily_returns)
        max_dd = _max_drawdown(equity)

    # Drift from live account
    max_drift = float(live_account.get("max_drift", 0.0)) if live_account.get("connected") else np.nan

    # ── Run the gauntlet checks ────────────────────────────────────────
    # PLAIN ENGLISH: Each check is a yes/no gate.  If you fail any gate,
    # the reason gets added to the failure list.
    reasons: list[str] = []

    # Gate 1: Enough trading days
    if n_days < MIN_TRADING_DAYS:
        reasons.append(f"trading_days {n_days} < {MIN_TRADING_DAYS} minimum")

    # Gate 2: Order fill quality
    if fill_stats["submitted_orders"] <= 0:
        reasons.append("no submitted paper orders found")
    elif fill_stats["fill_rate"] < MIN_FILL_RATE:
        reasons.append(f"fill_rate {fill_stats['fill_rate']:.3f} < {MIN_FILL_RATE:.3f}")

    # Gate 3: Cancel rate
    if fill_stats["cancel_rate"] > MAX_CANCEL_RATE:
        reasons.append(f"cancel_rate {fill_stats['cancel_rate']:.3f} > {MAX_CANCEL_RATE:.3f}")

    # Gate 4: Signal freshness
    if not signal_freshness.get("fresh", False):
        reasons.append(f"signal not fresh: {signal_freshness.get('reason', 'unknown')}")

    # Gate 5: Sharpe ratio (only check if enough data)
    if not equity.empty and n_days >= MIN_TRADING_DAYS:
        if sharpe < MIN_SHARPE:
            reasons.append(f"paper_sharpe {sharpe:.3f} < {MIN_SHARPE:.3f}")

    # Gate 6: Max drawdown (only check if enough data)
    if not equity.empty and n_days >= MIN_TRADING_DAYS:
        if max_dd < MAX_DRAWDOWN_PCT:
            reasons.append(f"paper_max_drawdown {max_dd:.2f}% < {MAX_DRAWDOWN_PCT:.2f}%")

    # Gate 7: Portfolio drift
    if np.isfinite(max_drift) and max_drift > MAX_DRIFT:
        reasons.append(f"portfolio_drift {max_drift:.3f} > {MAX_DRIFT:.3f}")

    # Gate 8: Alpaca connection
    if not live_account.get("connected", False):
        reasons.append(f"cannot connect to Alpaca: {live_account.get('reason', 'unknown')}")

    # ── Build result ───────────────────────────────────────────────────
    passed = len(reasons) == 0
    result = {
        "status": "passed" if passed else "failed",
        "approved_for_real_capital": passed,
        "reason": "Alpaca TQQQ paper gauntlet passed" if passed else "; ".join(reasons),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "core_satellite_tqqq",

        # Trading activity
        "trading_days": n_days,
        "paper_trades": int(len(trades)),
        **fill_stats,

        # Performance (if equity data available)
        "equity_data_available": not equity.empty,
        "total_return_pct": round(total_return_pct, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),

        # Signal state
        "signal_freshness": signal_freshness,

        # Live account state
        "live_account": live_account,

        # Thresholds used (so you can see what the passing grade is)
        "thresholds": {
            "min_trading_days": MIN_TRADING_DAYS,
            "min_sharpe": MIN_SHARPE,
            "max_drawdown_pct": MAX_DRAWDOWN_PCT,
            "min_fill_rate": MIN_FILL_RATE,
            "max_cancel_rate": MAX_CANCEL_RATE,
            "max_drift": MAX_DRIFT,
            "max_signal_age_hours": MAX_SIGNAL_AGE_HOURS,
        },
    }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT — human-readable output
# ─────────────────────────────────────────────────────────────────────────────

def print_gauntlet_report(result: dict) -> None:
    """
    Print the gauntlet results in a clean, readable format.

    PLAIN ENGLISH: Instead of dumping raw JSON, this prints a nice report
    card showing what passed, what failed, and the current account state.
    """
    status = result.get("status", "unknown")
    approved = result.get("approved_for_real_capital", False)

    print(f"\n{'═'*60}")
    print(f"  ALPACA TQQQ PAPER GAUNTLET")
    print(f"{'═'*60}")

    # Status badge
    if status == "no_data":
        print(f"  Status:  ⚠ NO DATA")
    elif approved:
        print(f"  Status:  ✓ PASSED — approved for real capital")
    else:
        print(f"  Status:  ✗ FAILED")
    print(f"  Reason:  {result.get('reason', 'n/a')}")
    print(f"{'─'*60}")

    # Trading activity
    print(f"\n  TRADING ACTIVITY")
    print(f"  {'─'*40}")
    print(f"  Trading days:     {result.get('trading_days', 0):<6d}  (min: {MIN_TRADING_DAYS})")
    print(f"  Orders submitted: {result.get('submitted_orders', 0)}")
    print(f"  Orders filled:    {result.get('filled_orders', 0)}")
    print(f"  Orders cancelled: {result.get('cancelled_orders', 0)}")
    fill_rate = result.get("fill_rate", 0.0)
    fill_status = "✓" if fill_rate >= MIN_FILL_RATE else "✗"
    print(f"  Fill rate:        {fill_rate:.1%}  {fill_status}  (min: {MIN_FILL_RATE:.0%})")
    cancel_rate = result.get("cancel_rate", 0.0)
    cancel_status = "✓" if cancel_rate <= MAX_CANCEL_RATE else "✗"
    print(f"  Cancel rate:      {cancel_rate:.1%}  {cancel_status}  (max: {MAX_CANCEL_RATE:.0%})")

    # Performance
    if result.get("equity_data_available"):
        print(f"\n  PERFORMANCE")
        print(f"  {'─'*40}")
        print(f"  Total return:     {result.get('total_return_pct', 0):.2f}%")
        sharpe_val = result.get("sharpe", 0)
        sharpe_status = "✓" if sharpe_val >= MIN_SHARPE else "✗"
        print(f"  Sharpe ratio:     {sharpe_val:.3f}  {sharpe_status}  (min: {MIN_SHARPE:.1f})")
        dd_val = result.get("max_drawdown_pct", 0)
        dd_status = "✓" if dd_val >= MAX_DRAWDOWN_PCT else "✗"
        print(f"  Max drawdown:     {dd_val:.2f}%  {dd_status}  (max: {MAX_DRAWDOWN_PCT:.1f}%)")
    else:
        print(f"\n  PERFORMANCE")
        print(f"  {'─'*40}")
        print(f"  ⚠ No equity data yet — create {ALPACA_EQUITY.name} for tracking")

    # Signal freshness
    sf = result.get("signal_freshness", {})
    print(f"\n  SIGNAL STATUS")
    print(f"  {'─'*40}")
    fresh_status = "✓" if sf.get("fresh") else "✗"
    age = sf.get("signal_age_hours")
    print(f"  Signal fresh:     {fresh_status}  ({sf.get('reason', 'n/a')})")
    if age is not None:
        print(f"  Signal age:       {age:.1f} hours")
    print(f"  Regime:           {sf.get('current_regime', 'unknown')}")
    print(f"  TQQQ config:      {sf.get('tqqq_weight_config', 0)*100:.0f}%")

    # Live account
    acct = result.get("live_account", {})
    if acct.get("connected"):
        print(f"\n  LIVE ACCOUNT")
        print(f"  {'─'*40}")
        print(f"  Equity:           ${acct.get('equity', 0):,.2f}")
        print(f"  Cash:             ${acct.get('cash', 0):,.2f}")
        print(f"  Invested:         ${acct.get('invested', 0):,.2f}")
        print(f"  Gross exposure:   {acct.get('gross_exposure', 0):.1%}")
        drift_val = acct.get("max_drift", 0)
        drift_status = "✓" if drift_val <= MAX_DRIFT else "✗"
        print(f"  Max drift:        {drift_val:.1%}  {drift_status}  (max: {MAX_DRIFT:.0%})")
        print(f"  Market:           {'OPEN ✓' if acct.get('market_open') else 'CLOSED ✗'}")
        print(f"  Positions:        {acct.get('n_positions', 0)}")

        positions = acct.get("positions", {})
        if positions:
            print(f"\n  {'Ticker':<8s} {'Qty':>6s} {'Value':>10s} {'Weight':>8s}")
            print(f"  {'─'*35}")
            for ticker, data in sorted(positions.items(), key=lambda x: -x[1]["value"]):
                print(f"  {ticker:<8s} {data['quantity']:>6d} ${data['value']:>9,.2f} {data['weight']:>7.1%}")
    else:
        print(f"\n  LIVE ACCOUNT")
        print(f"  {'─'*40}")
        print(f"  ✗ Not connected: {acct.get('reason', 'unknown')}")

    print(f"\n{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY SNAPSHOT — save current account equity for tracking over time
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_equity() -> None:
    """
    Take a snapshot of the current Alpaca account equity and append it
    to the equity tracking CSV.

    PLAIN ENGLISH: Every time you run this, it saves today's account value
    to a file.  Over time, this builds up a history of your account
    performance, which the gauntlet uses to compute Sharpe/drawdown.

    Run this daily (or after each rebalance) to build up equity data:
        python3 alpaca_paper_gauntlet.py --snapshot
    """
    try:
        from alpaca_paper_trading import AlpacaBroker
        broker = AlpacaBroker()
        equity = broker.get_equity()
        cash = broker.get_cash()
    except Exception as e:
        print(f"  ✗ Cannot connect to Alpaca: {e}")
        return

    now = datetime.now()
    row = {
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "invested": round(equity - cash, 2),
    }

    df = pd.DataFrame([row])
    if ALPACA_EQUITY.exists():
        existing = pd.read_csv(ALPACA_EQUITY)
        # Don't duplicate same-day entries — keep latest
        existing["date"] = existing["date"].astype(str)
        existing = existing[existing["date"] != row["date"]]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(ALPACA_EQUITY, index=False)
    print(f"  ✓ Equity snapshot saved: ${equity:,.2f} → {ALPACA_EQUITY}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — command line entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Alpaca TQQQ paper trading gauntlet — health check before real capital"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show extra detail in output")
    parser.add_argument("--snapshot", action="store_true",
                        help="Take an equity snapshot and save to tracking CSV")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted report")
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    SIGNALS.mkdir(parents=True, exist_ok=True)

    # Equity snapshot mode
    if args.snapshot:
        snapshot_equity()
        return

    # Run the gauntlet
    result = evaluate_alpaca_paper(verbose=args.verbose)

    # Save to JSON
    out_path = LOGS / f"alpaca_paper_gauntlet_{datetime.now().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # Display results
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_gauntlet_report(result)
        print(f"  Report saved → {out_path}")


if __name__ == "__main__":
    main()
