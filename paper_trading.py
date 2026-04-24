"""
paper_trading.py — Stateful paper-trading layer for the quant pipeline.

Manual workflow
---------------
1. Run `predict.py` so `signals/signals.csv` is fresh.
2. Run `paper_trading.py --run`.
   This will:
     - queue new orders from signals
     - fill pending orders at the next available market open
     - close positions whose planned holding period has expired
     - mark open positions to market at the latest close
     - update paper_state.json / paper_trades.csv / paper_equity.csv

Notes
-----
- This script is designed for manual daily use.
- It does not place real orders.
- It uses the same portfolio risk manager as the backtester.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd
import yfinance as yf

from settings import (
    SIGNALS_FILE, SIGNAL_DIR, SECTOR_MAP,
    PAPER_INITIAL_CAPITAL, PAPER_MAX_NEW_TRADES_PER_DAY, PAPER_ALLOW_SHORTS,
    PAPER_ENTRY_SLIPPAGE_PCT, PAPER_EXIT_SLIPPAGE_PCT, PAPER_COMMISSION_PER_SHARE,
    PAPER_DEFAULT_HOLD_DAYS, PAPER_MIN_CONFIDENCE,
    PAPER_STATE_FILE, PAPER_POSITIONS_FILE, PAPER_ORDERS_FILE,
    PAPER_TRADES_FILE, PAPER_EQUITY_FILE, PAPER_DAILY_SUMMARY_FILE,
)
from portfolio_manager import PortfolioRiskManager, ProposedTrade


@dataclass
class PendingOrder:
    ticker: str
    signal: str
    signal_date: str
    submit_date: str
    entry_date: str
    planned_exit_date: str
    confidence: float
    expected_return: float
    requested_position_pct: float
    sector: str
    regime_name: str = ""
    regime_vix: float = 0.0

@dataclass
class OpenPosition:
    ticker: str
    signal: str
    entry_date: str
    planned_exit_date: str
    confidence: float
    expected_return: float
    position_pct: float
    shares: float
    entry_price: float
    entry_value: float
    sector: str
    regime_name: str = ""
    regime_vix: float = 0.0

def _ensure_dirs():
    os.makedirs(SIGNAL_DIR, exist_ok=True)

def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, )):
        return obj.isoformat()
    raise TypeError

def load_state() -> dict:
    _ensure_dirs()
    if not os.path.exists(PAPER_STATE_FILE):
        state = {
            "cash": PAPER_INITIAL_CAPITAL,
            "realized_pnl": 0.0,
            "last_mark_date": None,
        }
        save_state(state)
        return state
    with open(PAPER_STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state: dict):
    with open(PAPER_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=_json_default)

def _load_csv_or_empty(path: str, columns: List[str]) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in columns:
            if c not in df.columns:
                df[c] = None
        return df[columns]
    return pd.DataFrame(columns=columns)

def load_orders() -> pd.DataFrame:
    cols = [f.name for f in PendingOrder.__dataclass_fields__.values()]
    return _load_csv_or_empty(PAPER_ORDERS_FILE, cols)

def save_orders(df: pd.DataFrame):
    df.to_csv(PAPER_ORDERS_FILE, index=False)

def load_positions() -> pd.DataFrame:
    cols = [f.name for f in OpenPosition.__dataclass_fields__.values()]
    return _load_csv_or_empty(PAPER_POSITIONS_FILE, cols)

def save_positions(df: pd.DataFrame):
    df.to_csv(PAPER_POSITIONS_FILE, index=False)

def load_trades() -> pd.DataFrame:
    cols = [
        "date","ticker","signal","entry_date","exit_date","confidence","expected_return",
        "position_pct","shares","entry_price","exit_price","gross_pnl","commission","net_pnl",
        "sector","regime_name","regime_vix"
    ]
    return _load_csv_or_empty(PAPER_TRADES_FILE, cols)

def save_trades(df: pd.DataFrame):
    df.to_csv(PAPER_TRADES_FILE, index=False)

def load_equity() -> pd.DataFrame:
    cols = ["date","cash","market_value","equity","realized_pnl","unrealized_pnl","open_positions"]
    return _load_csv_or_empty(PAPER_EQUITY_FILE, cols)

def save_equity(df: pd.DataFrame):
    df.to_csv(PAPER_EQUITY_FILE, index=False)

def _yf_download_ohlc(ticker: str, period: str = "3mo") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _latest_market_date(ticker: str = "SPY") -> pd.Timestamp:
    df = _yf_download_ohlc(ticker, period="15d")
    if df.empty:
        raise RuntimeError("Could not determine latest market date")
    return pd.Timestamp(df.index[-1]).normalize()

def _next_trading_day(ticker: str, after_date: pd.Timestamp) -> pd.Timestamp:
    df = _yf_download_ohlc(ticker, period="3mo")
    dates = pd.DatetimeIndex(df.index).normalize().unique()
    future = dates[dates > after_date.normalize()]
    if len(future) == 0:
        return after_date.normalize()
    return pd.Timestamp(future[0]).normalize()

def _nth_future_trading_day(ticker: str, start_date: pd.Timestamp, n: int) -> pd.Timestamp:
    df = _yf_download_ohlc(ticker, period="6mo")
    dates = pd.DatetimeIndex(df.index).normalize().unique()
    future = dates[dates >= start_date.normalize()]
    if len(future) == 0:
        return start_date.normalize()
    idx = min(max(n - 1, 0), len(future) - 1)
    return pd.Timestamp(future[idx]).normalize()

def _open_close_for_date(ticker: str, date: pd.Timestamp):
    df = _yf_download_ohlc(ticker, period="6mo")
    if df.empty:
        return None, None
    df.index = pd.DatetimeIndex(df.index).normalize()
    if date.normalize() not in df.index:
        return None, None
    row = df.loc[date.normalize()]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return float(row["Open"]), float(row["Close"])

def _latest_close(ticker: str):
    df = _yf_download_ohlc(ticker, period="15d")
    if df.empty:
        return None, None
    df.index = pd.DatetimeIndex(df.index).normalize()
    row = df.iloc[-1]
    return pd.Timestamp(df.index[-1]).normalize(), float(row["Close"])

def _requested_position_pct(confidence: float, expected_return: float) -> float:
    confidence_scale = max((confidence - 50.0) / 50.0, 0.10)
    return_scale = min(max(abs(expected_return) / 4.0, 0.30), 1.00)
    pct = 0.15 * confidence_scale * return_scale * 2.0
    return float(min(max(pct, 0.15), 0.30))

def load_signals() -> pd.DataFrame:
    if not os.path.exists(SIGNALS_FILE):
        raise FileNotFoundError(f"signals file not found: {SIGNALS_FILE}")
    df = pd.read_csv(SIGNALS_FILE)
    required = {"ticker","signal","confidence","expected_return"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"signals.csv missing columns: {sorted(missing)}")
    if "predicted_at" not in df.columns:
        df["predicted_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return df

def queue_new_orders_from_signals(state: dict, orders: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    signals = load_signals().copy()
    if not PAPER_ALLOW_SHORTS:
        signals = signals[signals["signal"].str.upper() == "LONG"].copy()

    signals["confidence"] = pd.to_numeric(signals["confidence"], errors="coerce").fillna(0.0)
    signals["expected_return"] = pd.to_numeric(signals["expected_return"], errors="coerce").fillna(0.0)
    signals = signals[signals["confidence"] >= PAPER_MIN_CONFIDENCE].copy()

    if signals.empty:
        return orders

    latest_market_date = _latest_market_date()
    price_history = {}
    for ticker in signals["ticker"].astype(str).unique().tolist():
        try:
            hist = _yf_download_ohlc(ticker, period="6mo")
            if not hist.empty:
                price_history[ticker] = hist["Close"]
        except Exception:
            pass

    open_tickers = set(positions["ticker"].astype(str).tolist()) if not positions.empty else set()
    queued_tickers = set(orders["ticker"].astype(str).tolist()) if not orders.empty else set()

    candidates = []
    for _, row in signals.sort_values(["signal_quality","confidence","expected_return"], ascending=[True, False, False]).iterrows():
        ticker = str(row["ticker"]).upper()
        if ticker in open_tickers or ticker in queued_tickers:
            continue
        req_pct = _requested_position_pct(float(row["confidence"]), float(row["expected_return"]))
        candidates.append(
            ProposedTrade(
                ticker=ticker,
                date=latest_market_date,
                signal=str(row["signal"]).upper(),
                confidence=float(row["confidence"]),
                expected_return=float(row["expected_return"]),
                requested_position_pct=req_pct,
            )
        )

    # Use portfolio manager to approve the best candidates
    equity_df = load_equity()
    equity_curve = pd.Series(dtype=float)
    if not equity_df.empty:
        equity_curve = pd.Series(index=pd.to_datetime(equity_df["date"]), data=pd.to_numeric(equity_df["equity"], errors="coerce"))

    approved = PortfolioRiskManager().approve_day(candidates[:PAPER_MAX_NEW_TRADES_PER_DAY * 3], price_history, equity_curve)

    rows = []
    for tr in approved[:PAPER_MAX_NEW_TRADES_PER_DAY]:
        entry_date = _next_trading_day(tr.ticker, latest_market_date)
        exit_date = _nth_future_trading_day(tr.ticker, entry_date, PAPER_DEFAULT_HOLD_DAYS)
        rows.append(asdict(PendingOrder(
            ticker=tr.ticker,
            signal=tr.signal,
            signal_date=str(latest_market_date.date()),
            submit_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            entry_date=str(entry_date.date()),
            planned_exit_date=str(exit_date.date()),
            confidence=float(tr.confidence),
            expected_return=float(tr.expected_return),
            requested_position_pct=float(tr.requested_position_pct),
            sector=SECTOR_MAP.get(tr.ticker, "OTHER"),
        )))
    if rows:
        add_df = pd.DataFrame(rows)
        orders = pd.concat([orders, add_df], ignore_index=True)
    return orders

def fill_pending_orders(state: dict, orders: pd.DataFrame, positions: pd.DataFrame):
    if orders.empty:
        return orders, positions
    latest_market_date = _latest_market_date()
    remaining = []
    opened = []
    cash = float(state["cash"])

    for _, row in orders.iterrows():
        entry_date = pd.Timestamp(str(row["entry_date"])).normalize()
        if entry_date > latest_market_date:
            remaining.append(row.to_dict())
            continue

        open_px, _ = _open_close_for_date(str(row["ticker"]), entry_date)
        if open_px is None:
            remaining.append(row.to_dict())
            continue

        fill_px = float(open_px) * (1 + PAPER_ENTRY_SLIPPAGE_PCT if str(row["signal"]).upper() == "LONG" else 1 - PAPER_ENTRY_SLIPPAGE_PCT)
        alloc_value = cash * float(row["requested_position_pct"])
        shares = alloc_value / max(fill_px, 1e-9)
        commission = shares * PAPER_COMMISSION_PER_SHARE
        cash -= (alloc_value + commission) if str(row["signal"]).upper() == "LONG" else commission

        opened.append(asdict(OpenPosition(
            ticker=str(row["ticker"]).upper(),
            signal=str(row["signal"]).upper(),
            entry_date=str(entry_date.date()),
            planned_exit_date=str(pd.Timestamp(str(row["planned_exit_date"])).date()),
            confidence=float(row["confidence"]),
            expected_return=float(row["expected_return"]),
            position_pct=float(row["requested_position_pct"]),
            shares=float(shares),
            entry_price=float(fill_px),
            entry_value=float(alloc_value),
            sector=str(row["sector"]),
            regime_name=str(row.get("regime_name","")),
            regime_vix=float(row.get("regime_vix",0.0) or 0.0),
        )))

    state["cash"] = float(cash)
    new_orders = pd.DataFrame(remaining, columns=orders.columns) if remaining else orders.iloc[0:0].copy()
    if opened:
        positions = pd.concat([positions, pd.DataFrame(opened)], ignore_index=True)
    return new_orders, positions

def close_due_positions(state: dict, positions: pd.DataFrame, trades: pd.DataFrame):
    if positions.empty:
        return positions, trades
    latest_market_date = _latest_market_date()
    keep = []
    cash = float(state["cash"])
    realized = float(state.get("realized_pnl", 0.0))

    for _, row in positions.iterrows():
        exit_due = pd.Timestamp(str(row["planned_exit_date"])).normalize()
        if exit_due > latest_market_date:
            keep.append(row.to_dict())
            continue

        _, close_px = _open_close_for_date(str(row["ticker"]), exit_due)
        if close_px is None:
            keep.append(row.to_dict())
            continue

        exit_px = float(close_px) * (1 - PAPER_EXIT_SLIPPAGE_PCT if str(row["signal"]).upper() == "LONG" else 1 + PAPER_EXIT_SLIPPAGE_PCT)
        shares = float(row["shares"])
        entry_px = float(row["entry_price"])
        commission = shares * PAPER_COMMISSION_PER_SHARE

        if str(row["signal"]).upper() == "LONG":
            gross = shares * (exit_px - entry_px)
            proceeds = shares * exit_px
            cash += proceeds - commission
        else:
            gross = shares * (entry_px - exit_px)
            cash += float(row["entry_value"]) + gross - commission

        net = gross - commission - (shares * PAPER_COMMISSION_PER_SHARE)  # include entry + exit commission
        realized += net

        trade_row = {
            "date": str(exit_due.date()),
            "ticker": str(row["ticker"]).upper(),
            "signal": str(row["signal"]).upper(),
            "entry_date": str(row["entry_date"]),
            "exit_date": str(exit_due.date()),
            "confidence": float(row["confidence"]),
            "expected_return": float(row["expected_return"]),
            "position_pct": float(row["position_pct"]),
            "shares": float(shares),
            "entry_price": float(entry_px),
            "exit_price": float(exit_px),
            "gross_pnl": float(gross),
            "commission": float((shares * PAPER_COMMISSION_PER_SHARE) + commission),
            "net_pnl": float(net),
            "sector": str(row["sector"]),
            "regime_name": str(row.get("regime_name","")),
            "regime_vix": float(row.get("regime_vix",0.0) or 0.0),
        }
        trades = pd.concat([trades, pd.DataFrame([trade_row])], ignore_index=True)

    state["cash"] = float(cash)
    state["realized_pnl"] = float(realized)
    new_positions = pd.DataFrame(keep, columns=positions.columns) if keep else positions.iloc[0:0].copy()
    return new_positions, trades

def mark_to_market(state: dict, positions: pd.DataFrame, equity: pd.DataFrame):
    latest_date = None
    market_value = 0.0
    unrealized = 0.0

    for _, row in positions.iterrows():
        dt, close_px = _latest_close(str(row["ticker"]))
        if close_px is None:
            continue
        latest_date = max(latest_date, dt) if latest_date is not None else dt
        shares = float(row["shares"])
        entry_px = float(row["entry_price"])

        if str(row["signal"]).upper() == "LONG":
            position_value = shares * float(close_px)
            pnl = shares * (float(close_px) - entry_px)
        else:
            position_value = float(row["entry_value"]) + shares * (entry_px - float(close_px))
            pnl = shares * (entry_px - float(close_px))

        market_value += position_value
        unrealized += pnl

    if latest_date is None:
        latest_date = pd.Timestamp.now().normalize()

    row = {
        "date": str(latest_date.date()),
        "cash": float(state["cash"]),
        "market_value": float(market_value),
        "equity": float(state["cash"] + market_value),
        "realized_pnl": float(state.get("realized_pnl", 0.0)),
        "unrealized_pnl": float(unrealized),
        "open_positions": int(len(positions)),
    }

    # upsert by date
    if not equity.empty and str(latest_date.date()) in equity["date"].astype(str).tolist():
        equity.loc[equity["date"].astype(str) == str(latest_date.date()), list(row.keys())] = list(row.values())
    else:
        equity = pd.concat([equity, pd.DataFrame([row])], ignore_index=True)

    state["last_mark_date"] = str(latest_date.date())
    return equity

def run_once():
    state = load_state()
    orders = load_orders()
    positions = load_positions()
    trades = load_trades()
    equity = load_equity()

    orders = queue_new_orders_from_signals(state, orders, positions)
    orders, positions = fill_pending_orders(state, orders, positions)
    positions, trades = close_due_positions(state, positions, trades)
    equity = mark_to_market(state, positions, equity)

    save_state(state)
    save_orders(orders)
    save_positions(positions)
    save_trades(trades)
    save_equity(equity)

    summary = {
        "state": state,
        "open_positions": int(len(positions)),
        "pending_orders": int(len(orders)),
        "closed_trades": int(len(trades)),
        "latest_equity": float(equity["equity"].iloc[-1]) if not equity.empty else float(state["cash"]),
    }
    with open(PAPER_DAILY_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

def show_status():
    state = load_state()
    orders = load_orders()
    positions = load_positions()
    trades = load_trades()
    equity = load_equity()
    summary = {
        "cash": state.get("cash"),
        "realized_pnl": state.get("realized_pnl"),
        "last_mark_date": state.get("last_mark_date"),
        "pending_orders": len(orders),
        "open_positions": len(positions),
        "closed_trades": len(trades),
        "latest_equity": float(equity["equity"].iloc[-1]) if not equity.empty else float(state["cash"]),
    }
    print(json.dumps(summary, indent=2))

def reset_all():
    for path in [
        PAPER_STATE_FILE, PAPER_POSITIONS_FILE, PAPER_ORDERS_FILE,
        PAPER_TRADES_FILE, PAPER_EQUITY_FILE, PAPER_DAILY_SUMMARY_FILE
    ]:
        if os.path.exists(path):
            os.remove(path)
    load_state()
    print("Paper trading state reset.")

def main():
    parser = argparse.ArgumentParser(description="Stateful paper trading for the quant pipeline")
    parser.add_argument("--run", action="store_true", help="Queue signals, fill due orders, close due positions, mark to market")
    parser.add_argument("--status", action="store_true", help="Show current paper trading status")
    parser.add_argument("--reset", action="store_true", help="Reset all paper trading files")
    args = parser.parse_args()

    if args.reset:
        reset_all()
        return
    if args.status:
        show_status()
        return
    if args.run or (not args.status and not args.reset):
        run_once()

if __name__ == "__main__":
    main()
