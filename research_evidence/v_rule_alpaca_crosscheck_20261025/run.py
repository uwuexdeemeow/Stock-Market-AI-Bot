"""
run.py — independent re-build of rule V (QQQ + BIL volatility target) from
Alpaca CLI data.  It does NOT import any project code, on purpose: if this
script and the project's own test (research_evidence/index_risk_20261016)
agree, the project code is very likely doing what the rule says.

PLAIN ENGLISH: on the last trading day of each month we look at how jumpy
QQQ was over its last 63 daily returns (its "volatility", as a yearly
number).  QQQ share = 20% / volatility, never above 100%.  The rest goes in
BIL (a T-bill fund, basically cash).  We trade at the NEXT trading day's
opening price and pay a small cost.  Between trades we just hold.

Inputs (made by fetch.sh):  raw/bars_QQQ_BIL_all_p*.json, raw/calendar.json
Outputs: equity.csv, benchmark_equity.csv, trades.csv, round_trips.csv,
         signals.csv, summary.json, report.md, data_fingerprint.json,
         fee_source.json, warnings.json, normalized/bars_*.csv

How to run (from this folder):  python run.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ── Confirmed settings (see notes.md) ──────────────────────────────────────
CONFIG = {
    "symbols": ["QQQ", "BIL"],
    "timeframe": "1Day",
    "feed": "sip",
    "adjustment": "all",          # splits AND dividends folded into prices
    "initial_cash": 100_000.0,
    "vol_sessions": 63,           # look-back: last 63 daily returns
    "vol_target": 0.20,           # 20% a year
    "trading_days": 252,
    "fill_model": "next_open",    # signal at month-end close, fill next open
    "slippage_bps": 5.0,          # per side, on every fill
    "fractional_shares": True,
    "sec_fee_rate": 0.0000206,    # x sell trade value (Alpaca fee PDF, rev. 2026-10-12)
    "cat_fee_per_share": 0.000003,  # buys and sells
    "taf_fee_per_share": 0.0,     # PDF shows $0 per share for equities
}


# ── Load data ──────────────────────────────────────────────────────────────

def load_bars() -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Read every raw page, return one daily table per symbol (New York dates)."""
    rows: dict[str, list[dict]] = {s: [] for s in CONFIG["symbols"]}
    pages = sorted(glob.glob(str(HERE / "raw" / "bars_QQQ_BIL_all_p*.json")))
    for page in pages:
        data = json.loads(Path(page).read_text())
        for symbol, bars in data["bars"].items():
            rows[symbol].extend(bars)
    frames = {}
    for symbol, bars in rows.items():
        df = pd.DataFrame(bars)
        # Alpaca stamps daily bars at New York midnight in UTC; convert back to a date.
        df["date"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        df = df.drop_duplicates("date").sort_values("date").set_index("date")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        frames[symbol] = df[["open", "high", "low", "close", "volume"]]
    return frames, pages


def load_calendar() -> pd.DatetimeIndex:
    cal = json.loads((HERE / "raw" / "calendar.json").read_text())
    return pd.DatetimeIndex(sorted(pd.to_datetime([d["date"] for d in cal])))


# ── Signal ─────────────────────────────────────────────────────────────────

def month_end_sessions(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last NYSE session of each month, from the exchange calendar."""
    s = sessions.to_series()
    return pd.DatetimeIndex(s.groupby([sessions.year, sessions.month]).max().values)


def qqq_target(closes_up_to_signal: pd.Series) -> tuple[float, float] | None:
    """(volatility, QQQ weight) from closes up to and including the signal day.

    Uses only completed closes, so no future data can leak in.
    Returns None while fewer than 63 returns exist (warm-up).
    """
    n = CONFIG["vol_sessions"]
    returns = closes_up_to_signal.pct_change().dropna().iloc[-n:]
    if len(returns) < n:
        return None
    vol = float(returns.std(ddof=1)) * math.sqrt(CONFIG["trading_days"])
    weight = min(1.0, CONFIG["vol_target"] / vol) if vol > 0 else 1.0
    return vol, weight


# ── Simulation ─────────────────────────────────────────────────────────────

class Book:
    """Cash + shares, with FIFO lots so we can report round trips."""

    def __init__(self, cash: float):
        self.cash = cash
        self.shares = {s: 0.0 for s in CONFIG["symbols"]}
        self.lots = {s: deque() for s in CONFIG["symbols"]}   # [qty, cost per share]
        self.trades: list[dict] = []
        self.round_trips: list[dict] = []
        self.fees_paid = 0.0
        self.slippage_paid = 0.0

    def value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(self.shares[s] * prices[s] for s in self.shares)

    def fill(self, day, symbol: str, side: str, qty: float, bar_open: float, signal_day, reason: str):
        if qty <= 1e-9:
            return
        slip = CONFIG["slippage_bps"] / 10_000
        if side == "buy":
            fill_price = bar_open * (1 + slip)
        else:
            fill_price = bar_open * (1 - slip)
        notional = qty * fill_price
        fees = qty * CONFIG["cat_fee_per_share"]
        if side == "sell":
            fees += notional * CONFIG["sec_fee_rate"] + qty * CONFIG["taf_fee_per_share"]
        self.fees_paid += fees
        self.slippage_paid += qty * bar_open * slip
        if side == "buy":
            self.cash -= notional + fees
            self.shares[symbol] += qty
            # Fees on a buy are folded into the lot's cost.
            self.lots[symbol].append([qty, (notional + fees) / qty, day])
        else:
            self.cash += notional - fees
            self.shares[symbol] -= qty
            remaining = qty
            proceeds_per_share = (notional - fees) / qty
            while remaining > 1e-9 and self.lots[symbol]:
                lot = self.lots[symbol][0]
                take = min(lot[0], remaining)
                self.round_trips.append({
                    "symbol": symbol, "entry_date": lot[2].date(), "exit_date": day.date(),
                    "qty": take, "entry_cost": lot[1], "exit_proceeds": proceeds_per_share,
                    "pnl": take * (proceeds_per_share - lot[1]),
                })
                lot[0] -= take
                remaining -= take
                if lot[0] <= 1e-9:
                    self.lots[symbol].popleft()
        self.trades.append({
            "date": day.date(), "signal_day": signal_day.date(), "symbol": symbol, "side": side,
            "qty": qty, "bar_open": bar_open, "fill_price": fill_price,
            "notional": notional, "fees": fees, "reason": reason,
        })


def rebalance(book: Book, day, opens: dict[str, float], qqq_weight: float, signal_day, reason: str):
    """Move to the target split at today's open.  Sell first, then buy."""
    equity = book.value(opens)
    target = {"QQQ": qqq_weight, "BIL": 1.0 - qqq_weight}
    # Leave a hair of cash for the buy-side slippage and fees so cash never goes negative.
    buffer = 1 - (CONFIG["slippage_bps"] / 10_000) - 0.0001
    target_shares = {s: equity * target[s] * buffer / opens[s] for s in target}
    for s in target:
        diff = target_shares[s] - book.shares[s]
        if diff < 0:
            book.fill(day, s, "sell", -diff, opens[s], signal_day, reason)
    for s in target:
        diff = target_shares[s] - book.shares[s]
        if diff > 0:
            # Never spend more cash than we have.
            affordable = book.cash / (opens[s] * (1 + CONFIG["slippage_bps"] / 10_000) + CONFIG["cat_fee_per_share"])
            book.fill(day, s, "buy", min(diff, affordable), opens[s], signal_day, reason)


def run_v(bars, sessions, warnings):
    qqq, bil = bars["QQQ"], bars["BIL"]
    days = qqq.index.intersection(bil.index)
    missing = sessions[(sessions >= days[0]) & (sessions <= days[-1])].difference(days)
    if len(missing):
        warnings.append(f"{len(missing)} calendar sessions have no bar: {[str(d.date()) for d in missing[:10]]}")
    signal_days = [d for d in month_end_sessions(sessions) if days[0] <= d <= days[-1]]

    # Work out each month-end target, and the session it trades on.
    signals = []
    for sd in signal_days:
        t = qqq_target(qqq["close"].loc[:sd])
        if t is None:
            continue
        pos = days.searchsorted(sd, side="right")
        trade_day = days[pos] if pos < len(days) else None
        signals.append({"signal_day": sd, "trade_day": trade_day, "volatility": t[0], "qqq_weight": t[1]})
    sig = pd.DataFrame(signals)
    trade_on = {r.trade_day: r for r in sig.itertuples() if r.trade_day is not None}

    first_trade_day = sig["trade_day"].dropna().iloc[0]
    book = Book(CONFIG["initial_cash"])
    curve = []
    for day in days[days >= first_trade_day]:
        opens = {s: float(bars[s].at[day, "open"]) for s in CONFIG["symbols"]}
        closes = {s: float(bars[s].at[day, "close"]) for s in CONFIG["symbols"]}
        if day in trade_on:
            r = trade_on[day]
            rebalance(book, day, opens, r.qqq_weight, r.signal_day, "month-end target")
        eq = book.value(closes)
        curve.append({"date": day, "equity": eq, "cash": book.cash,
                      "qqq_value": book.shares["QQQ"] * closes["QQQ"],
                      "bil_value": book.shares["BIL"] * closes["BIL"],
                      "qqq_exposure": book.shares["QQQ"] * closes["QQQ"] / eq})
    return pd.DataFrame(curve).set_index("date"), book, sig


def run_buy_hold(bars, start_day, weights: dict[str, float]) -> pd.Series:
    """Buy once at start_day's open (same slippage + fees), never trade again."""
    book = Book(CONFIG["initial_cash"])
    days = bars["QQQ"].index.intersection(bars["BIL"].index)
    days = days[days >= start_day]
    opens = {s: float(bars[s].at[start_day, "open"]) for s in CONFIG["symbols"]}
    eq0 = book.value(opens)
    buffer = 1 - (CONFIG["slippage_bps"] / 10_000) - 0.0001
    for s, w in weights.items():
        if w > 0:
            book.fill(start_day, s, "buy", eq0 * w * buffer / opens[s], opens[s], start_day, "buy and hold")
    return pd.Series([book.value({s: float(bars[s].at[d, "close"]) for s in CONFIG["symbols"]}) for d in days],
                     index=days)


# ── Metrics ────────────────────────────────────────────────────────────────

def metrics(equity: pd.Series, cash_returns: pd.Series | None = None) -> dict:
    daily = equity.pct_change().dropna()
    total = equity.iloc[-1] / CONFIG["initial_cash"] - 1
    n = len(equity)
    peak = equity.cummax()
    out = {
        "total_return_pct": 100 * total,
        "ann_return_pct": 100 * ((1 + total) ** (CONFIG["trading_days"] / n) - 1),
        "max_drawdown_pct": 100 * float((equity / peak - 1).min()),
        "sharpe_rf0": float(daily.mean() / daily.std(ddof=1) * math.sqrt(CONFIG["trading_days"])),
        "final_equity": float(equity.iloc[-1]),
        "trading_days": n,
    }
    if cash_returns is not None:
        excess = daily - cash_returns.reindex(daily.index).fillna(0.0)
        out["sharpe_over_bil"] = float(excess.mean() / excess.std(ddof=1) * math.sqrt(CONFIG["trading_days"]))
    return out


def yearly(equity: pd.Series) -> dict:
    ye = equity.groupby(equity.index.year).last()
    prev = ye.shift(1)
    prev.iloc[0] = CONFIG["initial_cash"]
    return {int(y): round(100 * (ye[y] / prev[y] - 1), 2) for y in ye.index}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    warnings: list[str] = []
    bars, pages = load_bars()
    sessions = load_calendar()

    fingerprint = {}
    for s, df in bars.items():
        norm = HERE / "normalized" / f"bars_{s}.csv"
        df.to_csv(norm, float_format="%.6f")
        fingerprint[s] = {
            "provider": "alpaca", "access_method": "alpaca_cli 0.0.14", "feed": CONFIG["feed"],
            "adjustment": CONFIG["adjustment"], "timeframe": "1Day", "extended_hours": False,
            "total_bars_fetched": int(len(df)), "bars_after_filter": int(len(df)),
            "first_bar_ts": str(df.index[0].date()), "last_bar_ts": str(df.index[-1].date()),
            "close_sum": round(float(df["close"].sum()), 4), "volume_sum": int(df["volume"].sum()),
            "calendar_filter": "NYSE sessions from alpaca calendar; daily bars are regular-hours aggregates",
            "raw_file_hash": {Path(p).name: sha256(Path(p)) for p in pages},
            "normalized_file_hash": sha256(norm),
        }
    fingerprint["calendar"] = {"raw_file_hash": sha256(HERE / "raw" / "calendar.json")}

    equity, book, sig = run_v(bars, sessions, warnings)
    start = equity.index[0]
    bil_ret = bars["BIL"]["close"].pct_change()
    bench = pd.DataFrame({
        "QQQ_buy_hold": run_buy_hold(bars, start, {"QQQ": 1.0, "BIL": 0.0}),
        "BIL_buy_hold": run_buy_hold(bars, start, {"QQQ": 0.0, "BIL": 1.0}),
        "equal_weight_no_rebalance": run_buy_hold(bars, start, {"QQQ": 0.5, "BIL": 0.5}),
    })

    m = {"V": metrics(equity["equity"], bil_ret)}
    for col in bench:
        m[col] = metrics(bench[col], bil_ret)

    rt = pd.DataFrame(book.round_trips)
    wins = rt[rt["pnl"] > 0]
    losses = rt[rt["pnl"] <= 0]
    m["V"].update({
        "fills": len(book.trades),
        "rebalance_days": int(len({t["date"] for t in book.trades})),
        "round_trips": len(rt),
        "win_rate_pct": 100 * len(wins) / len(rt) if len(rt) else None,
        "profit_factor": (float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum() != 0 else math.inf),
        "fees_paid": book.fees_paid,
        "slippage_paid": book.slippage_paid,
        "turnover": float(sum(abs(t["notional"]) for t in book.trades) / equity["equity"].mean()),
        "average_qqq_exposure": float(equity["qqq_exposure"].mean()),
    })

    # Write artifacts
    equity.to_csv(HERE / "equity.csv", float_format="%.4f")
    bench.to_csv(HERE / "benchmark_equity.csv", float_format="%.4f")
    pd.DataFrame(book.trades).to_csv(HERE / "trades.csv", index=False, float_format="%.6f")
    rt.to_csv(HERE / "round_trips.csv", index=False, float_format="%.6f")
    sig.to_csv(HERE / "signals.csv", index=False, float_format="%.6f")
    (HERE / "data_fingerprint.json").write_text(json.dumps(fingerprint, indent=2))
    fee_source = {
        "url": "https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf",
        "revision_date": "2026-10-12", "extracted_at": "2026-10-25",
        "modeled_categories": ["SEC Transaction Fee ($0.0000206 x sell value)",
                               "FINRA CAT ($0.000003 per share, buys and sells)",
                               "FINRA TAF (read as $0 per share for equities)"],
        "excluded_categories": ["commissions (none for this account type)", "options fees", "borrow/margin",
                                "transfer and account fees"],
    }
    (HERE / "fee_source.json").write_text(json.dumps(fee_source, indent=2))
    (HERE / "warnings.json").write_text(json.dumps(warnings, indent=2))

    trades = book.trades
    summary = {
        "strategy_name": "rule V: QQQ vol target 20% / 63 sessions, rest BIL, month-end signal",
        "start": str(start.date()), "end": str(equity.index[-1].date()),
        "symbols": CONFIG["symbols"], "timeframe": "1Day", "initial_cash": CONFIG["initial_cash"],
        "config": CONFIG, "metrics": m["V"],
        "benchmarks": {k: v for k, v in m.items() if k != "V"},
        "yearly_returns_pct": {"V": yearly(equity["equity"]), "QQQ_buy_hold": yearly(bench["QQQ_buy_hold"])},
        "first_trade": {k: str(v) for k, v in trades[0].items()},
        "last_trade": {k: str(v) for k, v in trades[-1].items()},
        "warnings": warnings, "data_fingerprint": fingerprint, "fee_source": fee_source,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"V": m["V"], **{k: v for k, v in m.items() if k != "V"}}, indent=1, default=str))


if __name__ == "__main__":
    main()
