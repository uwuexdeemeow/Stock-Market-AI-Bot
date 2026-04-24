from __future__ import annotations

"""
moomoo_paper_trading_complete.py
================================
Safer paper-trading wrapper.

Fixes:
- approved live universe enforced by default
- live SHORT orders disabled by default
- choose which trades to submit with include/exclude filters
- optional interactive confirmation per trade
- computes vol-aware recommended position size and borrow cost estimate
  so you can enter correct share counts directly into moomoo
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from risk_sizing import compute_position_size
from settings import BORROW_COST_ANNUAL_DEFAULT, BORROW_COSTS

SIGNAL_DIR = Path("signals")
SIGNALS_FILE = SIGNAL_DIR / "signals.csv"
APPROVED_TICKERS_FILE = SIGNAL_DIR / "approved_live_tickers.csv"

# Portfolio equity used for position sizing when no equity file exists.
# Update this to match your actual moomoo paper account balance.
DEFAULT_ACCOUNT_EQUITY = 10_000.0


def load_approved_tickers() -> set[str]:
    if not APPROVED_TICKERS_FILE.exists():
        return set()
    try:
        df = pd.read_csv(APPROVED_TICKERS_FILE)
        if "ticker" not in df.columns:
            return set()
        return {str(t).upper().strip() for t in df["ticker"].dropna().tolist()}
    except Exception:
        return set()


def _fetch_vol(ticker: str) -> float:
    """3-month realized annual vol for ticker. Returns 0.20 on failure."""
    try:
        hist = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        closes = hist["Close"].dropna()
        if len(closes) >= 20:
            vol = float(closes.pct_change().tail(63).std() * np.sqrt(252))
            return max(0.05, min(1.0, vol))
    except Exception:
        pass
    return 0.20


def _fetch_price(ticker: str) -> float:
    """Latest close price. Returns 0.0 on failure."""
    try:
        hist = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 0.0


def enrich_with_sizing(df: pd.DataFrame, equity: float) -> pd.DataFrame:
    """Add recommended_position_pct, recommended_shares, and est_borrow_cost columns."""
    rec_pcts, rec_shares, borrow_costs = [], [], []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        confidence = float(row.get("confidence", 60.0))
        expected_return = float(row.get("expected_return", 0.0))
        signal_quality = str(row.get("signal_quality", "MEDIUM")).upper()
        signal = str(row.get("signal", "LONG")).upper()
        hold_days = int(row.get("horizon_days", 5))

        asset_vol = _fetch_vol(ticker)
        price = float(row.get("price", 0.0)) or _fetch_price(ticker)

        pct = compute_position_size(confidence, expected_return, signal_quality, equity, asset_vol)
        rec_pcts.append(round(pct * 100, 2))

        shares = int((equity * pct) / price) if price > 0 else 0
        rec_shares.append(shares)

        # Estimated borrow cost shown for shorts so you know the carry cost upfront.
        if signal == "SHORT":
            annual_rate = BORROW_COSTS.get(ticker, BORROW_COST_ANNUAL_DEFAULT)
            cost = equity * pct * annual_rate * (hold_days / 365.0)
            borrow_costs.append(round(cost, 2))
        else:
            borrow_costs.append(0.0)

    df = df.copy()
    df["recommended_position_pct"] = rec_pcts
    df["recommended_shares"] = rec_shares
    df["est_borrow_cost_usd"] = borrow_costs
    return df


def filter_signals(
    include_tickers: Optional[set[str]],
    exclude_tickers: Optional[set[str]],
    enforce_approved_universe: bool,
    allow_shorts: bool,
) -> pd.DataFrame:
    if not SIGNALS_FILE.exists():
        raise FileNotFoundError(f"Missing signals file: {SIGNALS_FILE}")
    df = pd.read_csv(SIGNALS_FILE)
    if "ticker" not in df.columns:
        raise RuntimeError("signals.csv missing ticker column")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "actionable" in df.columns:
        df = df[df["actionable"].fillna(False).astype(bool)].copy()
    if include_tickers:
        include_tickers = {str(t).upper().strip() for t in include_tickers}
        df = df[df["ticker"].isin(include_tickers)].copy()
    if exclude_tickers:
        exclude_tickers = {str(t).upper().strip() for t in exclude_tickers}
        df = df[~df["ticker"].isin(exclude_tickers)].copy()
    if enforce_approved_universe:
        approved = load_approved_tickers()
        if approved:
            df = df[df["ticker"].isin(approved)].copy()
    if not allow_shorts and "signal" in df.columns:
        df = df[df["signal"].astype(str).str.upper() == "LONG"].copy()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Safer Moomoo paper-trading filter")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--exclude-tickers", nargs="*", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-approved-universe", action="store_true")
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument(
        "--equity",
        type=float,
        default=DEFAULT_ACCOUNT_EQUITY,
        help="Your moomoo paper account equity (used for position sizing)",
    )
    args = parser.parse_args()

    df = filter_signals(
        include_tickers=set(args.tickers) if args.tickers else None,
        exclude_tickers=set(args.exclude_tickers) if args.exclude_tickers else None,
        enforce_approved_universe=not args.no_approved_universe,
        allow_shorts=args.allow_shorts,
    )

    if df.empty:
        print("No live trades remain after filtering.")
        return

    print(f"Fetching vol data for {len(df)} tickers (may take a few seconds)...")
    df = enrich_with_sizing(df, equity=args.equity)

    out_path = SIGNAL_DIR / "signals_live_filtered.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved filtered live universe → {out_path}")

    # Print a readable summary table
    display_cols = ["ticker", "signal", "confidence", "expected_return",
                    "recommended_position_pct", "recommended_shares", "est_borrow_cost_usd"]
    display_cols = [c for c in display_cols if c in df.columns]
    print("\n" + df[display_cols].to_string(index=False))

    if args.interactive:
        keep_rows = []
        for _, row in df.iterrows():
            ticker = str(row.get("ticker", ""))
            signal = str(row.get("signal", ""))
            conf = row.get("confidence", None)
            shares = row.get("recommended_shares", "?")
            pct = row.get("recommended_position_pct", "?")
            print(f"\nCandidate: {ticker}  signal={signal}  confidence={conf}  → {shares} shares ({pct}% of equity)")
            choice = input(f"Submit this trade for {ticker}? [y/N]: ").strip().lower()
            if choice in {"y", "yes"}:
                keep_rows.append(row)
        df = pd.DataFrame(keep_rows)
        df.to_csv(out_path, index=False)
        print(f"Saved interactively approved trades → {out_path}")

    print("\nNext step: enter the recommended_shares from the CSV directly into moomoo.")
    print("Live shorts are disabled by default. Use --allow-shorts to enable.")


if __name__ == "__main__":
    main()
