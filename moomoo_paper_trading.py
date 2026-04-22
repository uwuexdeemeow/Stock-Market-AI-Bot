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
- keeps future auto-close / TP / SL as an improvement for later
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

SIGNAL_DIR = Path("signals")
SIGNALS_FILE = SIGNAL_DIR / "signals.csv"
APPROVED_TICKERS_FILE = SIGNAL_DIR / "approved_live_tickers.csv"


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


def filter_signals(include_tickers: Optional[set[str]], exclude_tickers: Optional[set[str]], enforce_approved_universe: bool, allow_shorts: bool) -> pd.DataFrame:
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
    args = parser.parse_args()

    df = filter_signals(
        include_tickers=set(args.tickers) if args.tickers else None,
        exclude_tickers=set(args.exclude_tickers) if args.exclude_tickers else None,
        enforce_approved_universe=not args.no_approved_universe,
        allow_shorts=args.allow_shorts,
    )

    out_path = SIGNAL_DIR / "signals_live_filtered.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved filtered live universe → {out_path}")

    if df.empty:
        print("No live trades remain after filtering.")
        return

    if args.interactive:
        keep_rows = []
        for _, row in df.iterrows():
            ticker = str(row.get("ticker", ""))
            signal = str(row.get("signal", ""))
            conf = row.get("confidence", None)
            print(f"Candidate: {ticker} signal={signal} confidence={conf}")
            choice = input(f"Submit this trade for {ticker}? [y/N]: ").strip().lower()
            if choice in {"y", "yes"}:
                keep_rows.append(row)
        df = pd.DataFrame(keep_rows)
        df.to_csv(out_path, index=False)
        print(f"Saved interactively approved trades → {out_path}")

    print("Next step: point your existing moomoo script at signals_live_filtered.csv or replace signals.csv with it.")
    print("Live shorts are disabled by default in this wrapper.")


if __name__ == "__main__":
    main()
