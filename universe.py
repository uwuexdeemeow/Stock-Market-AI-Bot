"""
universe.py — Symbol universe loader.

PLAIN ENGLISH:
Training on only 4 tickers is like judging a restaurant after one bite.
This module loads the S&P 500 constituent list (broad universe for TRAINING
and BACKTESTING) and keeps a separate narrow list for LIVE/paper trading.

USAGE:
  from universe import sp500_universe, live_universe
  train_tickers = sp500_universe()         # ~500 names
  live_tickers = live_universe()           # 4 approved names

NOTE: S&P 500 membership drifts over time. For a truly honest historical
backtest you need point-in-time constituents (survivorship-bias-free).
This function returns the CURRENT list as a pragmatic starting point —
flag in a TODO so future-you remembers to upgrade to a historical source
(e.g., Norgate, CRSP, or Wikipedia revision scraper) before live money.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SP500_CACHE = os.path.join(DATA_DIR, "sp500_tickers.csv")
APPROVED_LIVE = os.path.join(BASE_DIR, "approved_live_tickers.csv")


@lru_cache(maxsize=1)
def sp500_universe() -> list[str]:
    """Return S&P 500 tickers. Reads cached CSV if present; otherwise scrapes Wikipedia."""
    if os.path.exists(SP500_CACHE):
        with open(SP500_CACHE) as f:
            return [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]

    # one-time scrape — Wikipedia maintains a constantly-updated table
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = [str(t).replace(".", "-") for t in df["Symbol"].tolist()]  # BRK.B -> BRK-B for yfinance
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SP500_CACHE, "w", newline="") as f:
        w = csv.writer(f)
        for t in tickers:
            w.writerow([t])
    return tickers


@lru_cache(maxsize=1)
def live_universe() -> list[str]:
    """Return the tiny approved-for-live-trading list."""
    if not os.path.exists(APPROVED_LIVE):
        return ["AMZN", "MSFT", "TSLA", "GOOGL"]
    with open(APPROVED_LIVE) as f:
        return [r[0].strip().upper() for r in csv.reader(f) if r and r[0].strip() and not r[0].startswith("#")]


if __name__ == "__main__":
    print(f"SP500: {len(sp500_universe())} tickers")
    print(f"Live:  {live_universe()}")
