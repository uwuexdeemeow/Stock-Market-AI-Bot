"""
universe.py — Symbol universe loader.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SIGNAL_DIR = os.path.join(BASE_DIR, "signals")
SP500_CACHE = os.path.join(DATA_DIR, "sp500_tickers.csv")
APPROVED_LIVE = os.path.join(SIGNAL_DIR, "approved_live_tickers.csv")


@lru_cache(maxsize=1)
def sp500_universe() -> list[str]:
    """Return S&P 500 tickers. Reads cached CSV if present; otherwise scrapes Wikipedia.

    This is current-membership data, so it is not survivorship-bias-free for
    historical backtests. Use a point-in-time source before trusting broad OOS
    claims.
    """
    if os.path.exists(SP500_CACHE):
        with open(SP500_CACHE) as f:
            tickers = [r[0].strip().upper() for r in csv.reader(f) if r and r[0].strip()]
        if tickers:
            return tickers

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = [str(t).replace(".", "-").upper() for t in df["Symbol"].tolist()]
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SP500_CACHE, "w", newline="") as f:
            w = csv.writer(f)
            for t in tickers:
                w.writerow([t])
        return tickers
    except Exception:
        if os.path.exists(SP500_CACHE):
            with open(SP500_CACHE) as f:
                return [r[0].strip().upper() for r in csv.reader(f) if r and r[0].strip()]
        return DEFAULT_LIVE[:]


@lru_cache(maxsize=1)
def live_universe() -> list[str]:
    """Return only tickers approved by the objective model-quality gate."""
    if not os.path.exists(APPROVED_LIVE):
        return []
    with open(APPROVED_LIVE) as f:
        tickers = [
            r[0].strip().upper()
            for r in csv.reader(f)
            if r and r[0].strip() and not r[0].startswith("#") and r[0].strip().lower() != "ticker"
        ]
    return tickers


if __name__ == "__main__":
    print(f"SP500: {len(sp500_universe())} tickers")
    print(f"Live:  {live_universe()}")
