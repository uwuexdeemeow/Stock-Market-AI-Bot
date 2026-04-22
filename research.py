
"""
research.py — Build training parquet files using the shared production feature pipeline.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from settings import DATA_DIR, LOG_DIR, SHORTLIST_FILE, TRAIN_START, TRAIN_END
from pipeline_shared import build_research_feature_frame

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "research.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("research")

def research_ticker(ticker: str, start: str, end: str) -> bool:
    df = build_research_feature_frame(ticker, start, end)
    if df.empty:
        log.error("[%s] no data built", ticker)
        return False
    out = os.path.join(DATA_DIR, f"{ticker}.parquet")
    df.to_parquet(out, index=True)
    log.info("[%s] saved %s rows x %s cols -> %s", ticker, len(df), len(df.columns), out)
    return True

def load_shortlist() -> list[str]:
    if not os.path.exists(SHORTLIST_FILE):
        return []
    import pandas as pd
    df = pd.read_csv(SHORTLIST_FILE)
    if "ticker" not in df.columns:
        return []
    return [str(x).upper() for x in df["ticker"].dropna().tolist()]

def main():
    parser = argparse.ArgumentParser(description="Build research parquet files")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.test:
        tickers = ["AAPL"]
    else:
        tickers = load_shortlist()
        if not tickers:
            print("No shortlist found. Run scanner.py first or pass --ticker.")
            sys.exit(1)

    ok = True
    for t in tickers:
        ok = research_ticker(t, TRAIN_START, TRAIN_END) and ok
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
