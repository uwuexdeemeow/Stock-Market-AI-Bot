
"""
research.py — Build training parquet files using the shared production feature pipeline.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from settings import (
    DATA_DIR,
    LOG_DIR,
    SHORTLIST_FILE,
    TRAIN_START,
    TRAIN_END,
    WATCHLIST,
    SURVIVORSHIP_TRAINING_TICKERS,
)
from pipeline_shared import build_research_feature_frame
from fundamental_features import apply_sector_fundamental_zscores
from cross_sectional_features import apply_cross_sectional_rank_features

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

    # IMPORTANT: the current alpha_factor_backtest.load_factor_panel() expects
    # ticker parquet files named like AAPL.parquet, MSFT.parquet, QQQ.parquet.
    # Do not write *_features.parquet here unless the loader is changed too.
    out = os.path.join(DATA_DIR, f"{ticker}.parquet")

    df.to_parquet(out, index=True)

    log.info(
        "[%s] saved %s rows x %s cols -> %s",
        ticker,
        len(df),
        len(df.columns),
        out,
    )

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
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="Build parquets for the full settings.WATCHLIST (default)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--xs-only",
        action="store_true",
        help="Skip parquet rebuild; only run the cross-sectional rank post-pass on existing parquets.",
    )
    args = parser.parse_args()

    # ── Fast path: only re-run cross-sectional rank post-pass ────────────────
    # Used when feature engineering for xs_rank changed but raw parquet data is
    # already up-to-date.  Avoids the slow 28-ticker yfinance rebuild.
    if args.xs_only:
        try:
            xs_universe = sorted(set(WATCHLIST) | set(SURVIVORSHIP_TRAINING_TICKERS))
            xs_summary = apply_cross_sectional_rank_features(xs_universe, DATA_DIR)
            log.info(
                "[xs_rank] xs-only post-pass: updated=%d  new_cols=%d  skipped=%d  write_errors=%d",
                xs_summary.get("updated", 0),
                len(xs_summary.get("new_cols", [])),
                len(xs_summary.get("skipped", [])),
                len(xs_summary.get("write_errors", [])),
            )
            sys.exit(0)
        except Exception as exc:
            log.error("[xs_rank] xs-only post-pass failed: %s", exc)
            sys.exit(1)

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    elif args.test:
        tickers = ["AAPL"]
    else:
        tickers = [t.upper() for t in WATCHLIST]

    ok = True
    built_tickers: list[str] = []
    for t in tickers:
        built = research_ticker(t, TRAIN_START, TRAIN_END)
        if built:
            built_tickers.append(t)
        ok = built and ok
    if built_tickers:
        try:
            zscore_universe = sorted(set(WATCHLIST) | set(built_tickers))
            summary = apply_sector_fundamental_zscores(zscore_universe, DATA_DIR)
            log.info("[fundamentals] sector z-score post-pass: %s", summary)
        except Exception as exc:
            log.warning("[fundamentals] sector z-score post-pass failed: %s", exc)

        # Cross-sectional rank-within-sector / rank-within-market features.
        # These are the highest-leverage feature class for a cross-sectional
        # ranker: they encode "how does this stock compare to its peers TODAY?"
        # which the per-stock technical features cannot answer.  AUC=0.51 with
        # only per-stock features in the diagnostic ladder run; this post-pass
        # is the structural fix to inject rank-based discriminators.
        try:
            # IMPORTANT: include SURVIVORSHIP_TRAINING_TICKERS (e.g. FRC/BBBY)
            # in the post-pass.  train.py uses them as training-only rows; the
            # pooled feature set is the INTERSECTION across all training tickers,
            # so if FRC/BBBY lack xs_rank columns the entire xs_rank feature
            # group gets dropped from the candidate set silently.
            xs_universe = sorted(
                set(WATCHLIST) | set(built_tickers) | set(SURVIVORSHIP_TRAINING_TICKERS)
            )
            xs_summary = apply_cross_sectional_rank_features(xs_universe, DATA_DIR)
            log.info(
                "[xs_rank] cross-sectional rank post-pass: updated=%d  new_cols=%d  skipped=%d  write_errors=%d",
                xs_summary.get("updated", 0),
                len(xs_summary.get("new_cols", [])),
                len(xs_summary.get("skipped", [])),
                len(xs_summary.get("write_errors", [])),
            )
        except Exception as exc:
            log.warning("[xs_rank] cross-sectional rank post-pass failed: %s", exc)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
