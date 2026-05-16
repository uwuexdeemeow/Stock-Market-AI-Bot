
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
    """Build or rebuild the full parquet for a single ticker.

    PLAIN ENGLISH: Downloads all price data from `start` to `end`, computes
    every technical/factor/sentiment feature, and saves to data/{TICKER}.parquet.
    This is the FULL rebuild — used when you need to regenerate everything from
    scratch (new feature added, first run, etc.).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# INCREMENTAL REFRESH — only download new days since last parquet date
# ─────────────────────────────────────────────────────────────────────────────
# PLAIN ENGLISH: Instead of downloading 15 years of data every day, this checks
# what's already on disk and only downloads the missing recent days.  Features
# that need a lookback window (e.g. 252-day moving average) are recomputed for
# the last RECOMPUTE_WINDOW_DAYS so they stay accurate.  This cuts a 10-20 min
# full rebuild down to ~2-3 minutes for daily CI runs.

# How many calendar days of features to recompute from scratch.
# Must be larger than the longest lookback window used in any feature (252 days
# for the 52-week high factor) plus a safety margin for weekends/holidays.
INCREMENTAL_RECOMPUTE_WINDOW_DAYS = 380

# If an existing parquet is stale by more than this many calendar days,
# fall back to a full rebuild (incremental savings not worth the complexity).
INCREMENTAL_MAX_STALENESS_DAYS = 30


def research_ticker_incremental(ticker: str, start: str, end: str) -> bool:
    """Incrementally refresh a ticker's parquet — only fetch new days.

    PLAIN ENGLISH: Looks at the existing parquet for this ticker.  If it already
    has data up to yesterday, there's nothing to do (skip it).  If it's a few
    days behind, download only the missing days, rebuild features for a trailing
    window, and splice the new rows onto the existing data.

    Falls back to a full rebuild if:
      - No existing parquet exists
      - The parquet is more than INCREMENTAL_MAX_STALENESS_DAYS behind
      - The existing parquet has different columns (schema changed)

    Returns True if the parquet is up-to-date after this call.
    """
    import pandas as pd
    from datetime import datetime, timedelta

    out_path = os.path.join(DATA_DIR, f"{ticker}.parquet")

    # ── Check if parquet exists and how stale it is ─────────────────────────
    if not os.path.exists(out_path):
        log.info("[%s] no existing parquet — full rebuild", ticker)
        return research_ticker(ticker, start, end)

    try:
        existing = pd.read_parquet(out_path)
    except Exception as exc:
        log.warning("[%s] existing parquet unreadable (%s) — full rebuild", ticker, exc)
        return research_ticker(ticker, start, end)

    if existing.empty or not isinstance(existing.index, pd.DatetimeIndex):
        log.info("[%s] existing parquet empty or no DatetimeIndex — full rebuild", ticker)
        return research_ticker(ticker, start, end)

    last_date = existing.index.max()
    today = pd.Timestamp(end)
    staleness_days = (today - last_date).days

    # Already up-to-date (within 1 calendar day = same or next trading day)
    if staleness_days <= 1:
        log.info("[%s] already up-to-date (last=%s) — skipped", ticker, last_date.date())
        return True

    # Too stale — full rebuild is safer (catches schema changes, delistings, etc.)
    if staleness_days > INCREMENTAL_MAX_STALENESS_DAYS:
        log.info(
            "[%s] stale by %d days (max=%d) — full rebuild",
            ticker, staleness_days, INCREMENTAL_MAX_STALENESS_DAYS,
        )
        return research_ticker(ticker, start, end)

    # ── Incremental path: rebuild from (last_date - recompute_window) to today ─
    # We need a generous lookback because features like 252-day rolling max need
    # price history BEFORE the recompute window to produce valid values.
    recompute_start = last_date - timedelta(days=INCREMENTAL_RECOMPUTE_WINDOW_DAYS)
    # Clamp to original start date (don't go before available history)
    if recompute_start < pd.Timestamp(start):
        recompute_start = pd.Timestamp(start)

    log.info(
        "[%s] incremental: last=%s, stale=%dd, recomputing from %s",
        ticker, last_date.date(), staleness_days, recompute_start.date(),
    )

    # Build features for the recompute window + new days
    fresh = build_research_feature_frame(
        ticker, str(recompute_start.date()), end
    )
    if fresh.empty:
        log.error("[%s] incremental build returned empty — keeping existing", ticker)
        return True  # Don't destroy existing data

    # Schema compatibility check: if columns changed, do a full rebuild
    if set(fresh.columns) != set(existing.columns):
        missing_in_fresh = set(existing.columns) - set(fresh.columns)
        new_in_fresh = set(fresh.columns) - set(existing.columns)
        if missing_in_fresh:
            log.info(
                "[%s] schema changed (lost %d cols) — full rebuild",
                ticker, len(missing_in_fresh),
            )
            return research_ticker(ticker, start, end)
        # New columns added (e.g. new factor) — full rebuild to fill history
        if new_in_fresh:
            log.info(
                "[%s] schema expanded (+%d cols: %s) — full rebuild",
                ticker, len(new_in_fresh),
                ", ".join(sorted(new_in_fresh)[:5]),
            )
            return research_ticker(ticker, start, end)

    # ── Splice: keep old rows before recompute_start, use fresh for the rest ──
    # This preserves the full history while updating the tail with fresh features.
    old_rows = existing[existing.index < recompute_start]
    combined = pd.concat([old_rows, fresh], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    combined.to_parquet(out_path, index=True)
    log.info(
        "[%s] incremental save: %d rows (was %d, fresh contributed %d) -> %s",
        ticker, len(combined), len(existing), len(fresh), out_path,
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


def _validate_xs_rank_summary(summary: dict) -> None:
    updated = int(summary.get("updated", 0) or 0)
    new_cols = list(summary.get("new_cols", []) or [])
    write_errors = list(summary.get("write_errors", []) or [])
    if write_errors:
        preview = "; ".join(str(x) for x in write_errors[:5])
        suffix = f"; ... +{len(write_errors) - 5} more" if len(write_errors) > 5 else ""
        raise RuntimeError(
            f"cross-sectional rank post-pass had write_errors: {preview}{suffix}. "
            "Run `python3 research.py --xs-only` after checking parquet write permissions."
        )
    if updated <= 0:
        raise RuntimeError(
            "cross-sectional rank post-pass updated 0 parquet files. "
            "Run `python3 feature_research.py`, then `python3 research.py --xs-only`."
        )
    if not new_cols:
        raise RuntimeError(
            "cross-sectional rank post-pass produced no xs-rank columns. "
            "Run `python3 research.py --xs-only` and inspect logs/research.log."
        )


def main():
    parser = argparse.ArgumentParser(description="Build research parquet files")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="Build parquets for the full settings.WATCHLIST (default)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only download new days since last parquet date (fast daily refresh for CI).",
    )
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
            _validate_xs_rank_summary(xs_summary)
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

    # Choose build function: incremental (fast, for CI daily runs) or full rebuild.
    # PLAIN ENGLISH: --incremental skips tickers that are already up-to-date and
    # only downloads missing days for ones that are slightly behind.  This makes
    # daily CI runs 5-10x faster than a full rebuild.
    build_fn = research_ticker_incremental if args.incremental else research_ticker

    ok = True
    built_tickers: list[str] = []
    for t in tickers:
        built = build_fn(t, TRAIN_START, TRAIN_END)
        if built:
            built_tickers.append(t)
        ok = built and ok
    if built_tickers:
        try:
            zscore_universe = sorted(set(WATCHLIST) | set(built_tickers))
            summary = apply_sector_fundamental_zscores(zscore_universe, DATA_DIR)
            log.info("[fundamentals] sector z-score post-pass: %s", summary)
        except Exception as exc:
            log.error(
                "[fundamentals] sector z-score post-pass failed: %s. "
                "Run `python3 feature_research.py`, then retry `python3 research.py --xs-only`.",
                exc,
            )
            ok = False

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
            _validate_xs_rank_summary(xs_summary)
        except Exception as exc:
            log.error("[xs_rank] cross-sectional rank post-pass failed: %s", exc)
            ok = False
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
