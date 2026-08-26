
"""
research.py — Build training parquet files using the shared production feature pipeline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import sys
from datetime import timedelta

import pandas as pd
import data_provider

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
from safe_io import atomic_write_parquet
from data_manifest import (
    read_parquet_manifest,
    validate_provider_transition,
    write_parquet_manifest,
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "research.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("research")


def _write_ticker_manifest(
    ticker: str,
    frame: pd.DataFrame,
    out_path: str,
    *,
    provider_transition: dict | None = None,
) -> None:
    """Record the provider and checksum after one ticker parquet is saved."""
    previous = read_parquet_manifest(out_path)
    provider = data_provider.provider_for_ticker.get(
        str(ticker).upper(),
        str(previous.get("provider", "unknown")),
    )
    write_parquet_manifest(
        out_path,
        ticker=ticker,
        provider=provider,
        adjustment_mode="adjusted_ohlcv",
        frame=frame,
        provider_transition=provider_transition,
    )

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

    atomic_write_parquet(df, out, index=True)
    _write_ticker_manifest(ticker, df, out)

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
# fall back to a full rebuild.  PLAIN ENGLISH: the daily bot can safely catch
# up after a few missed months because the incremental rebuild already
# recomputes a 380-day window, which covers the longest lookback features.
INCREMENTAL_MAX_STALENESS_DAYS = int(os.environ.get("INCREMENTAL_MAX_STALENESS_DAYS", "120"))
POST_PASS_COLUMN_PREFIXES = ("xs_rank_",)


def _normalise_date(value: object) -> pd.Timestamp:
    """Convert any date-like value to a midnight Timestamp for day comparisons."""
    return pd.Timestamp(value).normalize()


def _is_nyse_session(day: object) -> bool:
    """Return True when `day` is a regular NYSE trading session."""
    ts = _normalise_date(day)
    try:
        import exchange_calendars as xcals

        return bool(xcals.get_calendar("XNYS").is_session(ts))
    except Exception:
        # Fallback keeps the script usable if exchange_calendars is missing.
        return ts.weekday() < 5


def _latest_nyse_session_on_or_before(day: object) -> pd.Timestamp:
    """Find the latest completed NYSE session on or before `day`."""
    ts = _normalise_date(day)
    for _ in range(14):
        if _is_nyse_session(ts):
            return ts
        ts -= pd.Timedelta(days=1)
    return _normalise_date(day) - pd.tseries.offsets.BDay(1)


def _provider_end_after_session(session: object) -> pd.Timestamp:
    """Provider end dates are exclusive, so ask for the business day after target."""
    return _normalise_date(session) + pd.tseries.offsets.BDay(1)


def _incremental_target_dates(end: object) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    """Return (latest_completed_session, provider_end, requested_day_closed)."""
    requested = _normalise_date(end)
    target_session = _latest_nyse_session_on_or_before(requested - pd.Timedelta(days=1))
    provider_end = _provider_end_after_session(target_session)
    return target_session, provider_end, not _is_nyse_session(requested)


def _latest_existing_parquet_date(ticker: str) -> pd.Timestamp | None:
    """Read one ticker parquet and return its latest saved date, if usable."""
    path = os.path.join(DATA_DIR, f"{ticker.upper()}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return _normalise_date(pd.DatetimeIndex(df.index).max())
    except Exception:
        return None


def _closed_market_incremental_noop(tickers: list[str], end: object) -> bool:
    """Skip closed-market incremental runs when every selected ticker is current."""
    target_session, _provider_end, market_closed = _incremental_target_dates(end)
    if not market_closed:
        return False

    missing_or_stale: list[str] = []
    for ticker in tickers:
        latest = _latest_existing_parquet_date(ticker)
        if latest is None or latest < target_session:
            missing_or_stale.append(ticker)

    if not missing_or_stale:
        log.info(
            "Market closed for %s; all %d ticker parquets already reach %s. Nothing to refresh.",
            _normalise_date(end).date(),
            len(tickers),
            target_session.date(),
        )
        return True

    preview = ", ".join(missing_or_stale[:10])
    suffix = f", ... +{len(missing_or_stale) - 10} more" if len(missing_or_stale) > 10 else ""
    log.info(
        "Market closed for %s; refreshing %d stale/missing ticker parquets toward %s: %s%s",
        _normalise_date(end).date(),
        len(missing_or_stale),
        target_session.date(),
        preview,
        suffix,
    )
    return False


def _is_post_pass_column(column: str) -> bool:
    """Return True for columns rebuilt by research post-pass stages."""
    return any(str(column).startswith(prefix) for prefix in POST_PASS_COLUMN_PREFIXES)


def research_ticker_incremental(
    ticker: str,
    start: str,
    end: str,
    *,
    backfill_new_columns: bool = False,
    target_end: object | None = None,
) -> bool:
    """Incrementally refresh a ticker's parquet — only fetch new days.

    PLAIN ENGLISH: Looks at the existing parquet for this ticker.  If it already
    has data up to yesterday, there's nothing to do (skip it).  If it's a few
    days behind, download only the missing days, rebuild features for a trailing
    window, and splice the new rows onto the existing data.

    Falls back to a full rebuild if:
      - No existing parquet exists
      - The parquet is more than INCREMENTAL_MAX_STALENESS_DAYS behind
      - The existing parquet loses non-post-pass columns

    New feature columns are tail-filled by default instead of forcing a full
    history rebuild.  That keeps scheduled CI refreshes bounded when a feature
    ships.  Use --backfill-new-columns when you need research-grade historical
    values for newly added factors.

    Returns True if the parquet is up-to-date after this call.
    """
    out_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    target_session, provider_end, market_closed = _incremental_target_dates(end)
    if target_end is not None:
        target_session = _normalise_date(target_end)
        provider_end = _provider_end_after_session(target_session)

    # ── Check if parquet exists and how stale it is ─────────────────────────
    if not os.path.exists(out_path):
        log.info("[%s] no existing parquet — full rebuild", ticker)
        return research_ticker(ticker, start, str(provider_end.date()))

    try:
        existing = pd.read_parquet(out_path)
    except Exception as exc:
        log.warning("[%s] existing parquet unreadable (%s) — full rebuild", ticker, exc)
        return research_ticker(ticker, start, str(provider_end.date()))

    if existing.empty or not isinstance(existing.index, pd.DatetimeIndex):
        log.info("[%s] existing parquet empty or no DatetimeIndex — full rebuild", ticker)
        return research_ticker(ticker, start, str(provider_end.date()))

    last_date = _normalise_date(existing.index.max())
    staleness_days = max(0, int((target_session - last_date).days))

    # Already reaches the latest real trading session.
    if last_date >= target_session:
        log.info("[%s] already up-to-date (last=%s) — skipped", ticker, last_date.date())
        if not read_parquet_manifest(out_path):
            _write_ticker_manifest(ticker, existing, out_path)
        return True

    # Too stale — full rebuild is safer (catches schema changes, delistings, etc.)
    if staleness_days > INCREMENTAL_MAX_STALENESS_DAYS:
        log.info(
            "[%s] stale by %d days (max=%d) — full rebuild",
            ticker, staleness_days, INCREMENTAL_MAX_STALENESS_DAYS,
        )
        return research_ticker(ticker, start, str(provider_end.date()))

    # ── Incremental path: rebuild from (last_date - recompute_window) to today ─
    # We need a generous lookback because features like 252-day rolling max need
    # price history BEFORE the recompute window to produce valid values.
    recompute_start = last_date - timedelta(days=INCREMENTAL_RECOMPUTE_WINDOW_DAYS)
    # Clamp to original start date (don't go before available history)
    if recompute_start < pd.Timestamp(start):
        recompute_start = pd.Timestamp(start)

    log.info(
        "[%s] incremental: last=%s, target=%s, stale=%dd, recomputing from %s",
        ticker, last_date.date(), target_session.date(), staleness_days, recompute_start.date(),
    )

    # Build features for the recompute window + new days
    fresh = build_research_feature_frame(
        ticker, str(recompute_start.date()), str(provider_end.date())
    )
    if fresh.empty:
        log.error("[%s] incremental build returned empty — keeping existing", ticker)
        return True  # Don't destroy existing data

    previous_manifest = read_parquet_manifest(out_path)
    previous_provider = str(previous_manifest.get("provider", ""))
    new_provider = data_provider.provider_for_ticker.get(str(ticker).upper(), "unknown")
    provider_transition = validate_provider_transition(
        existing,
        fresh,
        previous_provider=previous_provider,
        new_provider=new_provider,
    )
    if not provider_transition.get("ok", False):
        log.error(
            "[%s] provider changed %s -> %s but overlap failed: %s; keeping existing parquet",
            ticker,
            previous_provider or "unknown",
            new_provider,
            provider_transition,
        )
        return False

    # Schema compatibility check: if core pipeline columns changed, do a full
    # rebuild.  Post-pass columns such as xs_rank_* are expected to be absent
    # from a single-ticker fresh frame because they are rebuilt after all
    # tickers finish.  Treating those as a schema break turns every daily
    # incremental refresh into a full rebuild and can trip daily_run timeouts.
    if set(fresh.columns) != set(existing.columns):
        missing_in_fresh = set(existing.columns) - set(fresh.columns)
        new_in_fresh = set(fresh.columns) - set(existing.columns)
        blocking_missing = {col for col in missing_in_fresh if not _is_post_pass_column(col)}
        post_pass_missing = missing_in_fresh - blocking_missing
        if blocking_missing:
            log.info(
                "[%s] schema changed (lost %d core cols: %s) — full rebuild",
                ticker,
                len(blocking_missing),
                ", ".join(sorted(blocking_missing)[:5]),
            )
            return research_ticker(ticker, start, str(provider_end.date()))
        if new_in_fresh:
            preview = ", ".join(sorted(new_in_fresh)[:5])
            if backfill_new_columns:
                log.info(
                    "[%s] schema expanded (+%d cols: %s) — full rebuild requested",
                    ticker, len(new_in_fresh), preview,
                )
                return research_ticker(ticker, start, str(provider_end.date()))
            log.info(
                "[%s] schema expanded (+%d cols: %s) — tail-filling incremental window; "
                "run `python3 research.py --incremental --backfill-new-columns` for full history",
                ticker, len(new_in_fresh), preview,
            )
            for col in sorted(new_in_fresh):
                existing[col] = float("nan")
        if post_pass_missing:
            log.info(
                "[%s] preserving %d post-pass cols until post-pass refresh: %s",
                ticker,
                len(post_pass_missing),
                ", ".join(sorted(post_pass_missing)[:5]),
            )
            for col in sorted(post_pass_missing):
                fresh[col] = float("nan")
        if new_in_fresh or post_pass_missing:
            fresh = fresh.reindex(columns=existing.columns)

    # ── Splice: keep old rows before recompute_start, use fresh for the rest ──
    # This preserves the full history while updating the tail with fresh features.
    old_rows = existing[existing.index < recompute_start]
    combined = pd.concat([old_rows, fresh], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    atomic_write_parquet(combined, out_path, index=True)
    _write_ticker_manifest(
        ticker,
        combined,
        out_path,
        provider_transition=provider_transition,
    )
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
        "--backfill-new-columns",
        action="store_true",
        help=(
            "With --incremental, full-rebuild tickers when new feature columns appear. "
            "Slow; use for historical research after feature engineering, not daily CI."
        ),
    )
    parser.add_argument(
        "--xs-only",
        action="store_true",
        help="Skip parquet rebuild; only run the cross-sectional rank post-pass on existing parquets.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("RESEARCH_INCREMENTAL_WORKERS", "4")),
        help="Parallel ticker workers for incremental refresh (default: 4).",
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

    if args.incremental and _closed_market_incremental_noop(tickers, TRAIN_END):
        sys.exit(0)

    incremental_target_end = None
    if args.incremental:
        incremental_target_end, _provider_end, market_closed = _incremental_target_dates(TRAIN_END)
        if market_closed:
            log.info(
                "Market closed for %s; incremental target is latest NYSE session %s",
                _normalise_date(TRAIN_END).date(),
                incremental_target_end.date(),
            )

    ok = True
    built_tickers: list[str] = []

    def _build_one(ticker: str) -> bool:
        """Build one ticker using the selected full or incremental mode."""
        if args.incremental:
            return research_ticker_incremental(
                ticker,
                TRAIN_START,
                TRAIN_END,
                backfill_new_columns=bool(args.backfill_new_columns),
                target_end=incremental_target_end,
            )
        return research_ticker(ticker, TRAIN_START, TRAIN_END)

    worker_count = max(1, int(args.workers)) if args.incremental else 1
    if worker_count == 1:
        outcomes = [(ticker, _build_one(ticker)) for ticker in tickers]
    else:
        # PLAIN ENGLISH: Each ticker is independent until the two shared
        # cross-sectional post-passes below. A small worker pool cuts GitHub
        # refresh time without launching provider-internal thread storms.
        outcomes = []
        with ThreadPoolExecutor(max_workers=min(worker_count, len(tickers))) as pool:
            futures = {pool.submit(_build_one, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    outcomes.append((ticker, bool(future.result())))
                except Exception as exc:
                    log.error("[%s] incremental worker failed: %s", ticker, exc)
                    outcomes.append((ticker, False))
    for ticker, built in outcomes:
        if built:
            built_tickers.append(ticker)
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

        # The post-passes rewrite parquet bytes. Refresh sidecar checksums only
        # after both passes finish so manifests describe the final stored file.
        for ticker in built_tickers:
            path = os.path.join(DATA_DIR, f"{ticker}.parquet")
            try:
                final_frame = pd.read_parquet(path)
                _write_ticker_manifest(ticker, final_frame, path)
            except Exception as exc:
                log.error("[%s] manifest refresh failed: %s", ticker, exc)
                ok = False

    # ── Adaptive factor weight update ────────────────────────────────────
    # PLAIN ENGLISH: Now that parquets are fresh, recompute how well each
    # factor predicted returns over the last year.  Factors that worked well
    # recently get more weight in the composite score; those that lost their
    # edge get less.  This keeps the signal adaptive to regime changes.
    if built_tickers:
        try:
            from ranker_utils import compute_adaptive_factor_weights
            from settings import (
                SIMPLE_FACTOR_COLS, ADAPTIVE_WEIGHTS_FILE,
                ADAPTIVE_WEIGHT_HALFLIFE, ADAPTIVE_WEIGHT_FLOOR,
            )
            adaptive_w = compute_adaptive_factor_weights(
                data_dir=DATA_DIR,
                factor_cols=SIMPLE_FACTOR_COLS,
                lookback_days=252,
                halflife=ADAPTIVE_WEIGHT_HALFLIFE,
                floor=ADAPTIVE_WEIGHT_FLOOR,
                output_path=ADAPTIVE_WEIGHTS_FILE,
            )
            log.info("[adaptive_weights] Updated: %s", adaptive_w)
        except Exception as exc:
            log.warning("[adaptive_weights] Failed (non-fatal): %s", exc)

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
