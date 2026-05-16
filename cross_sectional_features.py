"""
cross_sectional_features.py — within-sector and within-market rank percentiles.

WHY THIS FILE EXISTS
--------------------
The 28-ticker pooled XGBoost run produced OOS Direction AUC = 0.51 (random) on
both the excess_return and direction targets, even after pooling, regime
features, macro protection, and triple-barrier ablation.  The technical
features in each parquet (returns, vol, MAs, MACD, RSI) describe a stock in
isolation — they say nothing about how it ranks RELATIVE to its peers on a
given day.  But cross-sectional ranking strategies live entirely on
relative-rank features: "is this stock the leader or laggard in its sector
right now?"

This file builds those features as a research-time post-pass: load every
ticker's parquet, build a date × ticker panel for a few source columns, then
for each date compute rank percentiles within sector and within the broad
market.  The new columns are written back to each ticker's parquet so train.py
and backtest.py see them automatically.

PLAIN ENGLISH FOR A BEGINNER
----------------------------
"Rank-within-sector" means: on day D, look at every tech stock in the universe,
sort them from worst to best by their 20-day return, and assign each one a
percentile from 0 (worst in sector) to 1 (best in sector).  That percentile is
the new feature.  Trees in XGBoost can then split on "this stock is in the
top quartile of its sector by momentum" — a much sharper question than
"this stock has 5% momentum" which doesn't mean anything without context.

Conventions
-----------
* All ranks are PERCENTILES in [0, 1].  0 = worst in group on that date,
  1 = best.  Lone tickers in a group on a given date default to 0.5 (median).
* Per-date groups smaller than MIN_GROUP_SIZE fall back to market-wide rank
  to avoid noisy "rank within 1 stock = always 0.5" cells.
* Output columns are prefixed with "xs_rank_sector_" and "xs_rank_market_"
  so allowed_prefixes lists in train.py / backtest.py can pick them up with
  a single prefix entry.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

# settings imports are wrapped because this module may be imported by tools that
# don't have the full settings.py environment available.
try:
    from settings import DATA_DIR, SECTOR_MAP
except Exception:  # pragma: no cover - defensive fallback for tests
    DATA_DIR = "data"
    SECTOR_MAP = {}

log = logging.getLogger("cross_sectional")

# Source columns we expect in every parquet.  If any are missing for a given
# ticker, we skip that ticker (the panel still works for the others).
# Derived columns (e.g. ret_60d) that aren't in the parquet are not used here —
# only what we already engineer in pipeline_shared.py.
SOURCE_COLS: list[str] = [
    "ret_5d",          # short-term momentum
    "ret_10d",         # short-term momentum
    "ret_20d",         # medium-term momentum (matches our return horizon)
    "factor_mom_12_1",                    # 12-1 momentum anomaly
    "factor_resid_mom_sector_12_1",       # sector-neutral 12-1 momentum
    "factor_beta_252_spy",                # trailing market beta
    "factor_idio_vol_252_spy",            # trailing idiosyncratic volatility
    "factor_liquidity_dollar_vol_20d",    # dollar-volume liquidity
    "factor_illiquidity_amihud_20d",      # Amihud illiquidity proxy
    "hvol_20d",        # 20-day historical volatility
    "dist_ma50",       # distance from 50-day moving average (trend strength)
    "dist_ma200",      # distance from 200-day moving average (long-term trend)
    "drawdown_60d",    # recent peak-to-trough drawdown — stress proxy
    "rsi_14",          # 14-day RSI (overbought / oversold)
    "macd",            # MACD line — momentum-of-momentum
]

# Minimum tickers in a (date, sector) group for sector rank to be trusted.
# Below this threshold we use market-wide rank instead.
# Raised from 3 to 6 now that BROAD_WATCHLIST has 10-18 names per sector —
# at MIN_GROUP_SIZE=3 ranks within (XLF: JPM/GS/BAC) collapsed to 0.25/0.5/0.75
# coin-flip, contributing noise.  At 6, sector ranks only fire when the group
# has real dispersion.
MIN_GROUP_SIZE: int = 6


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Convert a numeric series to a [0, 1] percentile rank.

    Plain English: assigns each value a number between 0 (smallest in the
    group) and 1 (largest in the group).  Ties are averaged (pandas default).
    A group of size 1 gets 0.5 by convention (no information).

    Args:
        series: numeric values within a single group.

    Returns:
        Series of the same length, dtype float, in [0, 1].
    """
    s = pd.to_numeric(series, errors="coerce")
    n = s.count()
    if n <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if n == 1:
        # Lone value — neither "best" nor "worst" makes sense; mid-rank.
        return pd.Series(0.5, index=series.index, dtype=float)
    # rank(pct=True) gives values in (0, 1].  Subtract a tiny offset so the
    # smallest non-NaN value is closer to 0; cap to [0, 1] for safety.
    ranks = s.rank(method="average", pct=True, na_option="keep")
    return ranks.clip(0.0, 1.0)


def apply_cross_sectional_rank_features(
    tickers: list[str],
    data_dir: str = DATA_DIR,
    sector_map: dict[str, str] | None = None,
    source_cols: list[str] | None = None,
) -> dict:
    """Compute within-sector and within-market rank percentiles for each ticker.

    For each (date, source_col) pair, this builds a panel of all tickers'
    values, ranks them within sector and across the whole market, and writes
    the percentiles back to each ticker's parquet as new columns:

        xs_rank_sector_<col>  in [0, 1]   per-date rank within sector
        xs_rank_market_<col>  in [0, 1]   per-date rank within full universe

    When a (date, sector) group has fewer than MIN_GROUP_SIZE tickers, the
    sector rank for that cell falls back to the market rank.

    Args:
        tickers:     list of tickers whose parquets to update.
        data_dir:    directory containing <ticker>.parquet files.
        sector_map:  optional override for SECTOR_MAP.
        source_cols: optional override for SOURCE_COLS.

    Returns:
        Summary dict with counts of updated parquets and any per-ticker errors.
    """
    sector_map = sector_map or SECTOR_MAP
    source_cols = source_cols or SOURCE_COLS

    # ── Step 1: load each parquet, collect rows for the panel ─────────────────
    # We only pull the source columns + index (date) to keep the panel small.
    frames: list[pd.DataFrame] = []
    paths: dict[str, str] = {}
    skipped: list[str] = []
    for ticker in tickers:
        ticker = ticker.upper()
        path = os.path.join(data_dir, f"{ticker}.parquet")
        if not os.path.exists(path):
            skipped.append(f"{ticker}:no_parquet")
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            skipped.append(f"{ticker}:read_error:{exc}")
            continue
        # Skip ticker if it doesn't have ANY source column.  Missing individual
        # columns are filled with NaN below, but a totally empty source set is
        # not actionable.
        present = [c for c in source_cols if c in df.columns]
        if not present:
            skipped.append(f"{ticker}:no_source_cols")
            continue
        sub = df[present].copy()
        # Ensure missing columns exist as NaN so the panel concat aligns.
        for c in source_cols:
            if c not in sub.columns:
                sub[c] = np.nan
        sub["ticker"] = ticker
        sub["sector"] = sector_map.get(ticker, "OTHER")
        sub["date"] = pd.to_datetime(df.index)
        frames.append(sub.reset_index(drop=True))
        paths[ticker] = path

    if not frames:
        return {"updated": 0, "skipped": skipped, "reason": "no_parquets"}

    # ── Step 2: build the (date × ticker) panel ───────────────────────────────
    panel = pd.concat(frames, ignore_index=True)
    log.info(
        "[xs_rank] Panel built: %d rows  %d tickers  %d source cols",
        len(panel), panel["ticker"].nunique(), len(source_cols),
    )

    # ── Step 3: compute per-date sector ranks and market ranks ────────────────
    # Working column-by-column keeps memory bounded and lets us isolate failures.
    new_cols: list[str] = []
    for col in source_cols:
        if col not in panel.columns:
            continue
        sector_col = f"xs_rank_sector_{col}"
        market_col = f"xs_rank_market_{col}"
        new_cols.extend([sector_col, market_col])

        # Market-wide rank: groupby date only.  Always defined (every date has
        # at least 1 ticker) so this is the safe fallback for sparse sectors.
        market_rank = (
            panel.groupby("date", group_keys=False)[col].apply(_percentile_rank)
        )
        panel[market_col] = market_rank.values

        # Sector rank: groupby (date, sector).  Cells where the (date, sector)
        # group has fewer than MIN_GROUP_SIZE tickers are replaced with the
        # market rank to avoid noisy lone-ticker = 0.5 cells.
        group_sizes = panel.groupby(["date", "sector"])["ticker"].transform("count")
        sector_rank = (
            panel.groupby(["date", "sector"], group_keys=False)[col]
            .apply(_percentile_rank)
        )
        # Where group size is too small, take market rank; else sector rank.
        too_small = group_sizes < MIN_GROUP_SIZE
        panel[sector_col] = np.where(too_small, panel[market_col], sector_rank.values)

    # Fill NaNs (caused by missing source cols on early dates) with 0.5 — neutral.
    for col in new_cols:
        panel[col] = panel[col].astype(float).fillna(0.5).clip(0.0, 1.0)

    # ── Step 4: write columns back to each ticker's parquet ───────────────────
    updated = 0
    write_errors: list[str] = []
    for ticker, path in paths.items():
        try:
            df = pd.read_parquet(path)
            sub = panel[panel["ticker"] == ticker].set_index("date")
            for col in new_cols:
                # Reindex to the parquet's date index; missing dates → 0.5.
                df[col] = sub[col].reindex(pd.DatetimeIndex(df.index)).fillna(0.5).values
            df.to_parquet(path, index=True)
            updated += 1
        except Exception as exc:
            write_errors.append(f"{ticker}:{exc}")

    log.info(
        "[xs_rank] Updated %d parquets with %d new rank columns each (skipped=%d, write_errors=%d)",
        updated, len(new_cols), len(skipped), len(write_errors),
    )
    return {
        "updated": updated,
        "new_cols": new_cols,
        "skipped": skipped,
        "write_errors": write_errors,
    }
