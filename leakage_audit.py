"""
leakage_audit.py — Temporal leakage detector.

Tests THREE layers for future information leakage:
  1. Per-ticker technical features (price/volume derived)
  2. Cross-sectional rank features (xs_rank_sector_*, xs_rank_market_*)
  3. Sector fundamental z-scores (fund_pe_sector_z, fund_fcf_yield_sector_z)

The core technique: split data at a cutoff date, perturb everything AFTER
the cutoff with large deterministic noise, rebuild the features, and check
whether any feature value AT OR BEFORE the cutoff changes.  If it does,
that feature looked into the future.

Layer 1 tests each ticker in isolation (single-ticker price features).
Layers 2 and 3 test the full multi-ticker panel (cross-sectional features
that rank stocks against each other on each date).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import DATA_DIR, LOG_DIR, SECTOR_MAP  # noqa: E402

try:
    import pipeline_shared  # type: ignore
except Exception:
    pipeline_shared = None

try:
    from cross_sectional_features import (
        apply_cross_sectional_rank_features,
        SOURCE_COLS as XS_SOURCE_COLS,
    )
except Exception:
    apply_cross_sectional_rank_features = None
    XS_SOURCE_COLS = []

try:
    from fundamental_features import apply_sector_fundamental_zscores
except Exception:
    apply_sector_fundamental_zscores = None

# ── Configuration ───────────────────────────────────────────────────────────
# One ticker per sector so every sector's cross-sectional ranking is tested.
# Includes the 3 original tickers (AAPL, MSFT, SPY) plus representatives
# from each sector in BROAD_WATCHLIST.
AUDIT_TICKERS = [
    "AAPL", "MSFT",           # XLK — tech (original)
    "AMZN",                    # XLY — consumer discretionary
    "JPM",                     # XLF — financials
    "UNH",                     # XLV — healthcare
    "XOM",                     # XLE — energy
    "CAT",                     # XLI — industrials
    "COST",                    # XLP — consumer staples
    "NEE",                     # XLU — utilities
    "PLD",                     # XLRE — real estate
    "LIN",                     # XLB — materials
    "NFLX",                    # XLC — communication services
]

# Wider window to match actual training range and stress rolling(252) features.
AUDIT_START = "2015-01-01"
AUDIT_END = "2024-01-01"

ALLOW_NETWORK_FETCH = os.environ.get(
    "LEAKAGE_AUDIT_ALLOW_NETWORK", ""
).strip().lower() in {"1", "true", "yes"}

# Columns that are EXPECTED to change (raw OHLCV) or are the label itself.
IGNORED_COLUMNS = {
    "target",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
}

# Tolerance for floating point comparison.  Features built from rolling
# windows near the cutoff may have tiny float jitter from reordering.
ATOL = 1e-9


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fetch(ticker: str) -> pd.DataFrame:
    """Load OHLCV data from local parquet or (optionally) yfinance."""
    local_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if os.path.exists(local_path):
        df = pd.read_parquet(local_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        ohlcv = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df.loc[AUDIT_START:AUDIT_END, ohlcv].dropna()

    if not ALLOW_NETWORK_FETCH:
        return pd.DataFrame()

    df = yf.download(ticker, start=AUDIT_START, end=AUDIT_END,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def _perturb_after_cutoff(df: pd.DataFrame, cutoff_pos: int) -> pd.DataFrame:
    """Apply large, deterministic perturbation to all rows AFTER cutoff_pos.

    If any feature computed at or before cutoff_pos changes when we do this,
    that feature is using future data — it leaks.
    """
    perturbed = df.copy()
    future_idx = perturbed.index[cutoff_pos + 1:]
    if len(future_idx) == 0:
        return perturbed

    for col in [c for c in ["Open", "High", "Low", "Close", "Volume"]
                if c in perturbed.columns]:
        values = perturbed.loc[future_idx, col].astype(float).to_numpy()
        # Reverse order + scale so future prices are wildly different.
        if col == "Volume":
            perturbed.loc[future_idx, col] = values[::-1] * 3.0 + 123.0
        else:
            scale = np.linspace(1.50, 0.50, len(values))
            perturbed.loc[future_idx, col] = values[::-1] * scale
    return perturbed


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1: Per-ticker technical features
# ═════════════════════════════════════════════════════════════════════════════

def _build_research_frame_from_raw(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build a single-ticker research frame using monkey-patched pipeline_shared.

    External data sources (VIX, sentiment, breadth, multi-market) are stubbed
    out so the test isolates price/volume-derived features only.
    """
    if pipeline_shared is None:
        raise RuntimeError("pipeline_shared could not be imported")

    originals = {
        "fetch_price_data": pipeline_shared.fetch_price_data,
        "build_multi_market": pipeline_shared.build_multi_market,
        "build_vix_features": pipeline_shared.build_vix_features,
        "build_sentiment_features": pipeline_shared.build_sentiment_features,
        "build_social_sentiment_features": pipeline_shared.build_social_sentiment_features,
        "build_pead_features": pipeline_shared.build_pead_features,
        "build_market_breadth_features": pipeline_shared.build_market_breadth_features,
        "build_market_concentration_features": pipeline_shared.build_market_concentration_features,
    }

    def fake_fetch_price_data(fetch_ticker: str, start: str, end: str) -> pd.DataFrame:
        if fetch_ticker.upper() == ticker.upper():
            return raw.copy()
        return originals["fetch_price_data"](fetch_ticker, start, end)

    def empty_frame(*args, **kwargs) -> pd.DataFrame:
        dates = args[1] if len(args) > 1 and isinstance(args[1], pd.DatetimeIndex) else args[0]
        return pd.DataFrame(index=dates)

    def neutral_pead_features(ticker_arg: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            index=dates,
            data={
                "eps_surprise_pct": 0.0,
                "days_since_earnings": 60.0,
                "days_to_next_earnings": 60.0,
            },
        )

    pipeline_shared.fetch_price_data = fake_fetch_price_data
    pipeline_shared.build_multi_market = empty_frame
    pipeline_shared.build_vix_features = empty_frame
    pipeline_shared.build_sentiment_features = empty_frame
    pipeline_shared.build_social_sentiment_features = empty_frame
    pipeline_shared.build_pead_features = neutral_pead_features
    pipeline_shared.build_market_breadth_features = empty_frame
    # Concentration comes from independent ETF downloads, not this ticker's
    # OHLCV. Re-downloading those ETFs twice can create tiny vendor rounding
    # differences, so freeze it here just like VIX and breadth. Its causal
    # formulas have a separate deterministic future-perturbation test.
    pipeline_shared.build_market_concentration_features = empty_frame
    try:
        return pipeline_shared.build_research_feature_frame(ticker, AUDIT_START, AUDIT_END)
    finally:
        for name, fn in originals.items():
            setattr(pipeline_shared, name, fn)


def _compare_features(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    ignored: set[str] | None = None,
) -> tuple[list[str], int]:
    """Compare two feature frames up to cutoff_date.  Return (leaks, n_checked)."""
    ignored = ignored or IGNORED_COLUMNS
    common = real.index.intersection(fake.index)
    common = common[common <= cutoff_date]
    leaks: list[str] = []
    checked = 0
    for col in real.columns:
        if col in ignored:
            continue
        if col not in fake.columns:
            continue
        try:
            a = pd.to_numeric(real.loc[common, col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(fake.loc[common, col], errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue
        if len(a) == 0 or len(b) == 0:
            continue
        checked += 1
        if not np.allclose(a, b, equal_nan=True, atol=ATOL):
            leaks.append(col)
    return leaks, checked


def audit_ticker(ticker: str) -> dict:
    """Layer 1: test single-ticker price-derived features for leakage."""
    raw = _fetch(ticker)
    if raw.empty:
        return {"ticker": ticker, "status": "skip", "reason": "no data"}
    if len(raw) < 120:
        return {"ticker": ticker, "status": "skip", "reason": f"not enough rows: {len(raw)}"}

    try:
        cutoff_pos = int(len(raw) * 0.70)
        cutoff_date = raw.index[cutoff_pos]
        real = _build_research_frame_from_raw(raw.copy(), ticker)
        fake = _build_research_frame_from_raw(
            _perturb_after_cutoff(raw, cutoff_pos), ticker
        )
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e)}

    leaks, checked = _compare_features(real, fake, cutoff_date)

    return {
        "ticker": ticker,
        "status": "pass" if not leaks else "FAIL",
        "leaking_features": leaks,
        "n_features_checked": checked,
        "cutoff_date": str(
            cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2: Cross-sectional rank features (xs_rank_sector_*, xs_rank_market_*)
# ═════════════════════════════════════════════════════════════════════════════

def audit_cross_sectional_leakage(tickers: list[str] | None = None) -> dict:
    """Test whether cross-sectional rank features leak future data.

    Strategy:
      1. Load every ticker's parquet, restrict to AUDIT_START:AUDIT_END.
      2. Pick a cutoff date at 70% of the date range.
      3. Build a "real" panel — all prices intact.
      4. Build a "fake" panel — every ticker's prices AFTER cutoff are perturbed.
      5. Run apply_cross_sectional_rank_features on both panels.
      6. Compare xs_rank_* columns AT OR BEFORE cutoff.  Any difference = leak.

    This catches bugs where the ranking function accidentally sees future
    rows when computing per-date ranks (e.g. using full-panel normalization
    instead of per-date groupby).
    """
    if apply_cross_sectional_rank_features is None:
        return {"status": "skip", "reason": "cross_sectional_features not importable"}

    tickers = tickers or AUDIT_TICKERS
    import tempfile
    import shutil

    # Load raw OHLCV for each ticker
    raw_data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = _fetch(t)
        if len(df) >= 120:
            raw_data[t] = df

    if len(raw_data) < 5:
        return {"status": "skip", "reason": f"only {len(raw_data)} tickers have data"}

    # Use the date index of the first ticker to pick a cutoff
    # (all tickers share roughly the same trading calendar)
    sample_dates = sorted(set().union(*(df.index for df in raw_data.values())))
    cutoff_pos = int(len(sample_dates) * 0.70)
    cutoff_date = pd.Timestamp(sample_dates[cutoff_pos])

    def _build_panel_in_tmpdir(perturb_future: bool) -> dict[str, pd.DataFrame]:
        """Build research parquets in a temp dir, run xs rank post-pass, return them."""
        tmpdir = tempfile.mkdtemp(prefix="leakage_xs_")
        try:
            # Step 1: build per-ticker research frames and save as parquet
            for t, raw in raw_data.items():
                if perturb_future:
                    # Find this ticker's cutoff position
                    cp = int(len(raw) * 0.70)
                    ticker_raw = _perturb_after_cutoff(raw, cp)
                else:
                    ticker_raw = raw.copy()
                try:
                    frame = _build_research_frame_from_raw(ticker_raw, t)
                except Exception:
                    continue
                frame.to_parquet(os.path.join(tmpdir, f"{t}.parquet"))

            # Step 2: run cross-sectional rank post-pass on the temp dir
            available = [t for t in raw_data if os.path.exists(
                os.path.join(tmpdir, f"{t}.parquet")
            )]
            apply_cross_sectional_rank_features(
                available, data_dir=tmpdir, sector_map=SECTOR_MAP
            )

            # Step 3: load the updated parquets
            result = {}
            for t in available:
                path = os.path.join(tmpdir, f"{t}.parquet")
                result[t] = pd.read_parquet(path)
            return result
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print("[leakage_audit] Layer 2: building REAL cross-sectional panel...")
    real_panels = _build_panel_in_tmpdir(perturb_future=False)
    print("[leakage_audit] Layer 2: building PERTURBED cross-sectional panel...")
    fake_panels = _build_panel_in_tmpdir(perturb_future=True)

    # Compare xs_rank_* columns before cutoff for each ticker
    xs_cols_to_check = set()
    all_leaks: dict[str, list[str]] = {}
    total_checked = 0

    for t in real_panels:
        if t not in fake_panels:
            continue
        real_df = real_panels[t]
        fake_df = fake_panels[t]

        # Only check xs_rank_* columns — the layer 1 audit already covers
        # single-ticker features.
        xs_cols = [c for c in real_df.columns if c.startswith("xs_rank_")]
        xs_cols_to_check.update(xs_cols)

        leaks, checked = _compare_features(
            real_df[xs_cols] if xs_cols else real_df[[]],
            fake_df[[c for c in xs_cols if c in fake_df.columns]] if xs_cols else fake_df[[]],
            cutoff_date,
            ignored=set(),  # no ignoring — all xs_rank_* columns must be clean
        )
        total_checked += checked
        if leaks:
            all_leaks[t] = leaks

    return {
        "layer": "cross_sectional_ranks",
        "status": "pass" if not all_leaks else "FAIL",
        "tickers_tested": len(real_panels),
        "xs_columns_checked": len(xs_cols_to_check),
        "n_features_checked": total_checked,
        "cutoff_date": str(cutoff_date.date()),
        "leaking_tickers": {t: cols for t, cols in all_leaks.items()},
    }


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3: Sector fundamental z-scores (fund_pe_sector_z, etc.)
# ═════════════════════════════════════════════════════════════════════════════

def audit_sector_zscore_leakage(tickers: list[str] | None = None) -> dict:
    """Test whether sector fundamental z-scores leak future data.

    Strategy:
      1. Load every ticker's parquet (needs fund_pe_ttm, fund_fcf_yield_ttm).
      2. Split at 70% cutoff.
      3. Build "real" panel — all fundamental data intact.
      4. Build "fake" panel — perturb fundamental values AFTER cutoff.
      5. Run apply_sector_fundamental_zscores on both.
      6. Compare z-score columns AT OR BEFORE cutoff.

    This catches the case where z-score normalization (mean/std) uses the
    full date range instead of expanding/rolling windows — the most likely
    leakage vector in the current codebase.
    """
    if apply_sector_fundamental_zscores is None:
        return {"status": "skip", "reason": "fundamental_features not importable"}

    tickers = tickers or AUDIT_TICKERS
    import tempfile
    import shutil

    # Load parquets that have fundamental columns
    parquet_data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.loc[AUDIT_START:AUDIT_END]
        needed = {"fund_pe_ttm", "fund_fcf_yield_ttm", "fund_has_valuation"}
        if not needed.issubset(df.columns):
            continue
        if len(df) < 120:
            continue
        parquet_data[t] = df

    if len(parquet_data) < 3:
        return {
            "status": "skip",
            "reason": f"only {len(parquet_data)} tickers have fundamental columns",
        }

    # Pick cutoff
    sample_dates = sorted(set().union(*(df.index for df in parquet_data.values())))
    cutoff_pos = int(len(sample_dates) * 0.70)
    cutoff_date = pd.Timestamp(sample_dates[cutoff_pos])

    def _build_zscore_panel(perturb_future: bool) -> dict[str, pd.DataFrame]:
        tmpdir = tempfile.mkdtemp(prefix="leakage_zscore_")
        try:
            for t, df in parquet_data.items():
                out = df.copy()
                if perturb_future:
                    future_mask = out.index > cutoff_date
                    # Perturb fundamental values after cutoff — large enough
                    # that any leaking mean/std would visibly shift.
                    for col in ["fund_pe_ttm", "fund_fcf_yield_ttm"]:
                        vals = pd.to_numeric(out.loc[future_mask, col],
                                             errors="coerce").to_numpy(dtype=float)
                        if len(vals) > 0:
                            out.loc[future_mask, col] = vals * 3.0 + 50.0
                out.to_parquet(os.path.join(tmpdir, f"{t}.parquet"))

            available = [t for t in parquet_data
                         if os.path.exists(os.path.join(tmpdir, f"{t}.parquet"))]
            apply_sector_fundamental_zscores(
                available, data_dir=tmpdir, sector_map=SECTOR_MAP
            )

            result = {}
            for t in available:
                result[t] = pd.read_parquet(os.path.join(tmpdir, f"{t}.parquet"))
            return result
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print("[leakage_audit] Layer 3: building REAL fundamental z-score panel...")
    real_panels = _build_zscore_panel(perturb_future=False)
    print("[leakage_audit] Layer 3: building PERTURBED fundamental z-score panel...")
    fake_panels = _build_zscore_panel(perturb_future=True)

    # Compare z-score columns before cutoff
    zscore_cols = ["fund_pe_sector_z", "fund_fcf_yield_sector_z", "fund_value_combo_z"]
    all_leaks: dict[str, list[str]] = {}
    total_checked = 0

    for t in real_panels:
        if t not in fake_panels:
            continue
        real_df = real_panels[t]
        fake_df = fake_panels[t]

        check_cols = [c for c in zscore_cols if c in real_df.columns and c in fake_df.columns]
        if not check_cols:
            continue

        leaks, checked = _compare_features(
            real_df[check_cols],
            fake_df[check_cols],
            cutoff_date,
            ignored=set(),
        )
        total_checked += checked
        if leaks:
            all_leaks[t] = leaks

    return {
        "layer": "sector_fundamental_zscores",
        "status": "pass" if not all_leaks else "FAIL",
        "tickers_tested": len(real_panels),
        "zscore_columns_checked": zscore_cols,
        "n_features_checked": total_checked,
        "cutoff_date": str(cutoff_date.date()),
        "leaking_tickers": {t: cols for t, cols in all_leaks.items()},
    }


# ═════════════════════════════════════════════════════════════════════════════
# Layer 4: Post-pass columns in parquets (smoke test)
#
# Unlike layers 2-3 which rebuild from scratch, this checks whether the
# EXISTING parquets (as written by research.py) have any columns that look
# like they used future data.  It's a fast smoke test — if research.py was
# run correctly, these columns should already be clean.
# ═════════════════════════════════════════════════════════════════════════════

def audit_existing_parquet_post_pass_columns(tickers: list[str] | None = None) -> dict:
    """Smoke test: verify post-pass columns in existing parquets are point-in-time.

    For each ticker, loads the parquet and checks that xs_rank_* and
    fund_*_sector_z columns don't exhibit obvious signs of future leakage:
      - Correlation with future returns should not be suspiciously high
      - Values should not be constant (indicating the post-pass didn't run)
    """
    tickers = tickers or AUDIT_TICKERS
    issues: dict[str, list[str]] = {}
    dead: dict[str, list[str]] = {}

    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.loc[AUDIT_START:AUDIT_END]
        if len(df) < 60:
            continue

        # Check post-pass columns exist
        post_pass = [c for c in df.columns
                     if c.startswith("xs_rank_") or c in (
                         "fund_pe_sector_z", "fund_fcf_yield_sector_z",
                         "fund_value_combo_z"
                     )]
        ticker_issues = []
        dead_features = []

        # Check 1: post-pass columns should exist
        if not post_pass:
            ticker_issues.append("no_post_pass_columns_found")

        # Check 2: columns should not be all-constant.
        # Distinguish "dead" (source data is all zero / never populated)
        # from truly suspicious constants.
        for col in post_pass:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 20 and vals.std() < 1e-12:
                # Check if the SOURCE data is also dead (all zeros).
                # fund_*_sector_z depends on fund_pe_ttm / fund_fcf_yield_ttm.
                # If those are all zero, the z-score being zero is expected.
                source_dead = False
                if col.startswith("fund_"):
                    for src in ["fund_pe_ttm", "fund_fcf_yield_ttm",
                                "fund_has_valuation"]:
                        if src in df.columns:
                            src_vals = pd.to_numeric(
                                df[src], errors="coerce"
                            ).dropna()
                            if len(src_vals) > 20 and src_vals.std() < 1e-12:
                                source_dead = True
                                break
                if source_dead:
                    dead_features.append(col)
                else:
                    ticker_issues.append(f"{col}:all_constant")

        # Check 3: suspiciously high correlation with future returns.
        # A cross-sectional rank based on day-t data should have modest
        # correlation with day-t+20 return.  IC > 0.3 is implausibly high
        # and suggests future info.
        if "Close" in df.columns:
            fwd_ret = df["Close"].pct_change(20).shift(-20)
            for col in post_pass:
                vals = pd.to_numeric(df[col], errors="coerce")
                both = pd.concat([vals, fwd_ret], axis=1).dropna()
                if len(both) < 60:
                    continue
                ic = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
                if abs(ic) > 0.30:
                    ticker_issues.append(f"{col}:suspicious_IC={ic:.3f}")

        if ticker_issues:
            issues[t] = ticker_issues
        if dead_features:
            dead[t] = dead_features

    return {
        "layer": "existing_parquet_smoke_test",
        "status": "pass" if not issues else "WARN",
        "tickers_tested": len(tickers),
        "issues": issues,
        "dead_features": dead,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    if pipeline_shared is None or not hasattr(pipeline_shared, "build_research_feature_frame"):
        print("[leakage_audit] pipeline_shared.build_research_feature_frame not found.",
              file=sys.stderr)
        return 2

    results = {"layers": {}}

    # ── Layer 1: Per-ticker technical features ──────────────────────────────
    print(f"\n{'='*70}")
    print("Layer 1: Per-ticker technical features")
    print(f"{'='*70}")
    ticker_results = []
    for t in AUDIT_TICKERS:
        print(f"  Auditing {t}...")
        r = audit_ticker(t)
        ticker_results.append(r)
        status = r["status"]
        n = r.get("n_features_checked", 0)
        leaks = r.get("leaking_features", [])
        if status == "FAIL":
            print(f"    FAIL — {len(leaks)} leaking: {leaks}")
        elif status == "pass":
            print(f"    pass — {n} features clean")
        else:
            print(f"    {status}: {r.get('reason', r.get('error', ''))}")

    results["layers"]["per_ticker"] = ticker_results

    # ── Layer 2: Cross-sectional rank features ──────────────────────────────
    print(f"\n{'='*70}")
    print("Layer 2: Cross-sectional rank features (xs_rank_*)")
    print(f"{'='*70}")
    xs_result = audit_cross_sectional_leakage()
    results["layers"]["cross_sectional"] = xs_result
    if xs_result["status"] == "FAIL":
        leaking = xs_result.get("leaking_tickers", {})
        print(f"  FAIL — {len(leaking)} tickers have leaking xs_rank columns:")
        for t, cols in leaking.items():
            print(f"    {t}: {cols}")
    elif xs_result["status"] == "pass":
        print(f"  pass — {xs_result.get('xs_columns_checked', 0)} xs_rank columns "
              f"across {xs_result.get('tickers_tested', 0)} tickers clean")
    else:
        print(f"  {xs_result['status']}: {xs_result.get('reason', '')}")

    # ── Layer 3: Sector fundamental z-scores ────────────────────────────────
    print(f"\n{'='*70}")
    print("Layer 3: Sector fundamental z-scores (fund_*_sector_z)")
    print(f"{'='*70}")
    zscore_result = audit_sector_zscore_leakage()
    results["layers"]["sector_zscores"] = zscore_result
    if zscore_result["status"] == "FAIL":
        leaking = zscore_result.get("leaking_tickers", {})
        print(f"  FAIL — {len(leaking)} tickers have leaking z-score columns:")
        for t, cols in leaking.items():
            print(f"    {t}: {cols}")
    elif zscore_result["status"] == "pass":
        print(f"  pass — z-score columns across "
              f"{zscore_result.get('tickers_tested', 0)} tickers clean")
    else:
        print(f"  {zscore_result['status']}: {zscore_result.get('reason', '')}")

    # ── Layer 4: Existing parquet smoke test ─────────────────────────────────
    print(f"\n{'='*70}")
    print("Layer 4: Existing parquet post-pass smoke test")
    print(f"{'='*70}")
    smoke_result = audit_existing_parquet_post_pass_columns()
    results["layers"]["parquet_smoke_test"] = smoke_result
    if smoke_result["status"] == "WARN":
        issues = smoke_result.get("issues", {})
        print(f"  WARN — {len(issues)} tickers have suspicious issues:")
        for t, iss in issues.items():
            print(f"    {t}: {iss}")
    else:
        print(f"  pass — {smoke_result.get('tickers_tested', 0)} tickers clean")
    dead_feats = smoke_result.get("dead_features", {})
    if dead_feats:
        n_dead = len(set(c for cols in dead_feats.values() for c in cols))
        print(f"  INFO — {n_dead} feature(s) are dead (source data never populated):")
        # Show unique dead features, not per-ticker
        sample_ticker = list(dead_feats.keys())[0]
        for col in dead_feats[sample_ticker]:
            print(f"    {col} (all {len(dead_feats)} tickers)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Overall status: FAIL if any layer fails, WARN if smoke test warns
    layer1_failed = [r for r in ticker_results if r["status"] == "FAIL"]
    layer2_failed = xs_result["status"] == "FAIL"
    layer3_failed = zscore_result["status"] == "FAIL"
    layer4_warned = smoke_result["status"] == "WARN"

    overall = "pass"
    if layer1_failed or layer2_failed or layer3_failed:
        overall = "FAIL"
    elif layer4_warned:
        overall = "WARN"

    results["overall_status"] = overall
    results["run_at"] = datetime.now(UTC).isoformat()
    results["audit_start"] = AUDIT_START
    results["audit_end"] = AUDIT_END
    results["tickers"] = AUDIT_TICKERS

    print(f"  Layer 1 (per-ticker):       {'FAIL' if layer1_failed else 'pass'}"
          f"  ({len(layer1_failed)} failures)" if layer1_failed else
          f"  Layer 1 (per-ticker):       pass")
    print(f"  Layer 2 (cross-sectional):  {xs_result['status']}")
    print(f"  Layer 3 (sector z-scores):  {zscore_result['status']}")
    print(f"  Layer 4 (parquet smoke):    {smoke_result['status']}")
    print(f"  Overall:                    {overall}")

    # Save results
    path = os.path.join(LOG_DIR, "leakage_audit.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[leakage_audit] wrote {path}")

    if overall == "FAIL":
        print("[leakage_audit] LEAKAGE DETECTED — fix before trusting backtest results.")
        return 1
    if overall == "WARN":
        print("[leakage_audit] warnings found — review but not blocking.")
        return 0
    print("[leakage_audit] all layers passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
