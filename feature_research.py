"""
feature_research.py — Situational analysis of alpha features.

PLAIN ENGLISH:  The existing scripts (feature_quality_diagnostic.py and
feature_ic_report.py) tell you WHETHER a feature predicts returns.  This
script tells you WHERE, WHEN, and UNDER WHAT CONDITIONS it works:

  1. Sector-specific IC   — does ret_5d work in tech but fail in energy?
  2. IC time trend        — is rsi_14 losing its edge recently?
  3. Feature interactions  — which feature pairs are complementary?
  4. Optimal holding period— are we trading the feature at the wrong horizon?
  5. Recent health         — has IC decayed in the last 6 months?
  6. Conditional IC        — does performance change with VIX / yield curve /
                             earnings proximity?

Usage:
    python3 feature_research.py                # analyse top 24 features
    python3 feature_research.py --top 10       # only top 10
    python3 feature_research.py --pairs 10     # limit interaction pairs
    python3 feature_research.py --skip-pairs   # skip expensive pair analysis

Output:
    signals/feature_research_report.json   — full structured report
    signals/feature_research_summary.csv   — one row per feature, flat columns
    Console summary grouped by recommendation category.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

import warnings

import numpy as np
import pandas as pd

# Suppress numpy warnings from Spearman correlations on constant-value
# sectors (e.g. a sector with 3 stocks where all have the same feature value).
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
# Suppress pandas 4.x concat sorting deprecation (harmless — we sort=False).
warnings.filterwarnings("ignore", message="Sorting by default when concatenating")

# ── Reusable IC functions from existing scripts ────────────────────────────
# These do the heavy lifting (daily Spearman IC, rolling stats, regime splits,
# IC decay curves).  No reason to rewrite them — they're already battle-tested.
from feature_quality_diagnostic import (
    compute_daily_ic,       # daily cross-sectional Spearman IC per feature
    rolling_ic_stats,       # hit_rate, stability_ratio over rolling windows
    regime_conditional_ic,  # bull vs bear IC using dist_ma200
    ic_decay_curve,         # IC at 5/10/20/40/60d horizons
)

# Feature list and horizon constant from the alpha backtest pipeline.
from alpha_factor_backtest import load_feature_specs, HORIZON_DAYS

# Project settings — paths, ticker lists, sector mapping.
from settings import DATA_DIR, SECTOR_MAP, SIGNAL_DIR, BROAD_WATCHLIST


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Forward return horizons to compute for holding-period analysis.
FORWARD_HORIZONS = [5, 10, 20]

# Default return column used for IC computation (5-day sector-excess return).
DEFAULT_RETURN_COL = f"fwd_sector_excess_{HORIZON_DAYS}"

# Rolling IC window for recent-health analysis (roughly 6 months of trading).
RECENT_LOOKBACK_DAYS = 126

# Number of equal subperiods for time-trend analysis.
N_SUBPERIODS = 4

# Minimum daily IC observations needed before we trust a result.
MIN_IC_OBS = 30


# ─────────────────────────────────────────────────────────────────────────────
# PANEL LOADING  (mirrors feature_ic_report.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

def load_broad_panel(tickers: Iterable[str]) -> pd.DataFrame:
    """
    Read every ticker's parquet and stack into one tall (date, ticker) frame.

    PLAIN ENGLISH: Each ticker has its own file (data/AAPL.parquet, etc.)
    with one row per trading day.  We glue them all together into a single
    big table so we can do cross-sectional analysis (comparing all stocks
    on the same day).
    """
    frames = []
    skipped = []
    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}.parquet")
        if not os.path.exists(path):
            skipped.append(t)
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            skipped.append(t)
            continue
        df = df.copy()
        df["ticker"] = t
        df["sector"] = SECTOR_MAP.get(t, "OTHER")
        # Some parquets store dates in the index, some in a column.
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df.index)
        else:
            df["date"] = pd.to_datetime(df["date"])
        frames.append(df.reset_index(drop=True))
    if not frames:
        raise RuntimeError("No parquets loaded — run research.py first")
    panel = pd.concat(frames, ignore_index=True)
    print(
        f"[panel] loaded {len(panel):,} rows  "
        f"{panel['ticker'].nunique()} tickers  "
        f"{panel.shape[1]} cols  "
        f"({len(skipped)} skipped)"
    )
    return panel


def add_forward_returns(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Compute forward N-day returns and sector-excess variants.

    PLAIN ENGLISH:
      fwd_ret_5   = how much the stock goes up/down over the next 5 days.
      fwd_sector_excess_5 = same thing, but minus the sector average.
        This isolates stock-picking skill from sector momentum.
    """
    panel = panel.sort_values(["ticker", "date"]).copy()
    for h in horizons:
        # Raw forward return: (price in h days / price today) - 1
        panel[f"fwd_ret_{h}"] = (
            panel.groupby("ticker")["Close"]
            .transform(lambda s: s.shift(-h) / s - 1.0)
        )
        # Cross-sectional excess: subtract the date-mean so we measure
        # stock-specific alpha, not market direction.
        date_mean = panel.groupby("date")[f"fwd_ret_{h}"].transform("mean")
        panel[f"fwd_excess_{h}"] = panel[f"fwd_ret_{h}"] - date_mean
        # Sector excess: subtract sector-mean so we measure within-sector skill.
        sector_mean = panel.groupby(["date", "sector"])[f"fwd_ret_{h}"].transform("mean")
        panel[f"fwd_sector_excess_{h}"] = panel[f"fwd_ret_{h}"] - sector_mean
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1: SECTOR-SPECIFIC IC
# ─────────────────────────────────────────────────────────────────────────────

def sector_specific_ic(
    panel: pd.DataFrame,
    feature: str,
    return_col: str,
    *,
    sector_groups: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """
    Compute IC separately for each sector.

    PLAIN ENGLISH: A feature with aggregate IC = 0.02 might have IC = 0.05
    in tech (XLK) and IC = -0.01 in energy (XLE).  If you only look at the
    aggregate number, you miss that the feature is sector-specific.

    We pass in pre-grouped DataFrames (sector_groups) so we don't re-filter
    the panel for every feature — big speed-up.
    """
    if sector_groups is None:
        sector_groups = dict(list(panel.groupby("sector")))

    sector_ics: dict[str, dict] = {}
    for sector, group in sector_groups.items():
        # Skip sectors with too few rows — IC is meaningless with 3 stocks.
        if group["ticker"].nunique() < 3:
            continue
        if feature not in group.columns or return_col not in group.columns:
            continue
        valid = group[[feature, return_col, "date"]].dropna()
        if len(valid) < MIN_IC_OBS:
            continue
        daily_ic = compute_daily_ic(valid, feature, return_col)
        if len(daily_ic) < MIN_IC_OBS:
            continue
        ic_mean = float(daily_ic.mean())
        ic_std = float(daily_ic.std()) if len(daily_ic) > 1 else 1.0
        t_stat = ic_mean / (ic_std / np.sqrt(len(daily_ic))) if ic_std > 1e-9 else 0.0
        hit_rate = float((daily_ic > 0).mean())
        sector_ics[sector] = {
            "ic": round(ic_mean, 6),
            "t_stat": round(t_stat, 2),
            "hit_rate": round(hit_rate, 4),
            "n_days": len(daily_ic),
        }

    if not sector_ics:
        return {
            "sector_ics": {},
            "best_sector": None,
            "worst_sector": None,
            "sector_dispersion": 0.0,
            "sector_specific": False,
        }

    # Find best / worst sectors by absolute IC.
    best = max(sector_ics.items(), key=lambda kv: kv[1]["ic"])
    worst = min(sector_ics.items(), key=lambda kv: kv[1]["ic"])
    ics = [v["ic"] for v in sector_ics.values()]
    dispersion = float(np.std(ics)) if len(ics) > 1 else 0.0

    # A feature is "sector specific" if the spread between best and worst
    # is more than 0.02 (= meaningful IC difference).
    sector_specific = bool((best[1]["ic"] - worst[1]["ic"]) > 0.02)

    return {
        "sector_ics": sector_ics,
        "best_sector": best[0],
        "worst_sector": worst[0],
        "best_sector_ic": best[1]["ic"],
        "worst_sector_ic": worst[1]["ic"],
        "sector_dispersion": round(dispersion, 6),
        "sector_specific": sector_specific,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: IC TIME TREND
# ─────────────────────────────────────────────────────────────────────────────

def ic_time_trend(daily_ic: pd.Series, n_subperiods: int = N_SUBPERIODS) -> dict:
    """
    Is the feature getting stronger or weaker over time?

    PLAIN ENGLISH: We chop the IC time series into 4 equal chunks (e.g.
    2010-2013, 2014-2016, 2017-2019, 2020-2023) and check whether the
    average IC in each chunk is trending up or down.  A negative slope
    means the feature is losing its edge — maybe the market adapted.
    """
    if len(daily_ic) < n_subperiods * MIN_IC_OBS:
        return {
            "subperiods": [],
            "trend_slope": 0.0,
            "recent_vs_full_ratio": 1.0,
            "trending_down": False,
        }

    # Split into n equal pieces.
    chunk_size = len(daily_ic) // n_subperiods
    subperiods = []
    means = []
    for i in range(n_subperiods):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_subperiods - 1 else len(daily_ic)
        chunk = daily_ic.iloc[start:end]
        mean_ic = float(chunk.mean())
        means.append(mean_ic)
        subperiods.append({
            "period": i + 1,
            "start": str(chunk.index[0]) if hasattr(chunk.index[0], 'strftime') else str(chunk.index[0]),
            "end": str(chunk.index[-1]) if hasattr(chunk.index[-1], 'strftime') else str(chunk.index[-1]),
            "mean_ic": round(mean_ic, 6),
            "n_days": len(chunk),
        })

    # Fit a linear slope to the subperiod means.
    # Positive slope = feature is strengthening over time (good).
    # Negative slope = feature is losing its edge (bad).
    x = np.arange(n_subperiods, dtype=float)
    y = np.array(means)
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0

    # Compare the last subperiod's IC to the overall average.
    full_mean = float(daily_ic.mean())
    recent_ratio = means[-1] / full_mean if abs(full_mean) > 1e-9 else 1.0

    return {
        "subperiods": subperiods,
        "trend_slope": round(slope, 6),
        "recent_vs_full_ratio": round(recent_ratio, 4),
        "trending_down": bool(slope < -0.001),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3: PAIRWISE INTERACTION IC
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_interaction_ic(
    panel: pd.DataFrame,
    features: list[str],
    return_col: str,
    *,
    max_pairs: int = 15,
) -> list[dict]:
    """
    Do any feature pairs predict better together than alone?

    PLAIN ENGLISH: If feature A ranks stocks 1-10 and feature B ranks them
    1-10, we create a combo rank (rank_A + rank_B).  If the combo has
    higher IC than either individual feature, the pair is SYNERGISTIC —
    they capture different information about future returns.

    We also check how correlated the two IC series are.  Low correlation
    means the features fire at different times (complementary).
    """
    # Pre-compute individual ICs and daily IC series for each feature.
    feature_ics: dict[str, float] = {}
    feature_ic_series: dict[str, pd.Series] = {}
    for f in features:
        if f not in panel.columns:
            continue
        valid = panel[[f, return_col, "date"]].dropna()
        if len(valid) < MIN_IC_OBS:
            continue
        daily_ic = compute_daily_ic(valid, f, return_col)
        if len(daily_ic) < MIN_IC_OBS:
            continue
        feature_ics[f] = float(daily_ic.mean())
        feature_ic_series[f] = daily_ic

    # Only consider features that have valid IC.
    valid_features = sorted(feature_ics.keys(), key=lambda f: abs(feature_ics[f]), reverse=True)
    # Limit to top features to keep runtime reasonable.
    valid_features = valid_features[:max_pairs]

    results = []
    for feat_a, feat_b in combinations(valid_features, 2):
        # Combo rank: rank(A) + rank(B).  Lower combined rank = stronger
        # signal in both features simultaneously.
        subset = panel[[feat_a, feat_b, return_col, "date"]].dropna()
        if len(subset) < MIN_IC_OBS:
            continue
        # Rank within each date — this is the cross-sectional rank combo.
        subset = subset.copy()
        subset["_combo_rank"] = (
            subset.groupby("date")[feat_a].rank(pct=True)
            + subset.groupby("date")[feat_b].rank(pct=True)
        )
        combo_ic_series = compute_daily_ic(subset, "_combo_rank", return_col)
        if len(combo_ic_series) < MIN_IC_OBS:
            continue
        combo_ic = float(combo_ic_series.mean())
        best_individual = max(abs(feature_ics.get(feat_a, 0)), abs(feature_ics.get(feat_b, 0)))
        # Synergy = how much better the combo is vs the best individual.
        synergy = abs(combo_ic) - best_individual if best_individual > 0 else 0.0

        # IC series correlation — low means the features fire at different
        # times, which is exactly what you want for diversification.
        ic_a = feature_ic_series.get(feat_a, pd.Series(dtype=float))
        ic_b = feature_ic_series.get(feat_b, pd.Series(dtype=float))
        aligned = pd.concat([ic_a.rename("a"), ic_b.rename("b")], axis=1).dropna()
        ic_corr = float(aligned["a"].corr(aligned["b"])) if len(aligned) > 30 else 0.0

        results.append({
            "feat_a": feat_a,
            "feat_b": feat_b,
            "combo_ic": round(combo_ic, 6),
            "ic_a": round(feature_ics.get(feat_a, 0.0), 6),
            "ic_b": round(feature_ics.get(feat_b, 0.0), 6),
            "best_individual_ic": round(best_individual, 6),
            "synergy": round(synergy, 6),
            "ic_correlation": round(ic_corr, 4),
            # Complementary = combo beats either individual AND IC series
            # are not highly correlated (they capture different info).
            "complementary": bool(synergy > 0.001 and ic_corr < 0.50),
        })

    # Sort by synergy (best combos first).
    results.sort(key=lambda r: r["synergy"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4: OPTIMAL HOLDING PERIOD
# ─────────────────────────────────────────────────────────────────────────────

def optimal_holding_period(
    panel: pd.DataFrame,
    feature: str,
    horizons: list[int] = FORWARD_HORIZONS,
) -> dict:
    """
    Is the feature being used at its best horizon?

    PLAIN ENGLISH: Some features predict 5-day returns well but not 20-day.
    Others are slow-burn signals that only show up at 20 days.  If we're
    using a feature at the 5-day horizon but it peaks at 20 days, we're
    leaving money on the table (or worse, trading noise).
    """
    horizon_ics: dict[int, float] = {}
    for h in horizons:
        # Look for sector-excess return at this horizon.
        ret_col = f"fwd_sector_excess_{h}"
        if ret_col not in panel.columns:
            continue
        valid = panel[[feature, ret_col, "date"]].dropna()
        if len(valid) < MIN_IC_OBS:
            continue
        daily_ic = compute_daily_ic(valid, feature, ret_col)
        if len(daily_ic) < MIN_IC_OBS:
            continue
        horizon_ics[h] = float(daily_ic.mean())

    if not horizon_ics:
        return {
            "optimal_horizon": HORIZON_DAYS,
            "current_horizon": HORIZON_DAYS,
            "ic_at_current": 0.0,
            "ic_at_optimal": 0.0,
            "horizon_mismatch": False,
            "horizon_ics": {},
        }

    # Find the horizon with the strongest IC (by absolute value).
    optimal_h = max(horizon_ics.items(), key=lambda kv: abs(kv[1]))
    current_ic = horizon_ics.get(HORIZON_DAYS, 0.0)
    optimal_ic = optimal_h[1]

    # Mismatch = the current horizon is noticeably worse than the optimal.
    mismatch = bool(
        optimal_h[0] != HORIZON_DAYS
        and abs(optimal_ic) > abs(current_ic) * 1.3  # >30% better
    )

    return {
        "optimal_horizon": int(optimal_h[0]),
        "current_horizon": HORIZON_DAYS,
        "ic_at_current": round(current_ic, 6),
        "ic_at_optimal": round(optimal_ic, 6),
        "horizon_mismatch": mismatch,
        "horizon_ics": {str(h): round(ic, 6) for h, ic in sorted(horizon_ics.items())},
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 5: RECENT IC HEALTH
# ─────────────────────────────────────────────────────────────────────────────

def recent_ic_health(daily_ic: pd.Series, lookback: int = RECENT_LOOKBACK_DAYS) -> dict:
    """
    Has the feature's IC decayed in the last 6 months?

    PLAIN ENGLISH: A feature might have great historical IC because of
    one or two golden periods years ago.  If the last 6 months' IC is
    much lower (or negative), the feature is DECAYING — maybe the market
    adapted, or the regime changed.

    We flag features where recent IC is less than 50% of the full-sample IC.
    """
    if len(daily_ic) < lookback:
        return {
            "recent_ic": 0.0,
            "full_ic": 0.0,
            "recent_t_stat": 0.0,
            "recent_vs_full_ratio": 1.0,
            "decaying": False,
            "strengthening": False,
        }

    recent = daily_ic.iloc[-lookback:]
    full_ic = float(daily_ic.mean())
    recent_ic = float(recent.mean())
    recent_std = float(recent.std()) if len(recent) > 1 else 1.0
    recent_t = recent_ic / (recent_std / np.sqrt(len(recent))) if recent_std > 1e-9 else 0.0

    ratio = recent_ic / full_ic if abs(full_ic) > 1e-9 else 1.0

    return {
        "recent_ic": round(recent_ic, 6),
        "full_ic": round(full_ic, 6),
        "recent_t_stat": round(recent_t, 2),
        "recent_vs_full_ratio": round(ratio, 4),
        # Decaying = recent IC is less than half the full-sample IC.
        "decaying": bool(ratio < 0.50 and abs(full_ic) > 0.002),
        # Strengthening = recent IC is more than 1.5x the full-sample IC.
        "strengthening": bool(ratio > 1.50 and abs(recent_ic) > 0.002),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 6: CONDITIONAL IC (VIX, YIELD CURVE, EARNINGS)
# ─────────────────────────────────────────────────────────────────────────────

def conditional_ic(
    panel: pd.DataFrame,
    feature: str,
    return_col: str,
) -> dict:
    """
    Does the feature's predictive power change with market conditions?

    PLAIN ENGLISH: Three conditions are tested:
      - VIX regime: does the feature work better in calm or volatile markets?
      - Yield curve: does it work differently when the yield curve is
        inverted (recession signal) vs normal?
      - Earnings proximity: does IC change near earnings announcements
        (when stock-specific info dominates)?

    A feature that only works in low-VIX environments is risky because
    you need it most during high-VIX (crisis) periods.
    """
    result: dict = {}

    # ── VIX regime (terciles: low / medium / high) ─────────────────────
    if "vix_percentile" in panel.columns:
        vix = panel["vix_percentile"].dropna()
        if len(vix) > MIN_IC_OBS:
            # Split into terciles: low (<33rd pctile), high (>66th pctile).
            low_thresh = vix.quantile(0.33)
            high_thresh = vix.quantile(0.66)
            low_mask = panel["vix_percentile"] <= low_thresh
            high_mask = panel["vix_percentile"] >= high_thresh
            low_panel = panel.loc[low_mask]
            high_panel = panel.loc[high_mask]
            low_ic = _safe_mean_ic(low_panel, feature, return_col)
            high_ic = _safe_mean_ic(high_panel, feature, return_col)
            result["vix"] = {
                "low_vol_ic": round(low_ic, 6),
                "high_vol_ic": round(high_ic, 6),
                "spread": round(abs(low_ic - high_ic), 6),
            }
        else:
            result["vix"] = {"low_vol_ic": 0.0, "high_vol_ic": 0.0, "spread": 0.0}
    else:
        result["vix"] = {"low_vol_ic": 0.0, "high_vol_ic": 0.0, "spread": 0.0}

    # ── Yield curve (normal vs inverted) ───────────────────────────────
    if "macro_yield_curve_10y_3m" in panel.columns:
        yc = panel["macro_yield_curve_10y_3m"].dropna()
        if len(yc) > MIN_IC_OBS:
            normal_panel = panel.loc[panel["macro_yield_curve_10y_3m"] > 0]
            inverted_panel = panel.loc[panel["macro_yield_curve_10y_3m"] <= 0]
            normal_ic = _safe_mean_ic(normal_panel, feature, return_col)
            inverted_ic = _safe_mean_ic(inverted_panel, feature, return_col)
            result["yield_curve"] = {
                "normal_ic": round(normal_ic, 6),
                "inverted_ic": round(inverted_ic, 6),
                "spread": round(abs(normal_ic - inverted_ic), 6),
            }
        else:
            result["yield_curve"] = {"normal_ic": 0.0, "inverted_ic": 0.0, "spread": 0.0}
    else:
        result["yield_curve"] = {"normal_ic": 0.0, "inverted_ic": 0.0, "spread": 0.0}

    # ── Earnings proximity (near <10d vs far >30d) ─────────────────────
    if "days_to_next_earnings" in panel.columns:
        near_mask = panel["days_to_next_earnings"] <= 10
        far_mask = panel["days_to_next_earnings"] > 30
        near_panel = panel.loc[near_mask]
        far_panel = panel.loc[far_mask]
        near_ic = _safe_mean_ic(near_panel, feature, return_col) if len(near_panel) > MIN_IC_OBS else 0.0
        far_ic = _safe_mean_ic(far_panel, feature, return_col) if len(far_panel) > MIN_IC_OBS else 0.0
        result["earnings"] = {
            "near_ic": round(near_ic, 6),
            "far_ic": round(far_ic, 6),
            "spread": round(abs(near_ic - far_ic), 6),
        }
    else:
        result["earnings"] = {"near_ic": 0.0, "far_ic": 0.0, "spread": 0.0}

    # Find which condition causes the most IC variation.
    spreads = {k: v.get("spread", 0.0) for k, v in result.items()}
    most_variable = max(spreads, key=spreads.get) if spreads else "none"
    result["most_variable_condition"] = most_variable
    # A feature is "conditional" if any condition spread exceeds 0.01.
    result["conditional"] = bool(max(spreads.values(), default=0.0) > 0.01)

    return result


def _safe_mean_ic(sub_panel: pd.DataFrame, feature: str, return_col: str) -> float:
    """Compute mean IC on a sub-panel, returning 0.0 if data is insufficient."""
    if feature not in sub_panel.columns or return_col not in sub_panel.columns:
        return 0.0
    valid = sub_panel[[feature, return_col, "date"]].dropna()
    if len(valid) < MIN_IC_OBS:
        return 0.0
    daily_ic = compute_daily_ic(valid, feature, return_col)
    return float(daily_ic.mean()) if len(daily_ic) > 10 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_research(
    panel: pd.DataFrame,
    features: list[str],
    return_col: str = DEFAULT_RETURN_COL,
    *,
    skip_pairs: bool = False,
    max_pairs: int = 15,
) -> dict:
    """
    Run all 6 analyses on each feature and return a structured report.

    PLAIN ENGLISH: This is the main function.  It loops through every
    feature, runs the 6 analyses above, and collects everything into
    a single JSON-friendly dict.
    """
    t0 = time.time()

    # Pre-group panel by sector once (big speed-up for sector IC analysis).
    sector_groups = dict(list(panel.groupby("sector")))

    feature_reports: list[dict] = []
    daily_ics: dict[str, pd.Series] = {}  # cache for time-trend / recent-health

    print(f"\nAnalysing {len(features)} features against {return_col}...")
    for i, feature in enumerate(features, 1):
        if feature not in panel.columns:
            print(f"  [{i}/{len(features)}] {feature} — SKIP (not in panel)")
            continue

        valid = panel[[feature, return_col, "date"]].dropna()
        if len(valid) < MIN_IC_OBS:
            print(f"  [{i}/{len(features)}] {feature} — SKIP (too few obs: {len(valid)})")
            continue

        # Compute daily IC once — reused by multiple analyses.
        daily_ic = compute_daily_ic(valid, feature, return_col)
        if len(daily_ic) < MIN_IC_OBS:
            print(f"  [{i}/{len(features)}] {feature} — SKIP (too few IC days: {len(daily_ic)})")
            continue
        daily_ics[feature] = daily_ic

        # Aggregate IC stats for context.
        agg_ic = float(daily_ic.mean())
        agg_std = float(daily_ic.std()) if len(daily_ic) > 1 else 1.0
        agg_t = agg_ic / (agg_std / np.sqrt(len(daily_ic))) if agg_std > 1e-9 else 0.0

        # ── Run all 6 analyses ────────────────────────────────────────
        report = {
            "feature": feature,
            "aggregate_ic": round(agg_ic, 6),
            "aggregate_t_stat": round(agg_t, 2),
            "n_ic_days": len(daily_ic),
            "sector_ic": sector_specific_ic(panel, feature, return_col, sector_groups=sector_groups),
            "time_trend": ic_time_trend(daily_ic),
            "holding_period": optimal_holding_period(panel, feature),
            "recent_health": recent_ic_health(daily_ic),
            "conditional_ic": conditional_ic(panel, feature, return_col),
            "regime_ic": regime_conditional_ic(panel, feature, return_col),
        }
        feature_reports.append(report)

        # Progress print.
        flags = []
        if report["sector_ic"]["sector_specific"]:
            flags.append("SECTOR")
        if report["time_trend"]["trending_down"]:
            flags.append("DECAY↓")
        if report["holding_period"]["horizon_mismatch"]:
            flags.append("HORIZON!")
        if report["recent_health"]["decaying"]:
            flags.append("RECENT↓")
        if report["conditional_ic"]["conditional"]:
            flags.append("COND")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"  [{i}/{len(features)}] {feature}: "
            f"IC={agg_ic:+.4f} t={agg_t:.1f}{flag_str}"
        )

    # ── Analysis 3: Pairwise interactions (optional, expensive) ────────
    interaction_pairs: list[dict] = []
    if not skip_pairs and len(daily_ics) >= 2:
        print(f"\nComputing pairwise interactions (top {max_pairs} features)...")
        interaction_pairs = pairwise_interaction_ic(
            panel, list(daily_ics.keys()), return_col, max_pairs=max_pairs,
        )
        complementary = [p for p in interaction_pairs if p["complementary"]]
        print(f"  {len(interaction_pairs)} pairs tested, {len(complementary)} complementary")

    elapsed = time.time() - t0

    # ── Build recommendations summary ──────────────────────────────────
    recommendations = _build_recommendations(feature_reports, interaction_pairs)

    return {
        "n_features": len(feature_reports),
        "return_col": return_col,
        "n_tickers": int(panel["ticker"].nunique()),
        "n_dates": int(panel["date"].nunique()),
        "elapsed_seconds": round(elapsed, 1),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": feature_reports,
        "interaction_pairs": interaction_pairs,
        "recommendations": recommendations,
    }


def _build_recommendations(
    reports: list[dict],
    pairs: list[dict],
) -> dict:
    """
    Group features by recommendation category for the console summary.

    PLAIN ENGLISH: Instead of dumping raw numbers, we bucket features into
    actionable categories so you know what to investigate or change.
    """
    recs: dict[str, list] = {
        "sector_specific": [],
        "decaying": [],
        "horizon_mismatch": [],
        "conditional": [],
        "strengthening": [],
        "synergistic_pairs": [],
    }

    for r in reports:
        f = r["feature"]
        if r["sector_ic"]["sector_specific"]:
            recs["sector_specific"].append({
                "feature": f,
                "best_sector": r["sector_ic"]["best_sector"],
                "best_ic": r["sector_ic"].get("best_sector_ic", 0),
                "worst_sector": r["sector_ic"]["worst_sector"],
                "worst_ic": r["sector_ic"].get("worst_sector_ic", 0),
            })
        if r["recent_health"]["decaying"]:
            recs["decaying"].append({
                "feature": f,
                "recent_ic": r["recent_health"]["recent_ic"],
                "full_ic": r["recent_health"]["full_ic"],
                "ratio": r["recent_health"]["recent_vs_full_ratio"],
            })
        if r["holding_period"]["horizon_mismatch"]:
            recs["horizon_mismatch"].append({
                "feature": f,
                "current": r["holding_period"]["current_horizon"],
                "optimal": r["holding_period"]["optimal_horizon"],
                "ic_current": r["holding_period"]["ic_at_current"],
                "ic_optimal": r["holding_period"]["ic_at_optimal"],
            })
        if r["conditional_ic"]["conditional"]:
            recs["conditional"].append({
                "feature": f,
                "most_variable": r["conditional_ic"]["most_variable_condition"],
                "vix_spread": r["conditional_ic"]["vix"].get("spread", 0),
            })
        if r["recent_health"]["strengthening"]:
            recs["strengthening"].append({
                "feature": f,
                "recent_ic": r["recent_health"]["recent_ic"],
                "full_ic": r["recent_health"]["full_ic"],
                "ratio": r["recent_health"]["recent_vs_full_ratio"],
            })

    for p in pairs:
        if p["complementary"]:
            recs["synergistic_pairs"].append(p)

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    """Handle numpy / pandas types that json.dump can't serialise."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def write_outputs(report: dict, output_dir: str = SIGNAL_DIR) -> tuple[Path, Path]:
    """
    Write the report to JSON and a flat CSV summary.

    PLAIN ENGLISH: The JSON has the full detailed report (nested dicts).
    The CSV has one row per feature with flat columns — easy to open in
    Excel or sort in a terminal with csvlook.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = Path(output_dir) / "feature_research_report.json"
    csv_path = Path(output_dir) / "feature_research_summary.csv"

    # ── JSON ───────────────────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"\n  JSON report: {json_path}")

    # ── Flat CSV (one row per feature) ─────────────────────────────────
    rows = []
    for feat in report.get("features", []):
        row = {
            "feature": feat["feature"],
            "aggregate_ic": feat["aggregate_ic"],
            "aggregate_t_stat": feat["aggregate_t_stat"],
            "n_ic_days": feat["n_ic_days"],
            # Sector IC
            "sector_specific": feat["sector_ic"]["sector_specific"],
            "best_sector": feat["sector_ic"]["best_sector"],
            "best_sector_ic": feat["sector_ic"].get("best_sector_ic", ""),
            "worst_sector": feat["sector_ic"]["worst_sector"],
            "worst_sector_ic": feat["sector_ic"].get("worst_sector_ic", ""),
            "sector_dispersion": feat["sector_ic"]["sector_dispersion"],
            # Time trend
            "trend_slope": feat["time_trend"]["trend_slope"],
            "trending_down": feat["time_trend"]["trending_down"],
            "recent_vs_full_trend": feat["time_trend"]["recent_vs_full_ratio"],
            # Holding period
            "optimal_horizon": feat["holding_period"]["optimal_horizon"],
            "current_horizon": feat["holding_period"]["current_horizon"],
            "horizon_mismatch": feat["holding_period"]["horizon_mismatch"],
            "ic_at_current": feat["holding_period"]["ic_at_current"],
            "ic_at_optimal": feat["holding_period"]["ic_at_optimal"],
            # Recent health
            "recent_ic": feat["recent_health"]["recent_ic"],
            "full_ic": feat["recent_health"]["full_ic"],
            "recent_t_stat": feat["recent_health"]["recent_t_stat"],
            "decaying": feat["recent_health"]["decaying"],
            "strengthening": feat["recent_health"]["strengthening"],
            # Conditional IC
            "conditional": feat["conditional_ic"]["conditional"],
            "most_variable_condition": feat["conditional_ic"]["most_variable_condition"],
            "vix_low_ic": feat["conditional_ic"]["vix"]["low_vol_ic"],
            "vix_high_ic": feat["conditional_ic"]["vix"]["high_vol_ic"],
            "yield_normal_ic": feat["conditional_ic"]["yield_curve"]["normal_ic"],
            "yield_inverted_ic": feat["conditional_ic"]["yield_curve"]["inverted_ic"],
            "earnings_near_ic": feat["conditional_ic"]["earnings"]["near_ic"],
            "earnings_far_ic": feat["conditional_ic"]["earnings"]["far_ic"],
            # Regime IC
            "bull_ic": feat["regime_ic"]["bull_ic"],
            "bear_ic": feat["regime_ic"]["bear_ic"],
            "regime_stable": feat["regime_ic"]["regime_stable"],
        }
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV summary: {csv_path}")

    return json_path, csv_path


def print_console_summary(report: dict) -> None:
    """
    Print a grouped summary to the terminal.

    PLAIN ENGLISH: Instead of scrolling through 24 features × 6 analyses,
    this prints only the ACTIONABLE findings — things you might want to
    change in your strategy.
    """
    recs = report.get("recommendations", {})
    print("\n" + "=" * 72)
    print("  FEATURE RESEARCH SUMMARY")
    print("=" * 72)
    print(f"  {report['n_features']} features | {report['n_tickers']} tickers | "
          f"{report['n_dates']} dates | {report['elapsed_seconds']:.0f}s")
    print()

    # ── SECTOR-SPECIFIC ────────────────────────────────────────────────
    items = recs.get("sector_specific", [])
    if items:
        print(f"  SECTOR-SPECIFIC ({len(items)} features only work in some sectors):")
        for item in items:
            print(f"    {item['feature']:40s}  best={item['best_sector']}({item['best_ic']:+.4f})  "
                  f"worst={item['worst_sector']}({item['worst_ic']:+.4f})")
        print()

    # ── DECAYING ───────────────────────────────────────────────────────
    items = recs.get("decaying", [])
    if items:
        print(f"  ⚠ DECAYING ({len(items)} features losing predictive power):")
        for item in items:
            print(f"    {item['feature']:40s}  recent={item['recent_ic']:+.4f}  "
                  f"full={item['full_ic']:+.4f}  ratio={item['ratio']:.2f}")
        print()

    # ── STRENGTHENING ──────────────────────────────────────────────────
    items = recs.get("strengthening", [])
    if items:
        print(f"  STRENGTHENING ({len(items)} features gaining power):")
        for item in items:
            print(f"    {item['feature']:40s}  recent={item['recent_ic']:+.4f}  "
                  f"full={item['full_ic']:+.4f}  ratio={item['ratio']:.2f}")
        print()

    # ── HORIZON MISMATCH ───────────────────────────────────────────────
    items = recs.get("horizon_mismatch", [])
    if items:
        print(f"  HORIZON MISMATCH ({len(items)} features used at wrong holding period):")
        for item in items:
            print(f"    {item['feature']:40s}  current={item['current']}d(IC={item['ic_current']:+.4f})  "
                  f"optimal={item['optimal']}d(IC={item['ic_optimal']:+.4f})")
        print()

    # ── CONDITIONAL ────────────────────────────────────────────────────
    items = recs.get("conditional", [])
    if items:
        print(f"  CONDITIONAL ({len(items)} features depend on market conditions):")
        for item in items:
            print(f"    {item['feature']:40s}  most_variable={item['most_variable']}  "
                  f"vix_spread={item['vix_spread']:.4f}")
        print()

    # ── SYNERGISTIC PAIRS ──────────────────────────────────────────────
    items = recs.get("synergistic_pairs", [])
    if items:
        print(f"  SYNERGISTIC PAIRS ({len(items)} complementary feature pairs):")
        for item in items[:10]:  # limit output
            print(f"    {item['feat_a']:20s} + {item['feat_b']:20s}  "
                  f"combo_ic={item['combo_ic']:+.4f}  synergy={item['synergy']:+.4f}  "
                  f"corr={item['ic_correlation']:.2f}")
        print()

    # ── CLEAN FEATURES (no flags) ──────────────────────────────────────
    flagged = set()
    for category in ("sector_specific", "decaying", "horizon_mismatch", "conditional", "strengthening"):
        for item in recs.get(category, []):
            flagged.add(item["feature"])
    all_features = [f["feature"] for f in report.get("features", [])]
    clean = [f for f in all_features if f not in flagged]
    if clean:
        print(f"  STABLE ({len(clean)} features with no flags — working as expected):")
        for f in clean:
            feat = next((r for r in report["features"] if r["feature"] == f), None)
            if feat:
                print(f"    {f:40s}  IC={feat['aggregate_ic']:+.4f}  t={feat['aggregate_t_stat']:.1f}")
        print()

    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Situational analysis of alpha features — sector, time, regime, interactions"
    )
    parser.add_argument("--top", type=int, default=24,
                        help="Number of top features to analyse (default: 24)")
    parser.add_argument("--pairs", type=int, default=15,
                        help="Max features to include in pairwise interaction analysis (default: 15)")
    parser.add_argument("--skip-pairs", action="store_true",
                        help="Skip pairwise interaction analysis (saves ~3 min)")
    args = parser.parse_args()

    print("=" * 72)
    print("  feature_research.py — Situational Alpha Feature Analysis")
    print("=" * 72)

    # ── Load the broad 147-ticker panel ────────────────────────────────
    # PLAIN ENGLISH: We use the full 147-ticker universe (not the smaller
    # 42-ticker watchlist) because sector-level analysis needs enough
    # stocks per sector to produce meaningful IC numbers.
    print("\n[1/3] Loading broad panel...")
    panel = load_broad_panel(BROAD_WATCHLIST)

    # ── Compute forward returns at multiple horizons ───────────────────
    print("[2/3] Computing forward returns...")
    panel = add_forward_returns(panel, FORWARD_HORIZONS)

    # Downcast float64 → float32 to save memory (~50% less RAM).
    for col in panel.select_dtypes(include=["float64"]).columns:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("float32")

    # ── Get feature list from the alpha backtest pipeline ──────────────
    print("[3/3] Loading feature specs...")
    specs = load_feature_specs(max_specs=int(args.top))
    features = [s["feature"] for s in specs if s.get("feature") in panel.columns]
    if not features:
        print("ERROR: No features found in panel. Run research.py first.")
        return 1
    print(f"  {len(features)} features to analyse")

    # ── Run the analysis ───────────────────────────────────────────────
    report = run_feature_research(
        panel,
        features,
        skip_pairs=bool(args.skip_pairs),
        max_pairs=int(args.pairs),
    )

    # ── Write outputs ──────────────────────────────────────────────────
    write_outputs(report)
    print_console_summary(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
