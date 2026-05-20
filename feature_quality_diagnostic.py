"""
feature_quality_diagnostic.py — Deep analysis of alpha factor quality.

This script addresses the "shallow predictive edge" problem by computing:
1. Rolling IC stability (does the factor work consistently or just in bursts?)
2. IC decay by horizon (does the signal persist or is it just noise?)
3. Regime-conditional IC (does the factor weaken in bull or bear markets?)
4. Feature correlation clusters (are we double-counting the same signal?)
5. Turnover-adjusted IC (after transaction costs, is there still edge?)

A professional quant system needs to know not just WHETHER a factor predicts
returns, but HOW STABLE and HOW ROBUST that prediction is across time and
market conditions.

Usage:
    python3 feature_quality_diagnostic.py
    python3 feature_quality_diagnostic.py --top 10  # only analyze top 10 factors

Output:
    signals/feature_quality_report.json  — full diagnostic report
    signals/feature_quality_summary.csv  — per-feature quality grades
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_backtest import (
    load_factor_panel,
    load_feature_specs,
    HORIZON_DAYS,
)
from settings import SIGNAL_DIR


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Rolling IC window sizes (trading days)
IC_ROLLING_WINDOWS = [63, 126, 252]  # ~3mo, 6mo, 1yr

# Minimum fraction of rolling windows where IC must be positive
MIN_IC_HIT_RATE = 0.55  # at least 55% of rolling windows show positive IC

# Minimum rolling IC stability (IC_mean / IC_std across windows)
MIN_IC_STABILITY_RATIO = 0.5  # IC should be at least half its std

# Feature correlation threshold for clustering
CORRELATION_CLUSTER_THRESHOLD = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# IC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_daily_ic(panel: pd.DataFrame, feature_col: str, return_col: str) -> pd.Series:
    """
    Compute daily cross-sectional Spearman IC for a factor.

    PLAIN ENGLISH: For each day, rank all stocks by the factor and by their
    future returns. The correlation between these rankings is the IC
    (Information Coefficient). High IC means the factor successfully predicts
    which stocks will outperform.

    Returns a Series indexed by date with one IC value per day.
    """
    def _daily_spearman(group):
        clean = group[[feature_col, return_col]].dropna()
        if len(clean) < 5:
            return np.nan
        if clean[feature_col].nunique() < 2 or clean[return_col].nunique() < 2:
            return np.nan
        x = clean[feature_col].rank()
        y = clean[return_col].rank()
        return float(x.corr(y))

    daily_ic = panel.groupby("date").apply(_daily_spearman)
    return daily_ic.dropna()


def add_oriented_feature(panel: pd.DataFrame, feature_col: str, preferred_direction: str) -> str:
    """Add a temporary feature column where higher values always mean better."""
    oriented_col = f"__quality_oriented_{feature_col}"
    values = pd.to_numeric(panel[feature_col], errors="coerce")
    if str(preferred_direction) == "low_is_good":
        values = -values
    panel[oriented_col] = values
    return oriented_col


def add_sector_excess_return_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add sector-excess forward-return columns matching the feature shortlist target."""
    out = panel.copy()
    horizons = {
        HORIZON_DAYS: "forward_return",
        10: "forward_return_10d",
        20: "forward_return_20d",
    }
    if "sector" not in out.columns:
        return out
    for horizon, raw_col in horizons.items():
        if raw_col not in out.columns:
            continue
        target_col = f"fwd_sector_excess_{horizon}"
        if target_col in out.columns:
            continue
        raw = pd.to_numeric(out[raw_col], errors="coerce")
        sector_mean = raw.groupby([out["date"], out["sector"]]).transform("mean")
        out[target_col] = raw - sector_mean
    return out


def rolling_ic_stats(daily_ic: pd.Series, window: int) -> dict:
    """
    Compute rolling IC statistics to measure signal stability.

    PLAIN ENGLISH: Instead of looking at the average IC over 16 years
    (which hides regime changes), we compute IC in rolling windows.
    A stable factor should have positive IC in most windows, not just
    a few lucky periods that inflate the long-run average.
    """
    rolling_mean = daily_ic.rolling(window, min_periods=window // 2).mean()
    rolling_std = daily_ic.rolling(window, min_periods=window // 2).std()

    positive_windows = (rolling_mean > 0).sum()
    total_windows = rolling_mean.notna().sum()
    hit_rate = positive_windows / total_windows if total_windows > 0 else 0

    # IC stability ratio: mean of rolling ICs / std of rolling ICs
    mean_of_rolling = float(rolling_mean.mean()) if not rolling_mean.empty else 0
    std_of_rolling = float(rolling_mean.std()) if not rolling_mean.empty else 1
    stability = mean_of_rolling / std_of_rolling if std_of_rolling > 1e-9 else 0

    # Worst rolling window (maximum drawdown of IC)
    worst_window = float(rolling_mean.min()) if not rolling_mean.empty else 0

    return {
        "window_days": window,
        "hit_rate": round(hit_rate, 4),
        "mean_rolling_ic": round(mean_of_rolling, 6),
        "std_rolling_ic": round(std_of_rolling, 6),
        "stability_ratio": round(stability, 4),
        "worst_window_ic": round(worst_window, 6),
        "positive_windows": int(positive_windows),
        "total_windows": int(total_windows),
    }


def regime_conditional_ic(
    panel: pd.DataFrame, feature_col: str, return_col: str
) -> dict:
    """
    Compute IC separately for bull vs bear regimes.

    PLAIN ENGLISH: A factor that only works in bull markets is dangerous —
    when the market crashes (exactly when you need protection), it stops
    working. We split the data by whether SPY is above/below its 200-day
    moving average and compute IC separately.
    """
    # Simple regime: SPY above/below 200MA
    spy_dates = panel.groupby("date").first()  # one row per date
    # We need SPY price — use the panel's market return as proxy
    # If dist_ma200 exists, use it directly
    if "dist_ma200" in panel.columns:
        panel_regime = panel.copy()
        # Get one dist_ma200 per date (median across tickers as market proxy)
        date_regime = panel.groupby("date")["dist_ma200"].median()
        bull_dates = set(date_regime[date_regime > 0].index)
        bear_dates = set(date_regime[date_regime <= 0].index)
    else:
        # Fallback: use top-half/bottom-half of dates by market return
        if return_col in panel.columns:
            date_returns = panel.groupby("date")[return_col].mean()
            median_ret = date_returns.median()
            bull_dates = set(date_returns[date_returns > median_ret].index)
            bear_dates = set(date_returns[date_returns <= median_ret].index)
        else:
            return {"bull_ic": 0, "bear_ic": 0, "regime_ratio": 1.0}

    bull_panel = panel[panel["date"].isin(bull_dates)]
    bear_panel = panel[panel["date"].isin(bear_dates)]

    bull_ic = compute_daily_ic(bull_panel, feature_col, return_col)
    bear_ic = compute_daily_ic(bear_panel, feature_col, return_col)

    bull_mean = float(bull_ic.mean()) if len(bull_ic) > 10 else 0
    bear_mean = float(bear_ic.mean()) if len(bear_ic) > 10 else 0

    # Regime ratio: how much does IC change between bull and bear?
    # Close to 1.0 = stable across regimes (good)
    # Close to 0 or negative = only works in one regime (bad)
    if abs(bull_mean) > 1e-9:
        regime_ratio = bear_mean / bull_mean
    else:
        regime_ratio = 0.0
    # Stable only counts when the factor is positively predictive in both
    # regimes and the weaker regime retains at least 30% of the stronger IC.
    stronger_ic = max(abs(bull_mean), abs(bear_mean))
    weaker_ic = min(abs(bull_mean), abs(bear_mean))
    strength_ratio = weaker_ic / stronger_ic if stronger_ic > 1e-9 else 0.0
    both_positive = bull_mean > 0.005 and bear_mean > 0.005
    regime_stable = both_positive and strength_ratio >= 0.3

    return {
        "bull_ic": round(bull_mean, 6),
        "bear_ic": round(bear_mean, 6),
        "bull_days": len(bull_ic),
        "bear_days": len(bear_ic),
        "regime_ratio": round(regime_ratio, 4),
        "regime_stable": bool(regime_stable),
    }


def regime_sensitivity_label(regime: dict) -> str:
    """Describe which market regime is weak for a factor.

    PLAIN ENGLISH: The old report called every unstable factor "bull only."
    That was misleading when bear IC was actually stronger than bull IC.
    This helper names the weak side directly.
    """
    bull_ic = float(regime.get("bull_ic", 0.0) or 0.0)
    bear_ic = float(regime.get("bear_ic", 0.0) or 0.0)
    if bull_ic > 0 and bear_ic <= 0:
        return "bull-only"
    if bear_ic > 0 and bull_ic <= 0:
        return "bear-only"
    if bull_ic > bear_ic:
        return "weaker in bear markets"
    if bear_ic > bull_ic:
        return "weaker in bull markets"
    return "regime-sensitive"


def ic_decay_curve(panel: pd.DataFrame, feature_col: str, horizons: list[int] | None = None) -> dict:
    """
    Compute IC at multiple forward-return horizons to see signal decay.

    PLAIN ENGLISH: A good factor predicts returns not just at one horizon
    (e.g., 10 days) but across a range of horizons.  If IC is high at 5 days
    but zero at 20 days, the signal decays too fast — you can't profitably
    trade it after costs because by the time you enter, the edge is gone.

    If IC INCREASES with horizon, the signal is a slow-burn (like value investing).
    If IC DECREASES with horizon, the signal is fast-burn (like momentum).

    We want signals with IC > 0 for at least 10-20 days (our holding period).
    """
    if horizons is None:
        horizons = [5, 10, 20, 40, 60]

    decay = {}
    for h in horizons:
        # Look for a return column at this horizon
        possible_cols = [
            f"forward_return_{h}d",
            "forward_return" if h == HORIZON_DAYS else None,
            f"fwd_return_{h}d",
            f"fwd_sector_excess_{h}",
        ]
        ret_col = None
        for col in possible_cols:
            if col and col in panel.columns:
                ret_col = col
                break

        if ret_col is None:
            # Try to compute from price if we have consecutive dates
            decay[h] = {"ic": 0.0, "available": False}
            continue

        daily_ic = compute_daily_ic(panel, feature_col, ret_col)
        mean_ic = float(daily_ic.mean()) if len(daily_ic) > 10 else 0.0
        decay[h] = {"ic": round(mean_ic, 6), "available": True, "n_days": len(daily_ic)}

    # Compute half-life: at what horizon does IC drop to half of its peak?
    available_ics = [(h, d["ic"]) for h, d in decay.items() if d.get("available")]
    if available_ics:
        peak_ic = max(abs(ic) for _, ic in available_ics)
        half_life = None
        if peak_ic > 0.005:
            # Find first horizon where IC drops below half of peak
            for h, ic in sorted(available_ics):
                if abs(ic) < peak_ic * 0.5:
                    half_life = h
                    break
    else:
        peak_ic = 0
        half_life = None

    return {
        "horizons": {str(h): d for h, d in decay.items()},
        "peak_ic": round(peak_ic, 6),
        "half_life_days": half_life,
        "decays_fast": half_life is not None and half_life <= 10,
    }


def find_return_column(panel: pd.DataFrame) -> str | None:
    """Return the primary forward-return column available in the factor panel."""
    candidates = [
        f"fwd_sector_excess_{HORIZON_DAYS}",
        f"forward_sector_excess_{HORIZON_DAYS}d",
        f"fwd_return_{HORIZON_DAYS}d",
        f"forward_return_{HORIZON_DAYS}d",
        "forward_return",
        "fwd_return",
    ]
    return next((col for col in candidates if col in panel.columns), None)


def turnover_impact(panel: pd.DataFrame, feature_col: str, top_n: int = 5) -> dict:
    """
    Measure how much turnover this feature causes.

    PLAIN ENGLISH: If a feature's ranking changes dramatically every day,
    using it to pick stocks will cause high turnover (constant buying/selling).
    High turnover = high trading costs = worse real-world performance.

    We measure: what fraction of the top-N stocks by this feature CHANGE
    from one rebalance to the next?  Low turnover (< 30% per period) is good.
    High turnover (> 60%) suggests the signal is noisy.
    """
    dates = sorted(panel["date"].unique())
    # Sample every 10 days (typical holding period)
    sample_dates = dates[::10]

    if len(sample_dates) < 10:
        return {"mean_turnover_pct": 0, "available": False}

    prev_top = set()
    turnovers = []

    for dt in sample_dates:
        day_data = panel[panel["date"] == dt]
        if len(day_data) < top_n:
            continue
        # Rank stocks by feature, get top N
        top_stocks = set(day_data.nlargest(top_n, feature_col)["ticker"].tolist())
        if prev_top:
            # Turnover = fraction of positions that changed
            changed = len(top_stocks - prev_top)
            turnover = changed / top_n
            turnovers.append(turnover)
        prev_top = top_stocks

    if not turnovers:
        return {"mean_turnover_pct": 0, "available": False}

    mean_turnover = float(np.mean(turnovers)) * 100
    std_turnover = float(np.std(turnovers)) * 100

    return {
        "mean_turnover_pct": round(mean_turnover, 1),
        "std_turnover_pct": round(std_turnover, 1),
        "n_periods": len(turnovers),
        "available": True,
        "high_turnover": mean_turnover > 60,  # >60% means too noisy
    }


def feature_correlation_matrix(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Compute cross-sectional correlation between features.

    PLAIN ENGLISH: If two features are 90% correlated, they're basically
    the same signal. Using both adds no new information but doubles your
    exposure to whatever that signal is. We identify clusters of highly
    correlated features so you only keep the best one from each cluster.

    This version is vectorized — instead of looping over each date and
    computing a separate correlation matrix, it ranks all stocks within
    each date in one shot, then computes the average correlation across
    all dates at once.  Much faster for large panels.
    """
    available = [f for f in features if f in panel.columns]
    if len(available) < 2:
        return pd.DataFrame()

    # Filter to rows with enough stocks for meaningful correlation
    date_counts = panel.groupby("date")[available[0]].transform("count")
    usable = panel.loc[date_counts >= 10, ["date"] + available].copy()
    if usable.empty:
        return pd.DataFrame()

    # Rank within each date (cross-sectional percentile rank) — vectorized
    ranked = usable.groupby("date")[available].rank(pct=True)

    # Compute the mean cross-sectional correlation across all dates.
    # pandas .corr() on the ranked data gives us what we want directly
    # because each date's ranks are independent cross-sections.
    # However, to properly average per-date correlations we use groupby.
    #
    # Vectorized approach: compute correlation of ranks directly.
    # This is equivalent to averaging per-date correlations when each
    # date has the same number of stocks.  For varying stock counts,
    # it's a weighted average (more stocks = more weight) which is
    # actually more statistically appropriate.
    avg_corr = ranked.corr()

    return pd.DataFrame(avg_corr.values, index=available, columns=available)


def compute_feature_quality_grade(stats: dict) -> str:
    """
    Assign A/B/C/D/F grade based on composite quality metrics.

    PLAIN ENGLISH:
      A = Strong, stable factor. Use with confidence.
      B = Good factor with minor weaknesses. Use it.
      C = Marginal factor. Might help, might not. Low weight.
      D = Weak factor. Probably noise. Consider dropping.
      F = Failed. Remove from the model.

    Scoring breakdown (0-100 points):
      - IC magnitude: 0-20 pts (is there signal at all?)
      - IC stability across time: 0-20 pts (consistent or burst-y?)
      - Regime robustness: 0-20 pts (works in bear AND bull?)
      - t-stat significance: 0-20 pts (statistically significant?)
      - Signal persistence/decay: 0-10 pts (does it last long enough to trade?)
      - Turnover penalty: 0-10 pts (low turnover = easier to trade profitably)
    """
    points = 0.0

    # (1) IC magnitude (0-20 pts)
    mean_ic = abs(float(stats.get("mean_ic", 0)))
    points += min(20, mean_ic * 1200)  # IC of ~0.017 → 20 pts

    # (2) Stability (0-20 pts) — rolling IC hit rate
    stability = stats.get("rolling_stats", [{}])
    if stability:
        avg_hit = np.mean([s.get("hit_rate", 0.5) for s in stability])
        points += 20 * max(0, (avg_hit - 0.5) * 4)  # 0.5→0, 0.75→20

    # (3) Regime robustness (0-20 pts)
    regime = stats.get("regime_conditional", {})
    if regime.get("regime_stable"):
        points += 12
    if regime.get("regime_ratio", 0) > 0.5:
        points += 8

    # (4) t-stat significance (0-20 pts)
    tstat = abs(float(stats.get("t_stat", 0)))
    points += min(20, tstat * 2.5)  # t=8 → 20 pts

    # (5) Signal persistence — IC decay (0-10 pts)
    # Reward factors whose signal persists to at least 10-day horizon
    decay = stats.get("ic_decay", {})
    if decay and not decay.get("decays_fast", False):
        points += 7  # signal persists beyond 10 days — tradeable
    if decay and decay.get("half_life_days") and decay["half_life_days"] >= 20:
        points += 3  # very persistent signal — excellent for our 10-day holding

    # (6) Turnover penalty (0-10 pts) — low turnover is better
    # High turnover means rankings are noisy day-to-day → expensive to trade
    turnover = stats.get("turnover_impact", {})
    if turnover.get("available"):
        mean_to = turnover.get("mean_turnover_pct", 50)
        if mean_to < 30:
            points += 10  # very low turnover — great
        elif mean_to < 50:
            points += 6  # moderate
        elif mean_to < 70:
            points += 3  # high but acceptable
        # > 70% turnover → 0 pts (too noisy)
    else:
        points += 5  # no data, assume moderate

    if points >= 75:
        return "A"
    elif points >= 55:
        return "B"
    elif points >= 35:
        return "C"
    elif points >= 18:
        return "D"
    else:
        return "F"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Feature quality diagnostic")
    parser.add_argument("--top", type=int, default=24, help="Analyze top N features")
    args = parser.parse_args()

    print("Loading factor panel...")
    specs = load_feature_specs(max_specs=args.top, write_health_outputs=False)
    # The quality report only needs raw feature columns plus forward returns.
    # PLAIN ENGLISH: Do not call the full score builder here; that also creates
    # walkforward/regime scores for trading, which is much slower and not needed
    # for measuring individual feature quality.
    panel = load_factor_panel(specs)
    panel = add_sector_excess_return_columns(panel)
    print(f"  Panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} dates")

    # Determine return column
    return_col = find_return_column(panel)
    if return_col is None:
        print("  ERROR: No forward return column found")
        sys.exit(1)
    print(f"  Return column: {return_col}")

    features = [s["feature"] for s in specs if s["feature"] in panel.columns]
    print(f"  Analyzing {len(features)} features...\n")

    results = []
    for i, spec in enumerate(specs):
        feat = spec["feature"]
        if feat not in panel.columns:
            continue
        analysis_feat = add_oriented_feature(panel, feat, spec.get("preferred_direction", "high_is_good"))

        print(f"  [{i+1}/{len(features)}] {feat}")

        # Basic IC
        daily_ic = compute_daily_ic(panel, analysis_feat, return_col)
        mean_ic = float(daily_ic.mean()) if len(daily_ic) > 0 else 0
        std_ic = float(daily_ic.std()) if len(daily_ic) > 0 else 1
        t_stat = mean_ic / (std_ic / np.sqrt(len(daily_ic))) if std_ic > 0 and len(daily_ic) > 0 else 0

        # Rolling IC stability
        rolling_stats = []
        for window in IC_ROLLING_WINDOWS:
            rs = rolling_ic_stats(daily_ic, window)
            rolling_stats.append(rs)
            print(f"    Rolling {window}d: hit_rate={rs['hit_rate']:.2f}, "
                  f"stability={rs['stability_ratio']:.2f}, "
                  f"worst={rs['worst_window_ic']:.4f}")

        # Regime conditional IC
        regime = regime_conditional_ic(panel, analysis_feat, return_col)
        print(f"    Regime: bull_ic={regime['bull_ic']:.4f}, "
              f"bear_ic={regime['bear_ic']:.4f}, "
              f"stable={regime['regime_stable']}")

        # IC decay curve (how signal persists across horizons)
        decay = ic_decay_curve(panel, analysis_feat)
        if decay.get("half_life_days"):
            print(f"    Decay: half_life={decay['half_life_days']}d, "
                  f"peak_ic={decay['peak_ic']:.4f}, "
                  f"fast_decay={decay['decays_fast']}")

        # Turnover impact (how much trading does this feature cause?)
        turnover = turnover_impact(panel, analysis_feat, top_n=5)
        if turnover.get("available"):
            print(f"    Turnover: {turnover['mean_turnover_pct']:.0f}%/period "
                  f"(high={turnover['high_turnover']})")

        # Compile stats
        stats = {
            "feature": feat,
            "preferred_direction": spec.get("preferred_direction", "high_is_good"),
            "mean_ic": round(mean_ic, 6),
            "std_ic": round(std_ic, 6),
            "t_stat": round(t_stat, 4),
            "n_days": len(daily_ic),
            "rolling_stats": rolling_stats,
            "regime_conditional": regime,
            "ic_decay": decay,
            "turnover_impact": turnover,
        }
        stats["grade"] = compute_feature_quality_grade(stats)
        results.append(stats)
        print(f"    Grade: {stats['grade']} (IC={mean_ic:.4f}, t={t_stat:.1f})")
        print()

    # Feature correlation analysis
    print("Computing feature correlations...")
    corr_matrix = feature_correlation_matrix(panel, features)
    clusters = []
    if not corr_matrix.empty:
        # Find highly correlated pairs
        for i, f1 in enumerate(corr_matrix.columns):
            for j, f2 in enumerate(corr_matrix.columns):
                if i < j and abs(corr_matrix.iloc[i, j]) > CORRELATION_CLUSTER_THRESHOLD:
                    clusters.append({
                        "feature_1": f1,
                        "feature_2": f2,
                        "correlation": round(float(corr_matrix.iloc[i, j]), 4),
                    })
        if clusters:
            print(f"  Found {len(clusters)} highly correlated pairs (>{CORRELATION_CLUSTER_THRESHOLD}):")
            for c in clusters[:10]:
                print(f"    {c['feature_1'][:40]} ↔ {c['feature_2'][:40]}: {c['correlation']:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("FEATURE QUALITY SUMMARY")
    print("=" * 60)
    grade_counts = {}
    for r in results:
        g = r["grade"]
        grade_counts[g] = grade_counts.get(g, 0) + 1
    for g in "ABCDF":
        count = grade_counts.get(g, 0)
        bar = "█" * count
        print(f"  Grade {g}: {count:>3} {bar}")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    a_features = [r for r in results if r["grade"] == "A"]
    d_features = [r for r in results if r["grade"] in ("D", "F")]
    regime_weak = [r for r in results if not r["regime_conditional"].get("regime_stable", True)]

    if a_features:
        print(f"  ✓ {len(a_features)} Grade-A features (strong, keep)")
    if d_features:
        print(f"  ⚠ {len(d_features)} Grade-D/F features (consider dropping):")
        for r in d_features:
            print(f"      {r['feature'][:50]} (IC={r['mean_ic']:.4f})")
    if regime_weak:
        print(f"  ⚠ {len(regime_weak)} regime-sensitive features:")
        for r in regime_weak:
            rc = r["regime_conditional"]
            label = regime_sensitivity_label(rc)
            print(
                f"      {r['feature'][:50]} ({label}; "
                f"bull={rc['bull_ic']:.4f}, bear={rc['bear_ic']:.4f})"
            )

    # Save report
    report = {
        "n_features": len(results),
        "grade_counts": grade_counts,
        "features": results,
        "correlation_clusters": clusters,
        "recommendations": {
            "keep": [r["feature"] for r in results if r["grade"] in ("A", "B")],
            "reduce_weight": [r["feature"] for r in results if r["grade"] == "C"],
            "consider_dropping": [r["feature"] for r in results if r["grade"] in ("D", "F")],
            "regime_sensitive": [r["feature"] for r in regime_weak],
            "regime_dependent": [r["feature"] for r in regime_weak],
        },
    }
    out_json = Path(SIGNAL_DIR) / "feature_quality_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved → {out_json}")

    # CSV summary
    summary_rows = []
    for r in results:
        rolling_252 = next((s for s in r["rolling_stats"] if s["window_days"] == 252), {})
        decay = r.get("ic_decay", {})
        turnover = r.get("turnover_impact", {})
        summary_rows.append({
            "feature": r["feature"],
            "grade": r["grade"],
            "mean_ic": r["mean_ic"],
            "t_stat": r["t_stat"],
            "rolling_252d_hit_rate": rolling_252.get("hit_rate", 0),
            "rolling_252d_stability": rolling_252.get("stability_ratio", 0),
            "regime_stable": r["regime_conditional"].get("regime_stable", False),
            "bull_ic": r["regime_conditional"].get("bull_ic", 0),
            "bear_ic": r["regime_conditional"].get("bear_ic", 0),
            "ic_half_life_days": decay.get("half_life_days"),
            "decays_fast": decay.get("decays_fast", False),
            "turnover_pct": turnover.get("mean_turnover_pct", None),
            "high_turnover": turnover.get("high_turnover", False),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("grade")
    out_csv = Path(SIGNAL_DIR) / "feature_quality_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Summary saved → {out_csv}")


if __name__ == "__main__":
    main()
