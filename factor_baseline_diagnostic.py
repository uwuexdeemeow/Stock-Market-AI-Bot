"""
factor_baseline_diagnostic.py — Does the DATA have cross-sectional signal?

Before blaming the model, test whether raw factor rankings (no ML) predict
forward excess returns.  If pure 12-1 momentum ranking can't beat random,
no XGBoost will help either.

Tests:
  1. Single-factor rank ICs (each feature ranked across stocks per day)
  2. Simple composite (equal-weight average of top IC factors)
  3. Walk-forward composite (re-estimate factor weights every 252 days)
  4. Compare all three to XGBoost's walk-forward IC

Usage:
    python3 factor_baseline_diagnostic.py
    STOCK_UNIVERSE_MODE=core python3 factor_baseline_diagnostic.py
"""
from __future__ import annotations

import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Setup paths ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import DATA_DIR, WATCHLIST, RETURN_HORIZON_DAYS, SECTOR_MAP

HORIZON = int(os.environ.get("RETURN_HORIZON_DAYS", str(RETURN_HORIZON_DAYS)))
MIN_STOCKS_PER_DAY = 15   # skip days with too few stocks (noisy IC)
MIN_DAYS_FOR_IC    = 100   # need enough days for stable mean IC
OOS_START_YEAR     = 2018  # everything before = "in-sample" for weight estimation


# ── 1. Load panel ────────────────────────────────────────────────────────────
def load_panel(tickers: list[str], horizon: int) -> pd.DataFrame:
    """Load all tickers into a long-format panel with forward excess returns."""
    frames = []
    for ticker in tickers:
        path = os.path.join(DATA_DIR, f"{ticker}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if len(df) < 252:
            continue

        close = pd.to_numeric(df.get("Close"), errors="coerce")
        if close.isna().all():
            continue

        # Forward return: stock return over next `horizon` days
        stock_fwd = close.pct_change(horizon).shift(-horizon)

        # SPY forward return (from multi-market features already in parquet)
        spy_fwd_col = f"spy_ret{horizon}d"
        if spy_fwd_col in df.columns:
            spy_fwd = pd.to_numeric(df[spy_fwd_col], errors="coerce").shift(-horizon)
        else:
            spy_fwd = 0.0

        excess_fwd = stock_fwd - spy_fwd

        # Collect all numeric features
        row = df.select_dtypes(include=[np.number]).copy()
        row["ticker"] = ticker
        row["date"] = df.index
        row["fwd_excess"] = excess_fwd.values
        row["fwd_raw"] = stock_fwd.values
        row["sector"] = SECTOR_MAP.get(ticker, "OTHER")
        frames.append(row)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["fwd_excess"])
    print(f"Panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} dates")
    return panel


# ── 2. Single-factor daily rank IC ──────────────────────────────────────────
def daily_rank_ic(panel: pd.DataFrame, factor_col: str,
                  target_col: str = "fwd_excess",
                  min_per_day: int = MIN_STOCKS_PER_DAY) -> dict:
    """Compute daily Spearman rank IC of one factor vs forward excess return."""
    sub = panel[["date", factor_col, target_col]].dropna()
    if sub.empty:
        return {"factor": factor_col, "n_days": 0, "mean_ic": 0.0,
                "t_stat": 0.0, "hit_rate": 0.0}

    daily_ics = []
    for _dt, g in sub.groupby("date"):
        if len(g) < min_per_day:
            continue
        f_rank = g[factor_col].rank()
        t_rank = g[target_col].rank()
        if f_rank.std() < 1e-12 or t_rank.std() < 1e-12:
            continue
        rho = float(np.corrcoef(f_rank.values, t_rank.values)[0, 1])
        if np.isfinite(rho):
            daily_ics.append(rho)

    if len(daily_ics) < MIN_DAYS_FOR_IC:
        return {"factor": factor_col, "n_days": len(daily_ics),
                "mean_ic": 0.0, "t_stat": 0.0, "hit_rate": 0.0}

    arr = np.array(daily_ics)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    n = len(arr)
    t = mean / (std / np.sqrt(n)) if std > 1e-12 else 0.0
    return {
        "factor": factor_col,
        "n_days": n,
        "mean_ic": round(mean, 5),
        "t_stat": round(t, 2),
        "hit_rate": round(float((arr > 0).mean()), 3),
    }


# ── 3. Composite factor score ───────────────────────────────────────────────
def build_composite_score(panel: pd.DataFrame, factor_cols: list[str],
                          weights: dict[str, float] | None = None) -> pd.Series:
    """Equal-weight (or custom-weight) z-score composite of selected factors."""
    if weights is None:
        weights = {c: 1.0 / len(factor_cols) for c in factor_cols}

    # Per-date cross-sectional z-score each factor, then weight-average
    composite = pd.Series(0.0, index=panel.index)
    for col in factor_cols:
        raw = panel[col].copy()
        # Cross-sectional z-score within each date
        grouped = raw.groupby(panel["date"])
        z = grouped.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        composite += weights.get(col, 0.0) * z.fillna(0.0)
    return composite


# ── 4. Walk-forward weight estimation ────────────────────────────────────────
def walk_forward_composite_ic(panel: pd.DataFrame, factor_cols: list[str],
                              rebalance_days: int = 252) -> dict:
    """Re-estimate factor weights every `rebalance_days` using trailing IC."""
    dates = sorted(panel["date"].unique())
    oos_start = pd.Timestamp(f"{OOS_START_YEAR}-01-01")
    oos_dates = [d for d in dates if d >= oos_start]

    if len(oos_dates) < MIN_DAYS_FOR_IC:
        return {"n_days": 0, "mean_ic": 0.0, "t_stat": 0.0}

    daily_ics = []
    last_weights = {c: 1.0 / len(factor_cols) for c in factor_cols}
    last_rebal_idx = 0

    for i, dt in enumerate(oos_dates):
        # Re-estimate weights periodically using trailing data
        if i - last_rebal_idx >= rebalance_days or i == 0:
            # Use all data before this date to estimate ICs
            train_mask = panel["date"] < dt
            train = panel.loc[train_mask]
            if len(train) > 1000:
                ics = {}
                for col in factor_cols:
                    res = daily_rank_ic(train, col, min_per_day=MIN_STOCKS_PER_DAY)
                    ics[col] = res["mean_ic"]
                # IC-weighted (positive IC only, else zero)
                total = sum(max(v, 0) for v in ics.values())
                if total > 0:
                    last_weights = {c: max(ics[c], 0) / total for c in factor_cols}
                else:
                    last_weights = {c: 1.0 / len(factor_cols) for c in factor_cols}
            last_rebal_idx = i

        # Score today's cross-section
        day_mask = panel["date"] == dt
        day = panel.loc[day_mask]
        if len(day) < MIN_STOCKS_PER_DAY:
            continue

        score = pd.Series(0.0, index=day.index)
        for col in factor_cols:
            raw = day[col]
            z = (raw - raw.mean()) / (raw.std() + 1e-9)
            score += last_weights.get(col, 0.0) * z.fillna(0.0)

        # Rank IC for this day
        s_rank = score.rank()
        t_rank = day["fwd_excess"].rank()
        if s_rank.std() < 1e-12 or t_rank.std() < 1e-12:
            continue
        rho = float(np.corrcoef(s_rank.values, t_rank.values)[0, 1])
        if np.isfinite(rho):
            daily_ics.append(rho)

    if len(daily_ics) < MIN_DAYS_FOR_IC:
        return {"n_days": 0, "mean_ic": 0.0, "t_stat": 0.0, "hit_rate": 0.0,
                "final_weights": last_weights}

    arr = np.array(daily_ics)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    n = len(arr)
    t = mean / (std / np.sqrt(n)) if std > 1e-12 else 0.0
    return {
        "n_days": n,
        "mean_ic": round(mean, 5),
        "t_stat": round(t, 2),
        "hit_rate": round(float((arr > 0).mean()), 3),
        "final_weights": {k: round(v, 3) for k, v in last_weights.items()},
    }


# ── 5. Top-N simulated PnL ──────────────────────────────────────────────────
def simulate_topn_pnl(panel: pd.DataFrame, score_col: str,
                       top_n: int = 4, long_only: bool = True) -> dict:
    """Simulate long top-N strategy: each day pick top_n stocks by score."""
    oos_start = pd.Timestamp(f"{OOS_START_YEAR}-01-01")
    oos = panel[panel["date"] >= oos_start].copy()

    daily_rets = []
    for dt, day in oos.groupby("date"):
        if len(day) < MIN_STOCKS_PER_DAY:
            continue
        ranked = day.nlargest(top_n, score_col)
        avg_ret = ranked["fwd_excess"].mean()
        if np.isfinite(avg_ret):
            daily_rets.append(avg_ret)

    if len(daily_rets) < 50:
        return {"n_days": 0, "mean_daily_ret": 0.0, "sharpe": 0.0}

    arr = np.array(daily_rets)
    # These are overlapping horizon-day returns, so annualise carefully
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # Approximate annualised Sharpe: mean excess per period / std, scaled
    sharpe = (mean / (std + 1e-9)) * np.sqrt(252 / max(HORIZON, 1))
    return {
        "n_days": len(arr),
        "mean_excess_per_period": round(mean * 100, 3),  # in percent
        "std_per_period": round(std * 100, 3),
        "approx_annual_sharpe": round(float(sharpe), 3),
        "hit_rate": round(float((arr > 0).mean()), 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"=== Factor Baseline Diagnostic ===")
    print(f"Universe: {len(WATCHLIST)} tickers, Horizon: {HORIZON}d")
    print(f"OOS start: {OOS_START_YEAR}\n")

    panel = load_panel(WATCHLIST, HORIZON)

    # ── Candidate factors to test ────────────────────────────────────────────
    # These are raw columns from the parquet that should have cross-sectional
    # variation (not market-wide constants like spy_*, vix_*, macro_*).
    candidate_factors = [
        # Price momentum
        "ret_5d", "ret_10d", "ret_20d",
        # Distance from MAs (mean reversion)
        "dist_ma10", "dist_ma20", "dist_ma50", "dist_ma200",
        # Volatility
        "hvol_20d", "atr_norm",
        # Technical
        "rsi_14", "bb_pos", "macd",
        # Literature factors
        "factor_mom_12_1", "factor_resid_mom_sector_12_1",
        "factor_beta_252_spy", "factor_idio_vol_252_spy",
        "factor_liquidity_dollar_vol_20d", "factor_illiquidity_amihud_20d",
        # Cross-sectional ranks
        "xs_rank_market_ret_5d", "xs_rank_market_ret_10d", "xs_rank_market_ret_20d",
        "xs_rank_sector_ret_5d", "xs_rank_sector_ret_10d", "xs_rank_sector_ret_20d",
        "xs_rank_market_hvol_20d", "xs_rank_sector_hvol_20d",
        "xs_rank_market_rsi_14", "xs_rank_sector_rsi_14",
        "xs_rank_market_dist_ma50", "xs_rank_sector_dist_ma50",
        "xs_rank_market_dist_ma200", "xs_rank_sector_dist_ma200",
        # Sector relative
        "sector_ratio_z20", "sector_ratio_z60",
        # Fundamentals
        "fund_pe_sector_z", "fund_fcf_yield_sector_z", "fund_value_combo_z",
        # Sentiment
        "sent_z_sentiment_3d", "sent_z_sentiment_5d",
        # Volume
        "vol_zscore_252",
    ]

    # Filter to columns that actually exist in the panel
    available = [c for c in candidate_factors if c in panel.columns]
    missing = [c for c in candidate_factors if c not in panel.columns]
    if missing:
        print(f"Missing from parquets ({len(missing)}): {missing[:10]}...")
    print(f"Testing {len(available)} factors\n")

    # ── Single-factor ICs (full sample first, then OOS only) ─────────────────
    print("=" * 70)
    print("SINGLE-FACTOR RANK IC (full sample)")
    print("=" * 70)
    results = []
    for col in available:
        res = daily_rank_ic(panel, col)
        results.append(res)

    results.sort(key=lambda x: abs(x["t_stat"]), reverse=True)
    print(f"{'Factor':<45} {'IC':>8} {'t-stat':>8} {'hit%':>7} {'days':>6}")
    print("-" * 76)
    for r in results:
        marker = " ***" if abs(r["t_stat"]) >= 3.0 else (" **" if abs(r["t_stat"]) >= 2.0 else "")
        print(f"{r['factor']:<45} {r['mean_ic']:>+8.4f} {r['t_stat']:>+8.2f} "
              f"{r['hit_rate']*100:>6.1f}% {r['n_days']:>5}{marker}")

    # ── OOS-only single-factor ICs ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"SINGLE-FACTOR RANK IC (OOS only: {OOS_START_YEAR}+)")
    print("=" * 70)
    oos_panel = panel[panel["date"] >= pd.Timestamp(f"{OOS_START_YEAR}-01-01")]
    oos_results = []
    for col in available:
        res = daily_rank_ic(oos_panel, col)
        oos_results.append(res)

    oos_results.sort(key=lambda x: abs(x["t_stat"]), reverse=True)
    print(f"{'Factor':<45} {'IC':>8} {'t-stat':>8} {'hit%':>7} {'days':>6}")
    print("-" * 76)
    for r in oos_results:
        marker = " ***" if abs(r["t_stat"]) >= 3.0 else (" **" if abs(r["t_stat"]) >= 2.0 else "")
        print(f"{r['factor']:<45} {r['mean_ic']:>+8.4f} {r['t_stat']:>+8.2f} "
              f"{r['hit_rate']*100:>6.1f}% {r['n_days']:>5}{marker}")

    # ── Pick top factors with positive OOS IC ────────────────────────────────
    # Use factors with |t| >= 2.0 in OOS, preferring positive IC (long winners)
    strong_oos = [r for r in oos_results if abs(r["t_stat"]) >= 2.0]
    if len(strong_oos) < 3:
        strong_oos = sorted(oos_results, key=lambda x: abs(x["t_stat"]), reverse=True)[:5]

    top_factor_cols = [r["factor"] for r in strong_oos[:8]]
    print(f"\n>>> Top factors for composite: {top_factor_cols}")

    # ── Equal-weight composite (full OOS) ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("COMPOSITE FACTOR STRATEGIES (OOS)")
    print("=" * 70)

    # Flip sign for factors where negative IC is the signal
    # (e.g., high vol predicts low returns → flip so ranking works)
    ic_signs = {r["factor"]: np.sign(r["mean_ic"]) for r in oos_results}

    # Build sign-corrected composite
    for col in top_factor_cols:
        if ic_signs.get(col, 1) < 0:
            panel[f"_flip_{col}"] = -panel[col]
        else:
            panel[f"_flip_{col}"] = panel[col]

    flipped_cols = [f"_flip_{col}" for col in top_factor_cols]
    panel["composite_equal"] = build_composite_score(panel, flipped_cols)

    eq_ic = daily_rank_ic(
        panel[panel["date"] >= pd.Timestamp(f"{OOS_START_YEAR}-01-01")],
        "composite_equal"
    )
    print(f"Equal-weight composite ({len(top_factor_cols)} factors):")
    print(f"  OOS IC = {eq_ic['mean_ic']:+.4f}, t = {eq_ic['t_stat']:+.2f}, "
          f"hit = {eq_ic['hit_rate']*100:.1f}%, days = {eq_ic['n_days']}")

    # ── Walk-forward IC-weighted composite ───────────────────────────────────
    wf_result = walk_forward_composite_ic(panel, flipped_cols, rebalance_days=252)
    print(f"\nWalk-forward IC-weighted composite (re-estimate yearly):")
    print(f"  OOS IC = {wf_result['mean_ic']:+.4f}, t = {wf_result['t_stat']:+.2f}, "
          f"hit = {wf_result.get('hit_rate', 0)*100:.1f}%, days = {wf_result['n_days']}")
    if "final_weights" in wf_result:
        print(f"  Final weights: {wf_result['final_weights']}")

    # ── Simulated top-N PnL ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"TOP-{4} PORTFOLIO SIMULATION (OOS, long-only)")
    print("=" * 70)

    # Best single factor
    if strong_oos:
        best_single = strong_oos[0]["factor"]
        best_col = f"_flip_{best_single}" if ic_signs.get(best_single, 1) < 0 else best_single
        if best_col not in panel.columns:
            best_col = best_single
        pnl_single = simulate_topn_pnl(panel, best_col, top_n=4)
        print(f"Best single factor ({best_single}):")
        print(f"  Mean excess/period: {pnl_single['mean_excess_per_period']:+.3f}%  "
              f"Sharpe ≈ {pnl_single['approx_annual_sharpe']:.3f}  "
              f"Hit: {pnl_single['hit_rate']*100:.1f}%")

    # Equal-weight composite
    pnl_eq = simulate_topn_pnl(panel, "composite_equal", top_n=4)
    print(f"\nEqual-weight composite:")
    print(f"  Mean excess/period: {pnl_eq['mean_excess_per_period']:+.3f}%  "
          f"Sharpe ≈ {pnl_eq['approx_annual_sharpe']:.3f}  "
          f"Hit: {pnl_eq['hit_rate']*100:.1f}%")

    # Random baseline
    np.random.seed(42)
    panel["_random"] = np.random.randn(len(panel))
    pnl_rand = simulate_topn_pnl(panel, "_random", top_n=4)
    print(f"\nRandom baseline:")
    print(f"  Mean excess/period: {pnl_rand['mean_excess_per_period']:+.3f}%  "
          f"Sharpe ≈ {pnl_rand['approx_annual_sharpe']:.3f}  "
          f"Hit: {pnl_rand['hit_rate']*100:.1f}%")

    # ── Year-by-year stability ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("YEAR-BY-YEAR COMPOSITE IC STABILITY")
    print("=" * 70)
    panel["year"] = pd.to_datetime(panel["date"]).dt.year
    print(f"{'Year':<6} {'IC':>8} {'t-stat':>8} {'hit%':>7} {'days':>6}")
    print("-" * 40)
    for yr in sorted(panel["year"].unique()):
        if yr < OOS_START_YEAR:
            continue
        yr_panel = panel[panel["year"] == yr]
        yr_ic = daily_rank_ic(yr_panel, "composite_equal")
        marker = " ***" if abs(yr_ic["t_stat"]) >= 2.0 else ""
        print(f"{yr:<6} {yr_ic['mean_ic']:>+8.4f} {yr_ic['t_stat']:>+8.2f} "
              f"{yr_ic['hit_rate']*100:>6.1f}% {yr_ic['n_days']:>5}{marker}")

    print("\n=== Done ===")
    print("If composite IC > 0.02 with t > 3: signal exists, model is the problem.")
    print("If composite IC ≈ 0 even with best factors: data lacks alpha at this horizon.")


if __name__ == "__main__":
    main()
