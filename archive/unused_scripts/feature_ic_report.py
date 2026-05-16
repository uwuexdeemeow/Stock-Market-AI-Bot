"""
feature_ic_report.py — measure each feature's predictive power directly.

WHY THIS FILE EXISTS
--------------------
Five rounds of model surgery (regime models, soft blends, target switches,
horizon switches, universe expansion 28→147, xs_rank features, market-constant
exclusion) all left OOS AUC stuck at ~0.51.  At that point the bottleneck is
almost certainly the FEATURES, not the model — but "AUC=0.51" tells you the
final score, not which features are alive and which are dead weight.

This script measures each feature DIRECTLY against forward returns, using
the same metric the backtest cares about: cross-sectional rank correlation.

PLAIN ENGLISH (no ML jargon)
----------------------------
For every feature (e.g. "ret_20d", "rsi_14", "xs_rank_sector_ret_20d") and
every trading day in history:
  1. Take that feature's value for all 147 stocks on day D.
  2. Take each stock's actual N-day forward return on day D.
  3. Ask: "did stocks with HIGHER feature values tend to have HIGHER forward
     returns?"  This is Spearman correlation — the "rank IC" — a number in
     [-1, +1].
  4. Average that daily IC across thousands of days.

If a feature has mean IC = 0.04 with t-stat 5, it has REAL signal: stocks it
rates highly outperform on average.  If mean IC = 0.001 with t-stat 0.3, the
feature is noise — the model can't extract signal that isn't there.

This is the diagnostic that should have been run BEFORE any model tuning.

OUTPUTS
-------
- logs/feature_ic_report.csv:    per-feature × target × horizon table of IC stats
- logs/feature_ic_family.csv:    per-family aggregate (mean of |t_stat| in family)
- logs/feature_ic_shortlist.csv: features that clear the IC/t-stat gates
- stdout:                        top-30 features by |t_stat| at primary horizon

Run:
    python3 feature_ic_report.py                 # default horizons {5,10,20}
    python3 feature_ic_report.py --horizon 5     # single horizon
    python3 feature_ic_report.py --target sector_excess spy_excess
    python3 feature_ic_report.py --top 50        # show top-50 instead of 30
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from settings import (
    DATA_DIR,
    LOG_DIR,
    SECTOR_MAP,
    WATCHLIST,
    BROAD_WATCHLIST,
    UNIVERSE_MODE,
    SURVIVORSHIP_TRAINING_TICKERS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "feature_ic_report.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("feature_ic")


# ── Knobs ─────────────────────────────────────────────────────────────────────
# Minimum number of tickers required on a single date for that date's IC to
# count.  With < 10 stocks the cross-sectional rank correlation is too noisy
# to be informative.  Days that fail this gate are skipped.
MIN_TICKERS_PER_DAY: int = 15
ALIVE_TSTAT: float = 3.0
ALIVE_MEAN_IC: float = 0.005
TARGET_CHOICES = ("xs_excess", "sector_excess", "spy_excess", "raw_return")

# Columns that are NOT features and must be excluded from scoring.
NON_FEATURE_COLS: set[str] = {
    "Open", "High", "Low", "Close", "Volume", "ma5", "ma10", "ma20", "ma50",
    "ma200", "target", "ticker", "sector", "date",
}


def _family_of(col: str) -> str:
    """Group features by the prefix before the first numeric / horizon suffix.

    Heuristic: take everything up to the first digit run.  E.g.
        "ret_5d"             → "ret_"
        "xs_rank_sector_rsi" → "xs_rank_sector_"
        "macd_signal"        → "macd_"
        "fund_pe_sector_z"   → "fund_"
    Imperfect but useful for at-a-glance family rollups.
    """
    if col.startswith("xs_rank_sector_"):
        return "xs_rank_sector_"
    if col.startswith("xs_rank_market_"):
        return "xs_rank_market_"
    if col.startswith("sent_z_"):
        return "sent_z_"
    if col.startswith("fund_"):
        return "fund_"
    if col.startswith("macro_"):
        return "macro_"
    if col.startswith("spy_"):
        return "spy_"
    if col.startswith("vix_"):
        return "vix_"
    if col.startswith("qqq_"):
        return "qqq_"
    # General: first underscore-separated token.
    head = col.split("_", 1)[0]
    return head + "_"


def _resolve_universe() -> list[str]:
    """Return the ticker universe matching the user's training settings."""
    if UNIVERSE_MODE == "broad":
        base = list(BROAD_WATCHLIST)
    else:
        base = list(WATCHLIST)
    # Include survivorship training tickers because they affect the panel
    # density (and thus IC stability) the same way training does.
    return sorted(set(t.upper() for t in base) | set(SURVIVORSHIP_TRAINING_TICKERS))


def load_panel(tickers: Iterable[str]) -> pd.DataFrame:
    """Read every ticker's parquet and stack into one tall (date, ticker) frame.

    Returns a DataFrame indexed by RangeIndex with columns:
        date (datetime64), ticker (str), sector (str), Close (float),
        ... all engineered features ...
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
        except Exception as exc:
            log.warning("[%s] read_parquet failed: %s", t, exc)
            skipped.append(t)
            continue
        df = df.copy()
        df["ticker"] = t
        df["sector"] = SECTOR_MAP.get(t, "OTHER")
        df["date"] = pd.to_datetime(df.index)
        frames.append(df.reset_index(drop=True))
    if not frames:
        raise RuntimeError("No parquets loaded — run research.py first")
    panel = pd.concat(frames, ignore_index=True)
    log.info(
        "[panel] loaded %d rows  %d tickers  %d cols  (%d skipped)",
        len(panel), panel["ticker"].nunique(), panel.shape[1], len(skipped),
    )
    return panel


def add_forward_targets(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Add forward N-day returns and neutralized variants.

    Forward total return:    Close[t+h] / Close[t] - 1   per (ticker, t)
    Forward XS-excess:       fwd_ret_h - mean(fwd_ret_h across tickers on date t)
    Forward sector-excess:   fwd_ret_h - mean(fwd_ret_h within sector/date)
    Forward SPY-excess:      fwd_ret_h - forward SPY return from the feature row

    The XS-excess is what a cross-sectional ranker actually targets — the part
    of the return that BEATS THE PEER GROUP on the same day, with market beta
    netted out automatically (every stock loses the date-mean equally).
    """
    panel = panel.sort_values(["ticker", "date"]).copy()
    for h in horizons:
        # Per-ticker shifted close: tomorrow's close in h business days.
        panel[f"fwd_ret_{h}"] = (
            panel.groupby("ticker")["Close"]
            .transform(lambda s: s.shift(-h) / s - 1.0)
        )
    # Date-mean per horizon → subtract to get cross-sectional excess.
    for h in horizons:
        date_mean = panel.groupby("date")[f"fwd_ret_{h}"].transform("mean")
        panel[f"fwd_excess_{h}"] = panel[f"fwd_ret_{h}"] - date_mean
        sector_mean = panel.groupby(["date", "sector"])[f"fwd_ret_{h}"].transform("mean")
        panel[f"fwd_sector_excess_{h}"] = panel[f"fwd_ret_{h}"] - sector_mean

        spy_col = f"spy_ret{h}d"
        if spy_col in panel.columns:
            panel[f"fwd_spy_ret_{h}"] = (
                panel.groupby("ticker")[spy_col]
                .transform(lambda s: s.shift(-h))
            )
            panel[f"fwd_spy_excess_{h}"] = panel[f"fwd_ret_{h}"] - panel[f"fwd_spy_ret_{h}"]
        else:
            panel[f"fwd_spy_ret_{h}"] = np.nan
            panel[f"fwd_spy_excess_{h}"] = np.nan
    return panel


def _target_col(target: str, horizon: int) -> str:
    if target == "raw_return":
        return f"fwd_ret_{horizon}"
    if target == "sector_excess":
        return f"fwd_sector_excess_{horizon}"
    if target == "spy_excess":
        return f"fwd_spy_excess_{horizon}"
    return f"fwd_excess_{horizon}"


def daily_rank_ic(
    panel: pd.DataFrame,
    feature: str,
    target: str,
    min_tickers: int = MIN_TICKERS_PER_DAY,
) -> dict | None:
    """Compute the daily Spearman rank correlation of `feature` vs `target`.

    Implementation note — vectorized across dates:
        1. Pivot to (date × ticker) wide frames for both feature and target.
        2. Rank each ROW (across tickers within that date) → ranks in [1, N].
        3. Pearson correlation between the two row-rank vectors per date is
           equivalent to Spearman.  This is much faster than groupby+spearmanr.
    """
    sub = panel[["date", "ticker", feature, target]].dropna()
    if sub.empty:
        return None
    feat = sub.pivot(index="date", columns="ticker", values=feature)
    targ = sub.pivot(index="date", columns="ticker", values=target)
    # Rank within row (axis=1).  NaNs stay NaN — pandas drops them from rank
    # automatically so we don't bias toward zero.
    feat_r = feat.rank(axis=1, method="average", na_option="keep")
    targ_r = targ.rank(axis=1, method="average", na_option="keep")
    # Per-day count of valid (feature, target) pairs.
    valid = (~feat_r.isna()) & (~targ_r.isna())
    n_per_day = valid.sum(axis=1)
    keep = n_per_day >= min_tickers
    if keep.sum() < 50:
        return None  # too few usable days for a stable estimate
    feat_r = feat_r.loc[keep]
    targ_r = targ_r.loc[keep]
    valid = valid.loc[keep]
    # Mask invalid cells so they don't drag means/stds.
    f = feat_r.where(valid)
    t = targ_r.where(valid)
    # Per-row Pearson on ranks = Spearman.
    f_mean = f.mean(axis=1)
    t_mean = t.mean(axis=1)
    f_centered = f.sub(f_mean, axis=0)
    t_centered = t.sub(t_mean, axis=0)
    num = (f_centered * t_centered).sum(axis=1, min_count=1)
    f_ss = (f_centered ** 2).sum(axis=1, min_count=1)
    t_ss = (t_centered ** 2).sum(axis=1, min_count=1)
    denom = np.sqrt(f_ss * t_ss)
    ic_series = (num / denom).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ic_series) < 50:
        return None
    arr = ic_series.values.astype(float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    n = len(arr)
    t_stat = mean / (std / np.sqrt(n)) if std > 1e-12 else 0.0
    return {
        "n_days": n,
        "mean_ic": mean,
        "std_ic": std,
        "t_stat": float(t_stat),
        "hit_rate": float((arr > 0).mean()),
    }


def score_all_features(
    panel: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    targets: list[str],
) -> pd.DataFrame:
    """Run daily_rank_ic for every feature × target × horizon."""
    rows = []
    total = len(feature_cols) * len(horizons) * len(targets)
    done = 0
    for h in horizons:
        for target_name in targets:
            target = _target_col(target_name, h)
            if target not in panel.columns:
                log.warning("[ic] target column missing: %s", target)
                done += len(feature_cols)
                continue
            for feat in feature_cols:
                done += 1
                if done % 50 == 0:
                    log.info("[ic] %d / %d  feature×target×horizon scored", done, total)
                res = daily_rank_ic(panel, feat, target)
                if res is None:
                    continue
                rows.append({
                    "feature": feat,
                    "family": _family_of(feat),
                    "target": target_name,
                    "target_col": target,
                    "horizon": h,
                    **res,
                    "abs_mean_ic": abs(res["mean_ic"]),
                    "abs_t_stat": abs(res["t_stat"]),
                    "preferred_direction": "high_is_good" if res["mean_ic"] >= 0 else "low_is_good",
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["horizon", "target", "abs_t_stat"], ascending=[True, True, False])


def aggregate_by_family(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Per-family rollup: mean |t_stat|, mean |IC|, count of features in family."""
    g = ic_df.groupby(["horizon", "target", "family"])
    agg = g.agg(
        n_features=("feature", "nunique"),
        mean_abs_t=("abs_t_stat", "mean"),
        max_abs_t=("abs_t_stat", "max"),
        mean_abs_ic=("mean_ic", lambda s: float(np.mean(np.abs(s)))),
    ).reset_index()
    return agg.sort_values(["horizon", "target", "mean_abs_t"], ascending=[True, True, False])


def build_shortlist(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Features strong enough to deserve model inclusion / focused A-B tests."""
    if ic_df.empty:
        return ic_df
    out = ic_df[
        (ic_df["abs_t_stat"] >= ALIVE_TSTAT)
        & (ic_df["abs_mean_ic"] >= ALIVE_MEAN_IC)
    ].copy()
    return out.sort_values(
        ["horizon", "target", "abs_t_stat", "abs_mean_ic"],
        ascending=[True, True, False, False],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-feature daily Spearman IC report")
    parser.add_argument(
        "--horizon", type=int, nargs="*", default=[5, 10, 20],
        help="Forward return horizons in trading days (default: 5 10 20)",
    )
    parser.add_argument(
        "--target",
        nargs="*",
        choices=TARGET_CHOICES,
        default=["xs_excess"],
        help="Targets to score features against (default: xs_excess)",
    )
    parser.add_argument("--top", type=int, default=30, help="How many top rows to print")
    parser.add_argument(
        "--min-tickers", type=int, default=MIN_TICKERS_PER_DAY,
        help="Minimum tickers required per date for that date's IC to count",
    )
    args = parser.parse_args()

    horizons = sorted(set(int(h) for h in args.horizon))
    targets = list(dict.fromkeys(args.target))
    log.info("[start] horizons=%s  targets=%s  min_tickers/day=%d", horizons, targets, args.min_tickers)

    universe = _resolve_universe()
    log.info("[start] universe size=%d  (UNIVERSE_MODE=%s)", len(universe), UNIVERSE_MODE)

    panel = load_panel(universe)
    panel = add_forward_targets(panel, horizons)

    # Discover candidate feature columns: numeric, not in NON_FEATURE_COLS,
    # not a forward-target column we just added.
    fwd_cols = {
        c for h in horizons
        for c in (
            f"fwd_ret_{h}",
            f"fwd_excess_{h}",
            f"fwd_sector_excess_{h}",
            f"fwd_spy_ret_{h}",
            f"fwd_spy_excess_{h}",
        )
    }
    feature_cols = [
        c for c in panel.columns
        if c not in NON_FEATURE_COLS
        and c not in fwd_cols
        and not str(c).startswith("fwd_")
        and pd.api.types.is_numeric_dtype(panel[c])
    ]
    log.info("[start] %d candidate feature columns", len(feature_cols))

    ic_df = score_all_features(panel, feature_cols, horizons, targets)
    if ic_df.empty:
        log.error("No usable IC scores produced — universe too sparse?")
        return 1

    # ── Persist ────────────────────────────────────────────────────────────────
    out_features = os.path.join(LOG_DIR, "feature_ic_report.csv")
    out_family = os.path.join(LOG_DIR, "feature_ic_family.csv")
    out_shortlist = os.path.join(LOG_DIR, "feature_ic_shortlist.csv")
    ic_df.to_csv(out_features, index=False)
    fam_df = aggregate_by_family(ic_df)
    fam_df.to_csv(out_family, index=False)
    shortlist = build_shortlist(ic_df)
    shortlist.to_csv(out_shortlist, index=False)
    log.info("[saved] per-feature: %s", out_features)
    log.info("[saved] per-family:  %s", out_family)
    log.info("[saved] shortlist:   %s", out_shortlist)

    # ── Console summary ───────────────────────────────────────────────────────
    primary_h = horizons[0]
    primary_target = targets[0]
    print(f"\n=== TOP {args.top} FEATURES BY |t-stat| @ horizon={primary_h}d target={primary_target} ===")
    top = (
        ic_df[(ic_df["horizon"] == primary_h) & (ic_df["target"] == primary_target)]
        .head(args.top)[["feature", "family", "n_days", "mean_ic", "t_stat", "hit_rate", "preferred_direction"]]
    )
    print(top.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print(f"\n=== FAMILY ROLLUP @ horizon={primary_h}d target={primary_target} ===")
    fam_primary = fam_df[(fam_df["horizon"] == primary_h) & (fam_df["target"] == primary_target)]
    print(fam_primary.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ── Verdict line ──────────────────────────────────────────────────────────
    # A feature with |t_stat| >= 3 has a >99% chance its mean IC is non-zero.
    # Count how many alive features each horizon has — that's the headline.
    print("\n=== ALIVE FEATURE COUNT (|t_stat| >= 3) ===")
    for h in horizons:
        for target_name in targets:
            h_df = ic_df[(ic_df["horizon"] == h) & (ic_df["target"] == target_name)]
            alive_df = build_shortlist(h_df)
            n_alive = len(alive_df)
            n_total = len(h_df)
            best = h_df.iloc[0] if len(h_df) else None
            best_str = (
                f"  best: {best['feature']:<35s} t={best['t_stat']:+.2f}  IC={best['mean_ic']:+.4f}"
                if best is not None else ""
            )
            print(f"horizon={h:>3d}d target={target_name:<13s}: {n_alive:>3d} / {n_total} alive{best_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
