"""
ranker_utils.py — small helpers for cross-sectional ranking models.

WHY THIS FILE EXISTS
--------------------
Switching from XGBClassifier (binary up/down) to XGBRanker (rank stocks within
a date) needs two pieces of glue:

    1. A "group" array telling XGBoost which rows belong to the same date.
       XGBRanker compares rows within a group against each other — without
       this it has no idea which stocks are competing on the same day.

    2. A daily Spearman IC evaluator.  AUC is meaningless for a ranker
       because the output is a continuous score, not a probability.  IC is
       what the backtest actually cares about: "do my rankings line up with
       forward returns on each day?"

Both helpers live here so train.py and backtest.py can import the same code
and stay in sync.

PLAIN ENGLISH
-------------
* "Group array" = on day 1 we ranked 147 stocks, on day 2 we ranked 146
  stocks (one had no data), etc.  XGBoost wants this as `[147, 146, ...]`.
* "Daily IC" = on each day, take the model's score for every stock and the
  actual N-day forward return for every stock, ask "did stocks ranked higher
  by the model actually return more?", output a number in [-1, +1].  Average
  that number over thousands of days to get the headline rank IC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_rank_groups_from_dates(dates: np.ndarray) -> np.ndarray:
    """Convert a sorted date array into XGBRanker's `group` argument.

    XGBRanker expects rows to be already sorted so that all rows from one
    group (one date) sit next to each other.  The `group` array then says
    "first 147 rows belong to day 1, next 146 to day 2, ..." — its sum must
    equal the total number of rows.

    Args:
        dates: 1-D array of dates (any orderable dtype, but pre-sorted).

    Returns:
        1-D int64 array of group sizes.

    Example:
        dates  = [2020-01-02, 2020-01-02, 2020-01-03]
        groups = [2, 1]
    """
    if len(dates) == 0:
        return np.zeros(0, dtype=np.int64)
    # Detect group boundaries: positions where the date differs from the
    # previous row.  np.diff returns an array shorter by 1, so we add an
    # explicit "1" at position 0 (always a boundary).
    series = pd.Series(dates)
    # `.ne` plus `.shift` gives a boolean Series with True at each new-group
    # position.  Cumsum then produces a stable group id.
    is_new = series.ne(series.shift())
    group_ids = is_new.cumsum().values
    # value_counts(sort=False) preserves group order.
    counts = pd.Series(group_ids).value_counts(sort=False).values
    return counts.astype(np.int64)


def daily_rank_ic(
    scores: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    min_per_day: int = 15,
    min_total_days: int = 10,
) -> dict:
    """Compute the average daily Spearman rank IC of `scores` vs `targets`.

    On every distinct date in `dates`, rank both `scores` and `targets`
    across the tickers present that day, then take the Pearson correlation
    of those ranks (which equals Spearman on the raw values).  Average that
    daily correlation across all eligible days.

    Days with fewer than `min_per_day` tickers are skipped — small panels
    produce noisy correlations that distort the mean.

    Args:
        scores:  predicted rank scores per row.
        targets: forward returns (or any continuous target) per row.
        dates:   date label per row.
        min_per_day: minimum tickers required for a day's IC to count.

    Returns:
        dict with keys:
            n_days     int   — number of usable days
            mean_ic    float — average daily Spearman IC (signed, [-1, 1])
            std_ic     float — sample stdev of daily ICs
            t_stat     float — mean / (std / sqrt(n)) — > 3 means real
            hit_rate   float — fraction of days where IC > 0
    """
    # Pandas frame is the easiest way to groupby + rank cleanly.
    df = pd.DataFrame({"date": pd.Series(dates), "score": scores, "target": targets})
    df = df.dropna(subset=["score", "target"])
    if df.empty:
        return _empty_ic()

    daily = []
    for _date, g in df.groupby("date"):
        if len(g) < min_per_day:
            continue
        # `.rank()` handles ties via average rank.  Then Pearson on ranks =
        # Spearman on the originals.  np.corrcoef is fast and avoids an
        # extra scipy import.
        r_score = g["score"].rank().values
        r_targ = g["target"].rank().values
        if r_score.std(ddof=1) < 1e-12 or r_targ.std(ddof=1) < 1e-12:
            continue
        rho = float(np.corrcoef(r_score, r_targ)[0, 1])
        if np.isfinite(rho):
            daily.append(rho)

    if len(daily) < min_total_days:
        return _empty_ic()
    arr = np.asarray(daily, dtype=float)
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


def _empty_ic() -> dict:
    return {"n_days": 0, "mean_ic": 0.0, "std_ic": 0.0, "t_stat": 0.0, "hit_rate": 0.0}


# ── Self-test (runs only when executed directly) ──────────────────────────────
if __name__ == "__main__":
    # Synthetic data: 5 dates, 4 tickers per date.  Score = target + noise so
    # IC should be strongly positive (close to 1).
    rng = np.random.default_rng(0)
    n_days = 5
    n_tickers = 4
    dates = np.repeat(np.arange(n_days), n_tickers)
    targets = rng.normal(size=n_days * n_tickers)
    scores = targets + 0.1 * rng.normal(size=n_days * n_tickers)

    groups = build_rank_groups_from_dates(dates)
    assert groups.sum() == len(dates), "group sum must equal total rows"
    assert (groups == n_tickers).all(), "every group should have n_tickers rows"
    print(f"[OK] groups for {n_days} dates × {n_tickers} tickers: {groups.tolist()}")

    # IC should be high and positive.
    res = daily_rank_ic(scores, targets, dates, min_per_day=2, min_total_days=3)
    print(f"[OK] daily IC (high-correlation toy): mean={res['mean_ic']:+.3f}  "
          f"t={res['t_stat']:+.2f}  n_days={res['n_days']}")
    assert res["mean_ic"] > 0.5, "expected high IC on near-perfect synthetic data"

    # Negative-correlation case: scores = -targets → IC ~ -1
    res_neg = daily_rank_ic(-targets, targets, dates, min_per_day=2, min_total_days=3)
    print(f"[OK] daily IC (negated): mean={res_neg['mean_ic']:+.3f}")
    assert res_neg["mean_ic"] < -0.5, "expected strongly negative IC"

    # Random case: IC should be ≈ 0.
    res_rand = daily_rank_ic(
        rng.normal(size=n_days * n_tickers), targets, dates, min_per_day=2, min_total_days=3
    )
    print(f"[OK] daily IC (random):  mean={res_rand['mean_ic']:+.3f}  (expect ≈ 0)")

    # Group-builder edge case: variable group sizes
    dates2 = np.array([1, 1, 1, 2, 2, 3])
    g2 = build_rank_groups_from_dates(dates2)
    assert g2.tolist() == [3, 2, 1], f"unexpected groups: {g2}"
    print(f"[OK] variable groups [1,1,1,2,2,3] → {g2.tolist()}")

    print("\nAll ranker_utils self-tests passed.")
