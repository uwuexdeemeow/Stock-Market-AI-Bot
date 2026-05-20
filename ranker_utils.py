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

import os
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


def load_adaptive_factor_weights(
    adaptive_weights_file: str,
    fallback_weights: dict[str, float],
    factor_cols: list[str],
    *,
    max_age_days: int = 7,
    now: pd.Timestamp | None = None,
    return_metadata: bool = False,
):
    """Load validated adaptive factor weights or fall back to static weights."""
    import json as _json
    from pathlib import Path

    fallback = {col: float(fallback_weights.get(col, 0.0) or 0.0) for col in factor_cols}
    total = sum(v for v in fallback.values() if np.isfinite(v))
    if total > 0:
        fallback = {col: round(float(fallback[col]) / total, 6) for col in factor_cols}
    else:
        fallback = {col: round(1.0 / max(len(factor_cols), 1), 6) for col in factor_cols}
    metadata = {
        "adaptive_weight_status": "fallback",
        "adaptive_weight_reason": "not_loaded",
        "adaptive_weights_file": str(adaptive_weights_file),
    }

    try:
        payload = _json.loads(Path(adaptive_weights_file).read_text(encoding="utf-8", errors="replace"))
        weights_raw = payload.get("weights", {})
        missing = [col for col in factor_cols if col not in weights_raw]
        if missing:
            raise ValueError("missing_weights:" + ",".join(missing))
        weights = {col: float(weights_raw[col]) for col in factor_cols}
        if not all(np.isfinite(v) and v >= 0.0 for v in weights.values()):
            raise ValueError("non_finite_or_negative_weights")
        weight_sum = float(sum(weights.values()))
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"weights_sum={weight_sum:.4f}")

        computed_at = pd.Timestamp(payload.get("computed_at"))
        if computed_at.tzinfo is not None:
            computed_at = computed_at.tz_convert(None)
        now_ts = pd.Timestamp.now() if now is None else pd.Timestamp(now)
        if now_ts.tzinfo is not None:
            now_ts = now_ts.tz_convert(None)
        age_days = float((now_ts - computed_at).total_seconds() / 86400.0)
        if age_days > float(max_age_days):
            raise ValueError(f"stale_weights:{age_days:.1f}d")

        weights = {col: round(float(weights[col]) / weight_sum, 6) for col in factor_cols}
        metadata.update({
            "adaptive_weight_status": "adaptive",
            "adaptive_weight_reason": "loaded",
            "adaptive_weight_age_days": round(age_days, 3),
            "adaptive_weight_computed_at": str(payload.get("computed_at", "")),
        })
        return (weights, metadata) if return_metadata else weights
    except Exception as exc:
        metadata["adaptive_weight_reason"] = str(exc)
        return (fallback, metadata) if return_metadata else fallback


# ── Adaptive factor weight computation ────────────────────────────────────────

def compute_adaptive_factor_weights(
    data_dir: str,
    factor_cols: list[str],
    lookback_days: int = 252,
    halflife: int = 63,
    floor: float = 0.05,
    target_horizon_days: int = 5,
    output_path: str | None = None,
) -> dict[str, float]:
    """Recompute factor weights from trailing IC with exponential decay.

    PLAIN ENGLISH: For each factor, look at the last ~year of data and
    measure how well it predicted future returns (daily rank correlation).
    Recent days count more (exponential weighting with given halflife).
    Factors with strong recent IC get more weight; those that lost their
    edge get less (but never below the floor — the IC might recover).

    Args:
        data_dir: path to directory with ticker parquet files (AAPL.parquet etc.)
        factor_cols: list of factor column names to weight
        lookback_days: how many trading days of history to use for IC calculation
        halflife: exponential decay halflife in days (63 = 1 quarter)
        floor: minimum weight per factor (prevents total zeroing)
        output_path: if set, writes JSON file with weights + diagnostics

    Returns:
        dict of factor_col → weight (sums to 1.0)
    """
    import glob
    import json as _json
    from pathlib import Path
    from datetime import datetime

    # Load all ticker parquets to build a cross-sectional panel
    parquets = glob.glob(os.path.join(data_dir, "*.parquet"))
    frames = []
    for pq in parquets:
        ticker = Path(pq).stem
        # Skip non-ticker files (e.g. _metadata, long names)
        if ticker.startswith("_") or len(ticker) > 6 or "-" in ticker:
            continue
        try:
            df = pd.read_parquet(pq)
        except Exception:
            continue
        if not {"Open", "Close"}.issubset(df.columns):
            continue
        available_factors = [c for c in factor_cols if c in df.columns]
        if not available_factors:
            continue
        keep_cols = ["Open", "Close"] + available_factors
        chunk = df[keep_cols].copy()
        chunk["_adaptive_forward_return"] = (
            pd.to_numeric(chunk["Close"], errors="coerce").shift(-int(target_horizon_days))
            / pd.to_numeric(chunk["Open"], errors="coerce").shift(-1)
            - 1.0
        )
        chunk["ticker"] = ticker
        chunk["date"] = chunk.index
        frames.append(chunk)

    if not frames:
        # No data available — return equal weights
        equal = {c: 1.0 / len(factor_cols) for c in factor_cols}
        return equal

    panel = pd.concat(frames, ignore_index=True)
    target_col = "_adaptive_forward_return"
    panel = panel.dropna(subset=[target_col])

    # Keep only the last lookback_days of unique trading dates
    all_dates = sorted(panel["date"].unique())
    if len(all_dates) > lookback_days:
        cutoff = all_dates[-lookback_days]
        panel = panel[panel["date"] >= cutoff]
        all_dates = sorted(panel["date"].unique())

    n_days = len(all_dates)
    if n_days < 30:
        equal = {c: 1.0 / len(factor_cols) for c in factor_cols}
        return equal

    # Exponential decay weights: most recent day gets highest weight.
    # PLAIN ENGLISH: A halflife of 63 means data from 63 days ago counts
    # half as much as today's data.  This makes the weights responsive to
    # recent regime changes without forgetting the past entirely.
    exp_weights = np.array([2.0 ** (-(n_days - 1 - i) / halflife) for i in range(n_days)])
    exp_weights /= exp_weights.sum()

    # Compute exponentially-weighted mean IC for each factor
    factor_ics: dict[str, float] = {}
    for col in factor_cols:
        if col not in panel.columns:
            factor_ics[col] = 0.0
            continue
        daily_ics = []
        for dt in all_dates:
            day = panel[panel["date"] == dt]
            valid = day[[col, target_col]].dropna()
            if len(valid) < 10:
                daily_ics.append(0.0)
                continue
            # Spearman rank correlation: does this factor's ranking match
            # the actual return ranking?
            scores = valid[col].rank().values
            targets = valid[target_col].rank().values
            if scores.std() < 1e-12 or targets.std() < 1e-12:
                daily_ics.append(0.0)
                continue
            rho = float(np.corrcoef(scores, targets)[0, 1])
            daily_ics.append(rho if np.isfinite(rho) else 0.0)

        # Weighted mean IC
        arr = np.array(daily_ics)
        factor_ics[col] = float((arr * exp_weights).sum())

    # IC-weighted with floor:
    # PLAIN ENGLISH: Factors with negative IC get zero raw weight (they're
    # hurting, not helping).  The floor ensures no factor drops below 5% —
    # it might recover next quarter and we don't want to fully abandon it.
    raw_weights = {c: max(factor_ics.get(c, 0.0), 0.0) for c in factor_cols}
    total = sum(raw_weights.values())
    if total > 0:
        floor_total = min(float(floor) * len(factor_cols), 0.95)
        variable_budget = max(0.0, 1.0 - floor_total)
        weights = {
            c: float(floor) + variable_budget * (raw_weights[c] / total)
            for c in factor_cols
        }
    else:
        # All factors have negative IC — equal weight (defensive)
        weights = {c: 1.0 / len(factor_cols) for c in factor_cols}

    # Re-normalize after applying floors (floors can push sum above 1.0)
    w_total = sum(weights.values())
    weights = {c: round(w / w_total, 6) for c, w in weights.items()}

    # Write to JSON for downstream consumers (backtest.py, core_satellite_alpha)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": weights,
            "ics": {c: round(v, 6) for c, v in factor_ics.items()},
            "lookback_days": lookback_days,
            "halflife": halflife,
            "floor": floor,
            "target_horizon_days": int(target_horizon_days),
            "target_source": "open_next_to_close_horizon",
            "weight_status": "adaptive",
            "n_trading_days_used": n_days,
            "n_tickers_in_panel": len(frames),
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        }
        Path(output_path).write_text(_json.dumps(payload, indent=2))

    return weights


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
