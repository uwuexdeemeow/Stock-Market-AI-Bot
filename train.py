from __future__ import annotations

"""
train.py
========
XGBoost-only trainer with strict temporal discipline.

Fixes applied (v2):
- walk_forward_summary() now clearly documented as a diagnostic CV over the
  full dataset (including test period). It is NOT the primary OOS validation —
  that comes from backtest.py. A note in the summary JSON makes this explicit.
- Calibrator fallback rate is now tracked and logged: if the isotonic
  calibrator cannot be fit (< 25 samples or single class), the threshold falls
  back to DEFAULT_FIXED_CONFIDENCE_THRESHOLD. The fallback is recorded in the
  saved metadata so backtest.py can surface it.
- model_version bumped to xgb_complete_v2 to distinguish from old artifacts.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,   # area under precision-recall curve — robust under class imbalance
    classification_report,
    roc_auc_score,             # ranking-quality metric the cross-sectional ranker actually uses
)
from sklearn.preprocessing import StandardScaler

from confidence_calibration import fit_direction_calibrator, save_direction_calibrator
from drift_monitor import snapshot_from_metadata
from labels import (
    make_direction_target as shared_make_direction_target,
    make_forward_return_target,
    make_return_bucket_target,
    make_spy_forward_return,
)
from model_quality import evaluate_model_quality, update_scaler_metadata, upsert_quality_report
from model_registry import register as register_model_run
from nested_cv import nested_walk_forward_search
from pipeline_shared import apply_sentiment_distribution_matching, fit_sentiment_zscore_stats
from safe_io import run_utf8
from settings import (
    DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    RETURN_HORIZON_DAYS,
    RETURN_BINS,
    RETURN_BIN_LABELS,
    TRAIN_CALIBRATION_SPLIT,
    CALIBRATION_TEST_SPLIT,
    EMBARGO_DAYS,
    CONFIDENCE_TARGET_PRECISION,
    DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
    DIRECTION_LABEL_THRESHOLD,
    PREDICTION_TARGET,
    EXCESS_RETURN_MIN_PCT,
    VOL_ADJUSTED_SHARPE_THRESHOLD,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOP_ROUNDS,
    XGB_MIN_CHILD_WEIGHT,
    XGB_GAMMA,
    XGB_REG_ALPHA,
    XGB_REG_LAMBDA,
    POOLED_TRAINING,
    FEATURE_IMPORTANCE_TOP_K,
    H5_SENTIMENT_FEATURES_ENABLED,
    CONFORMAL_ALPHA,
    CONFORMAL_MIN_CALIBRATION_ROWS,
    TUNE_HYPERPARAMS,
    MULTI_HORIZON_H20_WEIGHT,
    MULTI_HORIZON_H5_WEIGHT,
    SURVIVORSHIP_TRAINING_TICKERS,
    WATCHLIST,
    REGIME_CONDITIONAL_MODELS,
    REGIME_MIN_TRAIN_ROWS,
    EXCLUDE_MARKET_CONSTANTS,
    POOLED_RANKER_EXPERIMENT_ENABLED,
)
from xgb_feature_engineering import build_xgb_matrix

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "train.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("train")

N_RETURN_BINS = len(RETURN_BIN_LABELS)
RETURN_BIN_CENTRES = np.array([-0.04, -0.02, 0.00, 0.02, 0.04], dtype=float)
FEATURE_MIN_STD = 1e-9
FEATURE_MAX_MISSING_RATE = 0.50


def build_rank_groups_from_dates(date_array: np.ndarray | pd.DatetimeIndex | pd.Series) -> list[int]:
    """Return XGBRanker group sizes for rows already sorted by date."""
    dates = pd.DatetimeIndex(date_array)
    if len(dates) == 0:
        return []
    groups: list[int] = []
    current = dates[0]
    count = 0
    for dt in dates:
        if dt == current:
            count += 1
        else:
            groups.append(count)
            current = dt
            count = 1
    groups.append(count)
    return groups


def daily_rank_ic_from_scores(
    dates: np.ndarray | pd.DatetimeIndex | pd.Series,
    scores: np.ndarray,
    realized: np.ndarray,
    *,
    min_names_per_day: int = 5,
) -> dict:
    """Daily Spearman IC of ranker scores vs realized forward excess returns."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "score": pd.to_numeric(pd.Series(scores), errors="coerce"),
        "realized": pd.to_numeric(pd.Series(realized), errors="coerce"),
    }).dropna()
    rows: list[float] = []
    counts: list[int] = []
    for _, day in df.groupby("date"):
        if len(day) < min_names_per_day:
            continue
        if day["score"].nunique(dropna=True) < 2 or day["realized"].nunique(dropna=True) < 2:
            continue
        ic = day["score"].corr(day["realized"], method="spearman")
        if pd.notna(ic) and np.isfinite(ic):
            rows.append(float(ic))
            counts.append(int(len(day)))
    if not rows:
        return {
            "mean_ic": None,
            "t_stat": None,
            "median_ic": None,
            "std_ic": None,
            "positive_day_rate": None,
            "days": 0,
            "avg_names_per_day": None,
        }
    arr = np.asarray(rows, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    mean = float(arr.mean())
    t_stat = mean / (std / np.sqrt(len(arr))) if len(arr) > 1 and std > 0 else None
    return {
        "mean_ic": round(mean, 4),
        "t_stat": round(float(t_stat), 3) if t_stat is not None else None,
        "median_ic": round(float(np.median(arr)), 4),
        "std_ic": round(std, 4),
        "positive_day_rate": round(float((arr > 0).mean()), 4),
        "days": int(len(arr)),
        "avg_names_per_day": round(float(np.mean(counts)), 1) if counts else None,
    }


def _legacy_direction_target_wrapper(
    df: pd.DataFrame | pd.Series,
    stock_fwd: pd.Series | None = None,
    spy_fwd: pd.Series | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper around the shared label builder."""
    return shared_make_direction_target(
        df,
        prediction_target=PREDICTION_TARGET,
        horizon=RETURN_HORIZON_DAYS,
        stock_fwd=stock_fwd,
        spy_fwd=spy_fwd,
        direction_threshold=DIRECTION_LABEL_THRESHOLD,
        excess_return_min_pct=EXCESS_RETURN_MIN_PCT,
        vol_adjusted_sharpe_threshold=VOL_ADJUSTED_SHARPE_THRESHOLD,
    )


def build_feature_audit(df: pd.DataFrame, feature_cols: list[str], y_dir: np.ndarray | None = None) -> tuple[dict, list[str]]:
    rows = []
    dropped: list[str] = []
    target = pd.Series(y_dir, index=df.index) if y_dir is not None and len(y_dir) == len(df) else None

    for col in feature_cols:
        raw = pd.to_numeric(df[col], errors="coerce")
        finite = raw.replace([np.inf, -np.inf], np.nan)
        missing_rate = float(finite.isna().mean())
        inf_count = int(np.isinf(raw.to_numpy(dtype=float, na_value=np.nan)).sum())
        std = float(finite.fillna(0.0).std())
        years = df.index.year if isinstance(df.index, pd.DatetimeIndex) else pd.Series(index=df.index, data=0)
        yearly_means = finite.groupby(years).mean()
        stability_ratio = 0.0
        if len(yearly_means.dropna()) > 1:
            stability_ratio = float(yearly_means.std() / (abs(yearly_means.mean()) + 1e-9))

        corr_by_year: dict[str, float | None] = {}
        if target is not None:
            for year, values in finite.groupby(years):
                aligned_target = target.loc[values.index]
                if values.nunique(dropna=True) > 1 and aligned_target.nunique(dropna=True) > 1:
                    corr = values.fillna(0.0).corr(aligned_target)
                    corr_by_year[str(year)] = round(float(corr), 4) if pd.notna(corr) else None
                else:
                    corr_by_year[str(year)] = None

        reason = None
        if std <= FEATURE_MIN_STD:
            reason = "near_zero_variance"
        elif missing_rate > FEATURE_MAX_MISSING_RATE:
            reason = f"missing_rate>{FEATURE_MAX_MISSING_RATE:.0%}"
        if reason:
            dropped.append(col)

        rows.append({
            "feature": col,
            "missing_rate": round(missing_rate, 4),
            "inf_count": inf_count,
            "std": std,
            "stability_ratio": round(stability_ratio, 4),
            "target_corr_by_year": corr_by_year,
            "dropped_reason": reason,
        })

    report = {
        "feature_count_before": len(feature_cols),
        "feature_count_after": len(feature_cols) - len(dropped),
        "dropped_features": [{"feature": r["feature"], "reason": r["dropped_reason"]} for r in rows if r["dropped_reason"]],
        "features": rows,
    }
    return report, dropped


def make_return_target(close: pd.Series, stock_fwd: pd.Series | None = None) -> np.ndarray:
    return make_return_bucket_target(close, return_bins=RETURN_BINS, n_return_bins=N_RETURN_BINS, stock_fwd=stock_fwd)


def make_future_return(close: pd.Series, stock_fwd: pd.Series | None = None) -> np.ndarray:
    fwd = stock_fwd if stock_fwd is not None else make_forward_return_target(close, horizon=RETURN_HORIZON_DAYS)
    return fwd.values.astype(float)


def compute_split_indices(n_total: int) -> dict:
    split_train = int(n_total * TRAIN_CALIBRATION_SPLIT)
    split_test = int(n_total * CALIBRATION_TEST_SPLIT)
    train_end = max(0, split_train - EMBARGO_DAYS)
    calib_start = split_train
    calib_end = max(calib_start + 20, split_test - EMBARGO_DAYS)
    test_start = split_test
    return {
        "train_end": train_end,
        "calib_start": calib_start,
        "calib_end": calib_end,
        "test_start": test_start,
    }


def compute_dynamic_threshold(
    calibrator,
    target_precision: float = 0.58,
    default: float = 57.5,
    raw_probs: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> tuple[float, bool]:
    """
    Returns (threshold, used_fallback).

    Priority order:
    1. Isotonic calibrator → scan calibrated probability grid for target precision.
    2. Raw XGBoost probs + labels → bin by confidence percentile and find where
       precision >= target. Data-driven even without isotonic calibration.
    3. Hard-coded default (57.5). Last resort only.
    """
    observed_conf_ceiling = None
    if raw_probs is not None:
        _rp = np.asarray(raw_probs, dtype=float)
        _rp = _rp[np.isfinite(_rp)]
        if len(_rp):
            observed_conf_ceiling = float(np.max(np.maximum(_rp, 1.0 - _rp) * 100.0))

    def _cap_to_observed(value: float) -> float:
        if observed_conf_ceiling is None or observed_conf_ceiling >= value:
            return float(value)
        return float(max(52.0, np.floor(observed_conf_ceiling * 2.0) / 2.0))

    # --- Path 1: calibrator available ---
    if calibrator is not None:
        raw_grid = np.linspace(0.50, 0.99, 200)
        try:
            cal_grid = calibrator.predict(raw_grid)
            conf_grid = np.maximum(cal_grid, 1.0 - cal_grid) * 100.0
            idx = np.where(cal_grid >= target_precision)[0]
            if len(idx):
                return _cap_to_observed(float(np.clip(conf_grid[idx[0]], 52.0, 80.0))), False
        except Exception:
            pass

    # --- Path 2: scan raw XGBoost probs directly ---
    # Isotonic calibration failed its quality test, but we still have the raw
    # probabilities and true labels on the calibration split. Convert raw probs
    # to confidence scores (max(p, 1-p)*100) and find the lowest threshold where
    # precision meets the target with at least 20 samples above it.
    if raw_probs is not None and y_true is not None:
        probs = np.asarray(raw_probs, dtype=float)
        labels = np.asarray(y_true, dtype=int)
        conf = np.maximum(probs, 1.0 - probs) * 100.0
        pred_up = (probs >= 0.5).astype(int)
        # Scan thresholds from low to high; stop at first one meeting the target.
        for t in np.arange(52.0, 80.0, 0.5):
            mask = conf >= t
            if mask.sum() < 20:
                break  # too few samples — every higher threshold also has too few
            precision = float((pred_up[mask] == labels[mask]).mean())
            if precision >= target_precision:
                return float(t), True  # still flagged as fallback (no calibrator)

    # --- Path 3: fixed fallback, capped at the observed confidence ceiling ---
    # Returning 57.5 when the model never emits above 55.8 creates a zero-trade
    # dead zone. Keep this flagged as fallback so live code can refuse to treat
    # it as calibrated edge.
    if observed_conf_ceiling is not None and observed_conf_ceiling < default:
        return _cap_to_observed(default), True

    # --- Path 4: hard-coded default ---
    return default, True


def build_models(X_train, y_dir_train, X_cal, y_dir_cal, y_ret_train, y_ret_cal,
                 sample_weight: np.ndarray | None = None,
                 param_overrides: dict | None = None):
    """
    param_overrides: if provided by tune_xgb_best_tickers.py --apply, these
    values replace the corresponding settings.py defaults for this run only.
    Accepted keys: max_depth, learning_rate, subsample, colsample_bytree,
    min_child_weight.
    """
    common = dict(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        gamma=XGB_GAMMA,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        tree_method="hist",
        early_stopping_rounds=XGB_EARLY_STOP_ROUNDS,
        random_state=42,
    )
    if param_overrides:
        allowed = {"max_depth", "learning_rate", "subsample", "colsample_bytree",
                   "min_child_weight", "reg_lambda", "reg_alpha", "gamma"}
        applied = {k: v for k, v in param_overrides.items() if k in allowed}
        common.update(applied)
        log.info("build_models: applying tuned param overrides: %s", applied)

    # Class-balance weighting via per-row sample_weight — NOT scale_pos_weight.
    #
    # WHY NOT scale_pos_weight:
    #   scale_pos_weight multiplies the gradient of every positive example by spw.
    #   This is equivalent to replicating positives spw times in the training set,
    #   which shifts XGBoost's internal bias so raw output probabilities average
    #   around spw/(1+spw) instead of the true UP rate.  With UP rate ≈ 35% and
    #   spw ≈ 1.86, model outputs average ~0.65 even with zero signal → everything
    #   ≥ 0.5 → accuracy = UP rate = baseline.  This caused accuracy to look
    #   identical to baseline on every training run.
    #
    # WHY per-row sample_weight works:
    #   Rescales individual loss terms; does NOT shift the output probability scale.
    #   The 0.5 threshold stays correct.  Class balance = upweight positives:
    #   w_pos = n_neg/n_pos, w_neg = 1.0.
    n_pos = max(int(np.sum(y_dir_train == 1)), 1)
    n_neg = max(int(np.sum(y_dir_train == 0)), 1)
    class_w = np.where(y_dir_train == 1, float(n_neg) / n_pos, 1.0).astype(np.float32)
    # Combine with caller-supplied weights (e.g. time-decay); multiply element-wise.
    combined_weight = (sample_weight * class_w if sample_weight is not None else class_w).astype(np.float32)
    log.info("build_models: class balance via sample_weight  (n_pos=%d, n_neg=%d, w_pos=%.3f)",
             n_pos, n_neg, float(n_neg) / n_pos)

    dir_model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        **common   # no scale_pos_weight — class balance handled by combined_weight
    )
    dir_model.fit(X_train, y_dir_train, sample_weight=combined_weight,
                  eval_set=[(X_cal, y_dir_cal)], verbose=False)
    ret_model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=N_RETURN_BINS, eval_metric="mlogloss", **common
    )
    ret_model.fit(X_train, y_ret_train, sample_weight=sample_weight,
                  eval_set=[(X_cal, y_ret_cal)], verbose=False)
    return dir_model, ret_model


def compute_bucket_edges(conf: np.ndarray, n_buckets: int = 10) -> list[tuple[float, float]]:
    """Compute quantile-based confidence bucket boundaries from the real distribution.

    PLAIN ENGLISH: Instead of hardcoded cutoffs like [50-55, 55-60, ...], this
    splits the actual confidence values into n_buckets equal-count groups.
    If the model only ever outputs 50.0–52.5%, all buckets will fall inside
    that range — so every bucket will actually have observations and we can
    measure precision honestly instead of reporting empty fixed bins.

    Args:
        conf: 1-D array of confidence values in [50, 100].
        n_buckets: Number of equal-count buckets to create (default 10).

    Returns:
        List of (lo, hi) edge pairs — same shape as the fixed-edge list used by
        confidence_buckets(), but computed from the data.
    """
    conf_arr = np.asarray(conf, dtype=float)
    conf_arr = conf_arr[np.isfinite(conf_arr)]
    if conf_arr.size == 0:
        raw = np.linspace(50.0, 100.0, max(int(n_buckets), 1) + 1)
        return [(float(raw[i]), float(raw[i + 1])) for i in range(len(raw) - 1)]

    conf_arr = np.clip(conf_arr, 50.0, 100.0)
    n_buckets = max(int(n_buckets), 1)
    percentile_pts = np.linspace(0, 100, n_buckets + 1)
    raw = np.percentile(conf_arr, percentile_pts)
    # Keep full precision for bucket membership. Rounding here can collapse
    # nearby quantiles, creating artificial empty buckets in tight 50-52% regimes.
    raw[0] = 50.0
    raw[-1] = max(float(raw[-1]), float(conf_arr.max()))
    return [(float(raw[i]), float(raw[i + 1])) for i in range(n_buckets)]


def confidence_buckets(
    probs: np.ndarray,
    y_true: np.ndarray,
    edges: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Precision and coverage for each confidence bucket.

    PLAIN ENGLISH: Groups predictions by how confident the model was, then
    checks what fraction of those predictions were correct (precision). This
    tells us whether higher confidence actually means better predictions.

    Args:
        probs: Model output probabilities (values 0–1, one per prediction).
        y_true: Actual true labels (0 or 1).
        edges: Optional list of (lo, hi) confidence-score ranges in [50, 100].
               If None, uses fixed legacy edges [50-55, 55-60, 60-65, 65-70, 70-100].
               Pass the result of compute_bucket_edges() for distribution-adaptive bins.

    Returns:
        List of dicts — one per bucket — with keys: bucket, n, precision, avg_confidence.
    """
    # Confidence = how far the probability is from the uncertain 0.5 midpoint,
    # re-scaled to [50, 100] so 50 = coin flip, 100 = certain.
    conf = np.maximum(probs, 1.0 - probs) * 100.0
    pred = (probs >= 0.5).astype(int)

    # Fall back to hardcoded legacy edges when caller doesn't provide adaptive ones.
    edge_list: list[tuple[float, float]] = edges if edges is not None else [
        (50, 55), (55, 60), (60, 65), (65, 70), (70, 100)
    ]

    rows: list[dict] = []
    for i, (lo, hi) in enumerate(edge_list):
        # Last bucket is inclusive on the right; all others are [lo, hi).
        is_last = (i == len(edge_list) - 1)
        mask = (conf >= lo) & (conf <= hi if is_last else conf < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bucket": f"{lo:.1f}-{hi:.1f}", "n": 0,
                "precision": None, "avg_confidence": None,
                "lo": float(lo), "hi": float(hi),
            })
            continue
        rows.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "n": n,
            "precision": round(float((pred[mask] == y_true[mask]).mean()), 4),
            "avg_confidence": round(float(conf[mask].mean()), 2),
            "lo": float(lo),
            "hi": float(hi),
        })
    return rows


def threshold_sweep(probs: np.ndarray, y_true: np.ndarray) -> list[dict]:
    conf = np.maximum(probs, 1.0 - probs) * 100.0
    pred = (probs >= 0.5).astype(int)
    rows: list[dict] = []
    for thr in np.arange(52.5, 75.1, 2.5):
        mask = conf >= thr
        if mask.sum() == 0:
            rows.append({"threshold": round(float(thr), 1), "coverage": 0.0, "precision": None})
            continue
        rows.append(
            {
                "threshold": round(float(thr), 1),
                "coverage": round(float(mask.mean()), 4),
                "precision": round(float((pred[mask] == y_true[mask]).mean()), 4),
            }
        )
    return rows


def confidence_diagnostics(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    buckets: list[dict] | None = None,
) -> dict:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    mask = np.isfinite(probs)
    probs = probs[mask]
    labels = labels[mask]
    if len(probs) == 0:
        return {
            "confidence_validated": False,
            "reason": "no_probability_rows",
            "max_confidence": None,
            "threshold_coverage": 0.0,
            "confidence_correctness_spearman": None,
        }

    conf = np.maximum(probs, 1.0 - probs) * 100.0
    pred = (probs >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    corr = pd.Series(conf).corr(pd.Series(correct), method="spearman")
    threshold_coverage = float((conf >= float(threshold)).mean())

    bucket_rows = buckets or confidence_buckets(probs, labels)
    precisions = [
        float(row["precision"])
        for row in bucket_rows
        if row.get("precision") is not None and int(row.get("n", 0) or 0) >= 15
    ]
    monotonic = False
    if len(precisions) >= 3:
        x = pd.Series(np.arange(len(precisions)), dtype=float)
        y = pd.Series(precisions, dtype=float)
        bucket_corr = x.corr(y, method="spearman")
        monotonic = bool(pd.notna(bucket_corr) and float(bucket_corr) >= 0.20)
    else:
        bucket_corr = None

    validated = bool(
        threshold_coverage > 0.0
        and pd.notna(corr)
        and float(corr) >= 0.05
        and monotonic
    )
    reasons = []
    if threshold_coverage <= 0.0:
        reasons.append("threshold_above_confidence_ceiling")
    if pd.isna(corr) or float(corr) < 0.05:
        reasons.append("confidence_not_correlated_with_correctness")
    if not monotonic:
        reasons.append("bucket_precision_not_monotonic")

    return {
        "confidence_validated": validated,
        "reason": "ok" if validated else "; ".join(reasons),
        "max_confidence": round(float(np.max(conf)), 4),
        "min_confidence": round(float(np.min(conf)), 4),
        "threshold_coverage": round(threshold_coverage, 4),
        "confidence_correctness_spearman": round(float(corr), 4) if pd.notna(corr) else None,
        "bucket_precision_spearman": round(float(bucket_corr), 4) if bucket_corr is not None and pd.notna(bucket_corr) else None,
    }


def compute_conformal_qhat(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = CONFORMAL_ALPHA,
    min_rows: int = CONFORMAL_MIN_CALIBRATION_ROWS,
) -> float | None:
    """Split-conformal nonconformity threshold for binary direction probabilities."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    mask = np.isfinite(p) & np.isin(y, [0, 1])
    p = np.clip(p[mask], 0.0, 1.0)
    y = y[mask]
    n = len(p)
    if n < int(min_rows):
        return None
    scores = np.where(y == 1, 1.0 - p, p)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - float(alpha))) / n)
    try:
        return float(np.quantile(scores, q_level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, q_level, interpolation="higher"))


def conformal_prediction_info(p_up: float, qhat: float | None) -> dict:
    """Return conformal set diagnostics for one binary p(up)."""
    if qhat is None or not np.isfinite(qhat):
        return {
            "conformal_qhat": None,
            "conformal_set_size": None,
            "conformal_singleton": True,
            "conformal_contains_up": None,
            "conformal_contains_down": None,
        }
    p = float(np.clip(p_up, 0.0, 1.0))
    include_up = (1.0 - p) <= float(qhat)
    include_down = p <= float(qhat)
    set_size = int(include_up) + int(include_down)
    pred_up = p >= 0.5
    singleton = bool(set_size == 1 and ((pred_up and include_up) or ((not pred_up) and include_down)))
    return {
        "conformal_qhat": float(qhat),
        "conformal_set_size": int(set_size),
        "conformal_singleton": singleton,
        "conformal_contains_up": bool(include_up),
        "conformal_contains_down": bool(include_down),
    }


def evaluate_return_model(ret_model, X_test: np.ndarray, actual_returns: np.ndarray) -> dict:
    probs = ret_model.predict_proba(X_test)
    nrb = min(probs.shape[1], len(RETURN_BIN_CENTRES))
    expected_ret = (probs[:, :nrb] * RETURN_BIN_CENTRES[:nrb]).sum(axis=1)
    spearman = pd.Series(expected_ret).corr(
        pd.Series(actual_returns[: len(expected_ret)]), method="spearman"
    )
    return {
        "spearman_corr": round(float(spearman), 4) if pd.notna(spearman) else None,
        "avg_expected_return_pct": round(float(np.mean(expected_ret) * 100.0), 4),
        "avg_actual_return_pct": round(float(np.mean(actual_returns[: len(expected_ret)]) * 100.0), 4),
    }


def save_xgb_results_plot(
    ticker: str,
    probs: np.ndarray,
    y_true: np.ndarray,
    direction_accuracy: float,
    baseline_up_rate: float,
    rolling_window: int = 20,
) -> str:
    """Save a compact held-out performance chart for the XGBoost-only model."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    pred = (probs >= 0.5).astype(int)
    correct_pct = (pred == y_true).astype(float) * 100.0

    rolling = pd.Series(correct_pct).rolling(rolling_window, min_periods=1).mean()
    plot_path = os.path.join(MODEL_DIR, f"{ticker}_xgb_only_results.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=120)

    axes[0].plot(np.arange(len(rolling)), rolling, color="#1f77b4", linewidth=2)
    axes[0].axhline(50.0, color="#1f77b4", linestyle="--", linewidth=2)
    axes[0].set_title(f"Rolling accuracy ({rolling_window})", fontsize=15)
    axes[0].set_ylim(0, 100)
    axes[0].set_xlim(0, max(1, len(rolling) - 1))
    axes[0].set_yticks(np.arange(0, 101, 20))
    axes[0].grid(False)

    axes[1].bar(
        ["XGB Dir Acc", "Baseline UP"],
        [direction_accuracy, baseline_up_rate],
        color="#1f77b4",
    )
    axes[1].set_title("Held-out accuracy", fontsize=15)
    axes[1].set_ylim(0, 100)
    axes[1].set_yticks(np.arange(0, 101, 20))
    axes[1].grid(False)

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_pooled_ticker_plot(
    ticker_labels: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
) -> str:
    """
    Save a grouped bar chart — one bar-pair per ticker — showing each ticker's
    held-out direction accuracy vs its baseline UP rate.

    A bar taller than its grey baseline means the model has positive edge on
    that ticker.  Bars are coloured green when accuracy beats baseline, red
    when it doesn't.

    ticker_labels : 1-D array of ticker strings, one per test row
    probs         : calibrated p_up for each test row
    y_true        : true direction labels (0/1) for each test row
    """
    probs  = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    preds  = (probs >= 0.5).astype(int)

    # Compute per-ticker stats
    unique_tickers = sorted(set(ticker_labels))
    accs      = []
    baselines = []
    for t in unique_tickers:
        mask = ticker_labels == t
        if mask.sum() == 0:
            accs.append(50.0)
            baselines.append(50.0)
            continue
        acc  = float((preds[mask] == y_true[mask]).mean() * 100.0)
        base = float(y_true[mask].mean() * 100.0)
        accs.append(acc)
        baselines.append(base)

    n = len(unique_tickers)
    x = np.arange(n)
    width = 0.38

    # Green bar = model beats baseline; red = model lags baseline
    bar_colors = ["#2ca02c" if a >= b else "#d62728" for a, b in zip(accs, baselines)]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.85), 5), dpi=120)

    ax.bar(x - width / 2, accs,      width, color=bar_colors, label="Model acc", zorder=3)
    ax.bar(x + width / 2, baselines, width, color="#aec7e8",  label="Baseline",  zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(unique_tickers, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.axhline(50.0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_title("Held-out accuracy vs baseline — per ticker (pooled model)", fontsize=13)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9)
    ax.grid(False)

    fig.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "pooled_ticker_accuracy.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def walk_forward_summary(X_all: np.ndarray, y_all: np.ndarray, n_folds: int = 5) -> dict:
    """
    Diagnostic expanding-window CV over the FULL dataset (train + test period).

    This is NOT the primary OOS validation — it is a sanity check that the
    model learns something across the full history. Primary OOS evidence comes
    from backtest.py which runs a strict walk-forward that never touches
    future data during prediction generation.

    Each fold: train on rows 0..k, test on rows k..k+fold_size.
    Scaler is always fit on the training portion only.
    """
    n_total = len(X_all)
    fold_size = n_total // (n_folds + 1)
    if fold_size < 100:
        return {
            "fold_accuracies": [],
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
            "note": "diagnostic_cv_full_history_not_primary_oos",
        }
    accs: list[float] = []
    for fold in range(n_folds):
        train_end_raw = fold_size * (fold + 1)
        train_end = max(0, train_end_raw - EMBARGO_DAYS)
        test_start = train_end_raw
        test_end = min(fold_size * (fold + 2), n_total)
        if train_end < 100 or test_end <= test_start:
            continue
        scaler = StandardScaler()
        fold_idx = np.arange(0, train_end, max(1, RETURN_HORIZON_DAYS))
        X_train = scaler.fit_transform(X_all[fold_idx])
        X_test = scaler.transform(X_all[test_start:test_end])
        y_train = y_all[fold_idx]
        y_test = y_all[test_start:test_end]
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            min_child_weight=XGB_MIN_CHILD_WEIGHT,
            gamma=XGB_GAMMA,
            reg_alpha=XGB_REG_ALPHA,
            reg_lambda=XGB_REG_LAMBDA,
            tree_method="hist",
            random_state=42,
        )
        model.fit(X_train, y_train, verbose=False)
        accs.append(float(accuracy_score(y_test, model.predict(X_test)) * 100.0))
    return {
        "fold_accuracies": [round(x, 2) for x in accs],
        "mean_accuracy": round(float(np.mean(accs)), 2) if accs else 0.0,
        "std_accuracy": round(float(np.std(accs)), 2) if accs else 0.0,
        "note": "diagnostic_cv_full_history_not_primary_oos",
    }


def train_ticker(ticker: str, param_overrides: dict | None = None) -> bool:
    """
    Train XGBoost models for a single ticker.

    param_overrides: optional dict of XGBoost hyperparameters to use instead
    of the settings.py defaults. Accepted keys:
        max_depth, learning_rate, subsample, colsample_bytree, min_child_weight
    This is the clean entry point for tune_xgb_best_tickers.py --apply so it
    doesn't need to patch module globals.
    """
    parquet_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(parquet_path):
        log.error("Parquet not found for %s", ticker)
        return False

    df = pd.read_parquet(parquet_path).copy()
    if len(df) < 250:
        log.error("Not enough rows for %s: %s", ticker, len(df))
        return False

    stock_fwd_all = make_forward_return_target(df["Close"], horizon=RETURN_HORIZON_DAYS)
    spy_fwd_all = make_spy_forward_return(df, horizon=RETURN_HORIZON_DAYS)
    valid_labels = stock_fwd_all.notna() & spy_fwd_all.notna()
    if not bool(valid_labels.all()):
        log.info("[%s] Dropping %d rows with unknown future labels", ticker, int((~valid_labels).sum()))

    y_dir_full = shared_make_direction_target(
        df,
        prediction_target=PREDICTION_TARGET,
        horizon=RETURN_HORIZON_DAYS,
        stock_fwd=stock_fwd_all,
        spy_fwd=spy_fwd_all,
        direction_threshold=DIRECTION_LABEL_THRESHOLD,
        excess_return_min_pct=EXCESS_RETURN_MIN_PCT,
        vol_adjusted_sharpe_threshold=VOL_ADJUSTED_SHARPE_THRESHOLD,
    )
    df = df.loc[valid_labels].copy()
    stock_fwd = stock_fwd_all.loc[df.index]
    y_dir_all = y_dir_full[valid_labels.to_numpy()]
    y_ret_all = make_return_target(df["Close"], stock_fwd=stock_fwd)
    future_returns = make_future_return(df["Close"], stock_fwd=stock_fwd)
    if len(df) < 250:
        log.error("Not enough labelled rows for %s: %s", ticker, len(df))
        return False

    splits = compute_split_indices(len(df))
    train_end = splits["train_end"]
    calib_start = splits["calib_start"]
    calib_end = splits["calib_end"]
    test_start = splits["test_start"]
    if train_end < 100 or calib_end <= calib_start or test_start >= len(df):
        log.error("Invalid split for %s", ticker)
        return False

    sentiment_zscore_stats = fit_sentiment_zscore_stats(df.iloc[:train_end])
    df, sentiment_zscore_stats = apply_sentiment_distribution_matching(df, sentiment_zscore_stats)
    feature_cols_raw = _select_feature_cols(df)

    feature_audit, dropped_features = build_feature_audit(df, feature_cols_raw, y_dir_all)
    feature_audit_path = os.path.join(MODEL_DIR, f"{ticker}_feature_audit.json")
    with open(feature_audit_path, "w") as f:
        json.dump(feature_audit, f, indent=2)
    if dropped_features:
        log.info("[%s] Dropping %d weak features before training", ticker, len(dropped_features))
        feature_cols_raw = [c for c in feature_cols_raw if c not in set(dropped_features)]

    for col in feature_cols_raw:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_all, xgb_feature_cols = build_xgb_matrix(df, feature_cols_raw)

    # Non-overlapping subsample for training only.
    # With a 10-day return horizon, consecutive rows predict windows that share
    # 9 of the same 10 days.  Training on every row means XGBoost sees nearly
    # identical (X, y) pairs — it cannot learn to distinguish them and collapses
    # to predicting the base rate.  Keeping every Nth row (stride = horizon)
    # gives fully independent, non-overlapping return windows so each training
    # example is truly informative.  Calibration and test rows stay dense so
    # threshold-finding and accuracy reporting remain representative.
    stride = max(1, RETURN_HORIZON_DAYS)
    train_idx = np.arange(0, train_end, stride)
    X_train_sparse = X_all[train_idx]
    y_dir_train_sparse = y_dir_all[train_idx]
    y_ret_train_sparse = y_ret_all[train_idx]
    log.info(
        "[%s] Non-overlapping train rows: %d -> %d (stride=%d)",
        ticker, train_end, len(train_idx), stride,
    )

    # Scaler fit only on the subsampled training rows.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_sparse)
    X_cal = scaler.transform(X_all[calib_start:calib_end])
    X_test = scaler.transform(X_all[test_start:])

    # Exponential time-decay weights: rows from further in the past get lower
    # weight.  Half-life = 2 trading years (504 days) so a 4-year-old row gets
    # ~25% the weight of today's row.  The model therefore learns mostly from
    # recent market behaviour while still benefiting from older examples.
    half_life = 504.0
    age = train_end - train_idx          # how many rows ago each sample is
    sample_weight = np.exp(-np.log(2) / half_life * age)
    sample_weight = (sample_weight / sample_weight.mean()).astype(np.float32)

    # ── Initial fit (all features) for importance ranking ─────────────────────
    # A fast 100-tree pass to rank features by gain importance.  We don't need
    # a converged model here — just stable relative rankings.
    _init_dir = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
    )
    _init_dir.fit(X_train, y_dir_train_sparse, sample_weight=sample_weight, verbose=False)

    # Feature pruning: keep top-K by gain importance, drop the rest.
    # Per-ticker models have no ticker-dummy columns (n_ticker_dummies=0).
    keep_idx_ticker = _get_pruned_feature_indices(
        _init_dir.feature_importances_,
        n_base_feats=len(xgb_feature_cols),
        n_ticker_dummies=0,
        top_k=FEATURE_IMPORTANCE_TOP_K,
    )
    pruned_xgb_cols_ticker = [xgb_feature_cols[i] for i in keep_idx_ticker]
    n_dropped_ticker = len(xgb_feature_cols) - len(pruned_xgb_cols_ticker)
    if n_dropped_ticker > 0:
        log.info(
            "[%s] Feature pruning: %d → %d cols (%d dropped)",
            ticker, len(xgb_feature_cols), len(pruned_xgb_cols_ticker), n_dropped_ticker,
        )
        X_train = X_train[:, keep_idx_ticker]
        X_cal   = X_cal[:, keep_idx_ticker]
        X_test  = X_test[:, keep_idx_ticker]
        xgb_feature_cols = pruned_xgb_cols_ticker

    dir_model, ret_model = build_models(
        X_train,
        y_dir_train_sparse,
        X_cal,
        y_dir_all[calib_start:calib_end],
        y_ret_train_sparse,
        y_ret_all[calib_start:calib_end],
        sample_weight=sample_weight,
        param_overrides=param_overrides,
    )

    # Calibrator fit only on calibration split.
    cal_probs = dir_model.predict_proba(X_cal)[:, 1]
    calibrator = fit_direction_calibrator(cal_probs, y_dir_all[calib_start:calib_end])
    cal_path = os.path.join(MODEL_DIR, f"{ticker}_direction_calibrator.pkl")
    save_direction_calibrator(calibrator, cal_path)

    # Track whether the calibrator was usable or fell back to the fixed default.
    # Pass raw cal_probs + labels so compute_dynamic_threshold can derive a
    # data-driven threshold even when isotonic calibration fails its quality test.
    threshold, threshold_used_fallback = compute_dynamic_threshold(
        calibrator,
        target_precision=CONFIDENCE_TARGET_PRECISION,
        default=DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
        raw_probs=cal_probs,
        y_true=y_dir_all[calib_start:calib_end],
    )
    if threshold_used_fallback:
        log.info(
            "[%s] Calibrator skipped (Brier/rank test); threshold %.1f derived "
            "from raw probs on calibration set (%d rows)",
            ticker, threshold, calib_end - calib_start,
        )
    else:
        log.info("[%s] Dynamic threshold: %.1f (calibration set size: %d)", ticker, threshold, calib_end - calib_start)

    test_probs = dir_model.predict_proba(X_test)
    test_p_up_raw = test_probs[:, 1]
    test_p_up = np.array([
        float(calibrator.predict([float(p)])[0]) if calibrator is not None else float(p)
        for p in test_p_up_raw
    ], dtype=float)
    y_pred = (test_p_up >= 0.5).astype(np.int64)
    dir_acc = float(accuracy_score(y_dir_all[test_start:], y_pred) * 100.0)
    baseline_up_rate = float(np.mean(y_dir_all[test_start:]) * 100.0) if len(y_dir_all[test_start:]) else 50.0

    buckets = confidence_buckets(test_p_up, y_dir_all[test_start:])
    sweep = threshold_sweep(test_p_up, y_dir_all[test_start:])
    conf_diag = confidence_diagnostics(test_p_up, y_dir_all[test_start:], threshold, buckets)
    return_eval = evaluate_return_model(ret_model, X_test, future_returns[test_start:])
    wf = walk_forward_summary(X_all, y_dir_all)
    plot_path = save_xgb_results_plot(
        ticker,
        test_p_up,
        y_dir_all[test_start:],
        dir_acc,
        baseline_up_rate,
    )

    dir_model.save_model(os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.json"))
    ret_model.save_model(os.path.join(MODEL_DIR, f"{ticker}_xgb_ret.json"))

    metadata = {
        "scaler": scaler,
        "xgb_scaler": scaler,
        "feature_cols": feature_cols_raw,
        "feature_cols_raw": feature_cols_raw,
        "xgb_feature_cols": xgb_feature_cols,
        "xgb_interaction_features_enabled": True,
        "return_horizon": RETURN_HORIZON_DAYS,
        "return_bin_labels": RETURN_BIN_LABELS,
        "n_return_bins": N_RETURN_BINS,
        "ensemble_weights": {"xgboost": 1.0},
        "model_accuracies": {"xgboost": dir_acc},
        "ensemble_accuracy": dir_acc,
        "baseline_up_rate": baseline_up_rate,
        "selected_mode": "xgboost_only",
        "selected_model_name": None,
        "confidence_threshold": threshold,
        "threshold_used_fallback": threshold_used_fallback,
        "confidence_calibrator": cal_path,
        "xgb_model_format": "json",
        "train_split_end": train_end,
        "calib_split_start": calib_start,
        "calib_split_end": calib_end,
        "calib_set_size": calib_end - calib_start,
        "test_split_start": test_start,
        "walk_forward": wf,
        "confidence_buckets": buckets,
        "threshold_sweep": sweep,
        "confidence_diagnostics": conf_diag,
        "return_model_eval": return_eval,
        "results_plot": plot_path,
        "feature_audit": feature_audit,
        "feature_audit_path": feature_audit_path,
        "prediction_target": PREDICTION_TARGET,
        "model_version": "xgb_complete_v5",
        "trained_at": datetime.now().isoformat(),
        "sentiment_features_removed": False,
        "sentiment_features_distribution_matched": bool(sentiment_zscore_stats),
        "sentiment_zscore_stats": sentiment_zscore_stats,
        "param_overrides": param_overrides or {},
    }
    with open(os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    with open(os.path.join(MODEL_DIR, f"{ticker}_train_summary.json"), "w") as f:
        json.dump(
            {
                "ticker": ticker,
                "direction_accuracy": round(dir_acc, 2),
                "baseline_up_rate": round(baseline_up_rate, 2),
                "confidence_threshold": round(threshold, 1),
                "threshold_used_fallback": threshold_used_fallback,
                "calib_set_size": calib_end - calib_start,
                "walk_forward": wf,
                "confidence_buckets": buckets,
                "threshold_sweep": sweep,
                "confidence_diagnostics": conf_diag,
                "return_model_eval": return_eval,
                "results_plot": plot_path,
                "feature_audit_path": feature_audit_path,
                "feature_count_before_audit": feature_audit["feature_count_before"],
                "feature_count_after_audit": feature_audit["feature_count_after"],
                "dropped_feature_count": len(feature_audit["dropped_features"]),
                "prediction_target": PREDICTION_TARGET,
            },
            f,
            indent=2,
        )

    quality_row = evaluate_model_quality(
        ticker,
        train_summary={
            "ticker": ticker,
            "direction_accuracy": round(dir_acc, 2),
            "baseline_up_rate": round(baseline_up_rate, 2),
            "confidence_threshold": round(threshold, 1),
            "threshold_used_fallback": threshold_used_fallback,
            "confidence_diagnostics": conf_diag,
            "return_model_eval": return_eval,
        },
        trades=pd.DataFrame(),
        extra_metrics={"quality_source": "train_preliminary"},
    )
    update_scaler_metadata(ticker, quality_row)
    upsert_quality_report([quality_row])

    # Register only after every artifact and quality update succeeded.  This
    # keeps incomplete training attempts out of reproducibility history.
    drift_baseline_path = snapshot_from_metadata(ticker, metadata, tickers=[ticker])
    register_model_run(
        ticker,
        [
            os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.json"),
            os.path.join(MODEL_DIR, f"{ticker}_xgb_ret.json"),
            os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"),
            os.path.join(MODEL_DIR, f"{ticker}_direction_calibrator.pkl"),
            os.path.join(MODEL_DIR, f"{ticker}_train_summary.json"),
            feature_audit_path,
            drift_baseline_path,
        ],
        metrics={
            "direction_accuracy": round(dir_acc, 4),
            "baseline_up_rate": round(baseline_up_rate, 4),
            "test_auc": metadata.get("test_auc"),
            "confidence_threshold": round(threshold, 4),
        },
        metadata={
            "training_mode": "single_ticker",
            "prediction_target": PREDICTION_TARGET,
            "model_version": metadata["model_version"],
            "ticker": ticker,
        },
    )

    log.info(
        "RESULTS %s: acc=%.2f%% baseline=%.2f%% threshold=%.1f%% fallback=%s",
        ticker, dir_acc, baseline_up_rate, threshold, threshold_used_fallback,
    )
    log.info("WF-CV %s: %s", ticker, wf)
    log.info("Return model %s: %s", ticker, return_eval)
    log.info("Saved results plot %s", plot_path)
    log.info(
        "\n" + classification_report(y_dir_all[test_start:], y_pred, target_names=["DOWN", "UP"], zero_division=0)
    )
    return True


_RISKY_KEYWORDS = (
    "social", "news", "iv_", "put_call", "option", "earn", "eps_",
    "analyst", "recommend", "short_interest", "dark_pool", "dow_", "month_",
    "opex", "calendar", "sector_ret", "ret_vs_sector", "gld_", "hyg_", "tnx_",
    "tlt_", "eem_", "iwm_", "dia_", "uup_",
)
_ALLOWED_PREFIXES = (
    "ret_", "hvol_", "rsi_", "dist_ma", "ma_cross_", "macd", "atr_", "bb_",
    "vol_ratio", "volume_chg_", "vol_zscore_", "vol_pct_vs_", "vol_trend_",
    "hl_range", "spread_proxy", "roc_", "drawdown_",
    "weekly_", "monthly_", "tf_alignment", "spy_", "qqq_", "vix_", "macro_", "regime",
    "ret_vs_spy", "ret_vs_qqq", "breadth_", "pct_above_", "concentration_",
    "factor_", "sector_rel_",
    # Cross-sectional rank features written by cross_sectional_features.py
    # research-time post-pass.  These are rank-percentile-in-[0,1] features that
    # encode how this stock compares to its peers TODAY (within sector and
    # within the broad market).  They are the highest-leverage discriminator
    # class for a cross-sectional ranker because per-stock technicals alone
    # produced AUC=0.51 in the diagnostic ladder.
    "xs_rank_",
)
_ALLOWED_EXACT = {
    "obv_slope",
    "vwap_dist",
    "uptick_ratio",
    "variance_ratio",
    "fund_pe_sector_z",
    "fund_fcf_yield_sector_z",
    "fund_value_combo_z",
    "fund_has_valuation",
}
_EXCLUDE_COLS = {"target", "Open", "High", "Low", "Close", "Volume", "ma5", "ma10", "ma20", "ma50", "ma200"}


# Prefixes of "market context" features that move all stocks together
# (yield curve, credit spread, dollar, copper-gold, regime, broad indices,
# VIX).  Gain importance under the excess-return target ranks these LOW
# because they don't help DISCRIMINATE between stocks (they all see the
# same SPY).  But they ARE the strongest signal for absolute direction
# and bear-market avoidance.  Protected from pruning so the model always
# has macro context regardless of gain rank.
_MARKET_CTX_PREFIXES = ("macro_", "regime", "spy_", "qqq_", "vix_")


def _get_pruned_feature_indices(
    importances: np.ndarray,
    n_base_feats: int,
    n_ticker_dummies: int,
    top_k: int,
    sent_idx: np.ndarray | None = None,
    n_sent_keep: int = 16,   # raised from 3 — feature_ic_report shows 16/16 sentiment alive (mean |t|=4.49)
    macro_idx: np.ndarray | None = None,
    n_macro_keep: int = 10,
    xs_idx: np.ndarray | None = None,
    n_xs_keep: int = 30,     # raised from 12 — feature_ic_report shows ~30 xs_rank alive at horizon=5d
) -> np.ndarray:
    """
    Return sorted column indices to KEEP after gain-importance pruning.

    How it works
    ------------
    XGBoost records how much each feature reduces the model's loss (called
    "gain importance").  Features with near-zero gain contribute noise, not
    signal.  We keep only the top-K base features plus ALL ticker dummy
    columns (which encode the cross-sectional identity of each ticker —
    they must never be dropped).

    Args
    ----
    importances      : dir_model.feature_importances_ — shape (n_total_features,)
                       where n_total_features = n_base_feats + n_ticker_dummies.
    n_base_feats     : Number of non-dummy (rolling-transform) features.
    n_ticker_dummies : Number of ticker one-hot columns appended after base features.
    top_k            : How many of the top-ranked base features to keep.

    Returns
    -------
    Sorted numpy array of integer column indices to retain.
    """
    # Rank ONLY the base features; ticker dummies are always kept.
    base_importances = importances[:n_base_feats]
    top_k_actual     = min(top_k, n_base_feats)   # can't keep more than we have
    top_k_idx        = set(np.argsort(base_importances)[-top_k_actual:].tolist())

    # Sentiment has sparse, event-driven gain and can be pruned out even when
    # USE_NEWS_SENTIMENT=True. Keep the strongest sentiment columns explicitly so
    # the shared h20/h5 pruned matrix still contains news signal.
    if sent_idx is not None and len(sent_idx) > 0:
        valid_sent = sent_idx[sent_idx < n_base_feats]
        if len(valid_sent) > 0:
            sent_by_imp = valid_sent[np.argsort(base_importances[valid_sent])[::-1]]
            for idx in sent_by_imp[:n_sent_keep]:
                top_k_idx.add(int(idx))

    # Macro / regime / broad-index features get the same protection as sentiment.
    # Without this they get pruned out because excess-return labels reward
    # cross-sectional discrimination, not market-wide context — leaving the model
    # blind to bull/bear regime, yield-curve inversion, credit stress, etc.
    if macro_idx is not None and len(macro_idx) > 0:
        valid_macro = macro_idx[macro_idx < n_base_feats]
        if len(valid_macro) > 0:
            macro_by_imp = valid_macro[np.argsort(base_importances[valid_macro])[::-1]]
            for idx in macro_by_imp[:n_macro_keep]:
                top_k_idx.add(int(idx))

    # Cross-sectional rank features (xs_rank_sector_*, xs_rank_market_*) are
    # the strongest theoretical discriminators for a cross-sectional ranker —
    # they encode "where does this stock sit vs peers TODAY?".  Force-keep the
    # top-N so the model always has rank-based signal even when their absolute
    # gain importance is overshadowed by per-stock vol noise.
    if xs_idx is not None and len(xs_idx) > 0:
        valid_xs = xs_idx[xs_idx < n_base_feats]
        if len(valid_xs) > 0:
            xs_by_imp = valid_xs[np.argsort(base_importances[valid_xs])[::-1]]
            for idx in xs_by_imp[:n_xs_keep]:
                top_k_idx.add(int(idx))

    # Dummy columns sit at the very end: indices [n_base_feats, n_base_feats+n_ticker_dummies)
    dummy_idx = np.arange(n_base_feats, n_base_feats + n_ticker_dummies)

    # Combine and sort so column order is preserved (XGB requires consistent col order)
    return np.sort(np.unique(np.concatenate([list(top_k_idx), dummy_idx])))


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the allowed numeric feature columns from a DataFrame.

    Cross-sectional rank columns are whitelisted UNCONDITIONALLY ahead of the
    risky-keyword check.  They are named like 'xs_rank_sector_ret_5d', and the
    substring 'sector_ret' is in _RISKY_KEYWORDS — without the whitelist they
    get silently filtered out before they ever reach the model.
    """
    cols = [
        c for c in df.columns
        if c not in _EXCLUDE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
        and (
            c.startswith("sent_z_")
            or c.startswith("xs_rank_")     # whitelist before risky-keyword check
            or (
                not any(k in c.lower() for k in _RISKY_KEYWORDS)
                and (c in _ALLOWED_EXACT or any(c.startswith(p) for p in _ALLOWED_PREFIXES))
            )
        )
    ]
    # When EXCLUDE_MARKET_CONSTANTS is on, drop columns whose values are the
    # same across every ticker on a given date (spy_*, vix_*, macro_*, regime,
    # qqq_*).  See settings.py for full reasoning — short version: a cross-
    # sectional ranker cannot use constant-within-date features to discriminate
    # between stocks, so they waste pruner budget without contributing edge.
    # xs_rank_market_* features ENCODE market-relative position per-stock and
    # are NOT constant — they survive the filter.
    if EXCLUDE_MARKET_CONSTANTS:
        cols = [c for c in cols if not any(c.startswith(p) for p in _MARKET_CTX_PREFIXES)]
    return cols


def train_pooled(
    tickers: list[str],
    param_overrides: dict | None = None,
    tune_hyperparams: bool = True,
    training_only_tickers: list[str] | None = None,
    ranker_experiment: bool = False,
) -> bool:
    """
    Train ONE XGBoost model across ALL tickers (cross-sectional pooling).

    Why this matters
    ----------------
    Per-ticker models train on ~258 effective rows each (after non-overlapping
    stride on ~2,500 raw rows).  That is too few for XGBoost to generalise.
    Pooling all tickers gives ~9× more data (2,300+ effective rows) while still
    letting the model learn ticker-specific patterns via one-hot dummy features.

    Saved artefacts
    ---------------
    - models/pooled_xgb_dir.json  — shared direction classifier
    - models/pooled_xgb_ret.json  — shared return bucket classifier
    - models/pooled_scaler.pkl    — scaler + metadata
    - models/{ticker}_scaler.pkl  — per-ticker wrapper referencing pooled model
    - models/{ticker}_xgb_dir.json  — copy of pooled dir model (for compatibility)
    - models/{ticker}_xgb_ret.json  — copy of pooled ret model (for compatibility)
    """
    # Build the full ticker list: prediction tickers + training-only (crashed/delisted).
    # training_only_set tracks which tickers contribute training data but never
    # receive per-ticker artefacts or live signals.
    training_only_set: set[str] = set(training_only_tickers or [])
    all_tickers = list(tickers) + [t for t in (training_only_tickers or []) if t not in tickers]
    log.info(
        "[pooled] Starting cross-sectional training on %d tickers (%d prediction, %d training-only/survivorship)",
        len(all_tickers), len(tickers), len(training_only_set),
    )

    # ── Step 1: find common raw feature set across all tickers ────────────────
    per_ticker_info: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    feature_sets: list[set[str]] = []
    pooled_sentiment_stats: dict[str, dict[str, dict[str, float]]] = {}

    for ticker in all_tickers:
        parquet_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
        if not os.path.exists(parquet_path):
            log.warning("[pooled] No parquet for %s — skipping", ticker)
            continue
        df = pd.read_parquet(parquet_path).copy()
        if len(df) < 250:
            log.warning("[pooled] Too few rows for %s (%d) — skipping", ticker, len(df))
            continue
        split_for_stats = compute_split_indices(len(df))
        sent_stats = fit_sentiment_zscore_stats(df.iloc[:split_for_stats["train_end"]])
        df, sent_stats = apply_sentiment_distribution_matching(df, sent_stats)
        pooled_sentiment_stats[ticker] = sent_stats
        # ── Sentiment intersection-drop fix ──────────────────────────────────
        # Survivorship tickers (FRC, BBBY) and any ticker without historical
        # FinBERT scores produce ZERO `sent_z_*` columns from
        # apply_sentiment_distribution_matching above.  When the common-feature
        # intersection runs across all tickers (line ~1271), a single ticker
        # missing `sent_z_*` drops the entire family from candidates — even
        # though feature_ic_report shows sent_z_ is the second-strongest family
        # with mean |t|=4.49.  Force-create the canonical sent_z_* set here as
        # zeros for any ticker that lacks them, so intersection keeps them.
        # 0.0 in z-score space is "neutral / no signal" — safe default for
        # tickers without sentiment coverage.
        from pipeline_shared import RAW_SENTIMENT_COLUMNS  # local import — keeps top-of-file imports clean
        for _raw_col in RAW_SENTIMENT_COLUMNS:
            _z_col = f"sent_z_{_raw_col}"
            if _z_col not in df.columns:
                df[_z_col] = 0.0
        cols = _select_feature_cols(df)
        if not cols:
            log.warning("[pooled] No usable features for %s — skipping", ticker)
            continue
        per_ticker_info[ticker] = (df, cols)
        feature_sets.append(set(cols))

    if len(per_ticker_info) < 2:
        log.error("[pooled] Need at least 2 tickers, got %d", len(per_ticker_info))
        return False

    common_features: list[str] = sorted(feature_sets[0].intersection(*feature_sets[1:]))
    if not common_features:
        log.error("[pooled] No features common across all tickers")
        return False
    log.info("[pooled] Common feature set: %d features across %d tickers",
             len(common_features), len(per_ticker_info))

    # ── Step 2: build per-ticker XGB feature matrices ────────────────────────
    # Rolling transforms are applied PER-TICKER so that windows don't bleed
    # across ticker boundaries.  We then concatenate the resulting matrices.
    valid_tickers: list[str] = []
    all_dates_list: list[np.ndarray] = []        # each entry: (n_ticker_rows,) timestamps
    all_X_list: list[np.ndarray] = []            # each entry: (n_ticker_rows, n_xgb_features)
    all_y_dir_list: list[np.ndarray] = []
    all_y_rank_list: list[np.ndarray] = []
    all_y_ret_list: list[np.ndarray] = []
    all_y_dir_h5_list: list[np.ndarray] = []    # 5-day direction labels for ensemble
    all_ticker_labels_list: list[np.ndarray] = []  # ticker name repeated for each row
    xgb_cols: list[str] | None = None

    for ticker, (df_raw, _) in per_ticker_info.items():
        df = df_raw.copy()
        stock_fwd = make_forward_return_target(df["Close"], horizon=RETURN_HORIZON_DAYS)
        spy_fwd = make_spy_forward_return(df, horizon=RETURN_HORIZON_DAYS)
        valid_mask = stock_fwd.notna() & spy_fwd.notna()
        df = df.loc[valid_mask].copy()
        if len(df) < 100:
            log.warning("[pooled] Too few valid rows for %s after label filter", ticker)
            continue
        stock_fwd = stock_fwd.loc[df.index]
        spy_fwd   = spy_fwd.loc[df.index]

        y_dir = shared_make_direction_target(
            df,
            prediction_target=PREDICTION_TARGET,
            horizon=RETURN_HORIZON_DAYS,
            stock_fwd=stock_fwd,
            spy_fwd=spy_fwd,
            direction_threshold=DIRECTION_LABEL_THRESHOLD,
            excess_return_min_pct=EXCESS_RETURN_MIN_PCT,
            vol_adjusted_sharpe_threshold=VOL_ADJUSTED_SHARPE_THRESHOLD,
        )
        y_ret = make_return_target(df["Close"], stock_fwd=stock_fwd)
        y_rank = (stock_fwd - spy_fwd).to_numpy(dtype=np.float32)

        # 5-day forward direction label for the secondary short-horizon model.
        # 1 = price is higher 5 days later, 0 = flat or lower.
        # The h20 valid-mask excludes the last 20 rows so every included row
        # has at least 20 rows of future data, making 5-day futures always available.
        fwd_5d   = df["Close"].pct_change(5).shift(-5).fillna(0.0)
        y_dir_h5 = (fwd_5d.values > 0.0).astype(np.int64)

        # Ensure all common features exist (fill missing with 0)
        for col in common_features:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X, cols = build_xgb_matrix(df, common_features)
        if xgb_cols is None:
            xgb_cols = cols

        dates_np = np.array(df.index, dtype="datetime64[ns]")
        all_dates_list.append(dates_np)
        all_X_list.append(X)
        all_y_dir_list.append(y_dir)
        all_y_rank_list.append(y_rank)
        all_y_ret_list.append(y_ret)
        all_y_dir_h5_list.append(y_dir_h5)
        # Track which ticker each row belongs to (needed for per-ticker plot).
        all_ticker_labels_list.append(np.array([ticker] * len(X), dtype=object))
        valid_tickers.append(ticker)

    if not valid_tickers:
        log.error("[pooled] No tickers survived after validation")
        return False

    # ── Step 3: append ticker one-hot columns ─────────────────────────────────
    # Ticker dummies are NOT rolled; they are appended after rolling transforms.
    n_base_xgb = len(xgb_cols)
    ticker_classes = sorted(valid_tickers)
    ticker_to_idx  = {t: i for i, t in enumerate(ticker_classes)}
    n_tickers      = len(ticker_classes)
    ticker_dummy_xgb_cols = [f"tkr__{t}" for t in ticker_classes]

    all_X_with_dummy = []
    for i, (ticker, X) in enumerate(zip(valid_tickers, all_X_list)):
        dummy_block = np.zeros((len(X), n_tickers), dtype=np.float64)
        dummy_block[:, ticker_to_idx[ticker]] = 1.0
        all_X_with_dummy.append(np.concatenate([X, dummy_block], axis=1))

    all_xgb_cols = xgb_cols + ticker_dummy_xgb_cols

    # ── Step 4: concatenate and sort chronologically ───────────────────────────
    all_dates         = np.concatenate(all_dates_list)
    X_combined        = np.concatenate(all_X_with_dummy, axis=0)
    y_dir_comb        = np.concatenate(all_y_dir_list)
    y_rank_comb       = np.concatenate(all_y_rank_list)
    y_ret_comb        = np.concatenate(all_y_ret_list)
    y_dir_h5_comb     = np.concatenate(all_y_dir_h5_list)   # short-horizon labels
    ticker_labels_all = np.concatenate(all_ticker_labels_list)  # ticker name per row
    sort_order = np.argsort(all_dates, kind="stable")
    all_dates         = all_dates[sort_order]
    X_combined        = X_combined[sort_order]
    y_dir_comb        = y_dir_comb[sort_order]
    y_rank_comb       = y_rank_comb[sort_order]
    y_ret_comb        = y_ret_comb[sort_order]
    y_dir_h5_comb     = y_dir_h5_comb[sort_order]
    ticker_labels_all = ticker_labels_all[sort_order]

    # ── Step 5: date-based train / calib / test split ─────────────────────────
    # Row-count splits would leak ticker length differences into the cut points.
    # Date-based splits apply the same calendar cutoffs to every ticker.
    # Convert to pandas Timestamps for safe comparison across numpy/python datetime types
    all_ts = pd.DatetimeIndex(all_dates)
    unique_dates = sorted(all_ts.unique())
    n_dates = len(unique_dates)
    train_cutoff = unique_dates[int(n_dates * TRAIN_CALIBRATION_SPLIT)]
    calib_cutoff = unique_dates[int(n_dates * CALIBRATION_TEST_SPLIT)]

    train_mask = all_ts < train_cutoff
    calib_mask = (all_ts >= train_cutoff) & (all_ts < calib_cutoff)
    test_mask  = all_ts >= calib_cutoff

    # Non-overlapping stride: sample every Nth unique DATE so all tickers' rows
    # for the chosen dates are retained (preserves cross-sectional variation).
    train_dates_sorted = sorted(set(all_ts[train_mask].tolist()))
    stride_dates_set   = set(train_dates_sorted[::max(1, RETURN_HORIZON_DAYS)])
    stride_mask = train_mask & np.array([d in stride_dates_set for d in all_ts])

    X_train       = X_combined[stride_mask]
    y_dir_train   = y_dir_comb[stride_mask]
    y_rank_train  = y_rank_comb[stride_mask]
    y_ret_train   = y_ret_comb[stride_mask]
    y_dir_h5_train = y_dir_h5_comb[stride_mask]
    dates_train    = all_dates[stride_mask]
    X_cal          = X_combined[calib_mask]
    y_dir_cal      = y_dir_comb[calib_mask]
    y_rank_cal     = y_rank_comb[calib_mask]
    y_ret_cal      = y_ret_comb[calib_mask]
    y_dir_h5_cal   = y_dir_h5_comb[calib_mask]
    dates_cal      = all_dates[calib_mask]
    X_test         = X_combined[test_mask]
    y_dir_test = y_dir_comb[test_mask]
    y_rank_test = y_rank_comb[test_mask]
    dates_test = all_dates[test_mask]
    ticker_labels_test = ticker_labels_all[test_mask]  # ticker name for each test row
    log.info("[pooled] Train rows: %d  Calib rows: %d  Test rows: %d",
             len(X_train), len(X_cal), len(X_test))

    if len(X_train) < 50 or len(X_cal) < 20:
        log.error("[pooled] Not enough data for train/calib split")
        return False

    # ── Step 6: scale ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_cal_sc   = scaler.transform(X_cal)
    X_test_sc  = scaler.transform(X_test)

    # Exponential decay sample weights: rows from further in the past get
    # lower weight (half-life = 504 trading days ≈ 2 years).
    train_pos = np.where(stride_mask)[0].astype(float)
    age = train_pos.max() - train_pos
    half_life = 504.0
    sample_weight = np.exp(-np.log(2) / half_life * age)
    sample_weight = (sample_weight / sample_weight.mean()).astype(np.float32)

    if ranker_experiment:
        log.info(
            "[ranker] Experimental pooled XGBRanker enabled — continuous target=fwd_excess_%dd",
            RETURN_HORIZON_DAYS,
        )
        train_groups = build_rank_groups_from_dates(dates_train)
        cal_groups = build_rank_groups_from_dates(dates_cal)
        test_groups = build_rank_groups_from_dates(dates_test)
        if sum(train_groups) != len(X_train_sc) or sum(cal_groups) != len(X_cal_sc) or sum(test_groups) != len(X_test_sc):
            log.error("[ranker] group sizes do not match matrix rows")
            return False

        # ── Ranker hyperparameters — deliberately simpler than the classifier ──
        # The classifier uses depth=4, n_est=500, which produced train IC=0.14
        # but test IC=−0.002 — catastrophic overfit.  Two key differences for
        # the ranker:
        #
        # 1. LOWER COMPLEXITY: depth=3, n_est=200, lr=0.01, heavy reg.
        #    With ~500k rows but only ~600 unique dates, the model can overfit
        #    to date-specific patterns with even moderate tree depth.
        #
        # 2. NO TICKER DUMMIES: ticker one-hot columns let the model learn
        #    "always rank AAPL high" from historical outperformance — that's
        #    survivorship bias, not a cross-sectional pattern.  Rankers should
        #    learn "stocks with high momentum + low vol outperform" regardless
        #    of which ticker it is.  Dropping 149 dummy features removes the
        #    primary memorisation pathway.
        ranker_params = {
            "objective": "rank:pairwise",
            "eval_metric": "rmse",
            "n_estimators": 200,           # was 500 — fewer rounds, less overfit
            "max_depth": 3,                # was 4 — shallower trees, less interaction capacity
            "learning_rate": 0.01,         # was 0.05 — slower learning rate forces smoother fit
            "subsample": 0.6,             # was 0.8 — more row dropout per tree
            "colsample_bytree": 0.4,      # was 0.7 — see fewer features per tree → decorrelate
            "min_child_weight": 20,        # was 3 — need ≥20 rows in a leaf (more stable splits)
            "gamma": 1.0,                  # was 0.1 — higher min loss reduction to split
            "reg_alpha": 1.0,              # was 0.1 — L1 regularisation 10× higher
            "reg_lambda": 10.0,            # was 1.0 — L2 regularisation 10× higher
            "tree_method": "hist",
            "random_state": 42,
        }

        # ── Drop ticker dummies: use ONLY base features for the ranker ───────
        # The classifier keeps ticker one-hot dummies to model per-stock
        # intercepts — useful for "will this stock go up?" classification.
        # For a cross-sectional RANKER, ticker dummies are anti-signal: they
        # let the model memorise "JPM historically ranked 3rd" instead of
        # learning generalisable cross-sectional patterns.
        #
        # keep_idx for ranker: only base features, NO ticker dummy columns.
        ranker_base_cols = all_xgb_cols[:n_base_xgb]   # first n_base_xgb cols
        X_train_base_only = X_train_sc[:, :n_base_xgb]
        X_cal_base_only   = X_cal_sc[:, :n_base_xgb]
        X_test_base_only  = X_test_sc[:, :n_base_xgb]

        init_ranker = xgb.XGBRanker(
            **{**ranker_params, "n_estimators": 100}
        )
        init_ranker.fit(X_train_base_only, y_rank_train, group=train_groups, verbose=False)

        sent_base_idx = np.array(
            [i for i, c in enumerate(ranker_base_cols) if c.startswith("sent_z_")],
            dtype=int,
        )
        macro_base_idx = np.array(
            [
                i for i, c in enumerate(ranker_base_cols)
                if any(c.startswith(p) for p in _MARKET_CTX_PREFIXES)
            ],
            dtype=int,
        )
        xs_base_idx = np.array(
            [i for i, c in enumerate(ranker_base_cols) if c.startswith("xs_rank_")],
            dtype=int,
        )
        # Prune with n_ticker_dummies=0 since we're not using dummies.
        keep_idx = _get_pruned_feature_indices(
            init_ranker.feature_importances_,
            n_base_feats=n_base_xgb,
            n_ticker_dummies=0,            # NO ticker dummies for ranker
            top_k=FEATURE_IMPORTANCE_TOP_K,
            sent_idx=sent_base_idx,
            macro_idx=macro_base_idx,
            xs_idx=xs_base_idx,
        )
        ranker_xgb_cols = [ranker_base_cols[i] for i in keep_idx]
        n_ranker_base = len(ranker_xgb_cols)
        log.info(
            "[ranker] feature pruning: %d base → %d cols (NO ticker dummies)",
            n_base_xgb, n_ranker_base,
        )

        X_train_rank = X_train_base_only[:, keep_idx]
        X_cal_rank   = X_cal_base_only[:, keep_idx]
        X_test_rank  = X_test_base_only[:, keep_idx]

        ranker = xgb.XGBRanker(**ranker_params)
        ranker.fit(
            X_train_rank,
            y_rank_train,
            group=train_groups,
            eval_set=[(X_cal_rank, y_rank_cal)],
            eval_group=[cal_groups],
            verbose=False,
        )
        # ── Ranker IC diagnostics ─────────────────────────────────────────────
        # Compute daily rank IC on TRAIN, CAL, and TEST splits so we can
        # distinguish overfitting (train IC >> test IC) from regime change
        # (train IC ≈ test IC ≈ 0).  Also log date ranges to check whether
        # test lands on an exotic period (COVID, 2022 bear, etc.).
        train_scores = ranker.predict(X_train_rank)
        cal_scores   = ranker.predict(X_cal_rank)
        test_scores  = ranker.predict(X_test_rank)
        train_ic = daily_rank_ic_from_scores(dates_train, train_scores, y_rank_train)
        cal_ic   = daily_rank_ic_from_scores(dates_cal, cal_scores, y_rank_cal)
        test_ic  = daily_rank_ic_from_scores(dates_test, test_scores, y_rank_test)

        # Date ranges for each split — critical for diagnosing regime shifts.
        _fmt = "%Y-%m-%d"
        log.info(
            "[ranker] Split date ranges:  train=%s→%s  cal=%s→%s  test=%s→%s",
            pd.Timestamp(dates_train.min()).strftime(_fmt),
            pd.Timestamp(dates_train.max()).strftime(_fmt),
            pd.Timestamp(dates_cal.min()).strftime(_fmt),
            pd.Timestamp(dates_cal.max()).strftime(_fmt),
            pd.Timestamp(dates_test.min()).strftime(_fmt),
            pd.Timestamp(dates_test.max()).strftime(_fmt),
        )
        log.info(
            "[ranker] Train daily IC: mean=%.4f  t=%.3f  days=%s  avg_names=%s",
            train_ic.get("mean_ic") or 0, train_ic.get("t_stat") or 0,
            train_ic.get("days"), train_ic.get("avg_names_per_day"),
        )
        log.info(
            "[ranker] Cal daily IC:   mean=%.4f  t=%.3f  days=%s  avg_names=%s",
            cal_ic.get("mean_ic") or 0, cal_ic.get("t_stat") or 0,
            cal_ic.get("days"), cal_ic.get("avg_names_per_day"),
        )
        log.info(
            "[ranker] Test daily IC:  mean=%.4f  t=%.3f  days=%s  avg_names=%s",
            test_ic.get("mean_ic") or 0, test_ic.get("t_stat") or 0,
            test_ic.get("days"), test_ic.get("avg_names_per_day"),
        )
        # Overfit verdict: if train IC >> test IC, model memorized patterns.
        _train_m = train_ic.get("mean_ic") or 0
        _test_m  = test_ic.get("mean_ic") or 0
        if abs(_train_m) > 0.03 and abs(_test_m) < 0.005:
            log.warning(
                "[ranker] OVERFIT WARNING — train IC=%.4f >> test IC=%.4f.  "
                "Reduce model complexity (depth, n_estimators) or increase regularisation.",
                _train_m, _test_m,
            )
        elif abs(_train_m) < 0.005 and abs(_test_m) < 0.005:
            log.warning(
                "[ranker] WEAK SIGNAL — both train IC=%.4f and test IC=%.4f near zero.  "
                "Features may lack cross-sectional signal at this horizon.",
                _train_m, _test_m,
            )

        ranker_path = os.path.join(MODEL_DIR, "pooled_xgb_ranker.json")
        ranker.save_model(ranker_path)
        ranker_meta = {
            "scaler": scaler,
            "xgb_scaler": scaler,
            "feature_cols": common_features,
            "feature_cols_raw": common_features,
            "xgb_feature_cols": ranker_xgb_cols,
            "ticker_dummy_cols": [],          # ranker uses NO ticker dummies
            "ticker_classes": ticker_classes,
            "n_base_xgb_features": n_ranker_base,
            "feature_pruning_top_k": FEATURE_IMPORTANCE_TOP_K,
            "feature_pruning_keep_idx": keep_idx.tolist(),
            "pooled": True,
            "ranker_experiment": True,
            "ranker_objective": "rank:pairwise",
            "ranker_target": f"forward_excess_return_{RETURN_HORIZON_DAYS}d",
            "ranker_model_path": ranker_path,
            "return_horizon": RETURN_HORIZON_DAYS,
            "pooled_tickers": [t for t in valid_tickers if t not in training_only_set],
            "survivorship_training_tickers": sorted(training_only_set & set(valid_tickers)),
            "train_daily_rank_ic": train_ic,
            "cal_daily_rank_ic": cal_ic,
            "test_daily_rank_ic": test_ic,
            "ranker_params": ranker_params,
        }
        with open(os.path.join(MODEL_DIR, "pooled_ranker_scaler.pkl"), "wb") as f:
            pickle.dump(ranker_meta, f)
        with open(os.path.join(MODEL_DIR, "pooled_ranker_train_summary.json"), "w") as f:
            json.dump(
                {
                    "ranker_model_path": ranker_path,
                    "ranker_target": ranker_meta["ranker_target"],
                    "train_daily_rank_ic": train_ic,
                    "cal_daily_rank_ic": cal_ic,
                    "test_daily_rank_ic": test_ic,
                    "n_features": len(ranker_xgb_cols),
                    "n_base_features": n_ranker_base,
                    "n_tickers": len(valid_tickers),
                    "train_rows": int(len(X_train_rank)),
                    "calib_rows": int(len(X_cal_rank)),
                    "test_rows": int(len(X_test_rank)),
                },
                f,
                indent=2,
            )
        log.info("[ranker] Saved %s", ranker_path)
        log.info("[ranker] Saved pooled_ranker_scaler.pkl and pooled_ranker_train_summary.json")
        return True

    # ── Step 6a: nested walk-forward hyperparameter search (optional) ─────────
    # Grid: max_depth × min_child_weight × reg_lambda (4×4×3 = 48 combos).
    # Each combo is evaluated with inner_splits=3 rolling folds inside the
    # training window.  The best combo for each outer fold is recorded.
    # We pick the combo from the outer fold with the highest test accuracy.
    # The tuned params are saved to models/pooled_best_xgb_params.json so
    # backtest.py can reuse them without re-running the search.
    best_params_override = param_overrides  # start from caller-supplied overrides
    if tune_hyperparams and len(X_train_sc) >= 200:
        log.info("[pooled] Running nested walk-forward CV hyperparameter search "
                 "(%d train rows, 4 outer × 3 inner folds)...", len(X_train_sc))
        tunable_grid = {
            # max_depth=3 is too shallow for 30 features; start at 4.
            "max_depth":        [4, 5, 6],
            # min_child_weight=1 allows tiny leaves → overfits on small folds.
            # With ~2k training rows, keep ≥ 3 samples per leaf minimum.
            "min_child_weight": [3, 5, 10],
            # reg_lambda=0.1 is under-regularised at our data scale;
            # empirically 10.0 works best, so search [1, 5, 10].
            "reg_lambda":       [1.0, 5.0, 10.0],
        }
        # Fixed params are not searched — they wrap around every tunable combo.
        fixed_params = {
            "n_estimators":     300,   # fast fixed count for search; final fit uses 500
            "learning_rate":    XGB_LEARNING_RATE,
            "subsample":        XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "gamma":            XGB_GAMMA,
            "reg_alpha":        XGB_REG_ALPHA,
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "tree_method":      "hist",
            "random_state":     42,
        }
        try:
            fold_results = nested_walk_forward_search(
                pd.DataFrame(X_train_sc),
                pd.Series(y_dir_train),
                tunable_grid,
                fixed_params=fixed_params,
                outer_splits=4,
                inner_splits=3,
                embargo=RETURN_HORIZON_DAYS,
                min_test_size=500,
            )
            if fold_results:
                _observed_outer_folds = sorted({int(r.fold) for r in fold_results})
                if len(_observed_outer_folds) < 3:
                    log.warning(
                        "[pooled] Nested CV produced only %d outer folds (%s); "
                        "ignoring tuned params and keeping settings.py defaults",
                        len(_observed_outer_folds),
                        _observed_outer_folds,
                    )
                    fold_results = []
            if fold_results:
                # Select params by AVERAGE outer-fold AUC across all evaluated
                # folds for each candidate combo, rather than picking the
                # single-best outer fold.  This avoids choosing the combo that
                # happened to win one lucky fold.
                from collections import defaultdict as _dd
                _combo_scores: dict[tuple, list[float]] = _dd(list)
                for _r in fold_results:
                    # Make a hashable key from the tunable params only
                    _key = tuple(sorted((_k, _r.params[_k]) for _k in tunable_grid))
                    _combo_scores[_key].append(_r.score)
                # Choose the combo with the highest MEAN outer-fold AUC
                _best_key = max(_combo_scores, key=lambda k: float(np.mean(_combo_scores[k])))
                _best_mean = float(np.mean(_combo_scores[_best_key]))
                tuned = dict(_best_key)
                log.info(
                    "[pooled] Nested CV best params (avg of %d folds, mean AUC=%.4f): %s",
                    len(_combo_scores[_best_key]), _best_mean, tuned,
                )
                # Log all fold results for transparency
                for r in fold_results:
                    log.info("[pooled]   fold %d: score=%.4f  params=%s",
                             r.fold, r.score,
                             {k: r.params[k] for k in tunable_grid})
                # Merge: nested-CV params are the base; caller overrides (if any) win.
                merged = dict(tuned)
                if param_overrides:
                    merged.update(param_overrides)
                best_params_override = merged
                # Persist so backtest.py can reuse the same config without re-tuning.
                tuned_path = os.path.join(MODEL_DIR, "pooled_best_xgb_params.json")
                with open(tuned_path, "w") as f:
                    json.dump({
                        "params": tuned,
                        "fold_results": [
                            {"fold": r.fold, "score": round(r.score, 4),
                             "params": {k: r.params[k] for k in tunable_grid}}
                            for r in fold_results
                        ],
                        "tuned_at": datetime.now().isoformat(),
                    }, f, indent=2)
                log.info("[pooled] Saved tuned params to %s", tuned_path)
        except Exception as e:
            log.warning("[pooled] Nested CV failed (%s) — using settings.py defaults", e)

    # ── Step 6b: quick initial fit (all features) — only used for importance ──
    # We use a fast 100-tree model here just to rank features by gain importance.
    # A fully-converged model is NOT needed for ranking; 100 trees give stable
    # relative scores at a fraction of the compute cost.
    _init_dir = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=(best_params_override or {}).get("max_depth", XGB_MAX_DEPTH),
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
    )
    _init_dir.fit(X_train_sc, y_dir_train, sample_weight=sample_weight, verbose=False)
    raw_importances = _init_dir.feature_importances_
    X_train_sc_full = X_train_sc
    X_cal_sc_full = X_cal_sc
    X_test_sc_full = X_test_sc
    sent_base_idx = np.array(
        [i for i, c in enumerate(all_xgb_cols[:n_base_xgb]) if c.startswith("sent_z_")],
        dtype=int,
    )
    # Macro / regime / broad-index column indices for protection from gain pruning.
    # Without this, OOS Direction AUC = 0.51 because the model is trained on per-stock
    # technicals only and has no knowledge of yield curve, credit spread, regime, or
    # SPY/VIX context.  Verified against AAPL parquet: macro_yield_curve_10y_3m,
    # macro_credit_spread_20d, macro_dxy_proxy_20d_z, macro_copper_gold_20d, regime,
    # spy_dist_ma200, spy_dist_ma50, vix_*  all exist with 99-100% non-zero rates.
    macro_base_idx = np.array(
        [
            i for i, c in enumerate(all_xgb_cols[:n_base_xgb])
            if any(c.startswith(p) for p in _MARKET_CTX_PREFIXES)
        ],
        dtype=int,
    )
    # Cross-sectional rank feature indices (xs_rank_sector_*, xs_rank_market_*).
    # Written by cross_sectional_features.py at research time.  These are the
    # primary discriminators a cross-sectional ranker should use.
    xs_base_idx = np.array(
        [i for i, c in enumerate(all_xgb_cols[:n_base_xgb]) if c.startswith("xs_rank_")],
        dtype=int,
    )
    log.info(
        "[pooled] Found %d sentiment + %d macro/regime + %d xs_rank features in candidate set",
        len(sent_base_idx), len(macro_base_idx), len(xs_base_idx),
    )

    # ── Step 6c: XGB gain feature pruning — keep top-K base features + all dummies ──
    # Uses XGBoost's built-in "gain" importance (total reduction in loss from
    # splitting on each feature).  NOT SHAP / permutation importance — SHAP_MIN_IMPORTANCE
    # in settings.py is kept for backward compatibility only and is not used here.
    # XGBoost with hundreds of rolled features is overparameterised relative to
    # our ~1,990 training rows.  Dropping low-importance features reduces
    # variance and stabilises calibration.  Ticker dummies are ALWAYS retained.
    keep_idx = _get_pruned_feature_indices(
        raw_importances,
        n_base_feats=n_base_xgb,
        n_ticker_dummies=n_tickers,
        top_k=FEATURE_IMPORTANCE_TOP_K,
        sent_idx=sent_base_idx,             # protect top-3 sentiment features from pruning
        macro_idx=macro_base_idx,           # protect top-10 macro/regime/market-context features
        xs_idx=xs_base_idx,                 # protect top-12 cross-sectional rank features
    )
    pruned_xgb_cols = [all_xgb_cols[i] for i in keep_idx]
    n_dropped = len(all_xgb_cols) - len(pruned_xgb_cols)
    n_pruned_base = int((keep_idx < n_base_xgb).sum())  # base cols kept (excl. dummies)
    n_sent_kept = sum(1 for c in pruned_xgb_cols if c.startswith("sent_z_"))
    n_macro_kept = sum(
        1 for c in pruned_xgb_cols if any(c.startswith(p) for p in _MARKET_CTX_PREFIXES)
    )
    n_xs_kept = sum(1 for c in pruned_xgb_cols if c.startswith("xs_rank_"))
    log.info(
        "[pooled] Feature pruning: %d → %d cols  (%d base kept of %d, %d dropped, "
        "%d dummies always kept, %d sentiment kept, %d macro/regime kept, %d xs_rank kept)",
        len(all_xgb_cols), len(pruned_xgb_cols),
        n_pruned_base, n_base_xgb, n_dropped, n_tickers, n_sent_kept, n_macro_kept, n_xs_kept,
    )
    # Slice all three data splits to the pruned column set.
    X_train_sc = X_train_sc_full[:, keep_idx]
    X_cal_sc   = X_cal_sc_full[:, keep_idx]
    X_test_sc  = X_test_sc_full[:, keep_idx]
    all_xgb_cols_full = all_xgb_cols
    all_xgb_cols = pruned_xgb_cols

    # ── Step 6d: final model fit with pruned features + tuned (or default) params ─
    dir_model, ret_model = build_models(
        X_train_sc, y_dir_train, X_cal_sc, y_dir_cal,
        y_ret_train, y_ret_cal,
        sample_weight=sample_weight,
        param_overrides=best_params_override,
    )

    # ── Step 6f: regime-conditional direction models (bull / bear split) ────────
    # Train two extra XGBoost classifiers using the same pruned+scaled feature set
    # but restricted to rows where regime=+1 (bull: SPY above 200MA AND VIX<20)
    # or regime=-1 (bear: SPY below 200MA OR VIX>=28).  At live prediction time
    # the model matching the current market regime is selected; if neither regime
    # has enough training rows (< REGIME_MIN_TRAIN_ROWS) both are skipped and the
    # combined model is used instead.  This lets the model specialise: bull-regime
    # patterns (momentum, growth factor) differ from bear-regime patterns
    # (mean-reversion, quality/defensive factor).
    dir_model_bull: xgb.XGBClassifier | None = None   # bull-specific direction model
    dir_model_bear: xgb.XGBClassifier | None = None   # bear-specific direction model
    regime_models_trained = False                      # flag: did we actually train them?
    if REGIME_CONDITIONAL_MODELS:
        # Find the 'regime' column index inside the FULL (pre-pruning) feature matrix.
        # X_train is the unscaled version from Step 5 — its columns match all_xgb_cols_full.
        # Raw values: +1 = bull, 0 = neutral, -1 = bear.
        regime_col_full = next(
            (i for i, c in enumerate(all_xgb_cols_full) if c == "regime"), None
        )
        if regime_col_full is not None:
            # Extract raw (unscaled) regime label for every training row.
            # X_train is unscaled so values are exactly -1 / 0 / +1.
            regime_vals = X_train[:, regime_col_full]        # shape: (n_train_rows,)
            bull_mask   = regime_vals > 0.5                  # keeps +1 rows
            bear_mask   = regime_vals < -0.5                 # keeps -1 rows
            n_bull = int(np.sum(bull_mask))
            n_bear = int(np.sum(bear_mask))
            log.info(
                "[pooled] Regime split — bull rows: %d  bear rows: %d  "
                "(need >= %d each to train regime models)",
                n_bull, n_bear, REGIME_MIN_TRAIN_ROWS,
            )
            if n_bull >= REGIME_MIN_TRAIN_ROWS and n_bear >= REGIME_MIN_TRAIN_ROWS:
                # Slice the pruned+scaled training matrix to each regime subset.
                # X_train_sc is already pruned (via keep_idx) and scaled — the masks
                # select rows (not columns), so the feature layout stays the same.
                X_bull_sc = X_train_sc[bull_mask]
                y_bull    = y_dir_train[bull_mask]
                y_ret_bull = y_ret_train[bull_mask]
                sw_bull   = sample_weight[bull_mask] if sample_weight is not None else None
                X_bear_sc = X_train_sc[bear_mask]
                y_bear    = y_dir_train[bear_mask]
                y_ret_bear = y_ret_train[bear_mask]
                sw_bear   = sample_weight[bear_mask] if sample_weight is not None else None
                # Use the full calibration set for early-stopping eval_set — splitting
                # the calibration set by regime would leave too few rows for a stable
                # stopping signal.  The calibrator (isotonic) is still trained on all
                # calibration rows after this step.
                dir_model_bull, _ = build_models(
                    X_bull_sc, y_bull, X_cal_sc, y_dir_cal,
                    y_ret_bull, y_ret_cal,
                    sample_weight=sw_bull,
                    param_overrides=best_params_override,
                )
                dir_model_bear, _ = build_models(
                    X_bear_sc, y_bear, X_cal_sc, y_dir_cal,
                    y_ret_bear, y_ret_cal,
                    sample_weight=sw_bear,
                    param_overrides=best_params_override,
                )
                # Save regime models alongside the combined model.
                dir_model_bull.save_model(os.path.join(MODEL_DIR, "pooled_xgb_dir_bull.json"))
                dir_model_bear.save_model(os.path.join(MODEL_DIR, "pooled_xgb_dir_bear.json"))
                regime_models_trained = True
                log.info(
                    "[pooled] Regime-conditional models saved — "
                    "bull (%d rows) → pooled_xgb_dir_bull.json, "
                    "bear (%d rows) → pooled_xgb_dir_bear.json",
                    n_bull, n_bear,
                )
            else:
                log.warning(
                    "[pooled] Regime-conditional models SKIPPED — insufficient rows "
                    "(bull=%d, bear=%d; need >= %d each)",
                    n_bull, n_bear, REGIME_MIN_TRAIN_ROWS,
                )
        else:
            log.warning(
                "[pooled] 'regime' column not found in feature list — "
                "skipping regime-conditional model training"
            )

    # ── Step 6e: train secondary 5-day direction model (multi-horizon ensemble) ─
    # h5 gets its own pruned matrix so short-horizon sentiment can survive
    # pruning without forcing sparse news features into the h20/return models.
    h5_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_lambda=10.0,
        tree_method="hist",
        early_stopping_rounds=20,
        random_state=42,
    )
    # Class-balance weights for h5 — computed from h5 labels, not h20 labels.
    # h5 label = (5-day return > 0). In a 2010-2024 bull-market dataset this is
    # UP ~55% of the time. Without correction the heavily regularised h5 model
    # just outputs the base rate (≈0.55) for every row.
    # Blend = 0.65 × h20 + 0.35 × 0.55.  Even when h20 ≈ 0.50, blend ≈ 0.5175 —
    # always ≥ 0.5 → all predictions UP → accuracy = UP rate = baseline every time.
    n_pos_h5 = max(int(np.sum(y_dir_h5_train == 1)), 1)
    n_neg_h5 = max(int(np.sum(y_dir_h5_train == 0)), 1)
    class_w_h5 = np.where(
        y_dir_h5_train == 1, float(n_neg_h5) / n_pos_h5, 1.0
    ).astype(np.float32)
    combined_w_h5 = (
        (sample_weight * class_w_h5).astype(np.float32)
        if sample_weight is not None
        else class_w_h5
    )
    log.info("[pooled] h5 class balance: n_pos=%d  n_neg=%d  w_pos=%.3f",
             n_pos_h5, n_neg_h5, float(n_neg_h5) / n_pos_h5)
    if H5_SENTIMENT_FEATURES_ENABLED:
        _init_h5 = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.7,
            tree_method="hist",
            random_state=42,
            eval_metric="logloss",
        )
        _init_h5.fit(X_train_sc_full, y_dir_h5_train, sample_weight=combined_w_h5, verbose=False)
        h5_keep_idx = _get_pruned_feature_indices(
            _init_h5.feature_importances_,
            n_base_feats=n_base_xgb,
            n_ticker_dummies=n_tickers,
            top_k=FEATURE_IMPORTANCE_TOP_K,
            sent_idx=sent_base_idx,
            macro_idx=macro_base_idx,    # protect macro/regime features for h5 too
            xs_idx=xs_base_idx,          # protect cross-sectional rank features for h5 too
        )
        h5_xgb_cols = [all_xgb_cols_full[i] for i in h5_keep_idx]
        h5_sent_kept = sum(1 for c in h5_xgb_cols if c.startswith("sent_z_"))
        log.info(
            "[pooled] h5 feature pruning: %d → %d cols (%d sentiment kept)",
            len(all_xgb_cols_full), len(h5_xgb_cols), h5_sent_kept,
        )
        X_train_h5_sc = X_train_sc_full[:, h5_keep_idx]
        X_cal_h5_sc = X_cal_sc_full[:, h5_keep_idx]
        X_test_h5_sc = X_test_sc_full[:, h5_keep_idx]
    else:
        h5_keep_idx = None
        h5_xgb_cols = []
        h5_sent_kept = 0
        X_train_h5_sc = X_train_sc
        X_cal_h5_sc = X_cal_sc
        X_test_h5_sc = X_test_sc
    h5_model.fit(
        X_train_h5_sc, y_dir_h5_train,
        eval_set=[(X_cal_h5_sc, y_dir_h5_cal)],
        sample_weight=combined_w_h5,
        verbose=False,
    )
    log.info("[pooled] h5 model trained on %d rows", len(y_dir_h5_train))

    cal_probs_raw = dir_model.predict_proba(X_cal_sc)[:, 1]
    cal_probs_h5 = h5_model.predict_proba(X_cal_h5_sc)[:, 1]
    cal_probs_blend = MULTI_HORIZON_H20_WEIGHT * cal_probs_raw + MULTI_HORIZON_H5_WEIGHT * cal_probs_h5
    conformal_qhat = compute_conformal_qhat(cal_probs_blend, y_dir_cal)
    calibrator    = fit_direction_calibrator(cal_probs_raw, y_dir_cal)
    threshold, fallback = compute_dynamic_threshold(
        calibrator,
        target_precision=CONFIDENCE_TARGET_PRECISION,
        default=DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
        raw_probs=cal_probs_raw,
        y_true=y_dir_cal,
    )
    log.info("[pooled] Confidence threshold: %.1f (fallback=%s)", threshold, fallback)
    if conformal_qhat is not None:
        log.info("[pooled] Conformal qhat(alpha=%.2f): %.4f", CONFORMAL_ALPHA, conformal_qhat)

    # ── Step 7: evaluate on test set ──────────────────────────────────────────
    # Multi-horizon blend: h20 primary (65% weight) + h5 secondary (35% weight).
    # p_up_raw is the blended probability used for ranking; calibration is applied
    # only to the h20 component so the isotonic calibrator sees a consistent
    # distribution (calibrating a blend directly would require retraining calibrator).
    test_p_up_h20   = dir_model.predict_proba(X_test_sc)[:, 1]
    test_p_up_h5    = h5_model.predict_proba(X_test_h5_sc)[:, 1]
    test_p_up_raw   = MULTI_HORIZON_H20_WEIGHT * test_p_up_h20 + MULTI_HORIZON_H5_WEIGHT * test_p_up_h5
    if calibrator is not None:
        test_p_up = np.array(
            [float(calibrator.predict([float(p)])[0]) for p in test_p_up_h20],
            dtype=float,
        )
    else:
        test_p_up = test_p_up_raw
    dir_acc  = float(accuracy_score(y_dir_test, (test_p_up_raw >= 0.5).astype(int)) * 100.0)
    baseline = float(np.mean(y_dir_test) * 100.0)
    # Buckets, sweep, and plots all use the blended test_p_up_raw so that:
    # (a) the bucket boundaries saved to metadata match the live confidence scale
    #     used by predict.py's assign_signal_quality(), and
    # (b) diagnostics reflect what the backtest/live system actually trades on.
    # test_p_up (calibrated h20) is kept separately only for the threshold
    # calculation above, where the calibrator operates on h20 probabilities.
    #
    # Adaptive bucket edges: compute decile-based edges from the ACTUAL blended
    # confidence distribution instead of hardcoded [50-55, 55-60, ...] ranges.
    # If the model outputs everything in 50-52%, all ten buckets will sit inside
    # that range — so every bucket has observations and we can measure precision
    # honestly rather than reporting five empty fixed bins.
    _conf_arr  = np.maximum(test_p_up_raw, 1.0 - test_p_up_raw) * 100.0
    _bucket_edges = compute_bucket_edges(_conf_arr, n_buckets=10)
    buckets  = confidence_buckets(test_p_up_raw, y_dir_test, edges=_bucket_edges)
    sweep    = threshold_sweep(test_p_up_raw, y_dir_test)
    conf_diag = confidence_diagnostics(test_p_up_raw, y_dir_test, threshold, buckets)
    log.info("[pooled] Test accuracy: %.2f%%  baseline: %.2f%%  threshold: %.1f%%",
             dir_acc, baseline, threshold)
    # ── Ranking-quality metrics (AUC + AP) ─────────────────────────────────────
    # Accuracy is misleading for the excess_return target because baseline UP rate
    # is ~45% (most stock-days don't beat SPY by >0.5% over 20d). The cross-sectional
    # ranker actually consumes the RAW probability ordering, so the metrics that
    # matter are AUC (rank-correctness) and AP (precision under imbalance).
    # Treat AUC < 0.52 as a failed model — baseline AUC = 0.50.
    try:
        test_auc = float(roc_auc_score(y_dir_test, test_p_up_raw))
        test_ap  = float(average_precision_score(y_dir_test, test_p_up_raw))
        log.info(
            "[pooled] Test AUC: %.4f  AP: %.4f  (acc edge over baseline: %+.2fpp)",
            test_auc, test_ap, dir_acc - baseline,
        )
    except Exception as _auc_err:
        # roc_auc_score raises if y_dir_test has only one class. Guard so we don't
        # crash the training pipeline on a degenerate split.
        test_auc = float("nan")
        test_ap  = float("nan")
        log.warning("[pooled] Could not compute AUC/AP: %s", _auc_err)
    if not conf_diag.get("confidence_validated", False):
        log.warning("[pooled] Confidence diagnostics failed: %s", conf_diag.get("reason"))

    # ── Step 7c: Logistic-Regression-L1 sanity baseline ──────────────────────
    # The diagnostic ladder (universe size, horizon, target, architecture,
    # macro/xs_rank protection, regime split, h5 ensemble) all produced AUC ~0.50.
    # Before declaring the feature set dead, sanity-check with a totally different
    # model class: a simple linear logistic regression with L1 regularization.
    #
    # Logic:
    #   * If LR-L1 also produces test AUC ~0.50 → features are genuinely zero-info,
    #     no model class can extract signal, conclude with the cash-overlay product.
    #   * If LR-L1 gives ≥ 0.53 while XGBoost gives ~0.50 → XGBoost is broken
    #     (likely over-regularized by nested CV picking max_depth=6 + min_child_weight=10
    #     + reg_lambda=10). Worth investigating before giving up.
    #
    # Uses the SAME pruned + scaled feature matrix as XGBoost so the comparison is
    # apples-to-apples. C is chosen by calibration-set AUC (no peek at test set).
    lr_test_auc = float("nan")
    lr_test_ap = float("nan")
    lr_best_C = float("nan")
    lr_n_nonzero = 0
    try:
        from sklearn.linear_model import LogisticRegression
        # Search a narrow grid of regularization strengths.  L1 + liblinear gives
        # sparse coefficients, so we also report nonzero count per fit to see how
        # many features the linear model actually uses.
        lr_candidates: list[tuple[float, float, "LogisticRegression"]] = []
        for C_val in (0.01, 0.1, 1.0, 10.0):
            lr = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=C_val,
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )
            # sample_weight = exponential time-decay (same as XGB) so the LR
            # weighs recent rows more heavily, matching the XGB training regime.
            lr.fit(X_train_sc, y_dir_train, sample_weight=sample_weight)
            cal_p = lr.predict_proba(X_cal_sc)[:, 1]
            if len(set(y_dir_cal)) >= 2:
                cal_auc = float(roc_auc_score(y_dir_cal, cal_p))
            else:
                cal_auc = 0.5
            lr_candidates.append((C_val, cal_auc, lr))
        # Best by calibration AUC (no test-set peeking).
        lr_candidates.sort(key=lambda t: -t[1])
        lr_best_C, lr_best_cal_auc, best_lr = lr_candidates[0]
        test_lr_p = best_lr.predict_proba(X_test_sc)[:, 1]
        if len(set(y_dir_test)) >= 2:
            lr_test_auc = float(roc_auc_score(y_dir_test, test_lr_p))
            lr_test_ap = float(average_precision_score(y_dir_test, test_lr_p))
        # Count nonzero coefficients to see how many features the linear model uses.
        lr_n_nonzero = int(np.sum(np.abs(best_lr.coef_) > 1e-6))
        log.info(
            "[pooled] LR-L1 baseline: test AUC=%.4f  AP=%.4f  best_C=%.2f  "
            "cal_AUC=%.4f  nonzero_coefs=%d/%d",
            lr_test_auc, lr_test_ap, lr_best_C, lr_best_cal_auc,
            lr_n_nonzero, X_train_sc.shape[1],
        )
        # Comparison verdict — single-line summary the user can scan.
        if not np.isnan(lr_test_auc) and not np.isnan(test_auc):
            delta = lr_test_auc - test_auc
            if delta > 0.01:
                verdict = (
                    f"LR > XGB by {delta:+.4f} — XGB likely over-regularized; "
                    f"investigate nested-CV params"
                )
            elif delta < -0.01:
                verdict = (
                    f"XGB > LR by {-delta:+.4f} — XGB picks fine; "
                    f"signal is real but bounded"
                )
            else:
                verdict = (
                    f"XGB ≈ LR (Δ={delta:+.4f}) — both models extract the same "
                    f"signal level; feature set is the bottleneck"
                )
            log.info("[pooled] AUC verdict — XGB: %.4f  LR-L1: %.4f  →  %s",
                     test_auc, lr_test_auc, verdict)
    except Exception as _lr_err:
        log.warning("[pooled] LR-L1 baseline failed: %s", _lr_err)

    # ── Step 7b: save results plot (same chart as per-ticker training) ────────
    # Reuses the existing save_xgb_results_plot() with ticker="pooled" so the
    # image is written to models/pooled_xgb_only_results.png.
    pooled_plot_path = save_xgb_results_plot(
        "pooled",
        test_p_up_raw,
        y_dir_test,
        dir_acc,
        baseline,
    )
    log.info("[pooled] Saved results plot %s", pooled_plot_path)

    # Per-ticker grouped bar chart (model acc vs baseline, one pair per ticker)
    ticker_plot_path = save_pooled_ticker_plot(
        ticker_labels_test,
        test_p_up_raw,
        y_dir_test,
    )
    log.info("[pooled] Saved per-ticker plot %s", ticker_plot_path)

    # ── Step 8: save pooled artefacts ─────────────────────────────────────────
    dir_model.save_model(os.path.join(MODEL_DIR, "pooled_xgb_dir.json"))
    ret_model.save_model(os.path.join(MODEL_DIR, "pooled_xgb_ret.json"))
    # Save the secondary 5-day direction model alongside the primary h20 model.
    h5_model.save_model(os.path.join(MODEL_DIR, "pooled_xgb_dir_h5.json"))
    log.info("[pooled] Saved h5 model to pooled_xgb_dir_h5.json")
    cal_path = os.path.join(MODEL_DIR, "pooled_direction_calibrator.pkl")
    save_direction_calibrator(calibrator, cal_path)

    pooled_meta = {
        "scaler": scaler,
        "xgb_scaler": scaler,
        "feature_cols": common_features,
        "feature_cols_raw": common_features,
        # xgb_feature_cols now reflects the PRUNED column list (base subset + dummies).
        # Downstream code (predict.py, backtest.py) must use this exact list
        # when calling build_xgb_matrix so the feature matrix matches the model.
        "xgb_feature_cols": all_xgb_cols,
        "xgb_interaction_features_enabled": True,
        "ticker_dummy_cols": ticker_dummy_xgb_cols,
        "ticker_classes": ticker_classes,
        # After pruning, n_base_xgb_features is the count of KEPT base cols (no dummies).
        "n_base_xgb_features": n_pruned_base,
        "feature_pruning_top_k": FEATURE_IMPORTANCE_TOP_K,
        "feature_pruning_keep_idx": keep_idx.tolist(),
        "h5_xgb_feature_cols": h5_xgb_cols,
        "h5_feature_pruning_keep_idx": h5_keep_idx.tolist() if h5_keep_idx is not None else None,
        "h5_sentiment_features_enabled": bool(H5_SENTIMENT_FEATURES_ENABLED and h5_sent_kept > 0),
        "tune_hyperparams_used": tune_hyperparams,
        "best_params_override": best_params_override or {},
        "pooled": True,
        # Only prediction tickers are listed here — training-only (survivorship)
        # tickers trained the model but cannot generate live signals.
        "pooled_tickers": [t for t in valid_tickers if t not in training_only_set],
        "survivorship_training_tickers": sorted(training_only_set & set(valid_tickers)),
        "return_horizon": RETURN_HORIZON_DAYS,
        "return_bin_labels": RETURN_BIN_LABELS,
        "n_return_bins": N_RETURN_BINS,
        "ensemble_weights": {"xgboost": 1.0},
        "model_accuracies": {"xgboost": dir_acc},
        "ensemble_accuracy": dir_acc,
        "baseline_up_rate": baseline,
        # AUC + AP saved alongside accuracy so model_quality_report.csv and
        # downstream tooling can rank models on ranking quality instead of
        # absolute accuracy (which is biased low under excess_return target).
        "test_auc": test_auc,
        "test_average_precision": test_ap,
        # Logistic Regression L1 sanity baseline — proves whether the feature
        # set has signal that any model class can extract, vs XGBoost-specific
        # over-regularization.  Compare lr_test_auc to test_auc to decide.
        "lr_test_auc": lr_test_auc,
        "lr_test_ap": lr_test_ap,
        "lr_best_C": lr_best_C,
        "lr_nonzero_coefs": lr_n_nonzero,
        "selected_mode": "xgboost_only",
        "selected_model_name": None,
        "confidence_threshold": threshold,
        "threshold_used_fallback": fallback,
        "confidence_calibrator": cal_path,
        "xgb_model_format": "json",
        # Multi-horizon ensemble: path to the 5-day secondary direction model.
        # predict.py and backtest.py load this alongside pooled_xgb_dir.json
        # and blend according to MULTI_HORIZON_* weights from settings.py.
        "h5_model_path": os.path.join(MODEL_DIR, "pooled_xgb_dir_h5.json"),
        "multi_horizon_weights": {
            "h20": float(MULTI_HORIZON_H20_WEIGHT),
            "h5": float(MULTI_HORIZON_H5_WEIGHT),
        },
        # Regime-conditional models: separate bull/bear classifiers trained by Step 6f.
        # predict.py loads these alongside pooled_xgb_dir.json and routes each
        # prediction to the model matching the current market regime.
        "regime_conditional_models": regime_models_trained,
        "regime_bull_model_path": (
            os.path.join(MODEL_DIR, "pooled_xgb_dir_bull.json")
            if regime_models_trained else None
        ),
        "regime_bear_model_path": (
            os.path.join(MODEL_DIR, "pooled_xgb_dir_bear.json")
            if regime_models_trained else None
        ),
        "confidence_buckets": buckets,
        "confidence_diagnostics": conf_diag,
        "conformal_alpha": float(CONFORMAL_ALPHA),
        "conformal_qhat": float(conformal_qhat) if conformal_qhat is not None else None,
        "conformal_min_calibration_rows": int(CONFORMAL_MIN_CALIBRATION_ROWS),
        # bucket_edges: the (lo, hi) pairs used to build the above bucket report.
        # Saved so predict.py / backtest.py can reproduce the same split when
        # reporting live confidence statistics.  These are decile-based, computed
        # from the actual test-set confidence distribution rather than fixed bins.
        "bucket_edges": [[float(lo), float(hi)] for lo, hi in _bucket_edges],
        "threshold_sweep": sweep,
        "results_plot": pooled_plot_path,
        "ticker_accuracy_plot": ticker_plot_path,
        "prediction_target": PREDICTION_TARGET,
        "model_version": "xgb_pooled_v1",
        "trained_at": datetime.now().isoformat(),
        "sentiment_features_removed": False,
        "sentiment_features_distribution_matched": bool(pooled_sentiment_stats),
        "sentiment_zscore_stats_by_ticker": pooled_sentiment_stats,
    }
    with open(os.path.join(MODEL_DIR, "pooled_scaler.pkl"), "wb") as f:
        pickle.dump(pooled_meta, f)

    # ── Step 9: save per-ticker artefacts (compatibility with predict/backtest) -
    # Training-only (survivorship) tickers are skipped — they have no live signal
    # and no artefact consumers.  The pooled model already learned from their data.
    for ticker in valid_tickers:
        if ticker in training_only_set:
            log.info("[pooled] Skipping artefact save for training-only ticker %s", ticker)
            continue
        ticker_meta = dict(pooled_meta)
        ticker_meta["ticker"] = ticker
        ticker_meta["sentiment_zscore_stats"] = pooled_sentiment_stats.get(ticker, {})
        with open(os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"), "wb") as f:
            pickle.dump(ticker_meta, f)
        dir_model.save_model(os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.json"))
        ret_model.save_model(os.path.join(MODEL_DIR, f"{ticker}_xgb_ret.json"))
        save_direction_calibrator(calibrator, os.path.join(MODEL_DIR, f"{ticker}_direction_calibrator.pkl"))
        with open(os.path.join(MODEL_DIR, f"{ticker}_train_summary.json"), "w") as f:
            json.dump({
                "ticker": ticker,
                "direction_accuracy": round(dir_acc, 2),
                "baseline_up_rate": round(baseline, 2),
                "confidence_threshold": round(threshold, 1),
                "threshold_used_fallback": fallback,
                "calib_set_size": len(y_dir_cal),
                "confidence_buckets": buckets,
                "confidence_diagnostics": conf_diag,
                "threshold_sweep": sweep,
                "prediction_target": PREDICTION_TARGET,
                "pooled": True,
                "pooled_tickers": valid_tickers,
            }, f, indent=2)
        log.info("[pooled] Saved artefacts for %s", ticker)

    # One pooled registry entry points at the shared artifacts. Per-ticker
    # compatibility copies are reproducible from these files and metadata.
    pooled_artifacts = [
        os.path.join(MODEL_DIR, "pooled_xgb_dir.json"),
        os.path.join(MODEL_DIR, "pooled_xgb_ret.json"),
        os.path.join(MODEL_DIR, "pooled_xgb_dir_h5.json"),
        os.path.join(MODEL_DIR, "pooled_scaler.pkl"),
        os.path.join(MODEL_DIR, "pooled_direction_calibrator.pkl"),
    ]
    pooled_artifacts.extend(
        path for path in (
            os.path.join(MODEL_DIR, "pooled_xgb_dir_bull.json"),
            os.path.join(MODEL_DIR, "pooled_xgb_dir_bear.json"),
        )
        if os.path.exists(path)
    )
    drift_baseline_path = snapshot_from_metadata(
        "pooled",
        pooled_meta,
        tickers=valid_tickers,
    )
    register_model_run(
        "pooled",
        [*pooled_artifacts, drift_baseline_path],
        metrics={
            "direction_accuracy": round(dir_acc, 4),
            "baseline_up_rate": round(baseline, 4),
            "test_auc": pooled_meta.get("test_auc"),
            "test_average_precision": pooled_meta.get("test_average_precision"),
        },
        metadata={
            "training_mode": "pooled",
            "prediction_target": PREDICTION_TARGET,
            "model_version": pooled_meta["model_version"],
            "pooled_tickers": pooled_meta["pooled_tickers"],
            "survivorship_training_tickers": pooled_meta["survivorship_training_tickers"],
        },
    )

    log.info("[pooled] Done. %d tickers, %.2f%% acc vs %.2f%% baseline",
             len(valid_tickers), dir_acc, baseline)
    return True


def main() -> None:
    global PREDICTION_TARGET

    parser = argparse.ArgumentParser(description="XGBoost-only trainer")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument(
        "--target",
        choices=["excess_return", "triple_barrier", "vol_adjusted", "direction"],
        default=None,
        help="Override settings.PREDICTION_TARGET for this training run.",
    )
    parser.add_argument(
        "--skip-leakage-audit",
        action="store_true",
        help="Skip the pre-training leakage audit gate.",
    )
    parser.add_argument(
        "--params-from-tuning", action="store_true",
        help=(
            "Load best hyperparameters from models/tuning/{ticker}_best_xgb_params.json "
            "if available, instead of settings.py defaults. "
            "Use after running tune_xgb_best_tickers.py without --apply."
        ),
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help=(
            "Skip nested walk-forward hyperparameter search (faster run). "
            "The search adds ~5 minutes for a 10-ticker pooled model. "
            "When omitted, TUNE_HYPERPARAMS from settings.py controls the default."
        ),
    )
    parser.add_argument(
        "--ranker",
        action="store_true",
        help=(
            "Experimental pooled XGBRanker train-only path. Fits rank:pairwise "
            "on continuous forward excess returns and reports daily rank IC. "
            "Writes separate ranker artifacts without replacing the classifier."
        ),
    )
    args = parser.parse_args()
    if args.target:
        PREDICTION_TARGET = args.target
        log.info("Using prediction target override: %s", PREDICTION_TARGET)

    if not args.skip_leakage_audit:
        audit_cmd = [sys.executable, "leakage_audit.py"]
        audit = run_utf8(audit_cmd, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True)
        if audit.stdout:
            log.info("Leakage audit output:\n%s", audit.stdout.strip())
        if audit.stderr:
            log.warning("Leakage audit warnings:\n%s", audit.stderr.strip())
        if audit.returncode != 0:
            raise SystemExit(f"Leakage audit failed with rc={audit.returncode}; training aborted.")

    # Use WATCHLIST as the authoritative ticker source for pooled training.
    # Previously this scanned data/*.parquet, which meant any extra parquet downloaded
    # by research.py (expanded universe, phase-3 tickers) would silently enter the
    # training pool and corrupt the model for the core universe.
    # SURVIVORSHIP_TRAINING_TICKERS are added separately in train_pooled() below
    # so they contribute training rows without generating live artefacts.
    available_parquets = {
        f.replace(".parquet", "")
        for f in os.listdir(DATA_DIR)
        if f.endswith(".parquet")
    }
    tickers = (
        [args.ticker.upper()]
        if args.ticker
        else [t for t in WATCHLIST if t in available_parquets]
    )
    if not tickers:
        raise SystemExit(
            f"No WATCHLIST tickers have parquets in {DATA_DIR}. "
            "Run research.py first, or pass --ticker to train a single ticker."
        )

    # ── Pooled training path ───────────────────────────────────────────────────
    use_pooled = POOLED_TRAINING if not args.ticker else False  # single-ticker forces per-ticker
    # TUNE_HYPERPARAMS (settings.py) is the default; --no-tune overrides it to False.
    run_tune = TUNE_HYPERPARAMS and not args.no_tune
    run_ranker = bool(args.ranker or POOLED_RANKER_EXPERIMENT_ENABLED)
    if use_pooled:
        log.info(
            "POOLED_TRAINING=True — training cross-sectional model on all tickers "
            "(tune_hyperparams=%s, ranker_experiment=%s)", run_tune, run_ranker,
        )
        success = train_pooled(
            tickers,
            tune_hyperparams=run_tune,
            training_only_tickers=SURVIVORSHIP_TRAINING_TICKERS,
            ranker_experiment=run_ranker,
        )
        if not success:
            log.warning("Pooled training failed; falling back to per-ticker training")
            use_pooled = False

    if not use_pooled:
        for ticker in tickers:
            param_overrides = None
            if args.params_from_tuning:
                tuning_path = os.path.join(MODEL_DIR, "tuning", f"{ticker}_best_xgb_params.json")
                if os.path.exists(tuning_path):
                    with open(tuning_path) as f:
                        tuning = json.load(f)
                    param_overrides = tuning.get("params")
                    log.info("[%s] Loaded tuned params from %s: %s", ticker, tuning_path, param_overrides)
                else:
                    log.info("[%s] No tuning file found at %s — using settings.py defaults", ticker, tuning_path)
            train_ticker(ticker, param_overrides=param_overrides)


if __name__ == "__main__":
    main()
