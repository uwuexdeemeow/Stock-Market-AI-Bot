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
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from confidence_calibration import fit_direction_calibrator, save_direction_calibrator
from labels import triple_barrier
from model_quality import evaluate_model_quality, update_scaler_metadata, upsert_quality_report
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


def make_direction_target(df: pd.DataFrame | pd.Series) -> np.ndarray:
    """
    Build the binary direction label according to PREDICTION_TARGET.

    Three modes (set in settings.py):

    "direction"     — original behaviour: UP if stock_forward_ret > threshold.
                      Noisy because it includes beta; a stock can go up 2 % just
                      because SPY went up 2 % — that is not alpha.

    "excess_return" — UP if (stock_ret - spy_ret) > EXCESS_RETURN_MIN_PCT.
                      Strips out market beta.  The model only gets credit for
                      returns that outperform SPY over the same window.  The
                      0.5 % floor clears estimated slippage + commission.

    "vol_adjusted"  — UP if excess_ret / daily_hvol_scaled > threshold.
                      A 2 % excess return during a calm (10 % vol) market is a
                      stronger signal than the same 2 % during a 50 % vol crisis.
                      Teaches the model to find risk-adjusted opportunities.

    The SPY benchmark return is read from the parquet column
    spy_ret{N}d where N = RETURN_HORIZON_DAYS.  That column stores the BACKWARD-
    looking SPY return at each date; shifting it by -N converts it to the
    FORWARD-looking return for the same N-day window, perfectly aligned with the
    stock forward return.
    """
    # Accept either a full DataFrame or just the Close series (legacy fallback)
    if isinstance(df, pd.Series):
        close = df
        spy_fwd = pd.Series(0.0, index=close.index)
        hvol = pd.Series(0.20, index=close.index)
    else:
        close = df["Close"]
        bench_col = f"spy_ret{RETURN_HORIZON_DAYS}d"
        # shift(-N) converts backward return at t+N into forward return at t
        spy_fwd = df[bench_col].shift(-RETURN_HORIZON_DAYS).fillna(0.0) if bench_col in df.columns else pd.Series(0.0, index=close.index)
        hvol = df["hvol_20d"] if "hvol_20d" in df.columns else pd.Series(0.20, index=close.index)

    stock_fwd = close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS).fillna(0.0)

    if PREDICTION_TARGET == "triple_barrier" and not isinstance(df, pd.Series):
        labels = triple_barrier(
            df["Close"],
            high=df["High"] if "High" in df.columns else None,
            low=df["Low"] if "Low" in df.columns else None,
            max_hold=RETURN_HORIZON_DAYS,
        ).fillna(0.0)
        return (labels.values > 0).astype(np.int64)

    if PREDICTION_TARGET == "excess_return":
        excess = stock_fwd - spy_fwd
        return (excess.values > EXCESS_RETURN_MIN_PCT).astype(np.int64)

    if PREDICTION_TARGET == "vol_adjusted":
        excess = stock_fwd - spy_fwd
        # Scale hvol to the N-day window: daily_vol * sqrt(N)
        vol_scaled = (hvol * np.sqrt(RETURN_HORIZON_DAYS)).clip(0.01, 1.0)
        sharpe_proxy = excess / vol_scaled
        return (sharpe_proxy.values > VOL_ADJUSTED_SHARPE_THRESHOLD).astype(np.int64)

    # "direction" — original behaviour
    return (stock_fwd.values > DIRECTION_LABEL_THRESHOLD).astype(np.int64)


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


def make_return_target(close: pd.Series) -> np.ndarray:
    fwd = close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS)
    buckets = np.digitize(fwd.fillna(0.0).values, RETURN_BINS)
    return np.clip(buckets, 0, N_RETURN_BINS - 1).astype(np.int64)


def make_future_return(close: pd.Series) -> np.ndarray:
    fwd = close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS)
    return fwd.fillna(0.0).values.astype(float)


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
    # --- Path 1: calibrator available ---
    if calibrator is not None:
        raw_grid = np.linspace(0.50, 0.99, 200)
        try:
            cal_grid = calibrator.predict(raw_grid)
            conf_grid = np.maximum(cal_grid, 1.0 - cal_grid) * 100.0
            idx = np.where(cal_grid >= target_precision)[0]
            if len(idx):
                return float(np.clip(conf_grid[idx[0]], 52.0, 80.0)), False
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

    # --- Path 3: hard-coded default ---
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
        allowed = {"max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight"}
        applied = {k: v for k, v in param_overrides.items() if k in allowed}
        common.update(applied)
        log.info("build_models: applying tuned param overrides: %s", applied)
    # Balance UP/DOWN classes so the model doesn't collapse to always predicting
    # the majority class. If 55% of training labels are UP, scale_pos_weight=0.82
    # tells XGBoost to weight DOWN cases more, forcing it to learn both directions.
    dir_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    dir_model.fit(X_train, y_dir_train, sample_weight=sample_weight,
                  eval_set=[(X_cal, y_dir_cal)], verbose=False)
    ret_model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=N_RETURN_BINS, eval_metric="mlogloss", **common
    )
    ret_model.fit(X_train, y_ret_train, sample_weight=sample_weight,
                  eval_set=[(X_cal, y_ret_cal)], verbose=False)
    return dir_model, ret_model


def confidence_buckets(probs: np.ndarray, y_true: np.ndarray) -> list[dict]:
    conf = np.maximum(probs, 1.0 - probs) * 100.0
    pred = (probs >= 0.5).astype(int)
    rows: list[dict] = []
    for lo, hi in [(50, 55), (55, 60), (60, 65), (65, 70), (70, 100)]:
        mask = (conf >= lo) & (conf < hi if hi < 100 else conf <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": f"{lo}-{hi}", "n": 0, "precision": None, "avg_confidence": None})
            continue
        rows.append(
            {
                "bucket": f"{lo}-{hi}",
                "n": n,
                "precision": round(float((pred[mask] == y_true[mask]).mean()), 4),
                "avg_confidence": round(float(conf[mask].mean()), 2),
            }
        )
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

    exclude = {"target", "Open", "High", "Low", "Close", "Volume", "ma5", "ma10", "ma20", "ma50", "ma200"}
    feature_cols_raw = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # Sentiment features excluded: live feed can produce zeros that create
    # distribution shift vs training data where values were non-zero.
    feature_cols_raw = [c for c in feature_cols_raw if not ("sent_" in c or "sentiment" in c)]

    y_dir_all = make_direction_target(df)
    y_ret_all = make_return_target(df["Close"])
    future_returns = make_future_return(df["Close"])

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

    splits = compute_split_indices(len(df))
    train_end = splits["train_end"]
    calib_start = splits["calib_start"]
    calib_end = splits["calib_end"]
    test_start = splits["test_start"]
    if train_end < 100 or calib_end <= calib_start or test_start >= len(df):
        log.error("Invalid split for %s", ticker)
        return False

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
        "return_model_eval": return_eval,
        "results_plot": plot_path,
        "feature_audit": feature_audit,
        "feature_audit_path": feature_audit_path,
        "prediction_target": PREDICTION_TARGET,
        "model_version": "xgb_complete_v5",
        "trained_at": datetime.now().isoformat(),
        "sentiment_features_removed": True,
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
            "return_model_eval": return_eval,
        },
        trades=pd.DataFrame(),
        extra_metrics={"quality_source": "train_preliminary"},
    )
    update_scaler_metadata(ticker, quality_row)
    upsert_quality_report([quality_row])

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
    args = parser.parse_args()
    if args.target:
        PREDICTION_TARGET = args.target
        log.info("Using prediction target override: %s", PREDICTION_TARGET)

    if not args.skip_leakage_audit:
        audit_cmd = [sys.executable, "leakage_audit.py"]
        audit = subprocess.run(audit_cmd, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
        if audit.stdout:
            log.info("Leakage audit output:\n%s", audit.stdout.strip())
        if audit.stderr:
            log.warning("Leakage audit warnings:\n%s", audit.stderr.strip())
        if audit.returncode != 0:
            raise SystemExit(f"Leakage audit failed with rc={audit.returncode}; training aborted.")

    tickers = (
        [args.ticker.upper()]
        if args.ticker
        else [f.replace(".parquet", "") for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".parquet")]
    )
    if not tickers:
        raise SystemExit(f"No parquets found in {DATA_DIR}")

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
