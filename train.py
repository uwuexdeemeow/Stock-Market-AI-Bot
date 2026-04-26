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
from labels import (
    make_direction_target as shared_make_direction_target,
    make_forward_return_target,
    make_return_bucket_target,
    make_spy_forward_return,
)
from model_quality import evaluate_model_quality, update_scaler_metadata, upsert_quality_report
from nested_cv import nested_walk_forward_search
from pipeline_shared import apply_sentiment_distribution_matching, fit_sentiment_zscore_stats
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
    TUNE_HYPERPARAMS,
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


def _legacy_direction_target_wrapper(
    df: pd.DataFrame | pd.Series,
    stock_fwd: pd.Series | None = None,
    spy_fwd: pd.Series | None = None,
) -> np.ndarray:
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
        if spy_fwd is None:
            spy_fwd = pd.Series(0.0, index=close.index)
        hvol = pd.Series(0.20, index=close.index)
    else:
        close = df["Close"]
        if spy_fwd is None:
            # shift(-N) converts backward return at t+N into forward return at t
            spy_fwd = make_spy_forward_return(df)
        hvol = df["hvol_20d"] if "hvol_20d" in df.columns else pd.Series(0.20, index=close.index)

    if stock_fwd is None:
        stock_fwd = make_forward_return(close)

    if PREDICTION_TARGET == "triple_barrier" and not isinstance(df, pd.Series):
        labels = triple_barrier(
            df["Close"],
            high=df["High"] if "High" in df.columns else None,
            low=df["Low"] if "Low" in df.columns else None,
            max_hold=RETURN_HORIZON_DAYS,
        )
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
        allowed = {"max_depth", "learning_rate", "subsample", "colsample_bytree",
                   "min_child_weight", "reg_lambda", "reg_alpha", "gamma"}
        applied = {k: v for k, v in param_overrides.items() if k in allowed}
        common.update(applied)
        log.info("build_models: applying tuned param overrides: %s", applied)

    # Compute scale_pos_weight from the actual training labels.
    # This tells XGBoost to weight the minority class up so it can't shortcut
    # to majority-class prediction.  With triple-barrier symmetric barriers the
    # split should be ~50/50, but if it drifts (e.g. 40% UP / 60% DOWN) this
    # corrects for it automatically.
    n_pos = max(int(np.sum(y_dir_train == 1)), 1)
    n_neg = max(int(np.sum(y_dir_train == 0)), 1)
    spw   = n_neg / n_pos   # DOWN count / UP count
    # Cap at 3× to avoid extreme over-correction on very lopsided sets.
    spw   = float(np.clip(spw, 0.33, 3.0))
    log.info("build_models: scale_pos_weight=%.3f  (n_pos=%d, n_neg=%d)", spw, n_pos, n_neg)

    dir_model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=spw, **common
    )
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


_RISKY_KEYWORDS = (
    "social", "news", "iv_", "put_call", "option", "earn", "eps_",
    "analyst", "recommend", "short_interest", "dark_pool", "dow_", "month_",
    "opex", "calendar", "sector_ret", "ret_vs_sector", "gld_", "hyg_", "tnx_",
    "tlt_", "eem_", "iwm_", "dia_", "uup_",
)
_ALLOWED_PREFIXES = (
    "ret_", "hvol_", "rsi_", "dist_ma", "ma_cross_", "macd", "atr_", "bb_",
    "vol_ratio", "volume_chg_", "hl_range", "spread_proxy", "roc_", "drawdown_",
    "weekly_", "monthly_", "tf_alignment", "spy_", "qqq_", "vix_", "regime",
    "ret_vs_spy", "ret_vs_qqq", "breadth_", "pct_above_",
)
_ALLOWED_EXACT = {"obv_slope", "vwap_dist", "uptick_ratio", "variance_ratio"}
_EXCLUDE_COLS = {"target", "Open", "High", "Low", "Close", "Volume", "ma5", "ma10", "ma20", "ma50", "ma200"}


def _get_pruned_feature_indices(
    importances: np.ndarray,
    n_base_feats: int,
    n_ticker_dummies: int,
    top_k: int,
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
    top_k_idx        = np.argsort(base_importances)[-top_k_actual:]

    # Dummy columns sit at the very end: indices [n_base_feats, n_base_feats+n_ticker_dummies)
    dummy_idx = np.arange(n_base_feats, n_base_feats + n_ticker_dummies)

    # Combine and sort so column order is preserved (XGB requires consistent col order)
    return np.sort(np.unique(np.concatenate([top_k_idx, dummy_idx])))


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the allowed numeric feature columns from a DataFrame."""
    return [
        c for c in df.columns
        if c not in _EXCLUDE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
        and (
            c.startswith("sent_z_")
            or (
                not any(k in c.lower() for k in _RISKY_KEYWORDS)
                and (c in _ALLOWED_EXACT or any(c.startswith(p) for p in _ALLOWED_PREFIXES))
            )
        )
    ]


def train_pooled(
    tickers: list[str],
    param_overrides: dict | None = None,
    tune_hyperparams: bool = True,
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
    log.info("[pooled] Starting cross-sectional training on %d tickers", len(tickers))

    # ── Step 1: find common raw feature set across all tickers ────────────────
    per_ticker_info: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    feature_sets: list[set[str]] = []
    pooled_sentiment_stats: dict[str, dict[str, dict[str, float]]] = {}

    for ticker in tickers:
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
    all_y_ret_list: list[np.ndarray] = []
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
        all_y_ret_list.append(y_ret)
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
    all_dates        = np.concatenate(all_dates_list)
    X_combined       = np.concatenate(all_X_with_dummy, axis=0)
    y_dir_comb       = np.concatenate(all_y_dir_list)
    y_ret_comb       = np.concatenate(all_y_ret_list)
    ticker_labels_all = np.concatenate(all_ticker_labels_list)  # ticker name per row
    sort_order = np.argsort(all_dates, kind="stable")
    all_dates         = all_dates[sort_order]
    X_combined        = X_combined[sort_order]
    y_dir_comb        = y_dir_comb[sort_order]
    y_ret_comb        = y_ret_comb[sort_order]
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

    X_train = X_combined[stride_mask]
    y_dir_train = y_dir_comb[stride_mask]
    y_ret_train = y_ret_comb[stride_mask]
    X_cal   = X_combined[calib_mask]
    y_dir_cal = y_dir_comb[calib_mask]
    y_ret_cal = y_ret_comb[calib_mask]
    X_test  = X_combined[test_mask]
    y_dir_test = y_dir_comb[test_mask]
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
            )
            if fold_results:
                # Take the tunable params from the outer fold with highest accuracy.
                best_fold = max(fold_results, key=lambda r: r.score)
                tuned = {k: best_fold.params[k] for k in tunable_grid}
                log.info("[pooled] Nested CV best params: %s (fold %d score=%.4f)",
                         tuned, best_fold.fold, best_fold.score)
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

    # ── Step 6c: feature pruning — keep top-K base features + all dummies ────
    # XGBoost with hundreds of rolled features is overparameterised relative to
    # our ~1,990 training rows.  Dropping low-importance features reduces
    # variance and stabilises calibration.  Ticker dummies are ALWAYS retained.
    keep_idx = _get_pruned_feature_indices(
        raw_importances,
        n_base_feats=n_base_xgb,
        n_ticker_dummies=n_tickers,
        top_k=FEATURE_IMPORTANCE_TOP_K,
    )
    pruned_xgb_cols = [all_xgb_cols[i] for i in keep_idx]
    n_dropped = len(all_xgb_cols) - len(pruned_xgb_cols)
    n_pruned_base = int((keep_idx < n_base_xgb).sum())  # base cols kept (excl. dummies)
    log.info(
        "[pooled] Feature pruning: %d → %d cols  (%d base kept of %d, %d dropped, %d dummies always kept)",
        len(all_xgb_cols), len(pruned_xgb_cols),
        n_pruned_base, n_base_xgb, n_dropped, n_tickers,
    )
    # Slice all three data splits to the pruned column set.
    X_train_sc = X_train_sc[:, keep_idx]
    X_cal_sc   = X_cal_sc[:, keep_idx]
    X_test_sc  = X_test_sc[:, keep_idx]
    all_xgb_cols   = pruned_xgb_cols

    # ── Step 6d: final model fit with pruned features + tuned (or default) params ─
    dir_model, ret_model = build_models(
        X_train_sc, y_dir_train, X_cal_sc, y_dir_cal,
        y_ret_train, y_ret_cal,
        sample_weight=sample_weight,
        param_overrides=best_params_override,
    )

    cal_probs_raw = dir_model.predict_proba(X_cal_sc)[:, 1]
    calibrator    = fit_direction_calibrator(cal_probs_raw, y_dir_cal)
    threshold, fallback = compute_dynamic_threshold(
        calibrator,
        target_precision=CONFIDENCE_TARGET_PRECISION,
        default=DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
        raw_probs=cal_probs_raw,
        y_true=y_dir_cal,
    )
    log.info("[pooled] Confidence threshold: %.1f (fallback=%s)", threshold, fallback)

    # ── Step 7: evaluate on test set ──────────────────────────────────────────
    test_p_up_raw = dir_model.predict_proba(X_test_sc)[:, 1]
    if calibrator is not None:
        test_p_up = np.array(
            [float(calibrator.predict([float(p)])[0]) for p in test_p_up_raw],
            dtype=float,
        )
    else:
        test_p_up = test_p_up_raw
    dir_acc  = float(accuracy_score(y_dir_test, (test_p_up >= 0.5).astype(int)) * 100.0)
    baseline = float(np.mean(y_dir_test) * 100.0)
    buckets  = confidence_buckets(test_p_up, y_dir_test)
    sweep    = threshold_sweep(test_p_up, y_dir_test)
    log.info("[pooled] Test accuracy: %.2f%%  baseline: %.2f%%  threshold: %.1f%%",
             dir_acc, baseline, threshold)

    # ── Step 7b: save results plot (same chart as per-ticker training) ────────
    # Reuses the existing save_xgb_results_plot() with ticker="pooled" so the
    # image is written to models/pooled_xgb_only_results.png.
    pooled_plot_path = save_xgb_results_plot(
        "pooled",
        test_p_up,
        y_dir_test,
        dir_acc,
        baseline,
    )
    log.info("[pooled] Saved results plot %s", pooled_plot_path)

    # Per-ticker grouped bar chart (model acc vs baseline, one pair per ticker)
    ticker_plot_path = save_pooled_ticker_plot(
        ticker_labels_test,
        test_p_up,
        y_dir_test,
    )
    log.info("[pooled] Saved per-ticker plot %s", ticker_plot_path)

    # ── Step 8: save pooled artefacts ─────────────────────────────────────────
    dir_model.save_model(os.path.join(MODEL_DIR, "pooled_xgb_dir.json"))
    ret_model.save_model(os.path.join(MODEL_DIR, "pooled_xgb_ret.json"))
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
        "ticker_dummy_cols": ticker_dummy_xgb_cols,
        "ticker_classes": ticker_classes,
        # After pruning, n_base_xgb_features is the count of KEPT base cols (no dummies).
        "n_base_xgb_features": n_pruned_base,
        "feature_pruning_top_k": FEATURE_IMPORTANCE_TOP_K,
        "feature_pruning_keep_idx": keep_idx.tolist(),
        "tune_hyperparams_used": tune_hyperparams,
        "best_params_override": best_params_override or {},
        "pooled": True,
        "pooled_tickers": valid_tickers,
        "return_horizon": RETURN_HORIZON_DAYS,
        "return_bin_labels": RETURN_BIN_LABELS,
        "n_return_bins": N_RETURN_BINS,
        "ensemble_weights": {"xgboost": 1.0},
        "model_accuracies": {"xgboost": dir_acc},
        "ensemble_accuracy": dir_acc,
        "baseline_up_rate": baseline,
        "selected_mode": "xgboost_only",
        "selected_model_name": None,
        "confidence_threshold": threshold,
        "threshold_used_fallback": fallback,
        "confidence_calibrator": cal_path,
        "xgb_model_format": "json",
        "confidence_buckets": buckets,
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
    for ticker in valid_tickers:
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
                "threshold_sweep": sweep,
                "prediction_target": PREDICTION_TARGET,
                "pooled": True,
                "pooled_tickers": valid_tickers,
            }, f, indent=2)
        log.info("[pooled] Saved artefacts for %s", ticker)

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

    # ── Pooled training path ───────────────────────────────────────────────────
    use_pooled = POOLED_TRAINING if not args.ticker else False  # single-ticker forces per-ticker
    # TUNE_HYPERPARAMS (settings.py) is the default; --no-tune overrides it to False.
    run_tune = TUNE_HYPERPARAMS and not args.no_tune
    if use_pooled:
        log.info(
            "POOLED_TRAINING=True — training cross-sectional model on all tickers "
            "(tune_hyperparams=%s)", run_tune,
        )
        success = train_pooled(tickers, tune_hyperparams=run_tune)
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
