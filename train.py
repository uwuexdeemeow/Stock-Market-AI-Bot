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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from confidence_calibration import fit_direction_calibrator, save_direction_calibrator
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


def make_direction_target(close: pd.Series) -> np.ndarray:
    fwd = close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS)
    return (fwd.fillna(0.0).values > 0).astype(np.int64)


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


def compute_dynamic_threshold(calibrator, target_precision: float = 0.58, default: float = 57.5) -> tuple[float, bool]:
    """
    Returns (threshold, used_fallback).
    used_fallback=True means the calibrator was None or failed, and the
    default fixed threshold was used instead.
    """
    if calibrator is None:
        return default, True
    raw_grid = np.linspace(0.50, 0.99, 200)
    try:
        cal_grid = calibrator.predict(raw_grid)
        conf_grid = np.maximum(cal_grid, 1.0 - cal_grid) * 100.0
        idx = np.where(cal_grid >= target_precision)[0]
        if len(idx):
            return float(np.clip(conf_grid[idx[0]], 52.0, 80.0)), False
    except Exception:
        pass
    return default, True


def build_models(X_train, y_dir_train, X_cal, y_dir_cal, y_ret_train, y_ret_cal,
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
    dir_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    dir_model.fit(X_train, y_dir_train, eval_set=[(X_cal, y_dir_cal)], verbose=False)
    ret_model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=N_RETURN_BINS, eval_metric="mlogloss", **common
    )
    ret_model.fit(X_train, y_ret_train, eval_set=[(X_cal, y_ret_cal)], verbose=False)
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
        X_train = scaler.fit_transform(X_all[:train_end])
        X_test = scaler.transform(X_all[test_start:test_end])
        y_train = y_all[:train_end]
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

    for col in feature_cols_raw:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_all, xgb_feature_cols = build_xgb_matrix(df, feature_cols_raw)
    y_dir_all = make_direction_target(df["Close"])
    y_ret_all = make_return_target(df["Close"])
    future_returns = make_future_return(df["Close"])

    splits = compute_split_indices(len(df))
    train_end = splits["train_end"]
    calib_start = splits["calib_start"]
    calib_end = splits["calib_end"]
    test_start = splits["test_start"]
    if train_end < 100 or calib_end <= calib_start or test_start >= len(df):
        log.error("Invalid split for %s", ticker)
        return False

    # Scaler fit only on training rows — never sees calibration or test data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[:train_end])
    X_cal = scaler.transform(X_all[calib_start:calib_end])
    X_test = scaler.transform(X_all[test_start:])

    dir_model, ret_model = build_models(
        X_train,
        y_dir_all[:train_end],
        X_cal,
        y_dir_all[calib_start:calib_end],
        y_ret_all[:train_end],
        y_ret_all[calib_start:calib_end],
        param_overrides=param_overrides,
    )

    # Calibrator fit only on calibration split.
    cal_probs = dir_model.predict_proba(X_cal)[:, 1]
    calibrator = fit_direction_calibrator(cal_probs, y_dir_all[calib_start:calib_end])
    cal_path = os.path.join(MODEL_DIR, f"{ticker}_direction_calibrator.pkl")
    save_direction_calibrator(calibrator, cal_path)

    # Track whether the calibrator was usable or fell back to the fixed default.
    threshold, threshold_used_fallback = compute_dynamic_threshold(
        calibrator,
        target_precision=CONFIDENCE_TARGET_PRECISION,
        default=DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
    )
    if threshold_used_fallback:
        log.warning(
            "[%s] Calibrator fallback: isotonic calibrator was None or failed. "
            "Using fixed threshold %.1f. Calibration set size: %d",
            ticker, threshold, calib_end - calib_start,
        )
    else:
        log.info("[%s] Dynamic threshold: %.1f (calibration set size: %d)", ticker, threshold, calib_end - calib_start)

    test_probs = dir_model.predict_proba(X_test)
    test_p_up = test_probs[:, 1]
    y_pred = test_probs.argmax(axis=1)
    dir_acc = float(accuracy_score(y_dir_all[test_start:], y_pred) * 100.0)
    baseline_up_rate = float(np.mean(y_dir_all[test_start:]) * 100.0) if len(y_dir_all[test_start:]) else 50.0

    buckets = confidence_buckets(test_p_up, y_dir_all[test_start:])
    sweep = threshold_sweep(test_p_up, y_dir_all[test_start:])
    return_eval = evaluate_return_model(ret_model, X_test, future_returns[test_start:])
    wf = walk_forward_summary(X_all, y_dir_all)

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
        "model_version": "xgb_complete_v2",
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
            },
            f,
            indent=2,
        )

    log.info(
        "RESULTS %s: acc=%.2f%% baseline=%.2f%% threshold=%.1f%% fallback=%s",
        ticker, dir_acc, baseline_up_rate, threshold, threshold_used_fallback,
    )
    log.info("WF-CV %s: %s", ticker, wf)
    log.info("Return model %s: %s", ticker, return_eval)
    log.info(
        "\n" + classification_report(y_dir_all[test_start:], y_pred, target_names=["DOWN", "UP"], zero_division=0)
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost-only trainer")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument(
        "--params-from-tuning", action="store_true",
        help=(
            "Load best hyperparameters from models/tuning/{ticker}_best_xgb_params.json "
            "if available, instead of settings.py defaults. "
            "Use after running tune_xgb_best_tickers.py without --apply."
        ),
    )
    args = parser.parse_args()

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