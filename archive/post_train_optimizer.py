"""
post_train_optimizer.py — XGBoost-only metadata refresher
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from settings import (
    DATA_DIR,
    MODEL_DIR,
    DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
    CONFIDENCE_TARGET_PRECISION,
    RETURN_HORIZON_DAYS,
)
from confidence_calibration import fit_direction_calibrator, save_direction_calibrator
from xgb_feature_engineering import build_xgb_matrix


def make_direction_target(close: pd.Series) -> np.ndarray:
    fwd = close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS)
    return (fwd.fillna(0.0).values > 0).astype(int)


def compute_dynamic_threshold(
    calibrator,
    target_precision: float = 0.58,
    default: float = 57.5,
) -> float:
    if calibrator is None:
        return default
    try:
        raw_grid = np.linspace(0.50, 0.99, 200)
        cal_grid = calibrator.predict(raw_grid)
        conf_grid = np.maximum(cal_grid, 1.0 - cal_grid) * 100.0
        idx = np.where(cal_grid >= target_precision)[0]
        return float(np.clip(conf_grid[idx[0]], 52.0, 75.0)) if len(idx) else default
    except Exception:
        return default


def load_xgb_dir_model(ticker: str):
    pkl_path = os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.pkl")
    json_path = os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.json")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    if os.path.exists(json_path):
        model = xgb.XGBClassifier()
        model.load_model(json_path)
        return model

    return None


def optimize_ticker(ticker: str) -> None:
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"SKIP {ticker}: missing scaler file")
        return

    with open(scaler_path, "rb") as f:
        saved = pickle.load(f)

    parquet_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(parquet_path):
        print(f"SKIP {ticker}: missing parquet file")
        return

    df = pd.read_parquet(parquet_path).copy()
    feature_cols_raw = saved.get("feature_cols_raw", saved.get("feature_cols", []))
    xgb_feature_cols = saved.get("xgb_feature_cols")
    xgb_scaler = saved.get("xgb_scaler")
    dir_model = load_xgb_dir_model(ticker)

    if dir_model is None or xgb_scaler is None:
        raise RuntimeError(f"Missing XGBoost artifacts for {ticker}")

    X_all, _ = build_xgb_matrix(df, feature_cols_raw, xgb_feature_cols)
    y_all = make_direction_target(df["Close"])

    test_start = int(saved.get("test_split_start", int(len(df) * 0.85)))
    test_start = max(0, min(test_start, max(0, len(df) - 50)))

    X_test = xgb_scaler.transform(X_all[test_start:])
    y_test = y_all[test_start:]

    probs = dir_model.predict_proba(X_test)[:, 1]
    pred = (probs >= 0.5).astype(int)

    acc = float((pred == y_test).mean() * 100.0) if len(y_test) else 50.0
    baseline_up_rate = float(np.mean(y_test) * 100.0) if len(y_test) else 50.0

    calibrator = fit_direction_calibrator(probs, y_test)
    cal_path = os.path.join(MODEL_DIR, f"{ticker}_direction_calibrator.pkl")
    save_direction_calibrator(calibrator, cal_path)

    threshold = compute_dynamic_threshold(
        calibrator,
        target_precision=CONFIDENCE_TARGET_PRECISION,
        default=DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
    )

    saved.update(
        {
            "selected_mode": "xgboost_only",
            "selected_model_name": None,
            "ensemble_weights": {"xgboost": 1.0},
            "model_accuracies": {"xgboost": acc},
            "ensemble_accuracy": acc,
            "baseline_up_rate": baseline_up_rate,
            "confidence_threshold": threshold,
            "confidence_calibrator": cal_path,
            "model_version": "xgb_only_v1",
        }
    )

    with open(scaler_path, "wb") as f:
        pickle.dump(saved, f)

    print(f"[{ticker}] selected_mode = xgboost_only")
    print(f"[{ticker}] selected_model_name = None")
    print(f"[{ticker}] ensemble_weights = {{'xgboost': 1.0}}")
    print(f"[{ticker}] xgboost_accuracy = {acc:.2f}%")
    print(f"[{ticker}] baseline_up_rate = {baseline_up_rate:.2f}%")
    print(f"[{ticker}] confidence_threshold = {threshold:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=False, default=None)
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = sorted(
            f.replace("_scaler.pkl", "")
            for f in os.listdir(MODEL_DIR)
            if f.endswith("_scaler.pkl")
        )
        if not tickers:
            raise RuntimeError(f"No scaler files found in {MODEL_DIR}")

    for ticker in tickers:
        try:
            optimize_ticker(ticker)
        except Exception as e:
            print(f"ERROR - {ticker}: {e}")


if __name__ == "__main__":
    main()
