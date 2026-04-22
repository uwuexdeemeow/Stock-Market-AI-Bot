"""
compare_modes.py

Standalone held-out mode comparison for one ticker.

Compares:
- xgboost_only
- best_neural_only
- saved_ensemble

This is the easiest/reversible option because it does not modify train.py or predict.py.
You can delete this file later and your main pipeline stays untouched.

Usage:
    python compare_modes.py --ticker AMD
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from settings import DATA_DIR, MODEL_DIR, WINDOW_SIZE
from model import build_model
from xgb_feature_engineering import build_xgb_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_return_target(close: pd.Series, return_bins: list[float], n_return_bins: int, horizon: int) -> np.ndarray:
    fwd = close.pct_change(horizon).shift(-horizon)
    buckets = np.digitize(fwd.fillna(0).values, return_bins)
    return np.clip(buckets, 0, n_return_bins - 1).astype(np.int64)


def make_windows(features: np.ndarray, dir_targets: np.ndarray, ret_targets: np.ndarray, window: int):
    X, y_d, y_r = [], [], []
    for i in range(window, len(features)):
        X.append(features[i - window: i])
        y_d.append(dir_targets[i])
        y_r.append(ret_targets[i])
    return (
        np.array(X, dtype=np.float32),
        np.array(y_d, dtype=np.int64),
        np.array(y_r, dtype=np.int64),
    )


def prepare_test_data(ticker: str, saved: dict):
    parquet_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(parquet_path).copy()

    feature_cols = saved["feature_cols"]
    feature_cols_raw = saved.get("feature_cols_raw", feature_cols)
    window = saved.get("window_size", WINDOW_SIZE)
    horizon = saved.get("return_horizon", 5)
    return_bins = saved.get("return_bins", [-0.03, -0.01, 0.01, 0.03])
    n_return_bins = saved.get("n_return_bins", 5)

    test_start = saved["test_split_start"]

    # Neural features
    features = df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).values
    dir_target = df["target"].values
    ret_target = make_return_target(df["Close"], return_bins, n_return_bins, horizon)

    feat_test = features[test_start:]
    dir_test = dir_target[test_start:]
    ret_test = ret_target[test_start:]

    neural_scaler = saved["scaler"]
    feat_test = neural_scaler.transform(feat_test)
    X_test, y_dir_test, y_ret_test = make_windows(feat_test, dir_test, ret_test, window)

    # XGB features
    xgb_scaler = saved.get("xgb_scaler")
    xgb_feature_cols = saved.get("xgb_feature_cols")
    X_flat = None
    if xgb_scaler is not None:
        xgb_all, _ = build_xgb_matrix(df, feature_cols_raw, xgb_feature_cols)
        xgb_all = xgb_scaler.transform(xgb_all[test_start:])
        if len(xgb_all) > len(y_dir_test):
            xgb_all = xgb_all[-len(y_dir_test):]
        X_flat = xgb_all

    return X_test, y_dir_test, y_ret_test, X_flat


def load_xgb_models(ticker: str):
    dir_json = os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.json")
    ret_json = os.path.join(MODEL_DIR, f"{ticker}_xgb_ret.json")
    if not (os.path.exists(dir_json) and os.path.exists(ret_json)):
        return None, None

    dir_clf = xgb.XGBClassifier()
    dir_clf.load_model(dir_json)
    ret_clf = xgb.XGBClassifier()
    ret_clf.load_model(ret_json)
    return dir_clf, ret_clf


def compute_report(y_true, dir_preds, exp_returns, dir_probs, confidence_threshold: float) -> dict:
    y_true = np.asarray(y_true)
    dir_preds = np.asarray(dir_preds)
    exp_returns = np.asarray(exp_returns)
    dir_probs = np.asarray(dir_probs)

    direction_accuracy = float((dir_preds == y_true).mean() * 100.0) if len(y_true) else 0.0
    baseline_up_rate = float(np.mean(y_true) * 100.0) if len(y_true) else 0.0

    signal_preds = (exp_returns >= 0).astype(int)
    signal_accuracy = float((signal_preds == y_true).mean() * 100.0) if len(y_true) else 0.0

    if len(dir_probs) and dir_probs.ndim == 2 and dir_probs.shape[1] >= 2:
        confidence = np.maximum(dir_probs[:, 0], dir_probs[:, 1]) * 100.0
    else:
        confidence = np.full(len(y_true), 50.0)

    tradeable_mask = confidence >= confidence_threshold
    tradeable_signal_rate = float(tradeable_mask.mean() * 100.0) if len(tradeable_mask) else 0.0

    direction_vote = np.where(dir_preds == 1, "LONG", "SHORT")
    final_signal = np.where(signal_preds == 1, "LONG", "SHORT")
    high_quality_mask = (
        (final_signal == direction_vote) &
        (np.abs(exp_returns) * 100.0 >= 1.0) &
        (confidence >= confidence_threshold)
    )
    if high_quality_mask.any():
        high_quality_precision = float((signal_preds[high_quality_mask] == y_true[high_quality_mask]).mean() * 100.0)
    else:
        high_quality_precision = 0.0

    return {
        "direction_accuracy": round(direction_accuracy, 2),
        "baseline_up_rate": round(baseline_up_rate, 2),
        "signal_accuracy": round(signal_accuracy, 2),
        "tradeable_signal_rate": round(tradeable_signal_rate, 2),
        "high_quality_signal_precision": round(high_quality_precision, 2),
        "n_test_samples": int(len(y_true)),
        "n_tradeable_samples": int(tradeable_mask.sum()),
        "n_high_quality_samples": int(high_quality_mask.sum()),
    }


def neural_probs(model, X_test: np.ndarray):
    X = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        dir_logits, ret_logits = model(X)
        dir_p = torch.softmax(dir_logits, dim=1).cpu().numpy()
        ret_p = torch.softmax(ret_logits, dim=1).cpu().numpy()
    return dir_p, ret_p


def evaluate_mode(mode_name: str, saved: dict, ticker: str, X_test, y_true, X_flat, best_neural_name: str | None):
    nrb = saved.get("n_return_bins", 5)
    centres = np.array([-0.04, -0.02, 0.00, 0.02, 0.04], dtype=np.float32)[:nrb]
    threshold = float(saved.get("confidence_threshold", 57.5))

    dir_probs = None
    ret_probs = None

    if mode_name == "xgboost_only":
        dir_clf, ret_clf = load_xgb_models(ticker)
        if dir_clf is None or X_flat is None:
            return {"mode": mode_name, "error": "xgboost models unavailable"}
        dir_probs = dir_clf.predict_proba(X_flat)
        ret_probs = ret_clf.predict_proba(X_flat)

    elif mode_name == "best_neural_only":
        if not best_neural_name:
            return {"mode": mode_name, "error": "no neural models available"}
        pt_path = os.path.join(MODEL_DIR, f"{ticker}_{best_neural_name}.pt")
        model = build_model(best_neural_name, len(saved["feature_cols"]), saved).to(DEVICE)
        model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))
        model.eval()
        dir_probs, ret_probs = neural_probs(model, X_test)

    elif mode_name == "saved_ensemble":
        weights = saved.get("ensemble_weights", {})
        dir_parts = []
        ret_parts = []

        for model_type, weight in weights.items():
            if weight <= 0:
                continue
            if model_type == "xgboost":
                dir_clf, ret_clf = load_xgb_models(ticker)
                if dir_clf is None or X_flat is None:
                    continue
                dir_parts.append(dir_clf.predict_proba(X_flat) * weight)
                ret_parts.append(ret_clf.predict_proba(X_flat) * weight)
                continue

            pt_path = os.path.join(MODEL_DIR, f"{ticker}_{model_type}.pt")
            if not os.path.exists(pt_path):
                continue
            model = build_model(model_type, len(saved["feature_cols"]), saved).to(DEVICE)
            model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))
            model.eval()
            d, r = neural_probs(model, X_test)
            dir_parts.append(d * weight)
            ret_parts.append(r * weight)

        if not dir_parts:
            return {"mode": mode_name, "error": "no ensemble components available"}
        dir_probs = np.sum(dir_parts, axis=0)
        ret_probs = np.sum(ret_parts, axis=0)

    else:
        return {"mode": mode_name, "error": "unknown mode"}

    dir_preds = dir_probs.argmax(axis=1)
    exp_returns = (ret_probs[:, :len(centres)] * centres).sum(axis=1)
    report = compute_report(y_true, dir_preds, exp_returns, dir_probs, threshold)
    report["mode"] = mode_name
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    with open(scaler_path, "rb") as f:
        saved = pickle.load(f)

    X_test, y_true, _, X_flat = prepare_test_data(ticker, saved)

    model_accuracies = saved.get("model_accuracies", {})
    neural_accs = {k: v for k, v in model_accuracies.items() if k != "xgboost"}
    best_neural_name = max(neural_accs, key=neural_accs.get) if neural_accs else None

    reports = []
    for mode in ["xgboost_only", "best_neural_only", "saved_ensemble"]:
        reports.append(evaluate_mode(mode, saved, ticker, X_test, y_true, X_flat, best_neural_name))

    out_path = os.path.join(MODEL_DIR, f"{ticker}_mode_comparison.json")
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2)

    print(json.dumps(reports, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
