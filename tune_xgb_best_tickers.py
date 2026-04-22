from __future__ import annotations

"""
tune_xgb_best_tickers.py
=========================
Hyperparameter tuner for XGBoost direction model.

Fixes vs old version:
- Four-way temporal split: train / calib (early stopping) / val (param
  selection) / test (one final eval, never touched during search).
  The old version selected params on the test set across 270 trials —
  that is hyperparameter overfitting and makes the accuracy numbers
  meaningless.
- Sentiment features stripped with the same filter as train.py so the
  feature set seen during tuning matches the feature set used at training
  and inference time.
- Best params are written to models/tuning/{ticker}_best_xgb_params.json
  AND optionally applied immediately by calling train_ticker() with the
  found params via --apply. Without --apply the tuner is read-only and
  train.py is unchanged.
- Grid is run on the validation split only. Test split is evaluated once
  at the end purely for reporting — it plays no role in selection.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

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
    XGB_GAMMA,
    XGB_REG_ALPHA,
    XGB_REG_LAMBDA,
    XGB_EARLY_STOP_ROUNDS,
)
from xgb_feature_engineering import build_xgb_matrix

# Import train_ticker so --apply can retrain immediately with found params.
# This import is deferred inside apply_params() to avoid circular issues if
# train.py is ever refactored into a module.

N_RETURN_BINS = len(RETURN_BIN_LABELS)

# ── Tuning grid ───────────────────────────────────────────────────────────────
# Keep this focused. A 3×3×2×2×3 = 108-combination grid is already expensive
# when run per-ticker. Widen it only after you have a baseline.
GRID = {
    "max_depth":        [4, 6, 8],
    "learning_rate":    [0.03, 0.05, 0.08],
    "subsample":        [0.70, 0.85],
    "colsample_bytree": [0.60, 0.80],
    "min_child_weight": [1, 3, 5],
}

# Four-way split fractions.  Must sum to <= 1.0.
# train | calib | val | test
# 0.60  | 0.10  | 0.15| 0.15  (approximate — embargo gaps consume a few rows)
TUNE_TRAIN_FRAC  = 0.60
TUNE_CALIB_FRAC  = 0.10
TUNE_VAL_FRAC    = 0.15
# test is whatever remains after the three above + embargo gaps


def make_direction_target(close: pd.Series) -> np.ndarray:
    fwd = close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS)
    return (fwd.fillna(0.0).values > 0).astype(np.int64)


def compute_tune_splits(n_total: int) -> dict:
    """
    Four-way temporal split for tuning.

    train  → fit model weights
    calib  → XGBoost early stopping only
    val    → hyperparameter selection (the only split the grid sees)
    test   → one-shot final eval after best params are chosen

    Each boundary has an EMBARGO_DAYS gap so the forward-looking target
    computed at the boundary row cannot leak into the next split.
    """
    t0 = int(n_total * TUNE_TRAIN_FRAC)
    t1 = int(n_total * (TUNE_TRAIN_FRAC + TUNE_CALIB_FRAC))
    t2 = int(n_total * (TUNE_TRAIN_FRAC + TUNE_CALIB_FRAC + TUNE_VAL_FRAC))

    train_end   = max(0,  t0 - EMBARGO_DAYS)
    calib_start = t0
    calib_end   = max(calib_start + 20, t1 - EMBARGO_DAYS)
    val_start   = t1
    val_end     = max(val_start + 20, t2 - EMBARGO_DAYS)
    test_start  = t2

    return {
        "train_end":   train_end,
        "calib_start": calib_start,
        "calib_end":   calib_end,
        "val_start":   val_start,
        "val_end":     val_end,
        "test_start":  test_start,
    }


def strip_sentiment(cols: list[str]) -> list[str]:
    """Same filter used in train.py and backtest.py — must stay in sync."""
    return [c for c in cols if not ("sent_" in c or "sentiment" in c)]


def load_ticker_data(ticker: str):
    parquet_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(parquet_path):
        raise RuntimeError(f"Missing parquet for {ticker}")

    df = pd.read_parquet(parquet_path).copy()
    if len(df) < 400:
        raise RuntimeError(f"Not enough rows for {ticker}: {len(df)} (need ≥ 400)")

    exclude = {"target", "Open", "High", "Low", "Close", "Volume",
               "ma5", "ma10", "ma20", "ma50", "ma200"}
    feature_cols_raw = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    feature_cols_raw = strip_sentiment(feature_cols_raw)

    for col in feature_cols_raw:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_all, _ = build_xgb_matrix(df, feature_cols_raw)
    y_all    = make_direction_target(df["Close"])
    return df, X_all, y_all, feature_cols_raw


def fit_candidate(
    X_train, y_train,
    X_calib, y_calib,
    params: dict,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        gamma=XGB_GAMMA,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        tree_method="hist",
        early_stopping_rounds=XGB_EARLY_STOP_ROUNDS,
        random_state=42,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_calib, y_calib)], verbose=False)
    return model


def tune_ticker(ticker: str, verbose: bool = True) -> dict:
    df, X_all, y_all, feature_cols_raw = load_ticker_data(ticker)
    splits = compute_tune_splits(len(df))

    train_end   = splits["train_end"]
    calib_start = splits["calib_start"]
    calib_end   = splits["calib_end"]
    val_start   = splits["val_start"]
    val_end     = splits["val_end"]
    test_start  = splits["test_start"]

    if train_end < 100 or calib_end <= calib_start or val_end <= val_start or test_start >= len(df):
        raise RuntimeError(
            f"Split sizes too small for {ticker} "
            f"(train={train_end}, calib={calib_end - calib_start}, "
            f"val={val_end - val_start}, test={len(df) - test_start})"
        )

    if verbose:
        print(
            f"[{ticker}] split sizes — "
            f"train: {train_end}, "
            f"calib: {calib_end - calib_start}, "
            f"val: {val_end - val_start}, "
            f"test: {len(df) - test_start}"
        )

    # Scaler fit on training rows only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[:train_end])
    X_calib = scaler.transform(X_all[calib_start:calib_end])
    X_val   = scaler.transform(X_all[val_start:val_end])
    X_test  = scaler.transform(X_all[test_start:])

    y_train = y_all[:train_end]
    y_calib = y_all[calib_start:calib_end]
    y_val   = y_all[val_start:val_end]
    y_test  = y_all[test_start:]

    baseline_val  = float(np.mean(y_val)  * 100.0) if len(y_val)  else 50.0
    baseline_test = float(np.mean(y_test) * 100.0) if len(y_test) else 50.0

    # ── Grid search on VALIDATION split only ─────────────────────────────────
    combos = list(itertools.product(
        GRID["max_depth"],
        GRID["learning_rate"],
        GRID["subsample"],
        GRID["colsample_bytree"],
        GRID["min_child_weight"],
    ))
    if verbose:
        print(f"[{ticker}] searching {len(combos)} combinations on val split …")

    best_val_acc = -1.0
    best_params  = None
    results      = []

    for max_depth, learning_rate, subsample, colsample_bytree, min_child_weight in combos:
        params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
        )
        model    = fit_candidate(X_train, y_train, X_calib, y_calib, params)
        val_pred = model.predict(X_val)
        val_acc  = float((val_pred == y_val).mean() * 100.0)

        results.append({
            "ticker":           ticker,
            "val_accuracy":     round(val_acc, 4),
            "baseline_val":     round(baseline_val, 4),
            **params,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params  = params

    # ── One-shot test evaluation with the winning params ─────────────────────
    # Test split plays NO role in selection — this is purely for reporting.
    best_model    = fit_candidate(X_train, y_train, X_calib, y_calib, best_params)
    test_pred     = best_model.predict(X_test)
    test_acc      = float((test_pred == y_test).mean() * 100.0)

    output = {
        "ticker":            ticker,
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy":     round(test_acc, 4),
        "baseline_val":      round(baseline_val, 4),
        "baseline_test":     round(baseline_test, 4),
        "val_beats_baseline":  best_val_acc > baseline_val,
        "test_beats_baseline": test_acc > baseline_test,
        "feature_cols_raw":  feature_cols_raw,
        "params":            best_params,
        # Map to train.py / settings.py param names for --apply.
        "XGB_MAX_DEPTH":         best_params["max_depth"],
        "XGB_LEARNING_RATE":     best_params["learning_rate"],
        "XGB_SUBSAMPLE":         best_params["subsample"],
        "XGB_COLSAMPLE_BYTREE":  best_params["colsample_bytree"],
        "XGB_MIN_CHILD_WEIGHT":  best_params["min_child_weight"],
    }

    # ── Persist results ───────────────────────────────────────────────────────
    out_dir = os.path.join(MODEL_DIR, "tuning")
    os.makedirs(out_dir, exist_ok=True)

    results_df = pd.DataFrame(results).sort_values("val_accuracy", ascending=False)
    results_df.to_csv(os.path.join(out_dir, f"{ticker}_tuning_results.csv"), index=False)

    # Save as JSON (human-readable) and pickle (easy to load in train.py).
    summary_path = os.path.join(out_dir, f"{ticker}_best_xgb_params.json")
    with open(summary_path, "w") as f:
        # feature_cols_raw is a list — fine for JSON.
        json.dump(output, f, indent=2)

    if verbose:
        print(
            f"[{ticker}] best params (val acc={best_val_acc:.2f}% | "
            f"test acc={test_acc:.2f}% | baseline={baseline_test:.2f}%): "
            f"{best_params}"
        )
        if not output["test_beats_baseline"]:
            print(
                f"[{ticker}] WARNING: test accuracy {test_acc:.2f}% does not beat "
                f"baseline {baseline_test:.2f}%. Consider not applying these params."
            )

    return output


def apply_best_params(ticker: str, tuning_result: dict) -> None:
    """
    Retrain the ticker with the best-found hyperparameters by temporarily
    overriding the relevant settings and calling train_ticker() directly.

    This is the only correct way to apply tuning results — it goes through
    the full train.py pipeline (proper splits, calibration, metadata) rather
    than stitching artifacts together manually.
    """
    # Deferred import to avoid issues if train.py imports are slow.
    import train as train_module
    import settings as settings_module

    params = tuning_result["params"]

    # Temporarily patch the module-level constants that build_models() reads.
    original = {}
    patch = {
        "XGB_MAX_DEPTH":        params["max_depth"],
        "XGB_LEARNING_RATE":    params["learning_rate"],
        "XGB_SUBSAMPLE":        params["subsample"],
        "XGB_COLSAMPLE_BYTREE": params["colsample_bytree"],
        "XGB_MIN_CHILD_WEIGHT": params["min_child_weight"],
    }
    for k, v in patch.items():
        original[k] = getattr(settings_module, k)
        setattr(settings_module, k, v)
        # train_module reads from settings at call time via its own references,
        # so we patch both modules to be safe.
        if hasattr(train_module, k):
            setattr(train_module, k, v)

    try:
        print(f"\n[{ticker}] Applying tuned params and retraining …")
        success = train_module.train_ticker(ticker)
        if success:
            print(f"[{ticker}] Retrain with tuned params complete.")
        else:
            print(f"[{ticker}] Retrain failed — check train.log.")
    finally:
        # Always restore originals even if training throws.
        for k, v in original.items():
            setattr(settings_module, k, v)
            if hasattr(train_module, k):
                setattr(train_module, k, v)


def load_tickers(args) -> list[str]:
    if args.tickers:
        return [t.upper() for t in args.tickers]
    if args.from_breakdown:
        df = pd.read_csv(args.from_breakdown)
        if "ticker" not in df.columns:
            raise RuntimeError("Breakdown CSV must contain a 'ticker' column")
        col = "total_pnl" if "total_pnl" in df.columns else df.columns[1]
        df = df.sort_values(col, ascending=False)
        return [str(x).upper() for x in df["ticker"].head(args.top).tolist()]
    raise RuntimeError("Provide --tickers or --from-breakdown")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune XGBoost hyperparameters with a proper 4-way temporal split."
    )
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Tickers to tune, e.g. --tickers AAPL NVDA")
    parser.add_argument("--from-breakdown", type=str, default=None,
                        help="Path to ticker_breakdown.csv; tunes top --top tickers by PnL")
    parser.add_argument("--top", type=int, default=4,
                        help="How many tickers to pick from --from-breakdown (default 4)")
    parser.add_argument("--apply", action="store_true",
                        help="After tuning, retrain each ticker via train.py with the best params")
    parser.add_argument("--skip-if-no-improvement", action="store_true",
                        help="With --apply, skip retrain if tuned test acc <= baseline")
    args = parser.parse_args()

    tickers = load_tickers(args)
    os.makedirs(LOG_DIR, exist_ok=True)

    summary_rows = []
    for ticker in tickers:
        try:
            result = tune_ticker(ticker, verbose=True)
            summary_rows.append({
                "ticker":            result["ticker"],
                "best_val_accuracy": result["best_val_accuracy"],
                "test_accuracy":     result["test_accuracy"],
                "baseline_test":     result["baseline_test"],
                "test_beats_baseline": result["test_beats_baseline"],
                **result["params"],
            })

            if args.apply:
                if args.skip_if_no_improvement and not result["test_beats_baseline"]:
                    print(
                        f"[{ticker}] Skipping retrain — tuned params do not beat "
                        f"baseline on test split."
                    )
                else:
                    apply_best_params(ticker, result)

        except Exception as e:
            print(f"ERROR - {ticker}: {e}")

    if summary_rows:
        out_dir = os.path.join(MODEL_DIR, "tuning")
        os.makedirs(out_dir, exist_ok=True)
        summary_df = pd.DataFrame(summary_rows).sort_values("test_accuracy", ascending=False)
        summary_path = os.path.join(out_dir, "tuning_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nTuning summary saved → {summary_path}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()