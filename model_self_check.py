"""
model_self_check.py — XGBoost-only validator
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Tuple

import pandas as pd

from settings import DATA_DIR, MODEL_DIR

OPTIONAL_PREFIXES = ("sent_",)
OPTIONAL_EXACT = {
    "news_sentiment", "sentiment_disagreement", "headline_volume",
    "sentiment_3d", "sentiment_7d", "sentiment_delta", "sentiment_accel",
    # New timing/direction-split sentiment features
    "sent_pos", "sent_neg", "sent_premarket", "sent_afterhours",
    "sent_decay_1d", "sent_decay_3d", "sent_neg_decay", "sent_pos_decay",
    "headline_vol_spike",
    "social_bullish_ratio", "social_bearish_ratio", "social_bull_minus_bear",
    "social_message_volume", "social_combined", "x_mention_count", "x_sentiment_score",
    # Legacy Reddit column names — present in models trained before the Reddit→X rename.
    # New parquets use x_mention_count / x_sentiment_score instead.
    "reddit_mention_count", "reddit_upvote_score", "reddit_score", "reddit_mention_volume",
}
CRITICAL_EXACT_HINTS = {
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "rsi_7", "rsi_14", "macd", "macd_sig", "macd_hist",
    "atr_14", "atr_norm", "bb_pos", "bb_width", "vol_ratio",
    "hvol_5d", "hvol_20d", "weekly_ret", "monthly_ret",
    "tf_alignment", "vix_level", "vix_ratio", "Close",
}


def is_optional_feature(col: str) -> bool:
    return col in OPTIONAL_EXACT or col.startswith(OPTIONAL_PREFIXES)


def split_missing_features(cols: List[str]) -> Tuple[List[str], List[str]]:
    optional = [c for c in cols if is_optional_feature(c)]
    critical = [c for c in cols if c not in optional]
    return critical, optional


def find_xgb_artifact_paths(ticker: str):
    dir_path = next((p for p in [os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.pkl"), os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.json")] if os.path.exists(p)), None)
    ret_path = next((p for p in [os.path.join(MODEL_DIR, f"{ticker}_xgb_ret.pkl"), os.path.join(MODEL_DIR, f"{ticker}_xgb_ret.json")] if os.path.exists(p)), None)
    return dir_path, ret_path


def validate_ticker(ticker: str, verbose: bool = True):
    passed = True
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    parquet_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    calibrator_path = os.path.join(MODEL_DIR, f"{ticker}_direction_calibrator.pkl")

    def log(msg: str):
        if verbose:
            print(msg)

    log(f"\n=== SELF CHECK: {ticker} ===")
    if not os.path.exists(scaler_path):
        log(f"ERROR         - scaler missing: {scaler_path}")
        return False

    try:
        with open(scaler_path, "rb") as f:
            saved = pickle.load(f)
    except Exception as e:
        log(f"ERROR         - failed to load scaler pkl: {e}")
        return False

    feature_cols = saved.get("feature_cols", [])
    xgb_feature_cols = saved.get("xgb_feature_cols", [])
    log(f"model_version - {saved.get('model_version', 'unknown')}")
    log(f"feature_count - {len(feature_cols)}")
    log(f"selected_mode - {saved.get('selected_mode', 'xgboost_only')}")

    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            parquet_cols = set(df.columns)
            missing = [c for c in feature_cols if c not in parquet_cols]
            critical_missing, optional_missing = split_missing_features(missing)
            if critical_missing:
                passed = False
                log(f"ERROR         - parquet missing {len(critical_missing)} CRITICAL training feature columns (examples: {critical_missing[:10]})")
            if optional_missing:
                log(f"WARNING       - parquet missing {len(optional_missing)} OPTIONAL training feature columns (examples: {optional_missing[:10]})")
            if not critical_missing and not optional_missing:
                log("parquet       - OK (all training feature columns present)")
            hint_missing = [c for c in CRITICAL_EXACT_HINTS if c not in parquet_cols]
            if hint_missing:
                log(f"WARNING       - parquet is also missing some common core columns (examples: {hint_missing[:10]})")
        except Exception as e:
            passed = False
            log(f"ERROR         - failed to read parquet: {e}")
    else:
        log(f"WARNING       - parquet missing: {parquet_path}")

    dir_xgb_path, ret_xgb_path = find_xgb_artifact_paths(ticker)
    if dir_xgb_path is None:
        passed = False
        log(f"ERROR         - missing XGBoost direction model: {ticker}_xgb_dir.(pkl|json)")
    else:
        log(f"xgboost       - OK {os.path.basename(dir_xgb_path)}")
    if ret_xgb_path is None:
        passed = False
        log(f"ERROR         - missing XGBoost return model: {ticker}_xgb_ret.(pkl|json)")
    else:
        log(f"xgboost       - OK {os.path.basename(ret_xgb_path)}")

    xgb_scaler = saved.get("xgb_scaler") or saved.get("scaler")
    if xgb_scaler is None:
        passed = False
        log("ERROR         - xgb_scaler/scaler missing from scaler metadata")
    else:
        log("xgb_scaler    - OK present in scaler metadata")

    if not xgb_feature_cols:
        passed = False
        log("ERROR         - xgb_feature_cols missing from scaler metadata")
    else:
        log(f"xgb_features  - OK {len(xgb_feature_cols)} feature columns")

    if os.path.exists(calibrator_path):
        log(f"calibrator    - OK {os.path.basename(calibrator_path)}")
    elif bool(saved.get("threshold_used_fallback", False)):
        log("WARNING       - calibrator missing but threshold_used_fallback=True, so fixed threshold mode is active")
    else:
        log("WARNING       - calibrator missing; live confidence will use raw probabilities")

    if not saved.get("confidence_buckets"):
        log("WARNING       - confidence_buckets missing from scaler metadata")
    else:
        log(f"conf_buckets  - OK {len(saved.get('confidence_buckets', []))} buckets")

    quality = saved.get("model_quality") or {}
    if quality:
        status = quality.get("approval_status", "unknown")
        reason = quality.get("approval_reason", "")
        log(f"live_quality  - {status.upper()} ({reason})")
    else:
        log("WARNING       - model_quality missing; run backtest.py to approve/reject for live use")

    log(f"result        - {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if validate_ticker(args.ticker.upper(), verbose=True) else 1)


if __name__ == "__main__":
    main()
