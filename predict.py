from __future__ import annotations

"""
predict_complete.py
===================
Complete XGBoost-only live predictor.

Fixes:
- signal quality based on empirical confidence bucket precision, not circular logic
- degraded sentiment handled explicitly
- recent-accuracy gate only activates with enough samples
- live shorts disabled by default
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

# Multi-source data provider (yfinance → yahooquery → Stooq fallback)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from data_provider import download_prices as _dp_download_prices

from settings import (
    MODEL_DIR, SIGNAL_DIR, DEFAULT_FIXED_CONFIDENCE_THRESHOLD, RETURN_HORIZON_DAYS,
    MARKET_REGIME_FILTER_ENABLED, MARKET_REGIME_VIX_MAX, MARKET_REGIME_SPY_MA200_REQUIRED,
    MULTI_HORIZON_H20_WEIGHT, MULTI_HORIZON_H5_WEIGHT,
    CONFORMAL_TRADE_GATE_ENABLED,
    SINGLE_NAME_CRASH_FILTER_ENABLED, SINGLE_NAME_CRASH_MIN_PRICE,
    SINGLE_NAME_CRASH_MAX_DRAWDOWN_20D, SINGLE_NAME_CRASH_MAX_DRAWDOWN_60D,
    SINGLE_NAME_CRASH_MAX_DIST_MA200, SINGLE_NAME_DISTRESS_FILTER_ENABLED,
    SINGLE_NAME_DISTRESS_MAX_DRAWDOWN_252D, SINGLE_NAME_DISTRESS_MAX_DIST_MA50,
    SINGLE_NAME_DISTRESS_MAX_RET_20D,
    SPY_TIMING_MODE, SPY_TIMING_BULL_THRESHOLD, SPY_TIMING_INSTRUMENTS,
    SPY_TIMING_INVEST_PCT, SPY_TIMING_ENTER_THRESHOLD, SPY_TIMING_EXIT_THRESHOLD,
    SPY_TIMING_SMOOTH_WINDOW, SPY_TIMING_MIN_HOLD_DAYS,
    PAPER_MODE_STRATEGY, SINGLE_NAME_PAPER_TRADING_ENABLED, EARNINGS_BLACKOUT_DAYS,
    ETF_ROTATION_MODE, ETF_ROTATION_PRESETS, ETF_ROTATION_DEFENSIVE_MIXES,
    ETF_ROTATION_RISK_ON_ENTER, ETF_ROTATION_RISK_ON_EXIT,
    ETF_ROTATION_NEUTRAL_THRESHOLD, ETF_ROTATION_DEFAULT_VOL_TARGET,
    ETF_ROTATION_MAX_GROSS_EXPOSURE,
    SOCIAL_SENTIMENT_SAFETY_ENABLED,
)
from confidence_calibration import load_direction_calibrator, calibrate_p_up
from model_quality import read_quality_report
from model_self_check import validate_ticker, is_optional_feature
from pipeline_shared import build_live_features_with_latest_news
from alternative_data_features import build_all_live_alternative_features
from social_sentiment import get_live_social_signal
from trade_rules import load_trade_rule, passes_trade_rule
from xgb_feature_engineering import build_xgb_matrix

os.makedirs(SIGNAL_DIR, exist_ok=True)
RETURN_BIN_CENTRES = np.array([-0.04, -0.02, 0.00, 0.02, 0.04], dtype=float)
NON_TRADABLE_SCALER_PREFIXES = {"pooled"}

# Model staleness threshold — warn if models are older than this many days.
# PLAIN ENGLISH: ML models go stale over time as market dynamics shift.
# If your models haven't been retrained in a while, predictions may be
# unreliable.  This threshold triggers a warning (not a hard block).
MODEL_MAX_AGE_DAYS = int(os.environ.get("MODEL_MAX_AGE_DAYS", "30"))
SUMMARY_COLUMNS = [
    "ticker",
    "signal",
    "actionable",
    "signal_quality",
    "confidence",
    "expected_return",
    "price",
    "sentiment",
    "live_news_headlines",
    "model_approved",
    "suppressed_reason",
]


def build_live_xgb_matrix(
    df: pd.DataFrame,
    saved: dict,
    ticker: str,
    *,
    keep_idx_key: str = "feature_pruning_keep_idx",
) -> np.ndarray:
    """Build the live XGBoost matrix in the same shape used during training."""
    feature_cols = saved["feature_cols"]
    feature_cols_raw = saved.get("feature_cols_raw", feature_cols)
    xgb_feature_cols = saved.get("xgb_feature_cols")
    keep_idx = saved.get(keep_idx_key)
    if keep_idx is None and keep_idx_key != "feature_pruning_keep_idx":
        keep_idx = saved.get("feature_pruning_keep_idx")
    scaler = saved.get("xgb_scaler") or saved.get("scaler")
    include_interactions = bool(saved.get("xgb_interaction_features_enabled", False))

    if scaler is None:
        raise RuntimeError(f"No scaler found in saved metadata for {ticker}")

    scaler_n = int(getattr(scaler, "n_features_in_", 0) or 0)
    # Pooled model path: scaler was fit on full XGB features + ticker dummies,
    # then models were trained on a pruned keep-index. h5 may have its own
    # keep-index so sentiment can be active only for the short-horizon model.
    if saved.get("pooled") and keep_idx is not None and scaler_n:
        base_mat, _ = build_xgb_matrix(
            df,
            feature_cols_raw,
            None,
            include_interactions=include_interactions,
        )
        dummy_cols = list(saved.get("ticker_dummy_cols") or [])
        ticker_classes = [str(t).upper() for t in (saved.get("ticker_classes") or [])]
        if dummy_cols:
            dummies = np.zeros((base_mat.shape[0], len(dummy_cols)), dtype=np.float64)
            try:
                pos = ticker_classes.index(ticker.upper())
                if pos < dummies.shape[1]:
                    dummies[:, pos] = 1.0
            except ValueError:
                pass
            full_mat = np.concatenate([base_mat, dummies], axis=1)
        else:
            full_mat = base_mat

        if full_mat.shape[1] != scaler_n:
            raise RuntimeError(
                f"Live pooled feature shape mismatch for {ticker}: "
                f"built {full_mat.shape[1]} columns, scaler expects {scaler_n}"
            )
        scaled = scaler.transform(full_mat)
        keep = np.asarray(keep_idx, dtype=int)
        return scaled[:, keep]

    xgb_mat, _ = build_xgb_matrix(
        df,
        feature_cols_raw,
        xgb_feature_cols,
        include_interactions=include_interactions,
    )
    return scaler.transform(xgb_mat)


def load_xgb_models(ticker: str, artifact_suffix: str = ""):
    for ext in ("pkl", "json"):
        dir_path = os.path.join(MODEL_DIR, f"{ticker}{artifact_suffix}_xgb_dir.{ext}")
        ret_path = os.path.join(MODEL_DIR, f"{ticker}{artifact_suffix}_xgb_ret.{ext}")
        if os.path.exists(dir_path) and os.path.exists(ret_path):
            if ext == "pkl":
                with open(dir_path, "rb") as f:
                    dir_model = pickle.load(f)
                with open(ret_path, "rb") as f:
                    ret_model = pickle.load(f)
            else:
                dir_model = xgb.XGBClassifier(); dir_model.load_model(dir_path)
                ret_model = xgb.XGBClassifier(); ret_model.load_model(ret_path)
            return dir_model, ret_model
    return None, None


def load_h5_model(saved: dict) -> xgb.XGBClassifier | None:
    """
    Load the secondary 5-day direction model for the multi-horizon ensemble.

    The h5 model is saved pooled-only (not per-ticker) because it was trained
    on the full cross-sectional dataset in train.py.  We first try the path
    recorded in the saved metadata (h5_model_path), then fall back to the
    default pooled artifact name.

    Returns None silently when the h5 artifact doesn't exist yet — the caller
    should degrade gracefully to h20-only prediction in that case.
    """
    # Try the path recorded in metadata first (set by train.py Step 8).
    h5_path = saved.get("h5_model_path") or os.path.join(MODEL_DIR, "pooled_xgb_dir_h5.json")
    if not os.path.exists(h5_path):
        return None
    try:
        h5 = xgb.XGBClassifier()
        h5.load_model(h5_path)
        return h5
    except Exception:
        return None


def check_model_staleness() -> dict:
    """Check if prediction models are stale (older than MODEL_MAX_AGE_DAYS).

    PLAIN ENGLISH: Scans the models/ directory, finds the newest and oldest
    model file, and warns if ANY model is older than the threshold.  Returns
    a dict with status info.  This is informational — it won't block predictions,
    but it logs a warning so you know retraining is overdue.
    """
    import time as _time
    now = _time.time()
    threshold = MODEL_MAX_AGE_DAYS * 86400
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith((".pkl", ".json"))]
    if not model_files:
        return {"status": "no_models", "count": 0}

    ages = []
    for f in model_files:
        mtime = os.path.getmtime(os.path.join(MODEL_DIR, f))
        ages.append((now - mtime) / 86400)

    newest_days = min(ages)
    oldest_days = max(ages)
    stale_count = sum(1 for a in ages if a * 86400 > threshold)

    result = {
        "status": "stale" if stale_count > 0 else "fresh",
        "count": len(model_files),
        "newest_days": round(newest_days, 1),
        "oldest_days": round(oldest_days, 1),
        "stale_count": stale_count,
        "threshold_days": MODEL_MAX_AGE_DAYS,
    }

    if stale_count > 0:
        print(f"  ⚠ MODEL STALENESS: {stale_count}/{len(model_files)} model files are "
              f">{MODEL_MAX_AGE_DAYS} days old (oldest: {oldest_days:.0f}d). "
              f"Consider retraining.")

    return result


def discover_prediction_tickers() -> list[str]:
    tickers = []
    for filename in os.listdir(MODEL_DIR):
        if not filename.endswith("_scaler.pkl"):
            continue
        stem = filename.replace("_scaler.pkl", "")
        if "_h" in stem and stem.rsplit("_h", 1)[-1].isdigit():
            continue
        ticker = stem.upper()
        if ticker.lower() in NON_TRADABLE_SCALER_PREFIXES:
            continue
        tickers.append(ticker)
    return sorted(tickers)


def evaluate_sentiment_health(df: pd.DataFrame) -> tuple[str, float, list[str]]:
    cols = [c for c in df.columns if ("sent_" in c or "sentiment" in c)]
    if not cols:
        return "UNAVAILABLE", 0.0, []

    recent = df[cols].tail(min(5, len(df))).fillna(0.0)
    if recent.empty:
        return "UNAVAILABLE", 0.0, cols

    nonzero_ratio = float((recent.abs() > 1e-9).any(axis=1).mean())
    latest_headlines = float(df["live_news_headline_count"].iloc[-1]) if "live_news_headline_count" in df.columns else 0.0
    latest_breaking = float(df["live_news_breaking"].iloc[-1]) if "live_news_breaking" in df.columns else 0.0
    latest_score_abs = float(df["live_news_score_abs"].iloc[-1]) if "live_news_score_abs" in df.columns else 0.0
    latest_has_signal = float(df["live_news_has_signal"].iloc[-1]) if "live_news_has_signal" in df.columns else 0.0
    provider_cols = {
        "sentiment_primary_available",
        "sentiment_fallback_available",
        "sentiment_fresh_headline_count",
    }
    has_provider_diagnostics = bool(provider_cols.intersection(df.columns))
    primary_available = (
        float(df["sentiment_primary_available"].iloc[-1])
        if "sentiment_primary_available" in df.columns
        else 0.0
    )
    fallback_available = (
        float(df["sentiment_fallback_available"].iloc[-1])
        if "sentiment_fallback_available" in df.columns
        else 0.0
    )
    fresh_count = (
        float(df["sentiment_fresh_headline_count"].iloc[-1])
        if "sentiment_fresh_headline_count" in df.columns
        else latest_headlines
    )

    if has_provider_diagnostics:
        if primary_available > 0.0 and fresh_count > 0.0:
            return "FULL", max(nonzero_ratio, 1.0), cols
        if fallback_available > 0.0 and fresh_count > 0.0:
            return "PARTIAL", max(nonzero_ratio, 0.2), cols
        return "DEGRADED", nonzero_ratio, cols

    # If headlines were fetched today, do not treat a near-zero composite score as a
    # degraded feed. That can happen when the news mix is balanced/neutral.
    if latest_has_signal > 0.0 or latest_headlines > 0.0 or latest_breaking > 0.0 or latest_score_abs > 1e-9:
        if nonzero_ratio < 0.20:
            return "PARTIAL", nonzero_ratio, cols
        if nonzero_ratio < 0.50:
            return "PARTIAL", nonzero_ratio, cols
        return "FULL", nonzero_ratio, cols

    if nonzero_ratio < 0.20:
        return "DEGRADED", nonzero_ratio, cols
    if nonzero_ratio < 0.50:
        return "PARTIAL", nonzero_ratio, cols
    return "FULL", nonzero_ratio, cols


# ── OOD (Out-of-Distribution) Detection ──────────────────────────────────────
# PLAIN ENGLISH: The model was trained on historical data with certain feature
# ranges.  If live features are wildly outside those ranges, the model is
# making predictions in territory it's never seen — like asking someone who
# only studied addition to solve calculus.  We flag these cases so the
# prediction can be suppressed or discounted.
#
# How it works:
#   1. The StandardScaler saved during training stores the mean and standard
#      deviation of each feature column from the training set.
#   2. At prediction time, we check each live feature value: how many
#      standard deviations away from the training mean is it?
#   3. If more than OOD_ZSCORE_THRESHOLD (default: 5) stdevs away, that
#      feature is "out of distribution."
#   4. If more than OOD_MAX_FRACTION (default: 15%) of features are OOD,
#      the whole prediction is flagged as unreliable.

OOD_ZSCORE_THRESHOLD = 5.0    # feature z-score beyond this = out of distribution
OOD_MAX_FRACTION = 0.15       # if >15% of features are OOD, flag the prediction


def detect_ood_features(
    X_live: np.ndarray,
    saved: dict,
    *,
    zscore_threshold: float = OOD_ZSCORE_THRESHOLD,
) -> dict:
    """Check if live feature values are outside the training distribution.

    PLAIN ENGLISH: Compare each live feature to what the model saw during
    training.  If a feature's value is more than 5 standard deviations from
    the training mean, it's flagged as "out of distribution" (OOD).

    Args:
        X_live: 1-row numpy array of live feature values (already transformed
                by build_live_xgb_matrix, so BEFORE scaling).
        saved: the model metadata dict (contains the scaler with training stats).
        zscore_threshold: how many stdevs away counts as OOD.

    Returns a dict with:
        ood_fraction: fraction of features that are OOD (0.0 to 1.0)
        ood_count: number of OOD features
        n_features: total features checked
        ood_features: list of (index, zscore) for the worst OOD features
        is_ood: True if ood_fraction > OOD_MAX_FRACTION
    """
    scaler = saved.get("xgb_scaler") or saved.get("scaler")
    if scaler is None or not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        # No training distribution info available — can't check
        return {
            "ood_fraction": 0.0,
            "ood_count": 0,
            "n_features": 0,
            "ood_features": [],
            "is_ood": False,
        }

    train_mean = scaler.mean_
    train_scale = scaler.scale_

    # X_live shape might be (1, N) — flatten to 1D
    x = X_live.flatten()

    # Only check features up to the scaler's width (extras might be dummies/interactions)
    n_check = min(len(x), len(train_mean))
    if n_check == 0:
        return {
            "ood_fraction": 0.0,
            "ood_count": 0,
            "n_features": 0,
            "ood_features": [],
            "is_ood": False,
        }

    # Compute z-scores relative to training distribution
    # Avoid division by zero for constant features (scale_ == 0)
    safe_scale = np.where(train_scale[:n_check] > 1e-12, train_scale[:n_check], 1.0)
    z_scores = np.abs((x[:n_check] - train_mean[:n_check]) / safe_scale)

    # Find OOD features
    ood_mask = z_scores > zscore_threshold
    ood_count = int(ood_mask.sum())
    ood_fraction = ood_count / n_check

    # Top 5 worst OOD features (for diagnostics)
    worst_indices = np.argsort(z_scores)[-5:][::-1]
    ood_features = [
        {"index": int(idx), "zscore": round(float(z_scores[idx]), 1)}
        for idx in worst_indices
        if z_scores[idx] > zscore_threshold
    ]

    return {
        "ood_fraction": round(ood_fraction, 3),
        "ood_count": ood_count,
        "n_features": n_check,
        "ood_features": ood_features,
        "is_ood": ood_fraction > OOD_MAX_FRACTION,
    }


def compute_recent_accuracy(ticker: str, lookback_days: int = 20) -> tuple[float | None, int]:
    candidate_paths = [
        os.path.join(SIGNAL_DIR, "portfolio_long_only_trades.csv"),
        os.path.join(SIGNAL_DIR, "portfolio_trades.csv"),
    ]
    trades = None
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                tmp = pd.read_csv(path)
                if {"ticker", "net_pnl"}.issubset(tmp.columns):
                    trades = tmp
                    break
            except Exception:
                pass
    if trades is None:
        return None, 0
    recent = trades[trades["ticker"].astype(str).str.upper() == ticker.upper()].tail(lookback_days)
    if len(recent) < 5:
        return None, len(recent)
    return float((recent["net_pnl"] > 0).mean()), len(recent)


def assign_signal_quality(confidence: float, bucket_report: list[dict]) -> str:
    # Minimum sample counts per quality tier, matching the backtest bucket logic.
    MIN_N_HIGH = 30
    MIN_N_MEDIUM = 15
    for i, row in enumerate(bucket_report):
        try:
            if "lo" in row and "hi" in row:
                lo = float(row["lo"])
                hi = float(row["hi"])
            else:
                lo_s, hi_s = row["bucket"].split("-")
                lo = float(lo_s)
                hi = float(hi_s)
        except Exception:
            continue
        is_last = i == len(bucket_report) - 1
        if confidence >= lo and (confidence < hi or is_last):
            precision = row.get("precision")
            n = row.get("n", 0)
            if precision is None or n < MIN_N_MEDIUM:
                return "LOW"
            if precision >= 0.65 and n >= MIN_N_HIGH:
                return "HIGH"
            if precision >= 0.55:
                return "MEDIUM"
            return "LOW"
    return "MEDIUM"


def single_name_crash_filter_reason(df: pd.DataFrame, signal: str) -> str | None:
    """Live-side mirror of the backtest single-name crash/distress veto."""
    if not SINGLE_NAME_CRASH_FILTER_ENABLED or signal != "LONG" or "Close" not in df.columns:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 60:
        return None

    px = float(close.iloc[-1])
    if px < SINGLE_NAME_CRASH_MIN_PRICE:
        return "single_name_min_price"

    dd20 = px / max(float(close.tail(20).max()), 1e-9) - 1.0
    dd60 = px / max(float(close.tail(60).max()), 1e-9) - 1.0
    if dd20 <= SINGLE_NAME_CRASH_MAX_DRAWDOWN_20D:
        return "single_name_20d_crash"
    if dd60 <= SINGLE_NAME_CRASH_MAX_DRAWDOWN_60D:
        return "single_name_60d_crash"

    if len(close) >= 200:
        ma200 = float(close.tail(200).mean())
        dist_ma200 = px / max(ma200, 1e-9) - 1.0
        if dist_ma200 <= SINGLE_NAME_CRASH_MAX_DIST_MA200:
            return "single_name_ma200_crash"

    if SINGLE_NAME_DISTRESS_FILTER_ENABLED and len(close) >= 252:
        dd252 = px / max(float(close.tail(252).max()), 1e-9) - 1.0
        ma50 = float(close.tail(50).mean())
        dist_ma50 = px / max(ma50, 1e-9) - 1.0
        ret20 = px / max(float(close.iloc[-21]), 1e-9) - 1.0
        if (
            dd252 <= SINGLE_NAME_DISTRESS_MAX_DRAWDOWN_252D
            and dist_ma50 <= SINGLE_NAME_DISTRESS_MAX_DIST_MA50
            and ret20 <= SINGLE_NAME_DISTRESS_MAX_RET_20D
        ):
            return "single_name_long_term_distress"

    return None


def conformal_prediction_info(p_up: float, qhat: float | None) -> dict:
    if qhat is None or not np.isfinite(qhat):
        return {"set_size": None, "singleton": True}
    p = float(np.clip(p_up, 0.0, 1.0))
    include_up = (1.0 - p) <= float(qhat)
    include_down = p <= float(qhat)
    set_size = int(include_up) + int(include_down)
    pred_up = p >= 0.5
    singleton = bool(set_size == 1 and ((pred_up and include_up) or ((not pred_up) and include_down)))
    return {"set_size": int(set_size), "singleton": singleton}


def live_approval_for_ticker(ticker: str, saved: dict) -> tuple[bool, str]:
    report = read_quality_report()
    if not report.empty and {"ticker", "approved_for_live"}.issubset(report.columns):
        row = report[report["ticker"].astype(str).str.upper() == ticker.upper()]
        if row.empty:
            return False, "missing_from_model_quality_report"
        approved = str(row.iloc[0].get("approved_for_live", "")).lower() in {"true", "1"}
        reason = str(row.iloc[0].get("approval_reason", ""))
        return approved, reason or ("approved" if approved else "rejected")
    return bool(saved.get("approved_for_live", False)), str(saved.get("approval_reason", "not evaluated by backtest"))


def fallback_confidence_is_usable(saved: dict) -> bool:
    diagnostics = saved.get("confidence_diagnostics")
    if isinstance(diagnostics, dict):
        return bool(diagnostics.get("confidence_validated", False))
    if not saved.get("threshold_used_fallback", True):
        return True
    bucket_rows = saved.get("confidence_buckets", [])
    usable_precisions = []
    for row in bucket_rows:
        precision = row.get("precision")
        n = int(row.get("n", 0) or 0)
        if precision is not None and float(precision) >= 0.60 and n >= 20:
            usable_precisions.append(float(precision))
    if not usable_precisions:
        return False
    # One lucky high bucket is not enough; require some monotonic relationship
    # between confidence bucket order and observed precision.
    ordered = [
        float(row["precision"])
        for row in bucket_rows
        if row.get("precision") is not None and int(row.get("n", 0) or 0) >= 15
    ]
    if len(ordered) < 3:
        return False
    corr = pd.Series(range(len(ordered)), dtype=float).corr(pd.Series(ordered), method="spearman")
    return bool(pd.notna(corr) and float(corr) >= 0.20)


def predict_ticker(ticker: str, verbose: bool = False) -> dict:
    if not validate_ticker(ticker, verbose=verbose):
        raise RuntimeError(f"Self-check failed for {ticker}")
    with open(os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"), "rb") as f:
        saved = pickle.load(f)

    feature_cols = saved["feature_cols"]
    threshold = float(saved.get("confidence_threshold", DEFAULT_FIXED_CONFIDENCE_THRESHOLD))

    df = build_live_features_with_latest_news(
        ticker,
        feature_cols,
        sentiment_zscore_stats=saved.get("sentiment_zscore_stats"),
        verbose=verbose,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Could not build live features for {ticker}")

    missing = [c for c in feature_cols if c not in df.columns]
    optional = [c for c in missing if is_optional_feature(c)]
    critical = [c for c in missing if c not in optional]
    for col in optional:
        df[col] = 0.0
    if critical:
        raise RuntimeError(f"Critical live features missing: {critical[:10]}")

    sentiment_health, sentiment_nonzero_ratio, sentiment_cols = evaluate_sentiment_health(df)
    X_flat = build_live_xgb_matrix(df, saved, ticker)[[-1]]
    X_h5_flat = (
        build_live_xgb_matrix(df, saved, ticker, keep_idx_key="h5_feature_pruning_keep_idx")[[-1]]
        if saved.get("h5_feature_pruning_keep_idx") is not None
        else X_flat
    )

    # ── OOD (out-of-distribution) check ────────────────────────────────────
    # Before predicting, check if live features look like anything the model
    # saw during training.  If too many features are wildly out of range,
    # the prediction is unreliable and should be suppressed.
    ood_info = detect_ood_features(X_flat, saved)

    dir_model, ret_model = load_xgb_models(ticker)
    if dir_model is None or ret_model is None:
        raise RuntimeError(f"No XGBoost models loaded for {ticker}")

    raw_dir = dir_model.predict_proba(X_flat)[0]
    raw_ret = ret_model.predict_proba(X_flat)[0]

    # ── Multi-horizon ensemble blend ─────────────────────────────────────────
    # Try to load the 5-day secondary model trained alongside the primary h20
    # model.  Blending them gives p_up_raw — used for direction_vote and
    # confidence.  Calibration is applied to the h20 component only.
    p_up_h20 = float(raw_dir[1])
    h5_model  = load_h5_model(saved)
    if h5_model is not None:
        p_up_h5  = float(h5_model.predict_proba(X_h5_flat)[0, 1])
        p_up_raw = MULTI_HORIZON_H20_WEIGHT * p_up_h20 + MULTI_HORIZON_H5_WEIGHT * p_up_h5
    else:
        # h5 artifact not yet built — degrade gracefully to h20-only.
        p_up_raw = p_up_h20

    calibrator = load_direction_calibrator(saved.get("confidence_calibrator"))
    # Calibrate only h20; p_up_raw (blended) is used for direction + confidence.
    p_up = float(calibrate_p_up(calibrator, p_up_h20)) if calibrator is not None else p_up_h20
    # direction_vote and confidence come from the blended p_up_raw so both paths
    # (backtest walk-forward and live predict) are consistent.
    p_down = 1.0 - p_up_raw
    up_pct = p_up_raw * 100.0
    down_pct = p_down * 100.0
    confidence = min(max(max(up_pct, down_pct), 50.0), 99.0)

    nrb = int(saved.get("n_return_bins", len(RETURN_BIN_CENTRES)))
    if len(raw_ret) < nrb:
        pad = np.zeros(nrb)
        pad[:len(raw_ret)] = raw_ret
        raw_ret = pad
    expected_return = float((raw_ret[:nrb] * RETURN_BIN_CENTRES[:nrb]).sum()) * 100.0
    # direction_vote comes from the blended ensemble probability, not calibrated
    # h20 alone (calibrated h20 could collapse toward SHORT in imbalanced label runs).
    direction_vote = "LONG" if p_up_raw >= 0.5 else "SHORT"
    signal = direction_vote
    signal_quality = assign_signal_quality(confidence, saved.get("confidence_buckets", []))
    conformal_qhat = saved.get("conformal_qhat")
    conformal_info = conformal_prediction_info(
        p_up_raw,
        float(conformal_qhat) if conformal_qhat is not None else None,
    )
    model_approved, approval_reason = live_approval_for_ticker(ticker, saved)
    trade_rule = load_trade_rule(ticker)

    recent_accuracy, recent_n = compute_recent_accuracy(ticker)
    effective_threshold = max(threshold, float(trade_rule.confidence_threshold))
    actionable = confidence >= effective_threshold
    suppressed_reason = None

    if not model_approved:
        actionable = False
        suppressed_reason = f"model_not_approved: {approval_reason}"

    passed_rule, rule_reason = passes_trade_rule(
        {
            "signal": signal,
            "signal_quality": signal_quality,
            "confidence": confidence,
            "expected_return": expected_return,
        },
        trade_rule,
        mode="long_only",
    )
    if not passed_rule:
        actionable = False
        suppressed_reason = suppressed_reason or rule_reason

    if expected_return <= 0:
        actionable = False
        suppressed_reason = suppressed_reason or "expected_return_not_positive"

    if CONFORMAL_TRADE_GATE_ENABLED and not bool(conformal_info["singleton"]):
        actionable = False
        suppressed_reason = suppressed_reason or "conformal_not_singleton"

    if not fallback_confidence_is_usable(saved):
        actionable = False
        suppressed_reason = suppressed_reason or "fallback_confidence_not_validated"

    if recent_accuracy is not None and recent_n >= 5 and recent_accuracy < 0.50:
        actionable = False
        suppressed_reason = "recent_accuracy_below_threshold"

    if not saved.get("sentiment_features_removed", False) and sentiment_health == "DEGRADED":
        actionable = False
        suppressed_reason = "degraded_sentiment_features"

    # ── OOD gate: suppress prediction if features are out of training distribution
    if ood_info["is_ood"]:
        actionable = False
        suppressed_reason = (
            f"ood_features: {ood_info['ood_count']}/{ood_info['n_features']} "
            f"({ood_info['ood_fraction']:.0%}) features >5σ from training"
        )

    # Block trades when the broad market regime is unfavorable (high VIX or SPY below 200d MA).
    # This avoids taking new positions into a stressed or downtrending market.
    if MARKET_REGIME_FILTER_ENABLED and actionable:
        try:
            vix_level = float(df["vix_level"].iloc[-1]) if "vix_level" in df.columns else 0.0
            spy_ma200_dist = float(df["spy_dist_ma200"].iloc[-1]) if "spy_dist_ma200" in df.columns else 1.0
            if vix_level >= MARKET_REGIME_VIX_MAX:
                actionable = False
                suppressed_reason = f"regime_vix_too_high: {vix_level:.1f}>={MARKET_REGIME_VIX_MAX}"
            elif MARKET_REGIME_SPY_MA200_REQUIRED and spy_ma200_dist <= 0.0:
                actionable = False
                suppressed_reason = f"regime_spy_below_ma200: dist={spy_ma200_dist:.4f}"
        except Exception:
            pass  # If regime data is missing, don't block the trade

    crash_filter_reason = single_name_crash_filter_reason(df, signal)
    if actionable and crash_filter_reason is not None:
        actionable = False
        suppressed_reason = crash_filter_reason

    # ── Earnings blackout gate ────────────────────────────────────────────
    # PLAIN ENGLISH: If earnings are within EARNINGS_BLACKOUT_DAYS, suppress
    # the prediction.  Earnings are binary events with 5-15% gap risk that
    # the model cannot price.  Better to skip the trade entirely than gamble
    # on an announcement outcome.  This gate applies to per-ticker predictions
    # used by both SPY timing (direction votes) and single-name entries.
    if actionable and EARNINGS_BLACKOUT_DAYS > 0:
        days_to_earnings = None
        if "days_to_next_earnings" in df.columns:
            try:
                days_to_earnings = float(df["days_to_next_earnings"].iloc[-1])
            except (ValueError, TypeError, IndexError):
                pass
        if days_to_earnings is not None and 0 <= days_to_earnings <= EARNINGS_BLACKOUT_DAYS:
            actionable = False
            suppressed_reason = (
                f"earnings_blackout: {int(days_to_earnings)}d to earnings "
                f"(blackout={EARNINGS_BLACKOUT_DAYS}d)"
            )

    # Live safety: shorts are disabled until validated separately.
    live_actionable = actionable and signal == "LONG"
    if signal == "SHORT":
        suppressed_reason = suppressed_reason or "live_short_disabled"

    price = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0
    rsi = float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else 50.0
    sentiment = float(df["news_sentiment"].iloc[-1]) if "news_sentiment" in df.columns else 0.0
    sentiment_provider_status = (
        str(df["sentiment_provider_status_json"].iloc[-1])
        if "sentiment_provider_status_json" in df.columns
        else "[]"
    )

    result = {
        "ticker": ticker,
        "signal": signal,
        "direction_vote": direction_vote,
        "signal_quality": signal_quality,
        "actionable": live_actionable,
        "confidence": round(confidence, 1),
        "conf_threshold": round(effective_threshold, 1),
        "model_conf_threshold": round(threshold, 1),
        "rule_conf_threshold": round(float(trade_rule.confidence_threshold), 1),
        "up_pct": round(up_pct, 1),
        "down_pct": round(down_pct, 1),
        "expected_return": round(expected_return, 2),
        "horizon_days": RETURN_HORIZON_DAYS,
        "price": round(price, 2),
        "rsi": round(rsi, 1),
        "sentiment": round(sentiment, 3),
        "live_news_headlines": int(df["live_news_headline_count"].iloc[-1]) if "live_news_headline_count" in df.columns else 0,
        "live_news_breaking": int(df["live_news_breaking"].iloc[-1]) if "live_news_breaking" in df.columns else 0,
        "live_news_score_abs": round(float(df["live_news_score_abs"].iloc[-1]), 4) if "live_news_score_abs" in df.columns else 0.0,
        "live_news_has_signal": bool(float(df["live_news_has_signal"].iloc[-1])) if "live_news_has_signal" in df.columns else False,
        "sentiment_provider_status_json": sentiment_provider_status,
        "sentiment_primary_available": bool(float(df["sentiment_primary_available"].iloc[-1])) if "sentiment_primary_available" in df.columns else False,
        "sentiment_fallback_available": bool(float(df["sentiment_fallback_available"].iloc[-1])) if "sentiment_fallback_available" in df.columns else False,
        "sentiment_fresh_headline_count": int(float(df["sentiment_fresh_headline_count"].iloc[-1])) if "sentiment_fresh_headline_count" in df.columns else 0,
        "sentiment_health": sentiment_health,
        "sentiment_nonzero_ratio": round(sentiment_nonzero_ratio, 2),
        "sentiment_feature_count": len(sentiment_cols),
        "single_name_crash_filter_reason": crash_filter_reason or "",
        "conformal_qhat": round(float(conformal_qhat), 4) if conformal_qhat is not None else None,
        "conformal_set_size": conformal_info["set_size"],
        "conformal_singleton": conformal_info["singleton"],
        "conformal_gate_enabled": bool(CONFORMAL_TRADE_GATE_ENABLED),
        "recent_accuracy": round(recent_accuracy, 3) if recent_accuracy is not None else None,
        "recent_accuracy_n": recent_n,
        "model_approved": model_approved,
        "approval_reason": approval_reason,
        "trade_rule_min_expected_return": round(float(trade_rule.min_expected_return), 2),
        "trade_rule_qualities": "|".join(trade_rule.allowed_qualities),
        "trade_rule_exit_horizon_days": int(trade_rule.exit_horizon_days),
        "trade_rule_stop_loss_pct": round(float(trade_rule.stop_loss_pct), 4),
        "trade_rule_take_profit_pct": round(float(trade_rule.take_profit_pct), 4),
        "trade_rule_max_position_pct": round(float(trade_rule.max_position_pct), 4),
        "suppressed_reason": suppressed_reason,
        "selected_mode": "xgboost_only",
        "selected_model_name": None,
        "feature_health": sentiment_health,
        "ood_fraction": ood_info["ood_fraction"],
        "ood_count": ood_info["ood_count"],
        "ood_is_flagged": ood_info["is_ood"],
        "models_used": "xgboost_only",
        "model_version": saved.get("model_version", "unknown"),
        "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "live_short_enabled": False,
    }

    if SOCIAL_SENTIMENT_SAFETY_ENABLED:
        try:
            social = get_live_social_signal(ticker)
            if social.get("social_available"):
                result["social_combined"] = round(float(social.get("combined", 0.0)), 3)
        except Exception:
            pass

    # ── Alternative data: EPS revisions, analyst recs, short interest ────────
    # Fetches live snapshot data from yfinance and finnhub. These features
    # enrich the prediction output so the user can see fundamental context
    # alongside the model's technical signal.
    try:
        alt_feats = build_all_live_alternative_features(ticker)
        if alt_feats:
            for k, v in alt_feats.items():
                result[k] = round(float(v), 4) if isinstance(v, (int, float)) else v
    except Exception:
        pass  # alt data is supplementary — never block prediction on failure

    return result


def _shorten(value, max_len: int = 54) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_prediction_summary(out: pd.DataFrame, out_path: str) -> None:
    cols = [c for c in SUMMARY_COLUMNS if c in out.columns]
    summary = out[cols].copy()
    rename = {
        "signal_quality": "quality",
        "expected_return": "exp_ret_pct",
        "live_news_headlines": "news",
        "model_approved": "approved",
        "suppressed_reason": "blocked_by",
    }
    summary = summary.rename(columns=rename)
    for col in ("blocked_by",):
        if col in summary.columns:
            summary[col] = summary[col].map(_shorten)

    print("\nLive Signals")
    print(summary.to_string(index=False))

    actionable_n = int(out["actionable"].sum()) if "actionable" in out.columns else 0
    total_n = len(out)
    approved_n = int(out["model_approved"].sum()) if "model_approved" in out.columns else 0
    print(f"\nActionable: {actionable_n}/{total_n} | Model-approved: {approved_n}/{total_n}")
    print(f"Saved: {out_path}")


def compute_spy_timing_signal(predictions: list[dict]) -> dict:
    """Aggregate per-ticker predictions into a single SPY/QQQ timing signal.

    PLAIN ENGLISH: The model can't pick stocks (rank_spearman ≈ 0) but CAN
    time the market (NW t-stat 3.3 vs cash).  Instead of trading individual
    stocks, we count how many tickers the model says "LONG" on.  If the
    majority are bullish → buy SPY/QQQ.  If not → stay in cash.

    This eliminates single-stock blowup risk and captures the timing edge
    with one simple trade instead of juggling 10 positions.

    Returns a dict with the timing signal and supporting diagnostics.
    """
    if not predictions:
        return {
            "signal": "STAY_CASH",
            "market_timing_signal": "STAY_CASH",
            "reason": "no_predictions",
            "bull_fraction": 0.0,
            "threshold": SPY_TIMING_BULL_THRESHOLD,
            "instruments": SPY_TIMING_INSTRUMENTS,
            "invest_pct": 0.0,
            "target_spy_weight": 0.0,
            "target_qqq_weight": 0.0,
            "target_cash_weight": 1.0,
            "single_name_signals_enabled": False,
            "single_name_paper_trading_enabled": False,
            "paper_mode_strategy": PAPER_MODE_STRATEGY,
        }

    # Count direction votes across all tickers
    n_total = len(predictions)
    n_long = sum(1 for p in predictions if p.get("direction_vote") == "LONG")
    n_actionable_long = sum(
        1 for p in predictions
        if p.get("direction_vote") == "LONG" and p.get("actionable", False)
    )
    bull_fraction = n_long / n_total if n_total > 0 else 0.0

    # Average confidence across LONG votes — higher = stronger conviction
    long_confs = [
        float(p.get("confidence", 50.0))
        for p in predictions
        if p.get("direction_vote") == "LONG"
    ]
    avg_long_conf = float(np.mean(long_confs)) if long_confs else 50.0

    # Signal decision: hysteresis thresholds to reduce whipsaw.
    # Use ENTER threshold for fresh signals, EXIT threshold for closing.
    # Live state (was_invested_yesterday) is loaded from prior signal file.
    prior_signal_path = os.path.join(SIGNAL_DIR, "spy_timing_signal.csv")
    was_invested = False
    hold_days = 0  # How many consecutive days we've held the current position
    if os.path.exists(prior_signal_path):
        try:
            prev = pd.read_csv(prior_signal_path)
            if not prev.empty and "signal" in prev.columns:
                was_invested = str(prev["signal"].iloc[-1]).strip() == "BUY_SPY"
                # Count consecutive days in current position for min-hold enforcement.
                # PLAIN ENGLISH: Look at the history of signals.  Count how many days
                # in a row the signal has been the same as today's.  This tells us
                # whether we've held long enough to be allowed to flip.
                if "hold_days" in prev.columns:
                    hold_days = int(prev["hold_days"].iloc[-1])
                else:
                    # Infer from consecutive identical signals at the tail
                    signals = prev["signal"].tolist()
                    current = signals[-1]
                    hold_days = 0
                    for s in reversed(signals):
                        if str(s).strip() == str(current).strip():
                            hold_days += 1
                        else:
                            break
        except Exception:
            pass

    # Hysteresis: if currently invested, only exit below exit_threshold.
    # If not invested, only enter above enter_threshold.
    if was_invested:
        is_bullish = bull_fraction > SPY_TIMING_EXIT_THRESHOLD
    else:
        is_bullish = bull_fraction >= SPY_TIMING_ENTER_THRESHOLD

    # ── Minimum hold period enforcement ───────────────────────────────────
    # PLAIN ENGLISH: If we just entered or exited a position less than
    # SPY_TIMING_MIN_HOLD_DAYS ago, LOCK the signal to the current state.
    # This prevents costly whipsaw where the signal flips back and forth
    # daily — each flip costs ~1.5 bps in ETF slippage, and daily flipping
    # can erode ~7.5% annually.  The min-hold forces us to commit to a
    # direction for at least N days before reconsidering.
    min_hold_locked = False
    if hold_days < SPY_TIMING_MIN_HOLD_DAYS:
        # Lock: keep the current position regardless of what bull_fraction says.
        is_bullish = was_invested
        min_hold_locked = True

    # Scale position size by conviction strength:
    #   - Just above enter threshold (e.g. 56%) → invest at 60% of max
    #   - Strong majority (e.g. 80%) → invest at 100% of max
    if is_bullish:
        range_above = max(1.0 - SPY_TIMING_ENTER_THRESHOLD, 0.01)
        strength = min((bull_fraction - SPY_TIMING_ENTER_THRESHOLD) / range_above, 1.0)
        strength = max(strength, 0.0)
        invest_scale = 0.60 + 0.40 * strength
        invest_pct = SPY_TIMING_INVEST_PCT * invest_scale
    else:
        invest_pct = 0.0

    signal = "BUY_SPY" if is_bullish else "STAY_CASH"

    # ── Update hold counter ───────────────────────────────────────────────
    # PLAIN ENGLISH: If the signal is the same as yesterday, increment the
    # hold counter.  If it changed (or this is a fresh entry), reset to 1.
    # This counter is written to the CSV so tomorrow's run can read it back.
    prior_signal = "BUY_SPY" if was_invested else "STAY_CASH"
    if signal == prior_signal:
        new_hold_days = hold_days + 1
    else:
        new_hold_days = 1

    total_weight = sum(float(w) for w in SPY_TIMING_INSTRUMENTS.values() if float(w) > 0)
    instrument_weights = {
        str(sym).upper(): float(weight) / total_weight
        for sym, weight in SPY_TIMING_INSTRUMENTS.items()
        if float(weight) > 0 and total_weight > 0
    }
    target_weights = {
        sym: round(float(invest_pct) * float(weight), 4)
        for sym, weight in instrument_weights.items()
    }
    target_cash_weight = round(max(0.0, 1.0 - sum(target_weights.values())), 4)

    # Build the reason string — include min-hold lock info if active
    reason_parts = (
        f"{n_long}/{n_total} tickers LONG ({bull_fraction:.0%}) "
        f"{'≥' if is_bullish else '<'} {SPY_TIMING_BULL_THRESHOLD:.0%} threshold"
    )
    if min_hold_locked:
        reason_parts += f" [MIN-HOLD LOCKED: day {new_hold_days}/{SPY_TIMING_MIN_HOLD_DAYS}]"

    return {
        "signal": signal,
        "market_timing_signal": signal,
        "bull_fraction": round(bull_fraction, 3),
        "threshold": SPY_TIMING_BULL_THRESHOLD,
        "n_long": n_long,
        "n_total": n_total,
        "n_actionable_long": n_actionable_long,
        "avg_long_confidence": round(avg_long_conf, 1),
        "invest_pct": round(invest_pct, 3),
        "instruments": instrument_weights,
        "target_spy_weight": target_weights.get("SPY", 0.0),
        "target_qqq_weight": target_weights.get("QQQ", 0.0),
        "target_cash_weight": target_cash_weight,
        "single_name_signals_enabled": False,
        "single_name_paper_trading_enabled": False,
        "paper_mode_strategy": PAPER_MODE_STRATEGY,
        "paper_signal_type": "index_timing",
        "hold_days": new_hold_days,
        "min_hold_locked": min_hold_locked,
        "min_hold_days_setting": SPY_TIMING_MIN_HOLD_DAYS,
        "reason": reason_parts,
        "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def print_spy_timing_summary(timing: dict, predictions: list[dict]) -> None:
    """Print a clear, actionable SPY timing signal summary."""
    signal = timing["signal"]
    bull_pct = timing["bull_fraction"] * 100
    invest_pct = timing["invest_pct"] * 100

    print("\n" + "=" * 60)
    if signal == "BUY_SPY":
        print(f"  SIGNAL:  BUY SPY/QQQ")
        print(f"  Invest:  {invest_pct:.0f}% of portfolio")
        for instr, weight in timing["instruments"].items():
            alloc = invest_pct * weight
            print(f"    → {instr}: {alloc:.0f}% of portfolio ({weight:.0%} of invested)")
    else:
        print(f"  SIGNAL:  STAY IN CASH")
        print(f"  Invest:  0% — wait for bullish conditions")
    print(f"  {'─' * 56}")
    print(f"  Bull tickers: {timing['n_long']}/{timing['n_total']} ({bull_pct:.0f}%)")
    print(f"  Threshold:    {timing['threshold']:.0%}")
    print(f"  Avg LONG conf: {timing['avg_long_confidence']:.1f}%")
    print(f"  Reason:       {timing['reason']}")
    print("=" * 60)


def _latest_etf_market_state() -> dict:
    try:
        raw = _dp_download_prices(["SPY", "QQQ"], period="320d")
    except Exception:
        raw = pd.DataFrame()

    def _close(symbol: str) -> pd.Series:
        if raw.empty:
            return pd.Series(dtype=float)
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                return raw[(symbol, "Close")].dropna()
            if "Close" in raw.columns:
                return raw["Close"].dropna()
        except Exception:
            return pd.Series(dtype=float)
        return pd.Series(dtype=float)

    spy = _close("SPY")
    qqq = _close("QQQ")
    spy_ma200 = float(spy.rolling(200, min_periods=50).mean().iloc[-1]) if len(spy) else np.nan
    qqq_ma100 = float(qqq.rolling(100, min_periods=40).mean().iloc[-1]) if len(qqq) else np.nan
    qqq_vol = float(qqq.pct_change().rolling(20, min_periods=10).std().iloc[-1] * np.sqrt(252)) if len(qqq) else 0.0
    spy_last = float(spy.iloc[-1]) if len(spy) else np.nan
    qqq_last = float(qqq.iloc[-1]) if len(qqq) else np.nan
    return {
        "spy_trend_ok": bool(np.isfinite(spy_last) and np.isfinite(spy_ma200) and spy_last >= spy_ma200),
        "qqq_trend_ok": bool(np.isfinite(qqq_last) and np.isfinite(qqq_ma100) and qqq_last >= qqq_ma100),
        "qqq_realized_vol": qqq_vol if np.isfinite(qqq_vol) else 0.0,
        "spy_last": spy_last,
        "qqq_last": qqq_last,
    }


def _etf_rotation_paper_ready() -> bool:
    path = os.path.join(SIGNAL_DIR, "etf_rotation_metrics.json")
    try:
        with open(path) as f:
            metrics = json.load(f)
        return bool(metrics.get("paper_ready", False))
    except Exception:
        return False


def compute_etf_rotation_signal(predictions: list[dict]) -> dict:
    n_total = len(predictions)
    n_long = sum(1 for p in predictions if p.get("direction_vote") == "LONG")
    bull_fraction = n_long / n_total if n_total else 0.0
    market = _latest_etf_market_state()
    high_vol = bool(market["qqq_realized_vol"] > max(0.20, ETF_ROTATION_DEFAULT_VOL_TARGET * 1.75))

    if not market["spy_trend_ok"] or bull_fraction < ETF_ROTATION_RISK_ON_EXIT:
        regime = "risk_off"
    elif bull_fraction >= ETF_ROTATION_RISK_ON_ENTER and market["qqq_trend_ok"] and not high_vol:
        regime = "risk_on"
    elif bull_fraction >= ETF_ROTATION_NEUTRAL_THRESHOLD and market["spy_trend_ok"]:
        regime = "neutral"
    else:
        regime = "risk_off"

    preset_name = "limited_leverage"
    defensive_mix_name = "bil_ief_gld"
    preset = ETF_ROTATION_PRESETS[preset_name]
    base = (
        ETF_ROTATION_DEFENSIVE_MIXES[defensive_mix_name]
        if regime == "risk_off"
        else preset.get(regime, {})
    )
    max_gross = min(float(preset.get("max_gross", 1.0)), ETF_ROTATION_MAX_GROSS_EXPOSURE)
    if regime != "risk_on":
        max_gross = min(max_gross, 1.0)
    gross = sum(abs(float(v)) for v in base.values())
    cap_scale = min(1.0, max_gross / gross) if gross > 0 else 0.0
    vol_scale = 1.0
    if regime in {"risk_on", "neutral"} and market["qqq_realized_vol"] > 0:
        vol_scale = min(1.0, ETF_ROTATION_DEFAULT_VOL_TARGET / market["qqq_realized_vol"])
    target = {
        str(sym).upper(): round(float(weight) * cap_scale * vol_scale, 4)
        for sym, weight in base.items()
    }
    cash_weight = round(max(0.0, 1.0 - sum(target.values())), 4)
    gross_exposure = round(sum(abs(v) for v in target.values()), 4)
    paper_ready = _etf_rotation_paper_ready()

    return {
        "paper_signal_type": "etf_rotation",
        "market_timing_signal": regime.upper(),
        "regime": regime,
        "preset": preset_name,
        "defensive_mix": defensive_mix_name,
        "bull_fraction": round(bull_fraction, 3),
        "n_long": n_long,
        "n_total": n_total,
        "risk_on_enter_threshold": ETF_ROTATION_RISK_ON_ENTER,
        "risk_on_exit_threshold": ETF_ROTATION_RISK_ON_EXIT,
        "neutral_threshold": ETF_ROTATION_NEUTRAL_THRESHOLD,
        "vol_target": ETF_ROTATION_DEFAULT_VOL_TARGET,
        "qqq_realized_vol": round(float(market["qqq_realized_vol"]), 4),
        "spy_trend_ok": market["spy_trend_ok"],
        "qqq_trend_ok": market["qqq_trend_ok"],
        "target_spy_weight": target.get("SPY", 0.0),
        "target_qqq_weight": target.get("QQQ", 0.0),
        "target_bil_weight": target.get("BIL", 0.0),
        "target_ief_weight": target.get("IEF", 0.0),
        "target_gld_weight": target.get("GLD", 0.0),
        "target_cash_weight": cash_weight,
        "gross_exposure": gross_exposure,
        "max_gross_exposure": ETF_ROTATION_MAX_GROSS_EXPOSURE,
        "single_name_signals_enabled": False,
        "single_name_paper_trading_enabled": False,
        "paper_ready": paper_ready,
        "paper_mode_strategy": PAPER_MODE_STRATEGY,
        "reason": "ETF rotation gates pass" if paper_ready else "ETF rotation gates have not passed",
        "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def print_etf_rotation_summary(signal: dict) -> None:
    print("\n" + "=" * 60)
    print("  ETF ROTATION ALLOCATION")
    print(f"  Regime:       {signal['regime']}")
    print(f"  Paper ready:  {'YES' if signal['paper_ready'] else 'NO'}")
    print(f"  Bull tickers: {signal['n_long']}/{signal['n_total']} ({signal['bull_fraction']:.0%})")
    print(f"  Gross exp:    {signal['gross_exposure']:.0%}")
    for symbol in ("spy", "qqq", "bil", "ief", "gld", "cash"):
        key = f"target_{symbol}_weight"
        print(f"  {symbol.upper():<4} target: {float(signal.get(key, 0.0)):.0%}")
    print(f"  Reason:       {signal['reason']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete XGBoost-only predictor")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="Show self-check and per-source news diagnostics")
    parser.add_argument("--spy-timing", action="store_true",
                        help="Output SPY/QQQ market-timing signal instead of per-stock picks")
    parser.add_argument("--etf-rotation", action="store_true",
                        help="Output benchmark-relative ETF rotation allocation instead of per-stock picks")
    args = parser.parse_args()

    # Force SPY timing mode from CLI flag or settings
    use_spy_timing = args.spy_timing or SPY_TIMING_MODE
    use_etf_rotation = args.etf_rotation or ETF_ROTATION_MODE

    # Pre-flight: check if models are stale before running predictions
    check_model_staleness()

    tickers = [args.ticker.upper()] if args.ticker else discover_prediction_tickers()
    rows = []
    for ticker in tickers:
        try:
            rows.append(predict_ticker(ticker, verbose=args.verbose))
        except Exception as e:
            print(f"ERROR - {ticker}: {e}")
    if not rows:
        raise SystemExit("No predictions generated")
    out = pd.DataFrame(rows)

    # ── Sentiment feed health check — alert if degraded across many tickers ──
    # PLAIN ENGLISH: If most tickers got "DEGRADED" sentiment health, it means
    # the sentiment API (Finnhub/news feeds) is probably down.  The model will
    # still make predictions but they'll be based only on technical features,
    # which is less reliable.  Send one alert so you know to investigate.
    if "sentiment_health" in out.columns:
        _degraded_count = (out["sentiment_health"] == "DEGRADED").sum()
        _total = len(out)
        if _degraded_count > 0 and _degraded_count / _total > 0.5:
            _msg = (f"Sentiment feeds degraded for {_degraded_count}/{_total} tickers. "
                    f"Predictions running on technical features only.")
            print(f"\n  ⚠ {_msg}")
            try:
                from notifications import send_alert
                send_alert(_msg, title="Sentiment Degraded", priority="warning")
            except Exception:
                pass

    # ── SPY timing mode: aggregate predictions into one signal ───────────────
    if use_etf_rotation:
        rotation = compute_etf_rotation_signal(rows)
        print_etf_rotation_summary(rotation)
        rotation_path = os.path.join(SIGNAL_DIR, "etf_rotation_signal.csv")
        pd.DataFrame([rotation]).to_csv(rotation_path, index=False)
        print(f"ETF rotation signal saved: {rotation_path}")
        detail_path = os.path.join(SIGNAL_DIR, "etf_rotation_ticker_detail.csv")
        out.to_csv(detail_path, index=False)
    elif use_spy_timing:
        timing = compute_spy_timing_signal(rows)
        print_spy_timing_summary(timing, rows)
        # Save timing signal as a single-row CSV
        timing_path = os.path.join(SIGNAL_DIR, "spy_timing_signal.csv")
        pd.DataFrame([timing]).to_csv(timing_path, index=False)
        print(f"Timing signal saved: {timing_path}")
        # Still save the per-ticker detail for diagnostics
        detail_path = os.path.join(SIGNAL_DIR, "spy_timing_ticker_detail.csv")
        out.to_csv(detail_path, index=False)

    # ── Standard per-ticker output ───────────────────────────────────────────
    if "signal_quality" in out.columns:
        quality_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        out["_signal_quality_rank"] = out["signal_quality"].map(quality_rank).fillna(1)
        out = out.sort_values(["actionable", "_signal_quality_rank", "confidence"], ascending=[False, True, False]).drop(columns=["_signal_quality_rank"])
    else:
        out = out.sort_values(["actionable", "confidence"], ascending=[False, False])
    out_path = os.path.join(SIGNAL_DIR, "signals.csv")
    out.to_csv(out_path, index=False)
    if not use_spy_timing and not use_etf_rotation:
        print_prediction_summary(out, out_path)
        # ── Alternative data context for actionable tickers ──────────────
        # Show EPS revisions, analyst recs, short interest for stocks the
        # model flagged as actionable.  Gives the human trader fundamental
        # context alongside the purely-technical XGBoost signal.
        _alt_cols = [c for c in out.columns if c.startswith("alt_")]
        if _alt_cols:
            _show = out[out["actionable"] == True] if "actionable" in out.columns else out  # noqa: E712
            if not _show.empty:
                _key_alt = [c for c in [
                    "ticker", "alt_eps_revision_7d", "alt_eps_revision_30d",
                    "alt_eps_surprise_pct", "alt_rec_mean", "alt_rec_buy_pct",
                    "alt_rec_change_1m", "alt_short_ratio", "alt_short_pct_float",
                    "alt_short_change_mom", "alt_institutional_pct", "alt_target_upside",
                ] if c in _show.columns]
                if len(_key_alt) > 1:
                    print("\n── Alternative Data Context (actionable tickers) ──")
                    print(_show[_key_alt].to_string(index=False))


if __name__ == "__main__":
    main()
