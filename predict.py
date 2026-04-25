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
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from settings import (
    MODEL_DIR, SIGNAL_DIR, DEFAULT_FIXED_CONFIDENCE_THRESHOLD, RETURN_HORIZON_DAYS,
    MARKET_REGIME_FILTER_ENABLED, MARKET_REGIME_VIX_MAX, MARKET_REGIME_SPY_MA200_REQUIRED,
)
from confidence_calibration import load_direction_calibrator, calibrate_p_up
from model_quality import read_quality_report
from model_self_check import validate_ticker, is_optional_feature
from pipeline_shared import build_live_features_with_latest_news
from social_sentiment import get_live_social_signal
from trade_rules import load_trade_rule, passes_trade_rule
from xgb_feature_engineering import build_xgb_matrix

os.makedirs(SIGNAL_DIR, exist_ok=True)
RETURN_BIN_CENTRES = np.array([-0.04, -0.02, 0.00, 0.02, 0.04], dtype=float)
NON_TRADABLE_SCALER_PREFIXES = {"pooled"}


def load_xgb_models(ticker: str):
    for ext in ("pkl", "json"):
        dir_path = os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.{ext}")
        ret_path = os.path.join(MODEL_DIR, f"{ticker}_xgb_ret.{ext}")
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


def discover_prediction_tickers() -> list[str]:
    tickers = []
    for filename in os.listdir(MODEL_DIR):
        if not filename.endswith("_scaler.pkl"):
            continue
        ticker = filename.replace("_scaler.pkl", "").upper()
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
    # Minimum sample counts per quality tier — same as backtest.py MIN_BUCKET_N_*.
    # Without an n-floor the pooled bucket_report's in-sample precision (0.67)
    # falsely labels every live signal HIGH; n < threshold forces a downgrade.
    MIN_N_HIGH = 30
    MIN_N_MEDIUM = 15
    for row in bucket_report:
        try:
            lo, hi = row["bucket"].split("-")
            lo = float(lo); hi = float(hi)
        except Exception:
            continue
        if confidence >= lo and (confidence < hi or hi >= 100):
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
    if not saved.get("threshold_used_fallback", True):
        return True
    for row in saved.get("confidence_buckets", []):
        precision = row.get("precision")
        n = int(row.get("n", 0) or 0)
        if precision is not None and float(precision) >= 0.60 and n >= 20:
            return True
    return False


def predict_ticker(ticker: str) -> dict:
    if not validate_ticker(ticker, verbose=True):
        raise RuntimeError(f"Self-check failed for {ticker}")
    with open(os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"), "rb") as f:
        saved = pickle.load(f)

    feature_cols = saved["feature_cols"]
    xgb_scaler = saved.get("xgb_scaler") or saved.get("scaler")
    xgb_feature_cols = saved.get("xgb_feature_cols")
    threshold = float(saved.get("confidence_threshold", DEFAULT_FIXED_CONFIDENCE_THRESHOLD))

    df = build_live_features_with_latest_news(
        ticker,
        feature_cols,
        sentiment_zscore_stats=saved.get("sentiment_zscore_stats"),
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
    xgb_mat, _ = build_xgb_matrix(df, saved.get("feature_cols_raw", feature_cols), xgb_feature_cols)
    if xgb_scaler is None:
        raise RuntimeError(f"No scaler found in saved metadata for {ticker}")
    X_flat = xgb_scaler.transform(xgb_mat[[-1]])

    dir_model, ret_model = load_xgb_models(ticker)
    if dir_model is None or ret_model is None:
        raise RuntimeError(f"No XGBoost models loaded for {ticker}")

    raw_dir = dir_model.predict_proba(X_flat)[0]
    raw_ret = ret_model.predict_proba(X_flat)[0]

    calibrator = load_direction_calibrator(saved.get("confidence_calibrator"))
    p_up = float(calibrate_p_up(calibrator, float(raw_dir[1]))) if calibrator is not None else float(raw_dir[1])
    p_down = 1.0 - p_up
    up_pct = p_up * 100.0
    down_pct = p_down * 100.0
    confidence = min(max(max(up_pct, down_pct), 50.0), 99.0)

    nrb = int(saved.get("n_return_bins", len(RETURN_BIN_CENTRES)))
    if len(raw_ret) < nrb:
        pad = np.zeros(nrb)
        pad[:len(raw_ret)] = raw_ret
        raw_ret = pad
    expected_return = float((raw_ret[:nrb] * RETURN_BIN_CENTRES[:nrb]).sum()) * 100.0
    signal = "LONG" if expected_return >= 0 else "SHORT"
    direction_vote = "LONG" if up_pct >= down_pct else "SHORT"
    signal_quality = assign_signal_quality(confidence, saved.get("confidence_buckets", []))
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

    if signal_quality == "LOW":
        actionable = False
        suppressed_reason = suppressed_reason or "signal_quality_low"

    if expected_return <= 0:
        actionable = False
        suppressed_reason = suppressed_reason or "expected_return_not_positive"

    if not fallback_confidence_is_usable(saved):
        actionable = False
        suppressed_reason = suppressed_reason or "fallback_confidence_not_validated"

    if recent_accuracy is not None and recent_n >= 5 and recent_accuracy < 0.50:
        actionable = False
        suppressed_reason = "recent_accuracy_below_threshold"

    if not saved.get("sentiment_features_removed", False) and sentiment_health == "DEGRADED":
        actionable = False
        suppressed_reason = "degraded_sentiment_features"

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

    # Live safety: shorts are disabled until validated separately.
    live_actionable = actionable and signal == "LONG"
    if signal == "SHORT":
        suppressed_reason = suppressed_reason or "live_short_disabled"

    price = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0
    rsi = float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else 50.0
    sentiment = float(df["news_sentiment"].iloc[-1]) if "news_sentiment" in df.columns else 0.0

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
        "sentiment_health": sentiment_health,
        "sentiment_nonzero_ratio": round(sentiment_nonzero_ratio, 2),
        "sentiment_feature_count": len(sentiment_cols),
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
        "models_used": "xgboost_only",
        "model_version": saved.get("model_version", "unknown"),
        "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "live_short_enabled": False,
    }

    try:
        social = get_live_social_signal(ticker)
        if social.get("social_available"):
            result["social_combined"] = round(float(social.get("combined", 0.0)), 3)
    except Exception:
        pass

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete XGBoost-only predictor")
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    tickers = [args.ticker.upper()] if args.ticker else discover_prediction_tickers()
    rows = []
    for ticker in tickers:
        try:
            rows.append(predict_ticker(ticker))
        except Exception as e:
            print(f"ERROR - {ticker}: {e}")
    if not rows:
        raise SystemExit("No predictions generated")
    out = pd.DataFrame(rows)
    if "signal_quality" in out.columns:
        quality_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        out["_signal_quality_rank"] = out["signal_quality"].map(quality_rank).fillna(1)
        out = out.sort_values(["actionable", "_signal_quality_rank", "confidence"], ascending=[False, True, False]).drop(columns=["_signal_quality_rank"])
    else:
        out = out.sort_values(["actionable", "confidence"], ascending=[False, False])
    out_path = os.path.join(SIGNAL_DIR, "signals.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print("Saved →", out_path)


if __name__ == "__main__":
    main()
