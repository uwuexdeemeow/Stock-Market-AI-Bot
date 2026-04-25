from __future__ import annotations

"""
backtest.py — strict walk-forward validator aligned with train.py (xgb_complete_v2)

Key alignment points:
- Uses the same split logic as train.py (compute_split_indices)
- Fits scalers only on past training rows
- Fits calibrator only on the past calibration slice
- Predicts only on future rows
- Uses confidence buckets from past calibration data for signal_quality
- Supports long_short / long_only / long_only_bear_cash modes
- Writes ticker breakdown + 2022 diagnostics
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from confidence_calibration import fit_direction_calibrator, calibrate_p_up
from calibration_stability import calibration_stability as check_calibration_stability
from labels import (
    make_direction_target as shared_make_direction_target,
    make_forward_return_target,
    make_return_bucket_target,
    make_spy_forward_return,
)
from train import compute_dynamic_threshold as training_compute_dynamic_threshold
from model_quality import (
    evaluate_model_quality,
    read_quality_report,
    update_scaler_metadata,
    upsert_quality_report,
)
from trade_rules import load_trade_rule, passes_trade_rule, resolve_rule_exit
from settings import (
    DATA_DIR,
    SIGNAL_DIR,
    SECTOR_MAP,
    RETURN_HORIZON_DAYS,
    TRAIN_CALIBRATION_SPLIT,
    CALIBRATION_TEST_SPLIT,
    EMBARGO_DAYS,
    CONFIDENCE_TARGET_PRECISION,
    DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
    VIX_HIGH_THRESHOLD,
    VIX_EXTREME_THRESHOLD,
    BEAR_REGIME_MULT,
    HIGH_VIX_MULT,
    EXTREME_VIX_MULT,
    SLIPPAGE_BASE_PCT,
    COMMISSION_PER_SHARE,
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
    DIRECTION_LABEL_THRESHOLD,
    PREDICTION_TARGET,
    EXCESS_RETURN_MIN_PCT,
    VOL_ADJUSTED_SHARPE_THRESHOLD,
    MODEL_DIR,
    WATCHLIST,
    BACKTEST_MARKET_REGIME_FILTER_ENABLED,
    BACKTEST_REGIME_SIZE_MULTIPLIER_ENABLED,
    MARKET_REGIME_VIX_MAX,
    MARKET_REGIME_SPY_MA200_REQUIRED,
    BORROW_COST_ANNUAL_DEFAULT,
    BORROW_COSTS,
    POSITION_SIZING_MODE,
    POOLED_TRAINING,
    CROSS_SECTIONAL_TOP_N,
    BACKTEST_ALLOWED_SIGNAL_QUALITIES,
    FEATURE_IMPORTANCE_TOP_K,
)
from xgb_feature_engineering import build_xgb_matrix
from pipeline_shared import apply_sentiment_distribution_matching, fit_sentiment_zscore_stats
from portfolio_manager import PortfolioRiskManager, ProposedTrade
from execution_model import realistic_fill_price, commission as calc_commission, capacity_warning
from risk_sizing import compute_position_size
from experiment_ledger import append_experiment

INITIAL_CAPITAL = 10_000.0
BASE_POSITION_SIZE_PCT = 0.15
MAX_POSITION_SIZE_PCT = 0.30
# Boost disabled: diagnostics showed HIGH win rate (54%) < LOW win rate (56%).
# Boosting the worst-performing bucket amplified losses. Reset to 1.0 until
# signal quality is reliably calibrated out-of-sample.
HIGH_SIGNAL_BOOST = 1.0
# Separate n floors for each quality tier.
MIN_BUCKET_N_HIGH = 30
MIN_BUCKET_N_MEDIUM = 15
RETURN_BIN_CENTRES = np.array([-0.04, -0.02, 0.00, 0.02, 0.04], dtype=float)
N_RETURN_BINS = len(RETURN_BIN_CENTRES)

# Kill criteria: backtest must clear ALL of these gates to be considered valid.
# Sharpe ≥ 0.5: realistic bar for a selective/sparse strategy (in-market ~30% of time).
#   A fully-invested strategy targets ≥ 1.0; a selective one with 66%+ win rate
#   naturally has lower Sharpe because of cash drag on quiet days.
# NW t-stat ≥ 1.65: one-tailed 95% significance that daily returns > 0 (vs cash).
#   Measured vs 0 (not vs SPY) — SPY-relative t-stat is logged separately but not gated,
#   because a cash-holding strategy cannot be expected to beat a fully-invested index daily.
KILL_CRITERIA = {
    "min_sharpe": 0.5,       # annualised daily Sharpe must be ≥ 0.5
    "min_nw_tstat": 1.65,    # Newey-West t-stat of daily return vs 0 must be ≥ 1.65 (95% CI)
    "max_drawdown": -0.25,   # max drawdown must be better (less negative) than -25%
}


def _newey_west_tstat(excess: pd.Series, lag: int = 5) -> float:
    """
    Computes a Newey-West corrected t-statistic for the mean of `excess` returns.

    A plain t-stat assumes each day's return is independent, but stock returns
    have autocorrelation (today's big move can influence tomorrow's).  Newey-West
    adjusts the variance estimate to account for that, giving a more honest test.

    Returns 0.0 when there is not enough data (< 30 observations).
    """
    n = len(excess)
    if n < 30:
        return 0.0
    centered = excess.to_numpy(dtype=float) - float(excess.mean())
    # gamma0 is the basic variance of the centered series
    gamma0 = float(np.mean(centered * centered))
    lrv = gamma0
    # add weighted autocovariance terms up to `lag` lags
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)          # Bartlett (triangular) weight
        gk = float(np.mean(centered[k:] * centered[:-k]))
        lrv += 2 * w * gk
    se = float(np.sqrt(max(lrv, 0.0) / n))
    if se == 0:
        return 0.0
    return float(excess.mean() / se)


def _legacy_direction_target_wrapper(df: pd.DataFrame | pd.Series) -> np.ndarray:
    return shared_make_direction_target(
        df,
        prediction_target=PREDICTION_TARGET,
        horizon=RETURN_HORIZON_DAYS,
        direction_threshold=DIRECTION_LABEL_THRESHOLD,
        excess_return_min_pct=EXCESS_RETURN_MIN_PCT,
        vol_adjusted_sharpe_threshold=VOL_ADJUSTED_SHARPE_THRESHOLD,
    )

    # Must exactly mirror train.py so walk-forward accuracy measures the same label.
    if isinstance(df, pd.Series):
        close = df
        spy_fwd = pd.Series(0.0, index=close.index)
        hvol = pd.Series(0.20, index=close.index)
    else:
        close = df["Close"]
        bench_col = f"spy_ret{RETURN_HORIZON_DAYS}d"
        spy_fwd = df[bench_col].shift(-RETURN_HORIZON_DAYS) if bench_col in df.columns else pd.Series(0.0, index=close.index)
        hvol = df["hvol_20d"] if "hvol_20d" in df.columns else pd.Series(0.20, index=close.index)

    stock_fwd = close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS)

    if PREDICTION_TARGET == "triple_barrier" and not isinstance(df, pd.Series):
        labels = triple_barrier(
            df["Close"],
            high=df["High"] if "High" in df.columns else None,
            low=df["Low"] if "Low" in df.columns else None,
            max_hold=RETURN_HORIZON_DAYS,
        )
        return (labels.values > 0).astype(np.int64)

    if PREDICTION_TARGET == "excess_return":
        return ((stock_fwd - spy_fwd).values > EXCESS_RETURN_MIN_PCT).astype(np.int64)

    if PREDICTION_TARGET == "vol_adjusted":
        excess = stock_fwd - spy_fwd
        vol_scaled = (hvol * np.sqrt(RETURN_HORIZON_DAYS)).clip(0.01, 1.0)
        return ((excess / vol_scaled).values > VOL_ADJUSTED_SHARPE_THRESHOLD).astype(np.int64)

    return (stock_fwd.values > DIRECTION_LABEL_THRESHOLD).astype(np.int64)


def _legacy_return_target_wrapper(close: pd.Series) -> np.ndarray:
    return make_return_bucket_target(close, horizon=RETURN_HORIZON_DAYS)


def compute_split_indices(n_total: int) -> dict:
    split_train = int(n_total * TRAIN_CALIBRATION_SPLIT)
    split_test = int(n_total * CALIBRATION_TEST_SPLIT)
    train_end = max(0, split_train - EMBARGO_DAYS)
    calib_start = split_train
    calib_end = max(calib_start + 20, split_test - EMBARGO_DAYS)
    test_start = split_test
    return {"train_end": train_end, "calib_start": calib_start, "calib_end": calib_end, "test_start": test_start}


def build_models(
    X_train, y_dir_train, X_cal, y_dir_cal, y_ret_train, y_ret_cal,
    param_overrides: dict | None = None,
):
    """
    Fit the direction classifier and return-bucket classifier.

    param_overrides: optional dict of XGBoost hyperparameters that override
    the settings.py defaults.  Used to apply tuned params saved by train.py's
    nested CV search.  Accepted keys: max_depth, min_child_weight, reg_lambda
    (and any other valid XGBClassifier keyword).
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
    # Mirror train.py: compute scale_pos_weight from actual label distribution
    # so each walk-forward block corrects for any class imbalance automatically.
    n_pos = max(int(np.sum(y_dir_train == 1)), 1)
    n_neg = max(int(np.sum(y_dir_train == 0)), 1)
    spw   = float(np.clip(n_neg / n_pos, 0.33, 3.0))
    dir_model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=spw, **common
    )
    dir_model.fit(X_train, y_dir_train, eval_set=[(X_cal, y_dir_cal)], verbose=False)
    ret_model = xgb.XGBClassifier(objective="multi:softprob", num_class=N_RETURN_BINS, eval_metric="mlogloss", **common)
    ret_model.fit(X_train, y_ret_train, eval_set=[(X_cal, y_ret_cal)], verbose=False)
    return dir_model, ret_model


def confidence_buckets(probs: np.ndarray, y_true: np.ndarray) -> list[dict]:
    conf = np.maximum(probs, 1.0 - probs) * 100.0
    pred = (probs >= 0.5).astype(int)
    rows = []
    for lo, hi in [(50, 55), (55, 60), (60, 65), (65, 70), (70, 100)]:
        mask = (conf >= lo) & (conf < hi if hi < 100 else conf <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": f"{lo}-{hi}", "n": 0, "precision": None})
        else:
            rows.append({"bucket": f"{lo}-{hi}", "n": n, "precision": round(float((pred[mask] == y_true[mask]).mean()), 4)})
    return rows


def assign_signal_quality(confidence: float, bucket_report: list[dict],
                          direction: str = "LONG") -> str:
    """
    Map a confidence score to HIGH / MEDIUM / LOW using the calibration bucket report.

    For LONG signals  → use the bucket's UP precision directly.
    For SHORT signals → invert: the precision that matters is how often the DOWN
                        prediction was correct, which equals 1 - UP_precision.
                        A bucket with 35% UP precision is actually a strong SHORT
                        signal (65% DOWN precision).
    """
    for row in bucket_report:
        lo, hi = row["bucket"].split("-")
        lo = float(lo); hi = float(hi)
        if confidence >= lo and (confidence < hi or hi >= 100):
            up_precision = row.get("precision")
            n = row.get("n", 0)
            if up_precision is None or n < MIN_BUCKET_N_MEDIUM:
                return "LOW"
            precision = up_precision if direction == "LONG" else (1.0 - up_precision)
            if precision >= 0.65 and n >= MIN_BUCKET_N_HIGH:
                return "HIGH"
            if precision >= 0.57:
                return "MEDIUM"
            return "LOW"
    return "LOW"


def strip_sentiment(feature_cols: list[str]) -> list[str]:
    risky_keywords = (
        "sent", "social", "news", "iv_", "put_call", "option", "earn", "eps_",
        "analyst", "recommend", "short_interest", "dark_pool", "dow_", "month_",
        "opex", "calendar", "sector_ret", "ret_vs_sector", "gld_", "hyg_", "tnx_",
        "tlt_", "eem_", "iwm_", "dia_", "uup_",
    )
    allowed_prefixes = (
        "ret_", "hvol_", "rsi_", "dist_ma", "ma_cross_", "macd", "atr_", "bb_",
        "vol_ratio", "volume_chg_", "hl_range", "spread_proxy", "roc_", "drawdown_",
        "weekly_", "monthly_", "tf_alignment", "spy_", "qqq_", "vix_", "regime",
        "ret_vs_spy", "ret_vs_qqq", "breadth_", "pct_above_",
    )
    allowed_exact = {"obv_slope", "vwap_dist", "uptick_ratio", "variance_ratio"}
    return [
        c for c in feature_cols
        if c.startswith("sent_z_")
        or (
            not any(k in c.lower() for k in risky_keywords)
            and (c in allowed_exact or any(c.startswith(prefix) for prefix in allowed_prefixes))
        )
    ]


def load_ticker_data(ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(path).copy()
    stock_fwd_all = make_forward_return_target(df["Close"], horizon=RETURN_HORIZON_DAYS)
    spy_fwd_all = make_spy_forward_return(df, horizon=RETURN_HORIZON_DAYS)
    valid_labels = stock_fwd_all.notna() & spy_fwd_all.notna()
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
    y_dir = y_dir_full[valid_labels.to_numpy()]
    y_ret = make_return_bucket_target(df["Close"], horizon=RETURN_HORIZON_DAYS, stock_fwd=stock_fwd)
    split_for_stats = compute_split_indices(len(df))
    sent_stats = fit_sentiment_zscore_stats(df.iloc[:split_for_stats["train_end"]])
    df, _ = apply_sentiment_distribution_matching(df, sent_stats)
    exclude = {"target", "Open", "High", "Low", "Close", "Volume", "ma5", "ma10", "ma20", "ma50", "ma200"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = strip_sentiment(feature_cols)
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_all, _ = build_xgb_matrix(df, feature_cols)
    return df, X_all, y_dir, y_ret


BENCHMARK_SYMBOLS = ("SPY", "QQQ", "IWM")


def build_benchmark(index: pd.DatetimeIndex, symbol: str = "SPY") -> pd.Series:
    symbol = symbol.upper()
    local_path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    if os.path.exists(local_path):
        try:
            bench_df = pd.read_parquet(local_path)
            bench_df.index = pd.DatetimeIndex(bench_df.index)
            close = bench_df["Close"].reindex(index).ffill().bfill()
            if not close.empty and float(close.iloc[0]) != 0.0:
                return close / close.iloc[0]
        except Exception:
            pass
    start = index.min().strftime("%Y-%m-%d")
    end = (index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if not raw.empty and "Close" in raw.columns:
            close = raw["Close"].reindex(index).ffill().bfill()
            if not close.empty and float(close.iloc[0]) != 0.0:
                return close / close.iloc[0]
    except Exception:
        pass
    return pd.Series(index=index, data=1.0, dtype=float)


def build_benchmarks(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {symbol: build_benchmark(index, symbol=symbol) for symbol in BENCHMARK_SYMBOLS},
        index=index,
    ).ffill().bfill()


def build_vix_series(index: pd.DatetimeIndex) -> pd.Series:
    try:
        start = index.min().strftime("%Y-%m-%d")
        end = (index.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return vix["Close"].reindex(index, method="ffill").bfill()
    except Exception:
        return pd.Series(index=index, data=20.0)


def regime_state(spy_close: pd.Series, current_date) -> str:
    hist = spy_close[spy_close.index <= current_date].tail(50)
    if len(hist) < 50:
        return "unknown"
    curr = hist.iloc[-1]
    ma20 = hist.tail(20).mean()
    ma50 = hist.mean()
    if curr < ma20 < ma50:
        return "bear"
    if curr > ma20 > ma50:
        return "bull"
    return "neutral"


def regime_size_multiplier(spy_close: pd.Series, vix_level: float, current_date) -> float:
    mult = 1.0
    if regime_state(spy_close, current_date) == "bear":
        mult *= BEAR_REGIME_MULT
    if vix_level > VIX_EXTREME_THRESHOLD:
        mult *= EXTREME_VIX_MULT
    elif vix_level > VIX_HIGH_THRESHOLD:
        mult *= HIGH_VIX_MULT
    return max(mult, 0.20)


def requested_position_pct(confidence: float, expected_return: float, signal_quality: str) -> float:
    confidence_scale = max((confidence - 50.0) / 50.0, 0.1)
    return_scale = min(max(abs(expected_return) / 4.0, 0.3), 1.0)
    size = BASE_POSITION_SIZE_PCT * confidence_scale * return_scale * 2.0
    size = min(MAX_POSITION_SIZE_PCT, max(BASE_POSITION_SIZE_PCT, size))
    if signal_quality == "HIGH":
        size *= HIGH_SIGNAL_BOOST
    return min(MAX_POSITION_SIZE_PCT, size)


def historical_annual_vol(price_history: dict[str, pd.DataFrame], ticker: str, dt, lookback: int = 63) -> float:
    """Annualized realized vol using only prices known at or before dt."""
    if ticker not in price_history:
        return 0.20
    hist = price_history[ticker]
    if "Close" not in hist.columns:
        return 0.20
    closes = hist["Close"].loc[:pd.Timestamp(dt)].dropna()
    if len(closes) < 20:
        return 0.20
    vol = float(closes.pct_change().tail(lookback).std() * np.sqrt(252))
    return max(0.05, min(1.0, vol))


def _select_feature_cols_bt(df: pd.DataFrame) -> list[str]:
    """Same feature-filter logic used in train.py — must stay in sync."""
    exclude = {"target", "Open", "High", "Low", "Close", "Volume",
               "ma5", "ma10", "ma20", "ma50", "ma200"}
    risky = (
        "sent", "social", "news", "iv_", "put_call", "option", "earn", "eps_",
        "analyst", "recommend", "short_interest", "dark_pool", "dow_", "month_",
        "opex", "calendar", "sector_ret", "ret_vs_sector", "gld_", "hyg_", "tnx_",
        "tlt_", "eem_", "iwm_", "dia_", "uup_",
    )
    allowed_pfx = (
        "ret_", "hvol_", "rsi_", "dist_ma", "ma_cross_", "macd", "atr_", "bb_",
        "vol_ratio", "volume_chg_", "hl_range", "spread_proxy", "roc_", "drawdown_",
        "weekly_", "monthly_", "tf_alignment", "spy_", "qqq_", "vix_", "regime",
        "ret_vs_spy", "ret_vs_qqq", "breadth_", "pct_above_",
    )
    exact = {"obv_slope", "vwap_dist", "uptick_ratio", "variance_ratio"}
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        and (
            c.startswith("sent_z_")
            or (
                not any(k in c.lower() for k in risky)
                and (c in exact or any(c.startswith(p) for p in allowed_pfx))
            )
        )
    ]


def _get_pruned_feature_indices_bt(
    importances: np.ndarray,
    n_base_feats: int,
    n_ticker_dummies: int,
    top_k: int,
) -> np.ndarray:
    """
    Backtest mirror of train._get_pruned_feature_indices.

    At each walk-forward block we train a quick 100-tree model on all features,
    read gain importances, and keep only the top-K base features plus all ticker
    dummies.  The final block model is then retrained on just those columns.

    This ensures each block's model is lower-variance (fewer, more informative
    features) and consistent with what train.py produces for live trading.

    Args
    ----
    importances      : feature_importances_ from the initial 100-tree model.
    n_base_feats     : Base (non-dummy) feature count — set to n_base_feats from
                       the ticker_data shape before dummies are appended.
    n_ticker_dummies : Number of ticker one-hot columns (appended after base).
    top_k            : How many base features to retain.

    Returns
    -------
    Sorted array of integer column indices to keep.
    """
    base_imp  = importances[:n_base_feats]
    top_k_idx = np.argsort(base_imp)[-min(top_k, n_base_feats):]
    dummy_idx = np.arange(n_base_feats, n_base_feats + n_ticker_dummies)
    return np.sort(np.unique(np.concatenate([top_k_idx, dummy_idx])))


def _load_ticker_for_pooled(ticker: str, common_features: list[str]):
    """
    Load one ticker's data and return pre-computed XGB feature matrix + labels.

    Rolling transforms are applied here (once per ticker per backtest run)
    so the walk-forward loop can cheaply slice by row index.  Slicing is safe
    because rolling windows only look backward.

    Returns
    -------
    (dates_array, X_full, y_dir, y_ret, open_prices, close_prices)
    or None on failure.
    """
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(path).copy()
    stock_fwd = make_forward_return_target(df["Close"], horizon=RETURN_HORIZON_DAYS)
    spy_fwd   = make_spy_forward_return(df, horizon=RETURN_HORIZON_DAYS)
    valid     = stock_fwd.notna() & spy_fwd.notna()
    df        = df.loc[valid].copy()
    stock_fwd = stock_fwd.loc[df.index]
    spy_fwd   = spy_fwd.loc[df.index]
    if len(df) < 100:
        return None
    split_for_stats = compute_split_indices(len(df))
    sent_stats = fit_sentiment_zscore_stats(df.iloc[:split_for_stats["train_end"]])
    df, _ = apply_sentiment_distribution_matching(df, sent_stats)

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
    y_ret = make_return_bucket_target(df["Close"], horizon=RETURN_HORIZON_DAYS, stock_fwd=stock_fwd)

    for col in common_features:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X, _ = build_xgb_matrix(df, common_features)
    open_prices  = df["Open"].values  if "Open"  in df.columns else df["Close"].values
    close_prices = df["Close"].values

    return (
        np.array(df.index, dtype="datetime64[ns]"),
        X,
        y_dir,
        y_ret,
        open_prices,
        close_prices,
    )


def walk_forward_predictions_pooled(
    tickers: list[str],
    min_total_train_rows: int = 2000,
    test_block: int = 126,
) -> dict[str, pd.DataFrame]:
    """
    Cross-sectional pooled walk-forward.

    At every test block the model is retrained on ALL tickers' historical data
    combined, using only dates strictly before the embargo cutoff.  This gives
    ~N× more training rows than per-ticker models (where N = number of tickers).

    Ticker one-hot dummies let the model learn ticker-specific biases even when
    the feature set is shared.

    Returns
    -------
    dict mapping ticker → pd.DataFrame of OOS predictions (same schema as
    walk_forward_predictions_for_ticker so run_portfolio_backtest can consume it
    unchanged).
    """
    # ── 1. Determine common feature set across all tickers ────────────────────
    feature_sets: list[set[str]] = []
    valid_tickers: list[str] = []
    for ticker in tickers:
        path = os.path.join(DATA_DIR, f"{ticker}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        cols = _select_feature_cols_bt(df)
        if cols:
            feature_sets.append(set(cols))
            valid_tickers.append(ticker)
    if not valid_tickers:
        return {}
    common_features = sorted(feature_sets[0].intersection(*feature_sets[1:]))
    if not common_features:
        return {}
    print(f"[pooled-bt] Common features: {len(common_features)}  Tickers: {len(valid_tickers)}")

    # ── 2. Preload all ticker data (rolling transforms applied once) ───────────
    ticker_data: dict[str, tuple] = {}
    for ticker in valid_tickers:
        result = _load_ticker_for_pooled(ticker, common_features)
        if result is not None:
            ticker_data[ticker] = result
    valid_tickers = [t for t in valid_tickers if t in ticker_data]
    if len(valid_tickers) < 2:
        return {}

    ticker_classes = sorted(valid_tickers)
    ticker_to_idx  = {t: i for i, t in enumerate(ticker_classes)}
    n_tickers      = len(ticker_classes)
    n_base_feats   = ticker_data[valid_tickers[0]][1].shape[1]

    # ── 3. Build global date timeline ─────────────────────────────────────────
    all_dates_set: set = set()
    for dates, *_ in ticker_data.values():
        all_dates_set.update(dates.tolist())
    timeline = sorted(all_dates_set)   # all unique dates across all tickers

    # Convert timeline to numpy datetime64[ns] so comparisons work uniformly
    timeline_np = np.array(timeline, dtype="datetime64[ns]")

    # Find the first date where total rows (across all tickers) >= min_total_train_rows
    start_idx = None
    for i, dt_np in enumerate(timeline_np):
        total = sum(
            int((ticker_data[t][0] <= dt_np).sum())
            for t in valid_tickers
        )
        if total >= min_total_train_rows:
            start_idx = i
            break
    if start_idx is None:
        print("[pooled-bt] Not enough training data to start walk-forward")
        return {}

    # ── 4. Walk-forward loop ──────────────────────────────────────────────────
    records: dict[str, list[dict]] = {t: [] for t in valid_tickers}
    last_valid_idx = max(0, len(timeline) - RETURN_HORIZON_DAYS - 1)
    step = start_idx
    # Accumulate calibration raw probs and labels across blocks to run
    # calibration_stability check after the loop.
    _all_cal_probs_raw: list[float] = []
    _all_cal_labels: list[int] = []

    while step < last_valid_idx:
        block_end = min(step + test_block, last_valid_idx)
        oos_dates_np = timeline_np[step:block_end]
        # Embargo: exclude RETURN_HORIZON_DAYS rows immediately before step
        cutoff_idx  = max(0, step - RETURN_HORIZON_DAYS)
        cutoff_date = timeline_np[cutoff_idx]

        # Build pooled train + calib sets from all tickers up to cutoff
        train_X_parts, train_y_dir_parts, train_y_ret_parts = [], [], []
        calib_X_parts, calib_y_dir_parts, calib_y_ret_parts = [], [], []

        for ticker in valid_tickers:
            dates, X, y_dir, y_ret, _, _ = ticker_data[ticker]
            # Rows strictly before cutoff = training candidates
            train_rows = np.where(dates < cutoff_date)[0]
            if len(train_rows) < 10:
                continue
            # Split: first 85% for training (strided), last 15% for calibration
            calib_split = max(int(len(train_rows) * 0.85), len(train_rows) - 100)
            calib_rows  = train_rows[calib_split:]
            # Non-overlapping stride on the training portion
            train_rows_strided = train_rows[:calib_split:max(1, RETURN_HORIZON_DAYS)]
            if len(train_rows_strided) < 5:
                continue

            dummy = np.zeros((1, n_tickers), dtype=np.float64)
            dummy[0, ticker_to_idx[ticker]] = 1.0

            # Training rows with ticker dummy appended
            tr_X = np.concatenate(
                [X[train_rows_strided],
                 np.repeat(dummy, len(train_rows_strided), axis=0)], axis=1
            )
            train_X_parts.append(tr_X)
            train_y_dir_parts.append(y_dir[train_rows_strided])
            train_y_ret_parts.append(y_ret[train_rows_strided])

            if len(calib_rows) > 0:
                cal_X = np.concatenate(
                    [X[calib_rows],
                     np.repeat(dummy, len(calib_rows), axis=0)], axis=1
                )
                calib_X_parts.append(cal_X)
                calib_y_dir_parts.append(y_dir[calib_rows])
                calib_y_ret_parts.append(y_ret[calib_rows])

        if not train_X_parts or not calib_X_parts:
            step = block_end
            continue

        X_train_raw = np.concatenate(train_X_parts, axis=0)
        y_dir_train = np.concatenate(train_y_dir_parts)
        y_ret_train = np.concatenate(train_y_ret_parts)
        X_cal_raw   = np.concatenate(calib_X_parts, axis=0)
        y_dir_cal   = np.concatenate(calib_y_dir_parts)
        y_ret_cal   = np.concatenate(calib_y_ret_parts)

        # ── Load tuned hyperparameters saved by train.py (if available) ────────
        # train_pooled() persists the best params from nested CV to this file.
        # The backtest reuses them so both pipelines share the same hyperparameter
        # configuration — no per-block re-tuning needed.
        _tuned_path = os.path.join(MODEL_DIR, "pooled_best_xgb_params.json")
        _bt_param_overrides: dict | None = None
        if os.path.exists(_tuned_path):
            try:
                with open(_tuned_path) as _f:
                    _bt_param_overrides = json.load(_f).get("params")
            except Exception:
                pass

        # ── Apply the fixed pruned feature set from train.py (if available) ────
        # Per-block pruning via a fast 100-tree model is unstable: feature
        # rankings differ each block (small samples → noisy importances) and can
        # flip which columns are kept, producing inconsistent signals.
        # Instead, load the same pruned column indices that train_pooled() selected
        # on the full historical dataset.  This keeps the backtest aligned with
        # live trading (same features the production model uses) without introducing
        # block-by-block variance.
        _meta_path = os.path.join(MODEL_DIR, "pooled_scaler.pkl")
        block_keep_idx: np.ndarray | None = None
        if os.path.exists(_meta_path):
            try:
                import pickle as _pickle
                with open(_meta_path, "rb") as _mf:
                    _meta = _pickle.load(_mf)
                _raw_keep = _meta.get("feature_pruning_keep_idx")
                if _raw_keep is not None:
                    block_keep_idx = np.array(_raw_keep, dtype=int)
                    n_current_cols = X_train_raw.shape[1]
                    if block_keep_idx.size == 0 or int(block_keep_idx.max()) >= n_current_cols:
                        print(
                            "[pooled-bt] Ignoring stale pooled feature_pruning_keep_idx "
                            f"(max={int(block_keep_idx.max()) if block_keep_idx.size else 'empty'}, "
                            f"current_cols={n_current_cols}). Retrain pooled model to refresh metadata."
                        )
                        block_keep_idx = None
            except Exception:
                block_keep_idx = None

        if block_keep_idx is not None:
            # Slice raw matrices to the fixed pruned column set, then refit scaler.
            X_train_raw_p = X_train_raw[:, block_keep_idx]
            X_cal_raw_p   = X_cal_raw[:, block_keep_idx]
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw_p)
            X_cal   = scaler.transform(X_cal_raw_p)
        else:
            # No saved pruning info — fall back to all features.
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_cal   = scaler.transform(X_cal_raw)

        # ── Train block model with (optionally) tuned hyperparameters ────────
        dir_model, ret_model = build_models(
            X_train, y_dir_train, X_cal, y_dir_cal,
            y_ret_train, y_ret_cal,
            param_overrides=_bt_param_overrides,
        )
        cal_probs_raw = dir_model.predict_proba(X_cal)[:, 1]
        _all_cal_probs_raw.extend(cal_probs_raw.tolist())
        _all_cal_labels.extend(y_dir_cal.tolist())
        calibrator    = fit_direction_calibrator(cal_probs_raw, y_dir_cal)
        threshold, threshold_used_fallback = training_compute_dynamic_threshold(
            calibrator,
            target_precision=CONFIDENCE_TARGET_PRECISION,
            default=DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
            raw_probs=cal_probs_raw,
            y_true=y_dir_cal,
        )
        cal_probs = np.array([
            float(calibrate_p_up(calibrator, float(p))) for p in cal_probs_raw
        ], dtype=float)
        bucket_report = confidence_buckets(cal_probs, y_dir_cal)

        # OOS predictions for each ticker on each date in the block
        for oos_dt_np in oos_dates_np:
            for ticker in valid_tickers:
                dates, X, y_dir, y_ret, open_prices, close_prices = ticker_data[ticker]
                row_hits = np.where(dates == oos_dt_np)[0]
                if len(row_hits) == 0:
                    continue
                j = int(row_hits[0])

                dummy = np.zeros((1, n_tickers), dtype=np.float64)
                dummy[0, ticker_to_idx[ticker]] = 1.0
                X_row_full = np.concatenate([X[j:j+1], dummy], axis=1)
                # If train.py saved a fixed pruned column set, apply the same
                # slice here so the OOS row matches the training feature space.
                if block_keep_idx is not None:
                    X_row_full = X_row_full[:, block_keep_idx]
                X_row = scaler.transform(X_row_full)

                p_up_raw = float(dir_model.predict_proba(X_row)[0, 1])
                p_up = float(calibrate_p_up(calibrator, p_up_raw))
                p_down = 1.0 - p_up
                confidence = min(max(max(p_up, p_down) * 100.0, 50.0), 99.0)
                ret_probs = ret_model.predict_proba(X_row)[0, :N_RETURN_BINS]
                expected_return = float((ret_probs * RETURN_BIN_CENTRES).sum()) * 100.0
                direction_vote  = "LONG" if p_up >= p_down else "SHORT"
                signal_quality  = assign_signal_quality(confidence, bucket_report,
                                                        direction=direction_vote)

                entry_j  = min(j + 1, len(dates) - 1)
                exit_j   = min(entry_j + RETURN_HORIZON_DAYS, len(dates) - 1)
                # Convert numpy datetime64 → Python Timestamp for DataFrame compat
                oos_ts    = pd.Timestamp(oos_dt_np)
                entry_ts  = pd.Timestamp(dates[entry_j])
                exit_ts   = pd.Timestamp(dates[exit_j])

                records[ticker].append({
                    "date":              oos_ts,
                    "entry_date":        entry_ts,
                    "exit_date":         exit_ts,
                    "ticker":            ticker,
                    "sector":            SECTOR_MAP.get(ticker, "OTHER"),
                    "open_next":         float(open_prices[entry_j]),
                    "exit_future_close": float(close_prices[exit_j]),
                    "holding_days":      int(exit_j - entry_j),
                    "signal":            direction_vote,
                    "direction_vote":    direction_vote,
                    "confidence":        confidence,
                    "conf_threshold":    threshold,
                    "threshold_used_fallback": bool(threshold_used_fallback),
                    "expected_return":   expected_return,
                    "signal_quality":    signal_quality,
                    # Cross-sectional ranking uses expected_return rather than threshold;
                    # mark every row actionable so the ranker can see all candidates.
                    "actionable":        True,
                })

        step = block_end

    # Convert lists to DataFrames
    result: dict[str, pd.DataFrame] = {}
    for ticker, rows in records.items():
        if rows:
            df_out = pd.DataFrame(rows)
            df_out["date"] = pd.to_datetime(df_out["date"])
            result[ticker] = df_out.set_index("date")
    print(f"[pooled-bt] Walk-forward complete. Tickers with predictions: "
          f"{[t for t in result if not result[t].empty]}")

    # ── Calibration stability diagnostic ─────────────────────────────────────
    # Checks whether the isotonic calibration curve stays consistent across
    # walk-forward blocks.  A high KS distance means the calibration is drifting
    # (likely overfitting to each block's training distribution).
    if len(_all_cal_probs_raw) >= 50:
        _cal_stability = check_calibration_stability(
            pd.Series(_all_cal_probs_raw),
            pd.Series(_all_cal_labels),
        )
        _stable = _cal_stability.get("stable")
        _max_ks = _cal_stability.get("max_ks")
        _ks_str = f"{_max_ks:.4f}" if _max_ks is not None else "N/A"
        print(
            f"[pooled-bt] Calibration stability: stable={_stable}  "
            f"max_KS={_ks_str}  "
            f"valid_folds={_cal_stability.get('valid_folds')}  "
            f"skipped_folds={_cal_stability.get('skipped_folds')}"
        )
        if _stable is False:
            print(
                "[pooled-bt] WARNING: calibration is unstable across walk-forward "
                "blocks (max_KS > 0.10). Confidence thresholds may not generalise."
            )

    return result


def apply_cross_sectional_ranking(
    predictions_by_ticker: dict[str, pd.DataFrame],
    top_n: int,
    mode: str,
) -> dict[str, pd.DataFrame]:
    """
    Replace confidence-threshold filtering with cross-sectional ranking.

    At each date, rank all tickers by predicted `expected_return`.
    Mark the top-N tickers LONG, the bottom-N SHORT (if mode != long_only).
    Set `actionable=True` only for those ranked positions.

    This sidesteps the broken isotonic calibration entirely — relative ranking
    is far more robust than absolute probability thresholds on small samples.
    """
    # Collect all dates across all tickers
    all_dates: set = set()
    for pdf in predictions_by_ticker.values():
        if not pdf.empty:
            all_dates.update(pdf.index.tolist())

    # Reset actionable to False first (pooled backtest set all to True)
    updated: dict[str, pd.DataFrame] = {
        t: pdf.copy() for t, pdf in predictions_by_ticker.items()
    }
    for pdf in updated.values():
        pdf["actionable"] = False

    for dt in sorted(all_dates):
        # Collect (ticker, p_up, expected_return, direction_vote) for each ticker on this date.
        # p_up is derived from confidence + direction_vote:
        #   LONG  → p_up = confidence / 100   (p_up >= 0.5 means model leans UP)
        #   SHORT → p_up = 1 - confidence/100  (p_up <  0.5 means model leans DOWN)
        # We rank LONG candidates by p_up (highest UP-confidence first) and SHORT
        # candidates by (1 - p_up) (highest DOWN-confidence first).
        # This replaces the broken expected_return ranking: the return model's
        # Spearman is consistently near zero or negative, so sorting by it
        # produces random or inverted selection.
        day_rows: list[tuple[str, float, float, str]] = []
        for ticker, pdf in updated.items():
            if dt not in pdf.index:
                continue
            row = pdf.loc[dt]
            conf     = float(row.get("confidence", 50.0))
            dir_vote = str(row.get("direction_vote", "LONG"))
            exp_ret  = float(row.get("expected_return", 0.0))
            # p_up: probability model assigns to the UP outcome
            p_up = (conf / 100.0) if dir_vote == "LONG" else (1.0 - conf / 100.0)
            day_rows.append((ticker, p_up, exp_ret, dir_vote))

        if not day_rows:
            continue

        # Sort descending by p_up so the most-confident UP calls rank first.
        day_rows.sort(key=lambda x: x[1], reverse=True)
        longs  = [t for t, p, _, dv in day_rows if dv == "LONG"][:top_n]
        # For shorts: most-confident DOWN = lowest p_up → reverse order
        shorts = [t for t, p, _, dv in reversed(day_rows) if dv == "SHORT"][:top_n] if mode != "long_only" else []

        # Assign signal_quality by rank position:
        #   Rank 1 → HIGH  (most confident UP call)
        #   Rank 2 → MEDIUM
        #   Rank 3+ → LOW
        # With BACKTEST_ALLOWED_SIGNAL_QUALITIES = ("HIGH","MEDIUM") only top-2 trade.
        _quality_map = {0: "HIGH", 1: "MEDIUM"}
        for rank, ticker in enumerate(longs):
            quality = _quality_map.get(rank, "LOW")
            updated[ticker].loc[dt, "actionable"] = True
            updated[ticker].loc[dt, "signal"] = "LONG"
            updated[ticker].loc[dt, "signal_quality"] = quality
        for rank, ticker in enumerate(shorts):
            quality = _quality_map.get(rank, "LOW")
            updated[ticker].loc[dt, "actionable"] = True
            updated[ticker].loc[dt, "signal"] = "SHORT"
            updated[ticker].loc[dt, "signal_quality"] = quality

    return updated


def walk_forward_predictions_for_ticker(ticker: str, min_train_rows: int = 700, test_block: int = 126) -> pd.DataFrame:
    df, X_all, y_dir, y_ret = load_ticker_data(ticker)
    dates = pd.DatetimeIndex(df.index)
    records = []
    start = min_train_rows
    last_oos_exclusive = max(0, len(df) - RETURN_HORIZON_DAYS - 1)
    while start < last_oos_exclusive:
        end = min(start + test_block, last_oos_exclusive)
        train_cutoff = max(0, start - RETURN_HORIZON_DAYS)
        train_X = X_all[:train_cutoff]
        train_y_dir = y_dir[:train_cutoff]
        train_y_ret = y_ret[:train_cutoff]
        split = compute_split_indices(len(train_X))
        train_end, calib_start, calib_end = split["train_end"], split["calib_start"], split["calib_end"]
        if train_end < 100 or calib_end <= calib_start:
            break
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_X[:train_end])
        X_cal = scaler.transform(train_X[calib_start:calib_end])
        dir_model, ret_model = build_models(X_train, train_y_dir[:train_end], X_cal, train_y_dir[calib_start:calib_end], train_y_ret[:train_end], train_y_ret[calib_start:calib_end])
        cal_probs_raw = dir_model.predict_proba(X_cal)[:, 1]
        calibrator = fit_direction_calibrator(cal_probs_raw, train_y_dir[calib_start:calib_end])
        threshold, threshold_used_fallback = training_compute_dynamic_threshold(
            calibrator,
            target_precision=CONFIDENCE_TARGET_PRECISION,
            default=DEFAULT_FIXED_CONFIDENCE_THRESHOLD,
            raw_probs=cal_probs_raw,
            y_true=train_y_dir[calib_start:calib_end],
        )
        cal_probs = np.array([
            float(calibrate_p_up(calibrator, float(p))) if calibrator is not None else float(p)
            for p in cal_probs_raw
        ], dtype=float)
        bucket_report = confidence_buckets(cal_probs, train_y_dir[calib_start:calib_end])
        X_oos = scaler.transform(X_all[start:end])
        dir_probs = dir_model.predict_proba(X_oos)
        ret_probs = ret_model.predict_proba(X_oos)
        for j, idx in enumerate(range(start, end)):
            p_up = float(calibrate_p_up(calibrator, float(dir_probs[j][1]))) if calibrator is not None else float(dir_probs[j][1])
            p_down = 1.0 - p_up
            confidence = min(max(max(p_up, p_down) * 100.0, 50.0), 99.0)
            expected_return = float((ret_probs[j][:N_RETURN_BINS] * np.array([-0.04,-0.02,0.0,0.02,0.04])).sum()) * 100.0
            direction_vote = "LONG" if p_up >= p_down else "SHORT"
            signal = direction_vote
            signal_quality = assign_signal_quality(confidence, bucket_report, direction=direction_vote)
            entry_idx = min(idx + 1, len(df) - 1)
            exit_idx = min(entry_idx + RETURN_HORIZON_DAYS, len(df) - 1)
            records.append({
                "date": dates[idx],
                "entry_date": dates[entry_idx],
                "exit_date": dates[exit_idx],
                "ticker": ticker,
                "sector": SECTOR_MAP.get(ticker, "OTHER"),
                "open_next": float(df["Open"].iloc[entry_idx]) if "Open" in df.columns else float(df["Close"].iloc[entry_idx]),
                "exit_future_close": float(df["Close"].iloc[exit_idx]),
                "holding_days": int(exit_idx - entry_idx),
                "signal": signal,
                "direction_vote": direction_vote,
                "confidence": confidence,
                "conf_threshold": threshold,
                "threshold_used_fallback": bool(threshold_used_fallback),
                "expected_return": expected_return,
                "signal_quality": signal_quality,
                "actionable": bool(confidence >= threshold),
            })
        start = end
    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


def summarize_signal_quality(trades_df: pd.DataFrame) -> dict:
    summary = {}
    for quality in ["HIGH", "MEDIUM", "LOW"]:
        subset = trades_df[trades_df["signal_quality"] == quality].copy() if not trades_df.empty else pd.DataFrame()
        if subset.empty:
            summary[quality] = {"n": 0, "win_rate": None, "avg_pnl": None, "median_pnl": None, "total_pnl": 0.0}
        else:
            summary[quality] = {
                "n": int(len(subset)),
                "win_rate": round(float((subset["net_pnl"] > 0).mean()), 4),
                "avg_pnl": round(float(subset["net_pnl"].mean()), 4),
                "median_pnl": round(float(subset["net_pnl"].median()), 4),
                "total_pnl": round(float(subset["net_pnl"].sum()), 4),
            }
    return summary


def summarize_ticker_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["ticker","trades","win_rate","avg_pnl","median_pnl","total_pnl","long_trades","short_trades"])
    rows = []
    for ticker, subset in trades_df.groupby("ticker"):
        rows.append({
            "ticker": ticker,
            "trades": int(len(subset)),
            "win_rate": round(float((subset["net_pnl"] > 0).mean()), 4),
            "avg_pnl": round(float(subset["net_pnl"].mean()), 4),
            "median_pnl": round(float(subset["net_pnl"].median()), 4),
            "total_pnl": round(float(subset["net_pnl"].sum()), 4),
            "long_trades": int((subset["signal"] == "LONG").sum()),
            "short_trades": int((subset["signal"] == "SHORT").sum()),
        })
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)


def compute_period_metrics(equity: pd.Series, benchmark: pd.Series, start: str, end: str) -> dict:
    mask = (equity.index >= pd.Timestamp(start)) & (equity.index <= pd.Timestamp(end))
    eq = equity.loc[mask]
    if eq.empty:
        return {}
    bm = benchmark.reindex(eq.index).ffill().bfill()
    daily_ret = eq.pct_change().fillna(0.0)
    dd = eq / eq.cummax() - 1.0
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    bench_ret = float(bm.iloc[-1] / bm.iloc[0] - 1.0)
    sharpe = float((daily_ret.mean() / (daily_ret.std() + 1e-12)) * np.sqrt(252))
    return {"period": f"{start} to {end}", "total_return_pct": round(total_ret * 100, 2), "benchmark_return_pct": round(bench_ret * 100, 2), "alpha_pct": round((total_ret - bench_ret) * 100, 2), "sharpe_ratio_daily": round(sharpe, 3), "max_drawdown_pct": round(float(dd.min()) * 100, 2)}


def compute_benchmark_comparisons(equity: pd.Series, benchmarks: pd.DataFrame) -> dict:
    rows = {}
    if equity.empty or benchmarks.empty:
        return rows
    daily_ret = equity.pct_change().fillna(0.0)
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    for symbol in benchmarks.columns:
        bm = benchmarks[symbol].reindex(equity.index).ffill().bfill()
        if bm.empty or float(bm.iloc[0]) == 0.0:
            continue
        bm_ret = float(bm.iloc[-1] / bm.iloc[0] - 1.0)
        bm_daily = bm.pct_change().fillna(0.0)
        data_available = bool(bm.nunique(dropna=True) > 1)
        rows[symbol] = {
            "data_available": data_available,
            "benchmark_return_pct": round(bm_ret * 100.0, 2),
            "alpha_pct": round((total_ret - bm_ret) * 100.0, 2),
            "nw_tstat_vs_benchmark": round(_newey_west_tstat(daily_ret - bm_daily), 3),
        }
    return rows


def build_trade_attribution_report(trades_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "group_type", "group", "trades", "win_rate", "avg_pnl", "median_pnl",
        "total_pnl", "avg_holding_days", "avg_position_pct", "profit_factor",
    ]
    if trades_df.empty:
        return pd.DataFrame(columns=columns)

    df = trades_df.copy()
    if "entry_date" in df.columns:
        df["entry_year"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.year.astype("Int64").astype(str)
    group_specs = [
        ("ticker", "ticker"),
        ("sector", "sector"),
        ("signal", "signal"),
        ("signal_quality", "signal_quality"),
        ("exit_reason", "exit_reason"),
        ("entry_year", "entry_year"),
    ]
    rows = []
    for group_type, col in group_specs:
        if col not in df.columns:
            continue
        for group, g in df.groupby(col, dropna=False):
            pnl = pd.to_numeric(g["net_pnl"], errors="coerce").fillna(0.0)
            wins = pnl[pnl > 0].sum()
            losses = abs(pnl[pnl < 0].sum())
            rows.append({
                "group_type": group_type,
                "group": str(group),
                "trades": int(len(g)),
                "win_rate": round(float((pnl > 0).mean()), 4),
                "avg_pnl": round(float(pnl.mean()), 2),
                "median_pnl": round(float(pnl.median()), 2),
                "total_pnl": round(float(pnl.sum()), 2),
                "avg_holding_days": round(float(pd.to_numeric(g.get("holding_days", 0), errors="coerce").fillna(0).mean()), 2),
                "avg_position_pct": round(float(pd.to_numeric(g.get("position_pct", 0), errors="coerce").fillna(0).mean()), 2),
                "profit_factor": round(float(wins / losses), 3) if losses > 0 else None,
            })
    return pd.DataFrame(rows, columns=columns).sort_values(["group_type", "total_pnl"], ascending=[True, False])


def run_portfolio_backtest(
    predictions_by_ticker: dict[str, pd.DataFrame],
    mode: str,
    stress: float,
    use_trade_rules: bool = True,
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
):
    manager = PortfolioRiskManager()
    cash = INITIAL_CAPITAL
    open_positions = []
    trade_rows = []
    price_history = {}
    all_dates = set()
    for ticker in predictions_by_ticker:
        hist = pd.read_parquet(os.path.join(DATA_DIR, f"{ticker}.parquet")).copy()
        hist.index = pd.DatetimeIndex(hist.index)
        price_history[ticker] = hist
        all_dates.update(hist.index)
    trade_rules = {ticker: load_trade_rule(ticker) for ticker in predictions_by_ticker}
    all_dates = sorted(all_dates)
    if date_start is not None:
        start_ts = pd.Timestamp(date_start)
        all_dates = [dt for dt in all_dates if pd.Timestamp(dt) >= start_ts]
    if date_end is not None:
        # Keep enough future bars to let horizon/stop exits close naturally.
        end_ts = pd.Timestamp(date_end) + pd.offsets.BDay(RETURN_HORIZON_DAYS + 5)
        all_dates = [dt for dt in all_dates if pd.Timestamp(dt) <= end_ts]
    vix = build_vix_series(pd.DatetimeIndex(all_dates))
    benchmarks = build_benchmarks(pd.DatetimeIndex(all_dates))
    benchmark = benchmarks["SPY"]
    spy_close = benchmark * 100.0
    equity_curve, equity_index = [], []

    def mark_to_market(dt):
        total = cash
        for pos in open_positions:
            hist = price_history[pos["ticker"]]
            px = float(hist["Close"].reindex([dt], method="ffill").iloc[0]) if dt not in hist.index else float(hist.loc[dt, "Close"])
            if pos["signal"] == "LONG":
                total += pos["shares"] * px
            else:
                total += pos["shares"] * (2 * pos["entry_price"] - px)
        return float(total)

    for dt in all_dates:
        remaining = []
        for pos in open_positions:
            if dt >= pos["exit_date"]:
                exit_px = float(pos["exit_price"])
                if pos["signal"] == "LONG":
                    gross = pos["shares"] * (exit_px - pos["entry_price"])
                    cash_delta = pos["shares"] * pos["entry_price"] + gross - pos["commission_exit"]
                else:
                    gross = pos["shares"] * (pos["entry_price"] - exit_px)
                    cash_delta = pos["margin_reserved"] + gross - pos["commission_exit"]
                cash += cash_delta
                net = gross - pos["commission_entry"] - pos["commission_exit"] - pos.get("borrow_cost", 0.0)
                trade_rows.append({
                    "date": pos["signal_date"], "entry_date": pos["entry_date"], "exit_date": pos["exit_date"],
                    "ticker": pos["ticker"], "signal": pos["signal"], "signal_quality": pos["signal_quality"],
                    "confidence": round(pos["confidence"], 2), "expected_return": round(pos["expected_return"], 2),
                    "position_pct": round(pos["requested_position_pct"] * 100, 2), "regime_state": pos["regime_state"],
                    "entry_price": round(pos["entry_price"], 4), "exit_price": round(exit_px, 4),
                    "holding_days": int(pos["holding_days"]), "net_pnl": round(net, 2),
                    "exit_reason": pos.get("exit_reason", "time_exit"),
                    "sector": SECTOR_MAP.get(pos["ticker"], "OTHER"),
                    "capacity_warning": bool(pos.get("capacity_warning", False)),
                })
            else:
                remaining.append(pos)
        open_positions = remaining

        reg_state = regime_state(spy_close, dt)
        reg_mult = (
            regime_size_multiplier(spy_close, float(vix.get(dt, 20.0)), dt)
            if BACKTEST_REGIME_SIZE_MULTIPLIER_ENABLED else 1.0
        )
        current_equity = mark_to_market(dt)
        candidates = []
        for ticker, pdf in predictions_by_ticker.items():
            if pdf.empty or dt not in pdf.index:
                continue
            row = pdf.loc[dt]
            if not bool(row.get("actionable", False)):
                continue
            allowed_qualities = {str(q).upper() for q in BACKTEST_ALLOWED_SIGNAL_QUALITIES}
            if str(row.get("signal_quality", "LOW")).upper() not in allowed_qualities:
                continue
            signal = str(row["signal"])
            expected_return = float(row.get("expected_return", 0.0))
            rule = trade_rules.get(ticker)
            if use_trade_rules and rule is not None:
                passed, _reason = passes_trade_rule(row, rule, mode=mode)
                if not passed:
                    continue
            # In cross-sectional mode the expected_return is a RELATIVE rank signal —
            # "best of 10" is valid even if its absolute value is slightly negative.
            # Only veto when it's deeply negative (< -1%), which implies the return
            # model strongly disagrees with the direction vote.
            if signal == "LONG" and expected_return < -1.0:
                continue
            if signal == "SHORT" and expected_return > 1.0:
                continue
            if mode == "long_only" and signal != "LONG":
                continue
            if mode == "long_only_bear_cash":
                if reg_state == "bear":
                    continue
                if signal != "LONG":
                    continue
            # Universal regime filter: block new entries when VIX is elevated or
            # SPY is below its 200-day MA, regardless of backtest mode.
            if BACKTEST_MARKET_REGIME_FILTER_ENABLED:
                vix_now = float(vix.get(dt, 20.0))
                if vix_now >= MARKET_REGIME_VIX_MAX:
                    continue
                if MARKET_REGIME_SPY_MA200_REQUIRED and reg_state == "bear":
                    continue
            if POSITION_SIZING_MODE == "vol_kelly":
                # Compute realized vol only from history known on this signal date.
                asset_vol = historical_annual_vol(price_history, ticker, dt)
                requested_pct = compute_position_size(
                    float(row["confidence"]),
                    expected_return,
                    str(row["signal_quality"]),
                    current_equity,
                    asset_vol,
                ) * reg_mult
            else:
                requested_pct = requested_position_pct(
                    float(row["confidence"]),
                    expected_return,
                    str(row["signal_quality"]),
                ) * reg_mult
            if use_trade_rules and rule is not None:
                requested_pct = min(requested_pct, float(rule.max_position_pct))
            candidates.append(ProposedTrade(
                ticker=ticker,
                date=dt,
                signal=signal,
                confidence=float(row["confidence"]),
                expected_return=expected_return,
                requested_position_pct=requested_pct,
            ))
        eq_series = pd.Series(index=pd.DatetimeIndex(equity_index), data=equity_curve, dtype=float) if equity_curve else pd.Series(dtype=float)
        approved = manager.approve_day(candidates, {k: v["Close"] for k, v in price_history.items()}, eq_series)
        eff_slip = SLIPPAGE_BASE_PCT * stress
        for tr in approved:
            row = predictions_by_ticker[tr.ticker].loc[dt]
            entry = float(row["open_next"])
            hist = price_history[tr.ticker]
            rule = trade_rules.get(tr.ticker, load_trade_rule(tr.ticker))
            exit_date, exit_px, holding_days, exit_reason = (
                resolve_rule_exit(hist, row, rule) if use_trade_rules else (
                    pd.Timestamp(row["exit_date"]),
                    float(row["exit_future_close"]),
                    int(row["holding_days"]),
                    "time_exit",
                )
            )
            adv_entry = float(hist["Volume"].loc[:pd.Timestamp(row["entry_date"])].tail(20).mean()) if "Volume" in hist.columns else 0.0
            adv_exit = float(hist["Volume"].loc[:pd.Timestamp(exit_date)].tail(20).mean()) if "Volume" in hist.columns else adv_entry
            # Realized vol used to scale bid-ask spread, known only at entry.
            fill_vol = historical_annual_vol(price_history, tr.ticker, row["entry_date"])
            trade_value = current_equity * tr.requested_position_pct
            shares = trade_value / max(entry, 1e-9)
            if tr.signal == "LONG":
                entry = realistic_fill_price(entry, shares, adv_entry, side="buy", base_slippage_pct=SLIPPAGE_BASE_PCT * stress, asset_vol=fill_vol)
                exit_px = realistic_fill_price(exit_px, shares, adv_exit, side="sell", base_slippage_pct=SLIPPAGE_BASE_PCT * stress, asset_vol=fill_vol)
            else:
                entry = realistic_fill_price(entry, shares, adv_entry, side="sell", base_slippage_pct=SLIPPAGE_BASE_PCT * stress, asset_vol=fill_vol)
                exit_px = realistic_fill_price(exit_px, shares, adv_exit, side="buy", base_slippage_pct=SLIPPAGE_BASE_PCT * stress, asset_vol=fill_vol)
            commission_entry = calc_commission(int(round(shares)))
            commission_exit = calc_commission(int(round(shares)))
            # Borrow cost charged upfront for shorts: annual rate × holding period.
            # This is the fee paid to the broker for borrowing shares to sell short.
            borrow_cost = 0.0
            if tr.signal == "SHORT":
                borrow_annual = BORROW_COSTS.get(tr.ticker, BORROW_COST_ANNUAL_DEFAULT)
                borrow_cost = trade_value * borrow_annual * (int(holding_days) / 365.0)
            if tr.signal == "LONG":
                cash -= shares * entry + commission_entry
                margin_reserved = 0.0
            else:
                cash -= trade_value + commission_entry + borrow_cost
                margin_reserved = trade_value
            open_positions.append({
                "signal_date": dt,
                "entry_date": pd.Timestamp(row["entry_date"]),
                "exit_date": pd.Timestamp(exit_date),
                "ticker": tr.ticker,
                "signal": tr.signal,
                "signal_quality": str(row["signal_quality"]),
                "confidence": tr.confidence,
                "expected_return": tr.expected_return,
                "requested_position_pct": tr.requested_position_pct,
                "regime_state": reg_state,
                "entry_price": entry,
                "exit_price": exit_px,
                "holding_days": int(holding_days),
                "exit_reason": exit_reason,
                "shares": shares,
                "commission_entry": commission_entry,
                "commission_exit": commission_exit,
                "borrow_cost": borrow_cost,
                "margin_reserved": margin_reserved,
                "capacity_warning": bool(capacity_warning(shares, adv_entry)),
            })
        equity_index.append(dt)
        equity_curve.append(mark_to_market(dt))

    equity = pd.Series(index=pd.DatetimeIndex(equity_index), data=equity_curve, dtype=float)
    benchmarks = build_benchmarks(equity.index)
    benchmark = benchmarks["SPY"]
    daily_ret = equity.pct_change().fillna(0.0)
    dd = equity / equity.cummax() - 1.0
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0
    years = max((equity.index.max() - equity.index.min()).days / 365.25, 1 / 365.25)
    ann_ret = (1.0 + total_ret) ** (1.0 / years) - 1.0 if len(equity) else 0.0
    bench_ret = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0) if len(benchmark) else 0.0
    sharpe = float((daily_ret.mean() / (daily_ret.std() + 1e-12)) * np.sqrt(252)) if len(daily_ret) else 0.0
    # Compute Newey-West t-stat vs 0 (risk-free / cash benchmark).
    # A selective strategy that deliberately holds cash cannot be tested vs a
    # fully-invested SPY — every cash day would register as a "loss" against SPY's
    # positive drift, making the t-stat negative even when all trades are profitable.
    # The correct null hypothesis for a sparse trading strategy is:
    #   H0: mean daily return == 0  (is there any return above cash?)
    # We still log the SPY-relative t-stat separately for information.
    spy_daily = benchmark.pct_change().reindex(equity.index).fillna(0.0)
    nw_tstat = _newey_west_tstat(daily_ret)          # vs 0 (cash) — primary gate
    nw_tstat_vs_spy = _newey_west_tstat(daily_ret - spy_daily)  # informational only
    benchmark_comparisons = compute_benchmark_comparisons(equity, benchmarks)

    # Evaluate every kill-criteria gate and record which passed / failed.
    gate_results = {
        "sharpe_pass": bool(sharpe >= KILL_CRITERIA["min_sharpe"]),
        "nw_tstat_pass": bool(nw_tstat >= KILL_CRITERIA["min_nw_tstat"]),
        "drawdown_pass": bool((float(dd.min()) if len(dd) else 0.0) >= KILL_CRITERIA["max_drawdown"]),
    }
    gate_results["all_pass"] = all(gate_results.values())

    metrics = {
        "total_return_pct": round(total_ret * 100, 2),
        "annual_return_pct": round(ann_ret * 100, 2),
        "benchmark_return_pct": round(bench_ret * 100, 2),
        "alpha_pct": round((total_ret - bench_ret) * 100, 2),
        "benchmark_comparisons": benchmark_comparisons,
        "sharpe_ratio_daily": round(sharpe, 3),
        "nw_tstat_vs_cash": round(nw_tstat, 3),
        "nw_tstat_vs_spy": round(nw_tstat_vs_spy, 3),
        "max_drawdown_pct": round(float(dd.min()) * 100, 2) if len(dd) else 0.0,
        "slippage_assumption_bps": round(SLIPPAGE_BASE_PCT * stress * 10_000, 1),
        "execution_model": "spread_plus_sqrt_impact",
        "mode": mode,
        "kill_criteria": {
            "thresholds": KILL_CRITERIA,
            "results": gate_results,
        },
        "diagnostics_2022": compute_period_metrics(equity, benchmark, "2022-01-01", "2022-12-31"),
        "signal_quality": summarize_signal_quality(pd.DataFrame(trade_rows)),
    }
    return pd.DataFrame(trade_rows), equity, benchmarks, metrics


def default_backtest_tickers() -> list[str]:
    available = {f.replace(".parquet", "") for f in os.listdir(DATA_DIR) if f.endswith(".parquet")}
    report = read_quality_report()
    if not report.empty and {"ticker", "approved_for_live"}.issubset(report.columns):
        approved = report[report["approved_for_live"].astype(str).str.lower().isin({"true", "1"})]
        tickers = [str(t).upper() for t in approved["ticker"].tolist() if str(t).upper() in available]
        if tickers:
            return tickers
    # No manual live-approval fallback: if nothing is approved, the default
    # research run uses the watchlist but still writes rejected/approved status
    # from objective gates into model_quality_report.csv.
    return [t for t in WATCHLIST if t in available]


def write_quality_outputs(tickers: list[str], trades_df: pd.DataFrame, metrics: dict, quality_source: str) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        ticker_trades = (
            trades_df[trades_df["ticker"].astype(str).str.upper() == ticker.upper()].copy()
            if not trades_df.empty and "ticker" in trades_df.columns
            else pd.DataFrame()
        )
        row = evaluate_model_quality(
            ticker,
            trades=ticker_trades,
            extra_metrics={
                "quality_source": quality_source,
                "portfolio_sharpe": metrics.get("sharpe_ratio_daily"),
                "portfolio_nw_tstat_vs_cash": metrics.get("nw_tstat_vs_cash"),
                "portfolio_max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "portfolio_total_return_pct": metrics.get("total_return_pct"),
            },
        )
        rows.append(row)
        update_scaler_metadata(ticker, row)

    report = upsert_quality_report(rows)
    approved = report[report["approved_for_live"].astype(str).str.lower().isin({"true", "1"})].copy()
    approved_path = os.path.join(SIGNAL_DIR, "approved_live_tickers.csv")
    approved[["ticker", "approval_reason"]].to_csv(approved_path, index=False)
    print("Saved ->", os.path.join(MODEL_DIR, "model_quality_report.csv"))
    print("Saved ->", approved_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict walk-forward backtest aligned to train.py")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--mode", type=str, default="long_only", choices=["long_short", "long_only", "long_only_bear_cash"])
    parser.add_argument("--allow-shorts", action="store_true", help="Backward-compatible alias for --mode long_short.")
    parser.add_argument("--use-trade-rules", action="store_true", help="Apply optimized per-ticker trade rules. Off by default for pooled research backtests.")
    parser.add_argument("--ignore-trade-rules", action="store_true", help="Deprecated; trade rules are ignored unless --use-trade-rules is passed.")
    parser.add_argument("--stress", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(SIGNAL_DIR, exist_ok=True)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = default_backtest_tickers()

    # Compute effective_mode early so pooled path can use it for ranking.
    # Keep the default long-only, but make an explicit --mode long_short do
    # what it says. --allow-shorts remains for older runbooks/scripts.
    effective_mode = "long_short" if args.allow_shorts else args.mode
    use_trade_rules = bool(args.use_trade_rules and not args.ignore_trade_rules)
    use_pooled = POOLED_TRAINING and len(tickers) > 1

    if use_pooled:
        # ── Pooled cross-sectional walk-forward ───────────────────────────────
        print(f"[backtest] POOLED mode: training one model across {len(tickers)} tickers")
        predictions = walk_forward_predictions_pooled(tickers)
        # Apply cross-sectional ranking instead of confidence-threshold filtering
        predictions = apply_cross_sectional_ranking(
            predictions, top_n=CROSS_SECTIONAL_TOP_N, mode=effective_mode
        )
    else:
        # ── Per-ticker walk-forward (original path) ───────────────────────────
        predictions = {}
        for ticker in tickers:
            try:
                summary_path = os.path.join(MODEL_DIR, f"{ticker}_train_summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path) as _f:
                        _summary = json.load(_f)
                    _acc = _summary.get("direction_accuracy", 0)
                    _baseline = _summary.get("baseline_up_rate", 50)
                    if _acc < _baseline - 5.0:
                        print(f"SKIP {ticker}: dir_acc={_acc:.1f}% vs baseline={_baseline:.1f}% (>5pp below, model harmful)")
                        continue
                predictions[ticker] = walk_forward_predictions_for_ticker(ticker)
            except Exception as e:
                print(f"ERROR - {ticker}: {e}")

    # Save raw per-ticker prediction DataFrames so we can diagnose whether
    # predictions continue through later years and whether they stop being
    # actionable versus disappearing entirely.
    for ticker, pdf in predictions.items():
        if pdf is None or pdf.empty:
            continue
        pred_out = os.path.join(SIGNAL_DIR, f"{ticker}_walkforward_predictions.csv")
        pdf.reset_index().to_csv(pred_out, index=False)
        print("Saved ->", pred_out)

    predictions = {k: v for k, v in predictions.items() if not v.empty}
    if not predictions:
        raise SystemExit("No walk-forward predictions generated")

    trades_df, equity, benchmarks, metrics = run_portfolio_backtest(
        predictions,
        effective_mode,
        args.stress,
        use_trade_rules=use_trade_rules,
    )
    suffix = tickers[0] if args.ticker and len(tickers) == 1 else f"walkforward_{effective_mode}_{len(predictions)}tickers"
    trades_out = os.path.join(SIGNAL_DIR, f"{suffix}_trades.csv")
    equity_out = os.path.join(SIGNAL_DIR, f"{suffix}_equity.csv")
    metrics_out = os.path.join(SIGNAL_DIR, f"{suffix}_metrics.json")
    breakdown_out = os.path.join(SIGNAL_DIR, f"{suffix}_ticker_breakdown.csv")
    attribution_out = os.path.join(SIGNAL_DIR, f"{suffix}_trade_attribution.csv")
    trades_df.to_csv(trades_out, index=False)
    equity_frame = pd.DataFrame({"equity": equity})
    for symbol in benchmarks.columns:
        equity_frame[f"benchmark_{symbol.lower()}_norm"] = benchmarks[symbol].reindex(equity.index)
    equity_frame.to_csv(equity_out)
    summarize_ticker_breakdown(trades_df).to_csv(breakdown_out, index=False)
    build_trade_attribution_report(trades_df).to_csv(attribution_out, index=False)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    quality_report = write_quality_outputs(list(predictions.keys()), trades_df, metrics, "backtest_walkforward")
    append_experiment(
        name="backtest_walkforward",
        params={
            "tickers": list(predictions.keys()),
            "mode": effective_mode,
            "stress": args.stress,
            "use_trade_rules": use_trade_rules,
            "allow_shorts": args.allow_shorts,
            "horizon_days": RETURN_HORIZON_DAYS,
        },
        metrics={
            "total_return_pct": metrics.get("total_return_pct"),
            "alpha_pct": metrics.get("alpha_pct"),
            "sharpe_ratio_daily": metrics.get("sharpe_ratio_daily"),
            "nw_tstat_vs_cash": metrics.get("nw_tstat_vs_cash"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "trades": int(len(trades_df)),
            "approved_models": int(
                quality_report["approved_for_live"].astype(str).str.lower().isin({"true", "1"}).sum()
            )
            if "approved_for_live" in quality_report.columns
            else 0,
            "kill_all_pass": bool(metrics.get("kill_criteria", {}).get("results", {}).get("all_pass", False)),
        },
        artifacts={
            "trades": trades_out,
            "equity": equity_out,
            "metrics": metrics_out,
            "breakdown": breakdown_out,
            "attribution": attribution_out,
        },
    )
    print("Saved ->", trades_out)
    print("Saved ->", equity_out)
    print("Saved ->", metrics_out)
    print("Saved ->", breakdown_out)
    print("Saved ->", attribution_out)
    print(metrics)

    # ── Kill-criteria gate ────────────────────────────────────────────────────
    # Print a clear PASS/FAIL summary so the result is impossible to miss.
    # Exit with code 1 so CI / shell scripts can detect a failed backtest.
    kc = metrics["kill_criteria"]
    res = kc["results"]
    thr = kc["thresholds"]
    print("\n" + "=" * 60)
    print("KILL-CRITERIA GATE")
    print("=" * 60)
    print(f"  Sharpe        : {metrics['sharpe_ratio_daily']:.3f}  (need >= {thr['min_sharpe']})  "
          f"{'PASS' if res['sharpe_pass'] else 'FAIL'}")
    print(f"  NW t-stat(0)  : {metrics['nw_tstat_vs_cash']:.3f}  (need >= {thr['min_nw_tstat']})  "
          f"{'PASS' if res['nw_tstat_pass'] else 'FAIL'}  [vs SPY: {metrics['nw_tstat_vs_spy']:.3f}]")
    print(f"  Max drawdown  : {metrics['max_drawdown_pct']:.1f}%  (need >= {thr['max_drawdown']*100:.0f}%)  "
          f"{'PASS' if res['drawdown_pass'] else 'FAIL'}")
    print("-" * 60)
    if res["all_pass"]:
        print("RESULT: ALL GATES PASSED - strategy may advance to paper trading")
    else:
        failed = [k.replace("_pass", "") for k, v in res.items() if k != "all_pass" and not v]
        print(f"RESULT: FAILED - gates not cleared: {', '.join(failed)}")
        print("Strategy should NOT advance to paper trading.")
    print("=" * 60 + "\n")

    if not res["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
