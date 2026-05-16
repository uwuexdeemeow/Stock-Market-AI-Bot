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
import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
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
from train import compute_conformal_qhat, conformal_prediction_info
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
    TRIPLE_BARRIER_PT_MULT,
    TRIPLE_BARRIER_SL_MULT,
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
    META_LABELING_ENABLED,
    META_LABEL_MIN_CONFIDENCE,
    FEATURE_IMPORTANCE_TOP_K,
    MAX_SINGLE_NAME_EXPOSURE,
    CASH_OVERLAY_ENABLED,
    CASH_OVERLAY_MAX_ALLOC,
    CASH_OVERLAY_REQUIRE_SPY_MA200,
    CASH_OVERLAY_WEIGHTS,
    PRIMARY_BENCHMARK_WEIGHTS,
    TICKER_ELIGIBILITY_FILTER_ENABLED,
    TICKER_ELIGIBILITY_MIN_TRADES,
    TICKER_ELIGIBILITY_MIN_TOTAL_PNL,
    TICKER_ELIGIBILITY_MIN_WIN_RATE,
    TICKER_ELIGIBILITY_MIN_SHARPE,
    TICKER_ELIGIBILITY_MIN_PROFIT_FACTOR,
    TICKER_ELIGIBILITY_MAX_DRAWDOWN_PCT,
    H5_SENTIMENT_FEATURES_ENABLED,
    CONFORMAL_TRADE_GATE_ENABLED,
    CONFORMAL_ALPHA,
    CONFORMAL_MIN_CALIBRATION_ROWS,
    RANK_ORIENTATION_FLIP_ENABLED,
    SINGLE_NAME_CRASH_FILTER_ENABLED,
    SINGLE_NAME_CRASH_MIN_PRICE,
    SINGLE_NAME_CRASH_MAX_DRAWDOWN_20D,
    SINGLE_NAME_CRASH_MAX_DRAWDOWN_60D,
    SINGLE_NAME_CRASH_MAX_DIST_MA200,
    SINGLE_NAME_DISTRESS_FILTER_ENABLED,
    SINGLE_NAME_DISTRESS_MAX_DRAWDOWN_252D,
    SINGLE_NAME_DISTRESS_MAX_DIST_MA50,
    SINGLE_NAME_DISTRESS_MAX_RET_20D,
    MULTI_HORIZON_H20_WEIGHT,
    MULTI_HORIZON_H5_WEIGHT,
    VOL_TARGET_PCT,
    KELLY_SAFETY_FRACTION,
    SURVIVORSHIP_TRAINING_TICKERS,
    REGIME_CONDITIONAL_MODELS,
    REGIME_MIN_TRAIN_ROWS,
    EXCLUDE_MARKET_CONSTANTS,
    POOLED_RANKER_EXPERIMENT_ENABLED,
    SIMPLE_FACTOR_RANKING,
    SIMPLE_FACTOR_COLS,
    SIMPLE_FACTOR_WEIGHTS, ADAPTIVE_WEIGHTS_FILE,
    SPY_TIMING_MODE,
    SPY_TIMING_BULL_THRESHOLD,
    SPY_TIMING_INSTRUMENTS,
    SPY_TIMING_INVEST_PCT,
    SPY_TIMING_SMOOTH_WINDOW,
    SPY_TIMING_ENTER_THRESHOLD,
    SPY_TIMING_EXIT_THRESHOLD,
    SPY_TIMING_MIN_HOLD_DAYS,
    SPY_TIMING_ETF_COST_BPS,
    MIN_TIMING_SHARPE,
    MAX_TIMING_DRAWDOWN_PCT,
    MIN_TIMING_NW_TSTAT_VS_CASH,
    MIN_TIMING_TRADES_OR_EXPOSURE_DAYS,
    ETF_ROTATION_MODE,
    ETF_ROTATION_ASSETS,
    ETF_ROTATION_PRESETS,
    ETF_ROTATION_DEFENSIVE_MIXES,
    ETF_ROTATION_RISK_ON_ENTER,
    ETF_ROTATION_RISK_ON_EXIT,
    ETF_ROTATION_NEUTRAL_THRESHOLD,
    ETF_ROTATION_MIN_HOLD_DAYS,
    ETF_ROTATION_REBALANCE_BAND,
    ETF_ROTATION_DEFAULT_VOL_TARGET,
    ETF_ROTATION_MAX_GROSS_EXPOSURE,
    MIN_ETF_ROTATION_ALPHA_VS_SPY_PCT,
    MIN_ETF_ROTATION_ALPHA_VS_QQQ_PCT,
    MIN_ETF_ROTATION_ALPHA_VS_BLEND_PCT,
    MIN_ETF_ROTATION_NW_TSTAT_VS_BLEND,
    PAPER_MODE_STRATEGY,
    SINGLE_NAME_PAPER_TRADING_ENABLED,
    EDGE_AUDIT_ENABLED,
    REGIME_EDGE_GATE_ENABLED,
    RANK_DECILE_GATE_ENABLED,
    EDGE_GATE_MIN_TRADES,
    EDGE_GATE_MIN_PROFIT_FACTOR,
    EDGE_GATE_MIN_AVG_PNL,
)


def _download_yfinance(*args, **kwargs) -> pd.DataFrame:
    """Import yfinance only when a download fallback is actually needed."""
    from data_provider import _quiet_yfinance_logging
    import yfinance as yf

    with _quiet_yfinance_logging():
        return yf.download(*args, **kwargs)
from xgb_feature_engineering import build_xgb_matrix
from pipeline_shared import apply_sentiment_distribution_matching, fit_sentiment_zscore_stats
from ranker_utils import build_rank_groups_from_dates, daily_rank_ic
from portfolio_manager import PortfolioRiskManager, ProposedTrade
from execution_model import realistic_fill_price, commission as calc_commission, capacity_warning
from risk_sizing import compute_position_size
from experiment_ledger import append_experiment

INITIAL_CAPITAL = 10_000.0
BASE_POSITION_SIZE_PCT = 0.15
MAX_POSITION_SIZE_PCT = 0.30
HIGH_SIGNAL_BOOST = 1.0
MIN_BUCKET_N_HIGH = 30
MIN_BUCKET_N_MEDIUM = 15
_ETF_PRICE_FRAME_CACHE: dict[tuple[tuple[str, ...], str, str], pd.DataFrame] = {}
_ETF_VOTE_CACHE: dict[int, tuple[pd.Series, dict[pd.Timestamp, dict]]] = {}
_ETF_DOWNLOAD_FAILED_SYMBOLS: set[str] = set()

# ── Adaptive factor weights ───────────────────────────────────────────────────
# PLAIN ENGLISH: Load IC-based weights from JSON (written by research.py after
# each run).  If the file doesn't exist yet, fall back to static weights from
# settings.py.  This ensures the factor composite adapts to recent regime shifts.
def _load_factor_weights() -> dict[str, float]:
    """Load adaptive weights from JSON, fall back to static settings."""
    try:
        with open(ADAPTIVE_WEIGHTS_FILE) as f:
            import json as _json
            data = _json.load(f)
        weights = data.get("weights", {})
        if weights and abs(sum(weights.values()) - 1.0) < 0.01:
            return weights
    except (FileNotFoundError, KeyError, ValueError, OSError):
        pass
    return dict(SIMPLE_FACTOR_WEIGHTS)

_active_factor_weights = _load_factor_weights()

# ── ATR-based position sizing parameters ──────────────────────────────────────
# Instead of a flat % per trade, we risk a fixed fraction of capital per trade.
# Position size = RISK_BUDGET_PCT / (ATR_SL_MULT × atr_pct), capped at
# MAX_SINGLE_NAME_EXPOSURE.  This means high-volatility stocks (e.g. MU, NVDA)
# get smaller positions than low-volatility stocks (e.g. MSFT, UNH), so that
# each trade risks roughly the same dollar amount.
# ATR_SL_MULT = 1.5 reflects our -0.75×ATR stop plus a small slippage buffer.
ATR_RISK_BUDGET_PCT = 0.008   # risk 0.8% of capital per trade
ATR_SL_MULT         = 1.5     # multiplier covering -0.75×ATR stop + buffer
ATR_MIN_POSITION    = 0.05    # floor: never go below 5% on any signal
RETURN_BIN_CENTRES = np.array([-0.04, -0.02, 0.00, 0.02, 0.04], dtype=float)
N_RETURN_BINS = len(RETURN_BIN_CENTRES)


def print_model_summary(
    metrics: dict,
    *,
    mode: str,
    tickers: list[str],
    trades_path: str,
    equity_path: str,
    metrics_path: str,
) -> None:
    kc = metrics.get("kill_criteria", {})
    res = kc.get("results", {})
    status = "PASS" if bool(res.get("all_pass", False)) else "FAIL"
    failed = [
        k.replace("_pass", "")
        for k, v in res.items()
        if k != "all_pass" and not bool(v)
    ]

    print("\nMODEL BACKTEST")
    print("-" * 48)
    print(f"Status          : {status}" + (f" ({', '.join(failed)})" if failed else ""))
    print(f"Mode / horizon  : {mode} / {RETURN_HORIZON_DAYS}d")
    print(f"Tickers         : {len(tickers)}")
    print(f"Trades          : {metrics.get('trades', 0)}")
    print(f"Total return    : {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Benchmark       : {metrics.get('benchmark_return_pct', 0):.2f}% ({metrics.get('primary_benchmark', 'benchmark')})")
    print(f"Alpha           : {metrics.get('alpha_pct', 0):.2f}%")
    print(f"Sharpe          : {metrics.get('sharpe_ratio_daily', 0):.3f}")
    if metrics.get("cash_overlay_enabled"):
        print(f"Raw stock return: {metrics.get('raw_strategy_total_return_pct', 0):.2f}%")
        print(f"Raw stock Sharpe: {metrics.get('raw_strategy_sharpe_ratio_daily', 0):.3f}")
        print(f"Overlay avg alloc: {metrics.get('cash_overlay_avg_alloc', 0):.2%}")
        print(f"Overlay contrib : {metrics.get('overlay_contribution_pct', 0):.2f}%")
    print(f"NW t-stat       : {metrics.get('nw_tstat_vs_cash', 0):.3f}")
    print(f"Max drawdown    : {metrics.get('max_drawdown_pct', 0):.2f}%")
    rank_perf = metrics.get("rank_performance", {}) or {}
    if rank_perf.get("rank_spearman") is not None:
        print(f"Rank Spearman   : {rank_perf.get('rank_spearman')}")
    daily_ic = metrics.get("daily_rank_ic", {}) or {}
    if daily_ic.get("mean_ic") is not None:
        t_stat = daily_ic.get("t_stat")
        t_part = f", t={t_stat:.2f}" if t_stat is not None else ""
        print(f"Daily Rank IC   : {daily_ic.get('mean_ic'):.4f} ({daily_ic.get('days', 0)} days{t_part})")
    print("-" * 48)
    print(f"Trades file     : {trades_path}")
    print(f"Equity file     : {equity_path}")
    print(f"Metrics file    : {metrics_path}")

# Kill criteria: backtest must clear ALL of these gates to be considered valid.
# Sharpe ≥ 0.5: realistic bar for a selective/sparse strategy (in-market ~30% of time).
#   A fully-invested strategy targets ≥ 1.0; a selective one with 66%+ win rate
#   naturally has lower Sharpe because of cash drag on quiet days.
# NW t-stat ≥ 1.65: one-tailed 95% significance that daily returns > 0 (vs cash).
#   Measured vs 0 (not vs SPY) — SPY-relative t-stat is logged separately but not gated,
#   because a cash-holding strategy cannot be expected to beat a fully-invested index daily.
# Raw stock edge gates: keep the SPY/QQQ cash overlay from masking a degraded
# single-name model. These are intentionally modest; the stock strategy must at
# least be non-negative before the overlay is allowed to make the run "pass."
KILL_CRITERIA = {
    "min_sharpe": 0.5,       # annualised daily Sharpe must be ≥ 0.5
    "min_nw_tstat": 1.65,    # Newey-West t-stat of daily return vs 0 must be ≥ 1.65 (95% CI)
    "max_drawdown": -0.25,   # max drawdown must be better (less negative) than -25%
    "min_raw_stock_sharpe": 0.0,
    "min_raw_stock_return": 0.0,
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
    """Backward-compatible wrapper around the shared label builder."""
    return shared_make_direction_target(
        df,
        prediction_target=PREDICTION_TARGET,
        horizon=RETURN_HORIZON_DAYS,
        direction_threshold=DIRECTION_LABEL_THRESHOLD,
        excess_return_min_pct=EXCESS_RETURN_MIN_PCT,
        vol_adjusted_sharpe_threshold=VOL_ADJUSTED_SHARPE_THRESHOLD,
    )


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
    # Class-balance via per-row sample_weight — same fix as train.py build_models.
    # scale_pos_weight shifts the model's output probability scale upward (avg prob
    # → spw/(1+spw) instead of true UP rate), so all predictions end up ≥ 0.5 →
    # accuracy = UP rate = baseline.  Per-row weights don't shift the scale.
    n_pos = max(int(np.sum(y_dir_train == 1)), 1)
    n_neg = max(int(np.sum(y_dir_train == 0)), 1)
    class_w = np.where(y_dir_train == 1, float(n_neg) / n_pos, 1.0).astype(np.float32)
    dir_model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        **common   # no scale_pos_weight
    )
    dir_model.fit(X_train, y_dir_train, sample_weight=class_w,
                  eval_set=[(X_cal, y_dir_cal)], verbose=False)
    ret_model = xgb.XGBClassifier(objective="multi:softprob", num_class=N_RETURN_BINS, eval_metric="mlogloss", **common)
    ret_model.fit(X_train, y_ret_train, eval_set=[(X_cal, y_ret_cal)], verbose=False)
    return dir_model, ret_model


def build_meta_matrix(X: np.ndarray, p_up_raw: np.ndarray, p_up_cal: np.ndarray, expected_return: np.ndarray) -> np.ndarray:
    """Feature matrix for meta-labelling: original features plus primary signal diagnostics."""
    return np.concatenate(
        [
            np.asarray(X, dtype=float),
            np.asarray(p_up_raw, dtype=float).reshape(-1, 1),
            np.asarray(p_up_cal, dtype=float).reshape(-1, 1),
            np.asarray(expected_return, dtype=float).reshape(-1, 1),
            np.abs(np.asarray(p_up_cal, dtype=float).reshape(-1, 1) - 0.5),
        ],
        axis=1,
    )


def compute_bucket_edges(conf: np.ndarray, n_buckets: int = 10) -> list[tuple[float, float]]:
    """Quantile-based bucket edges computed from the real confidence distribution.

    PLAIN ENGLISH: Splits the actual confidence values into n_buckets equal-count
    groups. If the model only emits 50–52% confidence, all buckets sit in that
    range so each bucket actually has observations to report on — unlike fixed
    [50-55, 55-60, ...] bins that may be mostly empty.
    """
    conf_arr = np.asarray(conf, dtype=float)
    conf_arr = conf_arr[np.isfinite(conf_arr)]
    if conf_arr.size == 0:
        raw = np.linspace(50.0, 100.0, max(int(n_buckets), 1) + 1)
        return [(float(raw[i]), float(raw[i + 1])) for i in range(len(raw) - 1)]

    conf_arr = np.clip(conf_arr, 50.0, 100.0)
    n_buckets = max(int(n_buckets), 1)
    percentile_pts = np.linspace(0, 100, n_buckets + 1)
    raw = np.percentile(conf_arr, percentile_pts)
    # Keep full precision for bucket membership. Rounding can collapse nearby
    # quantiles and create fake empty buckets when confidence is tightly packed.
    raw[0] = 50.0
    raw[-1] = max(float(raw[-1]), float(conf_arr.max()))
    return [(float(raw[i]), float(raw[i + 1])) for i in range(n_buckets)]


def confidence_buckets(
    probs: np.ndarray,
    y_true: np.ndarray,
    edges: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Precision and coverage for each confidence bucket.

    Args:
        probs: Raw model probabilities in [0, 1].
        y_true: Actual direction labels (0=DOWN, 1=UP).
        edges: Optional list of (lo, hi) pairs in [50, 100]. When None, uses
               fixed legacy edges [50-55, 55-60, 60-65, 65-70, 70-100].
    """
    conf = np.maximum(probs, 1.0 - probs) * 100.0
    pred = (probs >= 0.5).astype(int)
    edge_list: list[tuple[float, float]] = edges if edges is not None else [
        (50, 55), (55, 60), (60, 65), (65, 70), (70, 100)
    ]
    rows = []
    for i, (lo, hi) in enumerate(edge_list):
        is_last = (i == len(edge_list) - 1)
        mask = (conf >= lo) & (conf <= hi if is_last else conf < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "n": 0,
                "precision": None,
                "lo": float(lo),
                "hi": float(hi),
            })
        else:
            rows.append({
                "bucket": f"{lo:.1f}-{hi:.1f}", "n": n,
                "precision": round(float((pred[mask] == y_true[mask]).mean()), 4),
                "lo": float(lo),
                "hi": float(hi),
            })
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
    for i, row in enumerate(bucket_report):
        if "lo" in row and "hi" in row:
            lo = float(row["lo"])
            hi = float(row["hi"])
        else:
            lo_s, hi_s = row["bucket"].split("-")
            lo = float(lo_s); hi = float(hi_s)
        is_last = i == len(bucket_report) - 1
        if confidence >= lo and (confidence < hi or is_last):
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
        "vol_ratio", "volume_chg_", "vol_zscore_", "vol_pct_vs_", "vol_trend_",
        "hl_range", "spread_proxy", "roc_", "drawdown_",
        "weekly_", "monthly_", "tf_alignment", "spy_", "qqq_", "vix_", "macro_", "regime",
        "ret_vs_spy", "ret_vs_qqq", "breadth_", "pct_above_",
        "factor_",
        # Cross-sectional rank features (xs_rank_sector_*, xs_rank_market_*).
        "xs_rank_",
    )
    allowed_exact = {
        "obv_slope",
        "vwap_dist",
        "uptick_ratio",
        "variance_ratio",
        "fund_pe_sector_z",
        "fund_fcf_yield_sector_z",
        "fund_value_combo_z",
        "fund_has_valuation",
    }
    return [
        c for c in feature_cols
        if c.startswith("sent_z_")
        or c.startswith("xs_rank_")     # whitelist before risky-keyword check
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


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {str(k).upper(): float(v) for k, v in (weights or {}).items() if float(v) > 0}
    total = sum(clean.values())
    if total <= 0:
        return {"SPY": 1.0}
    return {k: v / total for k, v in clean.items()}


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
        raw = _download_yfinance(symbol, start=start, end=end, progress=False, auto_adjust=True)
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
    benchmarks = pd.DataFrame(
        {symbol: build_benchmark(index, symbol=symbol) for symbol in BENCHMARK_SYMBOLS},
        index=index,
    ).ffill().bfill()
    primary_weights = _normalise_weights(PRIMARY_BENCHMARK_WEIGHTS)
    primary = pd.Series(index=benchmarks.index, data=0.0, dtype=float)
    for symbol, weight in primary_weights.items():
        if symbol not in benchmarks.columns:
            benchmarks[symbol] = build_benchmark(index, symbol=symbol)
        primary = primary.add(benchmarks[symbol].reindex(benchmarks.index).ffill().bfill() * weight, fill_value=0.0)
    benchmarks["BLEND"] = primary
    return benchmarks.ffill().bfill()


def apply_cash_overlay(
    equity: pd.Series,
    benchmarks: pd.DataFrame,
    gross_exposure: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Add a SPY/QQQ sleeve on idle capital when the broad trend is supportive."""
    if equity.empty or benchmarks.empty:
        return equity, pd.Series(index=equity.index, data=0.0, dtype=float)

    weights = _normalise_weights(CASH_OVERLAY_WEIGHTS)
    blend = pd.Series(index=equity.index, data=0.0, dtype=float)
    for symbol, weight in weights.items():
        if symbol in benchmarks.columns:
            blend = blend.add(benchmarks[symbol].reindex(equity.index).ffill().bfill() * weight, fill_value=0.0)

    if blend.nunique(dropna=True) <= 1:
        return equity, pd.Series(index=equity.index, data=0.0, dtype=float)

    trend_ok = pd.Series(index=equity.index, data=True, dtype=bool)
    if CASH_OVERLAY_REQUIRE_SPY_MA200 and "SPY" in benchmarks.columns:
        spy = benchmarks["SPY"].reindex(equity.index).ffill().bfill()
        trend_ok = spy > spy.rolling(200, min_periods=50).mean()

    idle_fraction = (1.0 - gross_exposure.reindex(equity.index).ffill().fillna(0.0)).clip(0.0, 1.0)
    overlay_alloc = (float(CASH_OVERLAY_MAX_ALLOC) * idle_fraction).where(trend_ok, 0.0).clip(0.0, 1.0)
    base_ret = equity.pct_change().fillna(0.0)
    blend_ret = blend.pct_change().fillna(0.0)
    combined_ret = base_ret + overlay_alloc.shift(1).fillna(0.0) * blend_ret
    overlay_equity = equity.iloc[0] * (1.0 + combined_ret).cumprod()
    return overlay_equity.astype(float), overlay_alloc.astype(float)


def build_vix_series(index: pd.DatetimeIndex) -> pd.Series:
    try:
        start = index.min().strftime("%Y-%m-%d")
        end = (index.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        vix = _download_yfinance("^VIX", start=start, end=end, progress=False, auto_adjust=True)
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


def atr_position_pct(
    price_history: dict[str, pd.DataFrame],
    ticker: str,
    dt,
    atr_window: int = 20,
) -> float:
    """
    Return a position size (as fraction of capital) that risks ATR_RISK_BUDGET_PCT
    of capital per trade, using ATR_SL_MULT × ATR as the dollar-stop estimate.

    position_pct = RISK_BUDGET / (ATR_SL_MULT × atr_pct)

    High-volatility tickers (NVDA, MU) get smaller positions; calm tickers
    (MSFT, UNH) get larger ones — but all are capped at MAX_SINGLE_NAME_EXPOSURE.
    Only prices known BEFORE dt are used so there is no look-ahead.
    """
    fallback = MAX_SINGLE_NAME_EXPOSURE * 0.75   # conservative fallback if no data
    if ticker not in price_history:
        return fallback
    hist = price_history[ticker]
    if "Close" not in hist.columns:
        return fallback
    closes = hist["Close"].loc[:pd.Timestamp(dt)].dropna()
    # Need at least atr_window+1 closes to compute ATR
    if len(closes) < atr_window + 1:
        return fallback
    last_price = float(closes.iloc[-1])
    if last_price <= 0:
        return fallback
    # ATR proxy: mean absolute close-to-close move over the last atr_window days
    atr_dollar = float(np.mean(np.abs(np.diff(closes.values[-(atr_window + 1):]))))
    if atr_dollar <= 0:
        return fallback
    atr_pct = atr_dollar / last_price
    raw = ATR_RISK_BUDGET_PCT / (ATR_SL_MULT * atr_pct)
    return float(np.clip(raw, ATR_MIN_POSITION, MAX_SINGLE_NAME_EXPOSURE))


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
        "vol_ratio", "volume_chg_", "vol_zscore_", "vol_pct_vs_", "vol_trend_",
        "hl_range", "spread_proxy", "roc_", "drawdown_",
        "weekly_", "monthly_", "tf_alignment", "spy_", "qqq_", "vix_", "macro_", "regime",
        "ret_vs_spy", "ret_vs_qqq", "breadth_", "pct_above_",
        "factor_",
        # Cross-sectional rank features (must stay in sync with train.py).
        "xs_rank_",
    )
    exact = {
        "obv_slope",
        "vwap_dist",
        "uptick_ratio",
        "variance_ratio",
        "fund_pe_sector_z",
        "fund_fcf_yield_sector_z",
        "fund_value_combo_z",
        "fund_has_valuation",
    }
    cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        and (
            c.startswith("sent_z_")
            or c.startswith("xs_rank_")     # whitelist before risky-keyword check
            or (
                not any(k in c.lower() for k in risky)
                and (c in exact or any(c.startswith(p) for p in allowed_pfx))
            )
        )
    ]
    # Match train.py: drop market-wide constants when the settings flag is on.
    # See settings.EXCLUDE_MARKET_CONSTANTS for full reasoning.  Must stay in
    # sync with train._select_feature_cols or the candidate set will diverge
    # between training and backtest, breaking the pruned-column reuse contract.
    if EXCLUDE_MARKET_CONSTANTS:
        cols = [c for c in cols if not any(c.startswith(p) for p in _MARKET_CTX_PREFIXES_BT)]
    return cols


# Same protected market-context prefixes as train._MARKET_CTX_PREFIXES.
# Kept in sync manually — both files compute macro_base_idx from this list.
_MARKET_CTX_PREFIXES_BT = ("macro_", "regime", "spy_", "qqq_", "vix_")


def _get_pruned_feature_indices_bt(
    importances: np.ndarray,
    n_base_feats: int,
    n_ticker_dummies: int,
    top_k: int,
    sent_idx: np.ndarray | None = None,
    n_sent_keep: int = 16,   # raised from 3 — kept in sync with train.py (IC report: 16/16 sentiment alive)
    macro_idx: np.ndarray | None = None,
    n_macro_keep: int = 10,
    xs_idx: np.ndarray | None = None,
    n_xs_keep: int = 30,     # raised from 12 — kept in sync with train.py (IC report: ~30 xs_rank alive)
) -> np.ndarray:
    """
    Backtest mirror of train._get_pruned_feature_indices.

    At each walk-forward block we train a quick 100-tree model on all features,
    read gain importances, and keep only the top-K base features plus all ticker
    dummies.  The final block model is then retrained on just those columns.

    Sentiment protection: even if sentiment features rank below top_k globally,
    the top-n_sent_keep of them (by gain importance) are always included.
    This prevents the pruner from silently discarding all news signal.
    Ticker dummies are always kept unconditionally (same as before).

    PLAIN ENGLISH: The model is allowed to vote on which technical features to
    keep. But sentiment features get reserved seats — we force at least 3 of
    the strongest sentiment features into every block's model regardless of the
    vote.  This is because sentiment has lower absolute gain (it only matters
    occasionally) but when it matters it matters a lot.

    Args
    ----
    importances   : feature_importances_ from the initial 100-tree model.
    n_base_feats  : Base (non-dummy) feature count.
    n_ticker_dummies : Number of ticker one-hot columns appended after base.
    top_k         : How many base features to retain via gain ranking.
    sent_idx      : Indices within [0, n_base_feats) that are sent_z_ columns.
                    Pass None to skip sentiment protection.
    n_sent_keep   : How many of those sentiment features to force-include.

    Returns
    -------
    Sorted array of integer column indices to keep.
    """
    base_imp  = importances[:n_base_feats]

    # Standard top-K selection by gain importance
    top_k_idx = set(np.argsort(base_imp)[-min(top_k, n_base_feats):].tolist())

    # Force-include the top-n_sent_keep sentiment features by their own gain
    # importance, regardless of global rank.  This mirrors how ticker dummies
    # are always kept — except we rank within the sentiment group first.
    if sent_idx is not None and len(sent_idx) > 0:
        valid_sent = sent_idx[sent_idx < n_base_feats]
        if len(valid_sent) > 0:
            # Sort sentiment indices by their gain importance (descending)
            sent_by_imp = valid_sent[np.argsort(base_imp[valid_sent])[::-1]]
            for idx in sent_by_imp[:n_sent_keep]:
                top_k_idx.add(int(idx))

    # Force-include the strongest macro/regime/market-context features.  Same
    # rationale as sentiment: gain importance under the excess-return target
    # demotes features that move all stocks together, so without protection the
    # model loses access to yield curve, credit spread, regime, etc.  Verified
    # missing in latest pruning: 0 of 30 base features kept were macro/regime,
    # which is the leading hypothesis for OOS Direction AUC = 0.51.
    if macro_idx is not None and len(macro_idx) > 0:
        valid_macro = macro_idx[macro_idx < n_base_feats]
        if len(valid_macro) > 0:
            macro_by_imp = valid_macro[np.argsort(base_imp[valid_macro])[::-1]]
            for idx in macro_by_imp[:n_macro_keep]:
                top_k_idx.add(int(idx))

    # Cross-sectional rank features: same protection logic as macro/sentiment.
    # These are the strongest theoretical discriminators for a cross-sectional
    # ranker (rank-within-sector by momentum / vol / drawdown / RSI).  Force-keep
    # the top-N each block so the model always has rank-based signal.
    if xs_idx is not None and len(xs_idx) > 0:
        valid_xs = xs_idx[xs_idx < n_base_feats]
        if len(valid_xs) > 0:
            xs_by_imp = valid_xs[np.argsort(base_imp[valid_xs])[::-1]]
            for idx in xs_by_imp[:n_xs_keep]:
                top_k_idx.add(int(idx))

    dummy_idx = np.arange(n_base_feats, n_base_feats + n_ticker_dummies)
    return np.sort(np.unique(np.concatenate([list(top_k_idx), dummy_idx])))


def _load_ticker_for_pooled(ticker: str, common_features: list[str]):
    """
    Load one ticker's data and return pre-computed XGB feature matrix + labels.

    Rolling transforms are applied here (once per ticker per backtest run)
    so the walk-forward loop can cheaply slice by row index.  Slicing is safe
    because rolling windows only look backward.

    Returns
    -------
    (dates_array, X_full, y_dir, y_ret, open_prices, close_prices, y_dir_h5)
    or None on failure.

    y_dir_h5 is the 5-day forward direction label used by the secondary
    short-horizon model in the multi-horizon ensemble.  Simple binary:
    1 = price is higher 5 trading days later, 0 = flat or lower.
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
    # Continuous forward excess return target for XGBRanker (same as train.py).
    # The ranker optimises pairwise ordering within each date — this is the
    # forward value it tries to rank correctly.
    y_rank = (stock_fwd - spy_fwd).to_numpy(dtype=np.float32)

    for col in common_features:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Capture feat_cols (second return of build_xgb_matrix) so the caller can
    # identify which feature indices correspond to sentiment columns and protect
    # them from being pruned out by gain-importance selection.
    X, feat_cols = build_xgb_matrix(df, common_features)
    open_prices  = df["Open"].values  if "Open"  in df.columns else df["Close"].values
    close_prices = df["Close"].values

    # 5-day forward direction label for the secondary short-horizon model.
    # Simple binary: 1 if close is higher 5 days later, 0 otherwise.
    # The h20 valid-mask already excludes the last 20 rows, so these 5-day
    # futures are always available (they look at most 5 rows ahead and we
    # have at least 20 rows of future data available for every included row).
    fwd_5d      = pd.Series(close_prices).pct_change(5).shift(-5).fillna(0.0).values
    y_dir_h5    = (fwd_5d > 0.0).astype(np.int64)

    return (
        np.array(df.index, dtype="datetime64[ns]"),
        X,
        y_dir,
        y_ret,
        open_prices,
        close_prices,
        y_dir_h5,
        feat_cols,   # list[str] — column names of X; used to identify sent_z_ indices
        y_rank,      # continuous forward excess return for XGBRanker
    )


def walk_forward_predictions_pooled(
    tickers: list[str],
    min_total_train_rows: int = 2000,
    test_block: int = 126,
    blend_weights: tuple[float, float] | None = None,
    training_only_tickers: list[str] | None = None,
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
    h20_weight, h5_weight = blend_weights or (MULTI_HORIZON_H20_WEIGHT, MULTI_HORIZON_H5_WEIGHT)
    weight_sum = float(h20_weight + h5_weight)
    if not np.isfinite(weight_sum) or abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"blend_weights must sum to 1.0, got {blend_weights}")

    # ── 1. Determine common feature set across all tickers ────────────────────
    # We check prediction tickers first, then training-only (survivorship) tickers.
    # Both groups must share the same common feature set so the matrix aligns.
    _training_only_set: set[str] = set(training_only_tickers or [])
    all_candidate_tickers = list(tickers) + [
        t for t in (training_only_tickers or []) if t not in tickers
    ]
    feature_sets: list[set[str]] = []
    valid_tickers: list[str] = []      # prediction tickers (will generate signals)
    fit_only_tickers: list[str] = []   # training-only (crashed/delisted) — no signals
    for ticker in all_candidate_tickers:
        path = os.path.join(DATA_DIR, f"{ticker}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        # Same fix as train.py: tickers without historical sentiment coverage
        # produce ZERO sent_z_* columns, which kills the entire family in the
        # cross-ticker intersection.  feature_ic_report shows sent_z_ has
        # mean |t|=4.49 — second-strongest family — so dropping it is
        # expensive.  Force-create canonical sent_z_* columns as zeros for
        # tickers that lack them, so intersection preserves the family.
        from pipeline_shared import RAW_SENTIMENT_COLUMNS  # local import — keeps imports clean
        for _raw_col in RAW_SENTIMENT_COLUMNS:
            _z_col = f"sent_z_{_raw_col}"
            if _z_col not in df.columns:
                df[_z_col] = 0.0
        cols = _select_feature_cols_bt(df)
        if cols:
            feature_sets.append(set(cols))
            if ticker in _training_only_set:
                fit_only_tickers.append(ticker)
            else:
                valid_tickers.append(ticker)
    if not valid_tickers:
        return {}
    common_features = sorted(feature_sets[0].intersection(*feature_sets[1:]))
    if not common_features:
        return {}
    print(
        f"[pooled-bt] Common features: {len(common_features)}  "
        f"Prediction tickers: {len(valid_tickers)}  "
        f"Training-only (survivorship) tickers: {len(fit_only_tickers)}"
    )

    # ── 2. Preload all ticker data (rolling transforms applied once) ───────────
    ticker_data: dict[str, tuple] = {}
    for ticker in valid_tickers + fit_only_tickers:
        result = _load_ticker_for_pooled(ticker, common_features)
        if result is not None:
            ticker_data[ticker] = result
    valid_tickers = [t for t in valid_tickers if t in ticker_data]
    fit_only_tickers = [t for t in fit_only_tickers if t in ticker_data]
    if len(valid_tickers) < 2:
        return {}

    # all_fit_tickers: every ticker whose data enters the model (including crashed ones).
    # ticker_classes / ticker_to_idx / n_tickers must reflect ALL fit tickers so that
    # one-hot dummy columns are consistent between training and prediction rows.
    all_fit_tickers: list[str] = valid_tickers + fit_only_tickers
    ticker_classes = sorted(all_fit_tickers)
    ticker_to_idx  = {t: i for i, t in enumerate(ticker_classes)}
    n_tickers      = len(ticker_classes)
    n_base_feats   = ticker_data[valid_tickers[0]][1].shape[1]

    # Identify which base-feature indices correspond to sentiment columns.
    # feat_cols is index 7 of the _load_ticker_for_pooled return tuple.
    # These indices are passed to _get_pruned_feature_indices_bt so it can
    # always include the top-N sentiment features by gain importance, even if
    # they don't rank in the global top-K.  This prevents the pruner from
    # silently dropping all sentiment signal every block.
    _feat_cols_base: list[str] = ticker_data[valid_tickers[0]][7]
    sent_base_idx = np.array(
        [i for i, c in enumerate(_feat_cols_base) if "sent_z_" in c],
        dtype=int,
    )
    # Macro / regime / market-context column indices.  Without this protection,
    # the per-block gain pruner silently drops all market-wide context (yield
    # curve, credit spread, regime, SPY/VIX state).  Confirmed in the latest
    # post-train pooled_scaler.pkl: 0 of 30 base features kept were macro/regime.
    macro_base_idx = np.array(
        [
            i for i, c in enumerate(_feat_cols_base)
            if any(c.startswith(p) for p in _MARKET_CTX_PREFIXES_BT)
        ],
        dtype=int,
    )
    xs_base_idx = np.array(
        [i for i, c in enumerate(_feat_cols_base) if c.startswith("xs_rank_")],
        dtype=int,
    )
    print(
        f"[pooled-bt] Sentiment feature indices: {len(sent_base_idx)} cols  "
        f"Macro/regime indices: {len(macro_base_idx)} cols  "
        f"XS-rank indices: {len(xs_base_idx)} cols  (protected where enabled)"
    )

    # Identify which base-feature column holds the market regime label.
    # 'regime' is computed by pipeline_shared.py and stored in the parquet as:
    #   +1 = bull (SPY above 200MA AND VIX < 20)
    #    0 = neutral
    #   -1 = bear  (SPY below 200MA OR VIX >= 28)
    # We record the column index here once (not per-block) because all tickers
    # share the same _feat_cols_base layout (common_features are intersection-selected).
    _regime_col_in_base: int | None = next(
        (i for i, c in enumerate(_feat_cols_base) if c == "regime"), None
    )
    if REGIME_CONDITIONAL_MODELS:
        if _regime_col_in_base is not None:
            print(
                f"[pooled-bt] Regime column found at base index {_regime_col_in_base} "
                f"('{_feat_cols_base[_regime_col_in_base]}') — regime-conditional routing enabled"
            )
        else:
            print("[pooled-bt] WARNING: 'regime' column not found in feature list — regime-conditional routing disabled")

    # ── 3. Build global date timeline ─────────────────────────────────────────
    all_dates_set: set = set()
    for dates, *_ in ticker_data.values():
        all_dates_set.update(dates.tolist())
    timeline = sorted(all_dates_set)   # all unique dates across all tickers

    # Convert timeline to numpy datetime64[ns] so comparisons work uniformly
    timeline_np = np.array(timeline, dtype="datetime64[ns]")

    # Find the first date where total rows (across ALL fit tickers, incl. training-only)
    # >= min_total_train_rows.  Survivorship tickers count toward the minimum because
    # they DO contribute training rows — that is the whole point of adding them.
    start_idx = None
    for i, dt_np in enumerate(timeline_np):
        total = sum(
            int((ticker_data[t][0] <= dt_np).sum())
            for t in all_fit_tickers
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
    # Accumulate calibrated probabilities and labels across blocks to run
    # calibration_stability check after the loop.
    _all_cal_probs: list[float] = []
    _all_cal_labels: list[int] = []

    while step < last_valid_idx:
        block_end = min(step + test_block, last_valid_idx)
        oos_dates_np = timeline_np[step:block_end]
        # Embargo: exclude RETURN_HORIZON_DAYS rows immediately before step
        cutoff_idx  = max(0, step - RETURN_HORIZON_DAYS)
        cutoff_date = timeline_np[cutoff_idx]

        # Build pooled train + calib sets from all tickers up to cutoff
        train_X_parts, train_y_dir_parts, train_y_ret_parts = [], [], []
        train_y_dir_h5_parts: list[np.ndarray] = []   # 5-day direction labels
        train_y_rank_parts: list[np.ndarray] = []      # continuous excess return for ranker
        train_dates_parts: list[np.ndarray] = []       # dates for ranker group building
        calib_X_parts, calib_y_dir_parts, calib_y_ret_parts = [], [], []
        calib_y_dir_h5_parts: list[np.ndarray] = []   # 5-day direction labels
        calib_rows_by_ticker: dict[str, np.ndarray] = {}

        # Assemble training data from ALL fit tickers (prediction + survivorship).
        # Survivorship tickers contribute pre-crash patterns to X_train but will
        # never appear in the OOS prediction loop below (only valid_tickers predict).
        for ticker in all_fit_tickers:
            dates, X, y_dir, y_ret, _, _, y_dir_h5, _, y_rank_tk = ticker_data[ticker]
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
            train_y_dir_h5_parts.append(y_dir_h5[train_rows_strided])
            # Ranker: collect base-feature-only X, rank target, and dates
            # (no ticker dummies — ranker learns cross-sectional patterns).
            train_y_rank_parts.append(y_rank_tk[train_rows_strided])
            train_dates_parts.append(dates[train_rows_strided])

            if len(calib_rows) > 0:
                calib_rows_by_ticker[ticker] = calib_rows
                cal_X = np.concatenate(
                    [X[calib_rows],
                     np.repeat(dummy, len(calib_rows), axis=0)], axis=1
                )
                calib_X_parts.append(cal_X)
                calib_y_dir_parts.append(y_dir[calib_rows])
                calib_y_ret_parts.append(y_ret[calib_rows])
                calib_y_dir_h5_parts.append(y_dir_h5[calib_rows])

        if not train_X_parts or not calib_X_parts:
            step = block_end
            continue

        X_train_raw    = np.concatenate(train_X_parts, axis=0)
        y_dir_train    = np.concatenate(train_y_dir_parts)
        y_ret_train    = np.concatenate(train_y_ret_parts)
        y_dir_h5_train = np.concatenate(train_y_dir_h5_parts)
        X_cal_raw      = np.concatenate(calib_X_parts, axis=0)
        y_dir_cal      = np.concatenate(calib_y_dir_parts)
        y_ret_cal      = np.concatenate(calib_y_ret_parts)
        y_dir_h5_cal   = np.concatenate(calib_y_dir_h5_parts)

        # Extract per-row regime labels from X_train_raw BEFORE pruning / scaling.
        # The regime column sits in the base-feature portion (index 0..n_base_feats-1),
        # so X_train_raw[:, _regime_col_in_base] gives raw -1/0/+1 values.
        # These boolean masks are used below to train separate bull/bear models.
        _block_bull_mask: np.ndarray | None = None   # True for bull-regime training rows
        _block_bear_mask: np.ndarray | None = None   # True for bear-regime training rows
        _n_bull_blk = 0
        _n_bear_blk = 0
        if REGIME_CONDITIONAL_MODELS and _regime_col_in_base is not None:
            _regime_raw = X_train_raw[:, _regime_col_in_base]   # raw regime values
            _block_bull_mask = _regime_raw > 0.5    # +1 rows
            _block_bear_mask = _regime_raw < -0.5   # -1 rows
            _n_bull_blk = int(np.sum(_block_bull_mask))
            _n_bear_blk = int(np.sum(_block_bear_mask))

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

        # ── Per-block XGB gain feature pruning (OOS-clean) ────────────────────
        # PREVIOUS APPROACH (removed): loaded feature_pruning_keep_idx from
        # models/pooled_scaler.pkl, which train.py selected on the FULL historical
        # dataset.  That introduced look-ahead: a block whose training window ends in
        # 2018 was using feature importance computed on data through 2024.  It aligned
        # live trading but made the research backtest non-causal.
        #
        # CURRENT APPROACH: fit a fast 100-tree model on this block's training data
        # only, rank features by XGB gain importance, and keep top-K.  Each block
        # therefore uses only features known to be informative up to its own cutoff.
        # The set is noisier fold-to-fold but OOS-clean.
        _init_for_pruning = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        _init_for_pruning.fit(X_train_raw, y_dir_train, verbose=False)
        block_keep_idx = _get_pruned_feature_indices_bt(
            _init_for_pruning.feature_importances_,
            n_base_feats=n_base_feats,          # base (non-dummy) feature count
            n_ticker_dummies=n_tickers,          # one-hot ticker dummies always kept
            top_k=FEATURE_IMPORTANCE_TOP_K,
            sent_idx=sent_base_idx,             # protect top-3 sentiment features from pruning
            macro_idx=macro_base_idx,           # protect top-10 macro/regime/market-context features
        )
        # Slice raw matrices to the pruned column set, then fit the block scaler.
        X_train_raw_p = X_train_raw[:, block_keep_idx]
        X_cal_raw_p   = X_cal_raw[:, block_keep_idx]
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw_p)
        X_cal   = scaler.transform(X_cal_raw_p)

        # ── Train block model with (optionally) tuned hyperparameters ────────
        dir_model, ret_model = build_models(
            X_train, y_dir_train, X_cal, y_dir_cal,
            y_ret_train, y_ret_cal,
            param_overrides=_bt_param_overrides,
        )

        # ── Train regime-conditional direction models (bull / bear) ───────────
        # If REGIME_CONDITIONAL_MODELS is on and each regime has enough rows,
        # train two specialised direction classifiers on the same pruned+scaled
        # feature matrix (X_train) but restricted to bull or bear rows only.
        # At prediction time we route each OOS row to the matching model.
        # Falls back to dir_model (combined) if either regime is too small.
        dir_model_bull: xgb.XGBClassifier | None = None
        dir_model_bear: xgb.XGBClassifier | None = None
        if (
            REGIME_CONDITIONAL_MODELS
            and _block_bull_mask is not None
            and _n_bull_blk >= REGIME_MIN_TRAIN_ROWS
            and _n_bear_blk >= REGIME_MIN_TRAIN_ROWS
        ):
            # X_train is the pruned+scaled matrix from this block.
            # _block_bull_mask / _block_bear_mask select rows from X_train_raw,
            # which has the same row count as X_train (masks are row-selectors).
            dir_model_bull, _ = build_models(
                X_train[_block_bull_mask], y_dir_train[_block_bull_mask],
                X_cal, y_dir_cal,
                y_ret_train[_block_bull_mask], y_ret_cal,
                param_overrides=_bt_param_overrides,
            )
            dir_model_bear, _ = build_models(
                X_train[_block_bear_mask], y_dir_train[_block_bear_mask],
                X_cal, y_dir_cal,
                y_ret_train[_block_bear_mask], y_ret_cal,
                param_overrides=_bt_param_overrides,
            )
        else:
            if REGIME_CONDITIONAL_MODELS and _regime_col_in_base is not None:
                pass  # Not enough rows this block — silently fall back to combined model

        # ── Train secondary 5-day direction model (multi-horizon ensemble) ────
        # h5 gets its own pruned matrix with sentiment protection. This activates
        # news for the short horizon without forcing sentiment into h20/ret.
        h5_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_lambda=10.0,
            tree_method="hist",
            early_stopping_rounds=20,
            random_state=42,
        )
        # Class-balance weights for h5: same fix as train.py.
        # h5 UP rate (~55% bull-market) with no correction → model outputs ~0.55
        # for all rows → blend always ≥ 0.5 → all predictions UP.
        _n_pos_h5 = max(int(np.sum(y_dir_h5_train == 1)), 1)
        _n_neg_h5 = max(int(np.sum(y_dir_h5_train == 0)), 1)
        _class_w_h5 = np.where(
            y_dir_h5_train == 1, float(_n_neg_h5) / _n_pos_h5, 1.0
        ).astype(np.float32)
        if H5_SENTIMENT_FEATURES_ENABLED:
            _init_h5_for_pruning = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                tree_method="hist",
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
            _init_h5_for_pruning.fit(X_train_raw, y_dir_h5_train, sample_weight=_class_w_h5, verbose=False)
            h5_block_keep_idx = _get_pruned_feature_indices_bt(
                _init_h5_for_pruning.feature_importances_,
                n_base_feats=n_base_feats,
                n_ticker_dummies=n_tickers,
                top_k=FEATURE_IMPORTANCE_TOP_K,
                sent_idx=sent_base_idx,
                macro_idx=macro_base_idx,    # protect macro/regime features for h5 too
            )
            X_train_h5_raw_p = X_train_raw[:, h5_block_keep_idx]
            X_cal_h5_raw_p = X_cal_raw[:, h5_block_keep_idx]
            h5_scaler = StandardScaler()
            X_train_h5 = h5_scaler.fit_transform(X_train_h5_raw_p)
            X_cal_h5 = h5_scaler.transform(X_cal_h5_raw_p)
        else:
            h5_block_keep_idx = block_keep_idx
            h5_scaler = scaler
            X_train_h5 = X_train
            X_cal_h5 = X_cal
        h5_model.fit(
            X_train_h5, y_dir_h5_train,
            eval_set=[(X_cal_h5, y_dir_h5_cal)],
            sample_weight=_class_w_h5,
            verbose=False,
        )

        cal_probs_raw = dir_model.predict_proba(X_cal)[:, 1]
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
        cal_ret_probs = ret_model.predict_proba(X_cal)[:, :N_RETURN_BINS]
        cal_expected_return = (cal_ret_probs * RETURN_BIN_CENTRES[:cal_ret_probs.shape[1]]).sum(axis=1) * 100.0

        # Blended calibration probabilities: same formula used for OOS direction_vote.
        # Meta-label must ask "was the ENSEMBLE direction correct?" not "was calibrated
        # h20 correct?"  If we used cal_probs (h20-only) here, the meta-model would
        # learn to predict h20 correctness but then be applied to blended signals —
        # a train/apply mismatch that degrades meta-model precision.
        cal_probs_h5_blk  = h5_model.predict_proba(X_cal_h5)[:, 1]
        cal_probs_blend    = h20_weight * cal_probs_raw + h5_weight * cal_probs_h5_blk
        conformal_qhat = compute_conformal_qhat(
            cal_probs_blend,
            y_dir_cal,
            alpha=CONFORMAL_ALPHA,
            min_rows=CONFORMAL_MIN_CALIBRATION_ROWS,
        )
        cal_primary_pred   = (cal_probs_blend >= 0.5).astype(int)
        y_meta_cal         = (cal_primary_pred == y_dir_cal).astype(int)
        meta_model = None
        if META_LABELING_ENABLED and len(np.unique(y_meta_cal)) >= 2 and len(y_meta_cal) >= 500:
            meta_model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_lambda=5.0,
                tree_method="hist",
                random_state=42,
            )
            meta_model.fit(
                build_meta_matrix(X_cal, cal_probs_blend, cal_probs, cal_expected_return),
                y_meta_cal,
                verbose=False,
            )
        _all_cal_probs.extend(cal_probs.tolist())
        _all_cal_labels.extend(y_dir_cal.tolist())
        # Per-ticker confidence-bucket reports removed: assign_signal_quality() is
        # no longer called here.  signal_quality is now set purely by rank position
        # inside apply_cross_sectional_ranking() — rank 0 → HIGH, rank 1 → MEDIUM,
        # rank 2+ → LOW.  Bucket reports were the only consumer of this computation,
        # so the whole per-ticker calib loop is gone.  Calibration stability check
        # still uses _all_cal_probs / _all_cal_labels accumulated above.

        # ── Train XGBRanker for this block (gated by flag) ────────────────────
        # When POOLED_RANKER_EXPERIMENT_ENABLED is on, a separate XGBRanker is
        # trained per walk-forward block alongside the existing classifier.
        # The ranker predicts continuous forward excess return and outputs a raw
        # score per (date, ticker) row.  This score is stored in the records dict
        # and used by apply_cross_sectional_ranking as the ranking signal — it
        # replaces the z-scored expected_return + direction_edge formula with a
        # model that directly optimises pairwise stock ordering.
        #
        # KEY DIFFERENCES from the classifier path:
        #   - NO ticker dummies (prevents memorising "AAPL always ranks 3rd")
        #   - Lower complexity (depth=3, n_est=200, heavy reg) to prevent overfit
        #   - Continuous target (forward excess return), not binary UP/DOWN
        #   - group array tells XGBoost which rows belong to the same date
        block_ranker = None
        block_ranker_scaler = None
        block_ranker_keep_idx = None
        if POOLED_RANKER_EXPERIMENT_ENABLED and train_y_rank_parts:
            y_rank_train_blk = np.concatenate(train_y_rank_parts)
            dates_train_blk  = np.concatenate(train_dates_parts)

            # Sort by date for group building (should already be ~sorted by
            # construction, but make it explicit for the ranker contract).
            sort_idx = np.argsort(dates_train_blk)
            y_rank_train_blk = y_rank_train_blk[sort_idx]
            dates_train_blk  = dates_train_blk[sort_idx]

            train_groups_blk = build_rank_groups_from_dates(dates_train_blk)
            # Base features only — strip ticker dummies from X_train_raw.
            # Match train.py ranker path: scale base features first, prune on
            # a low-complexity init ranker, then fit the final ranker on the
            # scaled/pruned columns.  The previous backtest path used raw,
            # unpruned features, so walk-forward results were not comparable
            # to the train-time IC signal.
            X_train_rank_raw_blk = X_train_raw[sort_idx, :n_base_feats]
            if sum(train_groups_blk) == len(X_train_rank_raw_blk):
                ranker_params_blk = {
                    "objective": "rank:pairwise",
                    "eval_metric": "rmse",
                    "n_estimators": 200,
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "subsample": 0.6,
                    "colsample_bytree": 0.4,
                    "min_child_weight": 20,
                    "gamma": 1.0,
                    "reg_alpha": 1.0,
                    "reg_lambda": 10.0,
                    "tree_method": "hist",
                    "random_state": 42,
                }
                block_ranker_scaler = StandardScaler()
                X_train_rank_sc_full = block_ranker_scaler.fit_transform(X_train_rank_raw_blk)

                init_ranker_blk = xgb.XGBRanker(**{**ranker_params_blk, "n_estimators": 100})
                init_ranker_blk.fit(
                    X_train_rank_sc_full,
                    y_rank_train_blk,
                    group=train_groups_blk,
                    verbose=False,
                )
                block_ranker_keep_idx = _get_pruned_feature_indices_bt(
                    init_ranker_blk.feature_importances_,
                    n_base_feats=n_base_feats,
                    n_ticker_dummies=0,
                    top_k=FEATURE_IMPORTANCE_TOP_K,
                    sent_idx=sent_base_idx,
                    macro_idx=macro_base_idx,
                    xs_idx=xs_base_idx,
                )
                X_train_rank_blk = X_train_rank_sc_full[:, block_ranker_keep_idx]

                block_ranker = xgb.XGBRanker(**ranker_params_blk)
                block_ranker.fit(
                    X_train_rank_blk,
                    y_rank_train_blk,
                    group=train_groups_blk,
                    verbose=False,
                )

        # OOS predictions for each ticker on each date in the block. Eligibility
        # is applied later inside apply_cross_sectional_ranking(), after ranked
        # trade outcomes mature by exit_date.
        for oos_dt_np in oos_dates_np:
            for ticker in valid_tickers:
                dates, X, y_dir, y_ret, open_prices, close_prices, _, _, _ = ticker_data[ticker]
                row_hits = np.where(dates == oos_dt_np)[0]
                if len(row_hits) == 0:
                    continue
                j = int(row_hits[0])

                dummy = np.zeros((1, n_tickers), dtype=np.float64)
                dummy[0, ticker_to_idx[ticker]] = 1.0
                X_row_full = np.concatenate([X[j:j+1], dummy], axis=1)
                # Apply the per-block pruned column set (always set; never None after
                # the fix that replaced pkl-based global pruning with per-block pruning).
                if block_keep_idx is not None:
                    X_row_full = X_row_full[:, block_keep_idx]
                X_row = scaler.transform(X_row_full)

                X_row_h5_full = np.concatenate([X[j:j+1], dummy], axis=1)
                X_row_h5 = h5_scaler.transform(X_row_h5_full[:, h5_block_keep_idx])

                # ── Multi-horizon ensemble blend ──────────────────────────
                # h20 primary model captures medium-term regime patterns;
                # h5 secondary model captures short-term momentum.
                # Weighted sum gives p_up_raw — used for ranking and direction.
                # ── Regime-conditional model routing (SOFT BLEND) ────────────
                # Hard-routing to a specialist (e.g. all-bear or all-bull) flipped
                # rank_spearman from +0.067 to -0.083 in the 28-ticker backtest:
                # the two specialists output probabilities on different scales,
                # so the cross-sectional ordering of ranks broke.
                #
                # Soft blend keeps the combined model as a stabilising anchor:
                #   bull-regime row → 0.6 * bull_specialist + 0.4 * combined
                #   bear-regime row → 0.6 * bear_specialist + 0.4 * combined
                #   neutral / specialist missing → 100% combined model
                # The combined model is always evaluated; specialists only tilt.
                # X is the RAW (unscaled, pre-dummy) base feature matrix;
                # _regime_col_in_base is the column index of the 'regime' feature.
                _oos_regime = (
                    float(X[j, _regime_col_in_base])
                    if (REGIME_CONDITIONAL_MODELS and _regime_col_in_base is not None)
                    else 0.0
                )
                _p_combined = float(dir_model.predict_proba(X_row)[0, 1])   # ranking anchor
                if _oos_regime > 0.5 and dir_model_bull is not None:
                    _p_specialist = float(dir_model_bull.predict_proba(X_row)[0, 1])
                    p_up_h20 = 0.6 * _p_specialist + 0.4 * _p_combined
                elif _oos_regime < -0.5 and dir_model_bear is not None:
                    _p_specialist = float(dir_model_bear.predict_proba(X_row)[0, 1])
                    p_up_h20 = 0.6 * _p_specialist + 0.4 * _p_combined
                else:
                    p_up_h20 = _p_combined   # neutral regime or specialist unavailable
                p_up_h5  = float(h5_model.predict_proba(X_row_h5)[0, 1])
                p_up_raw = h20_weight * p_up_h20 + h5_weight * p_up_h5
                # Calibration is applied to h20 only — calibrator was fit on h20 probs.
                p_up = float(calibrate_p_up(calibrator, p_up_h20))

                # ── Ranker score for cross-sectional ranking ─────────────────
                # Compute before direction/exits so ranker-mode trades get LONG
                # triple-barrier exits.  Previously this happened after the
                # barrier scan, so a ranker-selected LONG could inherit a SHORT
                # exit path from the classifier's p_up_raw vote.
                _ranker_score = float("nan")
                if (
                    block_ranker is not None
                    and block_ranker_scaler is not None
                    and block_ranker_keep_idx is not None
                ):
                    X_base_row = X[j:j+1, :n_base_feats]  # base features only
                    X_rank_row = block_ranker_scaler.transform(X_base_row)[:, block_ranker_keep_idx]
                    _ranker_score = float(block_ranker.predict(X_rank_row)[0])
                _ranker_active = POOLED_RANKER_EXPERIMENT_ENABLED and np.isfinite(_ranker_score)

                # direction_vote must come from the blended p_up_raw, not calibrated
                # h20 alone.  Using calibrated h20 for direction_vote was a bug: with
                # asymmetric barriers (or a prior imbalance run) the calibrator maps most
                # h20 probs below 0.5, making direction_vote always SHORT even when the
                # ensemble is bullish.  p_up_raw is symmetric by design — it averages
                # two uncalibrated probabilities so 0.5 is the true midpoint.
                direction_vote  = "LONG" if (_ranker_active or p_up_raw >= 0.5) else "SHORT"
                # Confidence reported as how far p_up_raw is from 0.5 (range 50–100).
                confidence = min(max(max(p_up_raw, 1.0 - p_up_raw) * 100.0, 50.0), 99.0)
                conformal_info = conformal_prediction_info(p_up_raw, conformal_qhat)
                p_down = 1.0 - p_up_raw
                ret_probs = ret_model.predict_proba(X_row)[0, :N_RETURN_BINS]
                expected_return = float((ret_probs * RETURN_BIN_CENTRES).sum()) * 100.0
                meta_confidence = 1.0
                if meta_model is not None:
                    meta_X = build_meta_matrix(
                        X_row,
                        np.array([p_up_raw], dtype=float),
                        np.array([p_up], dtype=float),
                        np.array([expected_return], dtype=float),
                    )
                    meta_confidence = float(meta_model.predict_proba(meta_X)[0, 1])
                # signal_quality is set to "UNRANKED" here and overridden by
                # apply_cross_sectional_ranking() using rank position (0=HIGH,
                # 1=MEDIUM, 2+=LOW).  Confidence-bucket bucketing removed — it
                # was pure noise when probabilities collapse around 0.5.
                signal_quality  = "UNRANKED"

                entry_j  = min(j + 1, len(dates) - 1)
                # Time-based exit boundary (10 trading days out)
                time_exit_j = min(entry_j + RETURN_HORIZON_DAYS, len(dates) - 1)

                # ── Triple-barrier exit ───────────────────────────────────────
                # The model was trained with triple-barrier labels:
                #   UP   = price hits +PT×ATR first  (profit target)
                #   DOWN = price hits -SL×ATR first  (stop-loss)
                #   FLAT = neither hit in 10 days   (time exit)
                # Without this matching exit logic, longs hold through crashes
                # and the backtest dramatically understates real-world stop-losses.
                #
                # ATR = average absolute close-to-close move over 14 days.
                # We use the close prices available in ticker_data (no High/Low).
                atr_window = 14
                if j >= atr_window:
                    # |close[i] - close[i-1]| for the 14 days ending at j
                    recent = close_prices[j - atr_window: j + 1]
                    atr_price = float(np.mean(np.abs(np.diff(recent))))
                else:
                    # Not enough history — fall back to 2% of current close
                    atr_price = float(close_prices[j]) * 0.02

                entry_price  = float(open_prices[entry_j])   # enter at next-day open
                # Barrier prices depend on direction.
                # LONG:  stop BELOW entry (lose if price falls), target ABOVE.
                # SHORT: stop ABOVE entry (lose if price rises), target BELOW.
                if direction_vote == "LONG":
                    stop_price   = entry_price - TRIPLE_BARRIER_SL_MULT * atr_price
                    target_price = entry_price + TRIPLE_BARRIER_PT_MULT * atr_price
                else:  # SHORT
                    stop_price   = entry_price + TRIPLE_BARRIER_SL_MULT * atr_price
                    target_price = entry_price - TRIPLE_BARRIER_PT_MULT * atr_price

                # Scan forward day-by-day to see if a barrier is touched first
                actual_exit_j      = time_exit_j
                actual_exit_price  = float(close_prices[time_exit_j])
                actual_exit_reason = "time_exit"
                for k in range(entry_j + 1, time_exit_j + 1):
                    c = float(close_prices[k])
                    if direction_vote == "LONG":
                        if c <= stop_price:
                            # Stop-loss hit: exit at stop price (conservative)
                            actual_exit_j      = k
                            actual_exit_price  = stop_price
                            actual_exit_reason = "stop_loss"
                            break
                        if c >= target_price:
                            # Profit target hit: exit at target price
                            actual_exit_j      = k
                            actual_exit_price  = target_price
                            actual_exit_reason = "take_profit"
                            break
                    else:  # SHORT
                        if c >= stop_price:
                            actual_exit_j      = k
                            actual_exit_price  = stop_price
                            actual_exit_reason = "stop_loss"
                            break
                        if c <= target_price:
                            actual_exit_j      = k
                            actual_exit_price  = target_price
                            actual_exit_reason = "take_profit"
                            break

                exit_j   = actual_exit_j
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
                    "horizon_future_close": float(close_prices[time_exit_j]),
                    "exit_future_close": actual_exit_price,
                    "holding_days":      int(exit_j - entry_j),
                    "exit_reason":       actual_exit_reason,
                    "signal":            direction_vote,
                    "direction_vote":    direction_vote,
                    "p_up_raw":          p_up_raw,
                    # Store the individual horizon probabilities so --ablate-blend
                    # can recompute p_up_raw with different weight combos without
                    # re-running the full walk-forward loop.
                    "p_up_h20":          p_up_h20,
                    "p_up_h5":           p_up_h5,
                    "p_up_calibrated":   p_up,
                    "meta_confidence":   meta_confidence,
                    "confidence":        confidence,
                    **conformal_info,
                    "conf_threshold":    threshold,
                    "threshold_used_fallback": bool(threshold_used_fallback),
                    "expected_return":   expected_return,
                    "signal_quality":    signal_quality,
                    "ranker_score":      _ranker_score,
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

    # ── OOS ranking-quality diagnostic (AUC + AP) ────────────────────────────
    # train.py reports AUC on the held-out test slice (single split).  The
    # walk-forward backtest is the real OOS test: every prediction here was
    # produced by a model that never saw the corresponding date.  Computing AUC
    # over ALL backtest predictions (concatenated across tickers and blocks)
    # tells us whether the model has any persistent ranking edge.
    #
    # Two AUCs are reported:
    #   1. Direction AUC — does p_up_raw rank realized stock UP/DOWN moves?
    #      Realized = exit_future_close > open_next.  This is what the trade
    #      actually pays off on.
    #   2. PnL-sign AUC — does p_up_raw rank profitable LONG trades?
    #      Computed only over LONG votes (where p_up_raw > 0.5).
    # Treat AUC < 0.52 as no edge.  AUC ≈ 0.50 means random ranking.
    #
    # IMPORTANT: prints go to STDERR (not stdout).  The main backtest runner
    # wraps this whole function in contextlib.redirect_stdout(io.StringIO())
    # when verbose_output=False, which would otherwise eat these diagnostics.
    # Using stderr makes the AUC lines visible regardless of verbose flag.
    def _auc_print(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        _all_p = []     # raw blended probabilities
        _all_dir = []   # 1 if exit > entry (realized stock UP), else 0
        _long_p = []    # subset for LONG votes only
        _long_pnl_pos = []   # 1 if LONG trade was profitable
        for tk, df_tk in result.items():
            if df_tk.empty:
                continue
            entry_px = df_tk["open_next"].astype(float).values
            exit_px  = df_tk["exit_future_close"].astype(float).values
            p_raw    = df_tk["p_up_raw"].astype(float).values
            dirs     = df_tk["direction_vote"].astype(str).values
            realized_up = (exit_px > entry_px).astype(int)
            _all_p.extend(p_raw.tolist())
            _all_dir.extend(realized_up.tolist())
            # LONG-only PnL sign — exit > entry means profit on a long.
            for i in range(len(p_raw)):
                if dirs[i] == "LONG":
                    _long_p.append(float(p_raw[i]))
                    _long_pnl_pos.append(int(exit_px[i] > entry_px[i]))
        n_total = len(_all_p)
        if n_total >= 50 and len(set(_all_dir)) >= 2:
            _dir_auc = float(roc_auc_score(_all_dir, _all_p))
            _dir_ap  = float(average_precision_score(_all_dir, _all_p))
            _base_up_rate = float(np.mean(_all_dir))
            _auc_print(
                f"[pooled-bt] OOS Direction AUC: {_dir_auc:.4f}  "
                f"AP: {_dir_ap:.4f}  "
                f"(n={n_total}, baseline UP rate={_base_up_rate:.2%})  "
                f"{'[edge]' if _dir_auc >= 0.52 else '[no edge]'}"
            )
        else:
            _auc_print(f"[pooled-bt] OOS Direction AUC: skipped (n={n_total}, classes={len(set(_all_dir))})")
        if len(_long_p) >= 50 and len(set(_long_pnl_pos)) >= 2:
            _long_auc = float(roc_auc_score(_long_pnl_pos, _long_p))
            _long_ap  = float(average_precision_score(_long_pnl_pos, _long_p))
            _long_win = float(np.mean(_long_pnl_pos))
            _auc_print(
                f"[pooled-bt] OOS LONG-trade PnL-sign AUC: {_long_auc:.4f}  "
                f"AP: {_long_ap:.4f}  "
                f"(n={len(_long_p)}, win rate={_long_win:.2%})  "
                f"{'[edge]' if _long_auc >= 0.52 else '[no edge]'}"
            )
    except Exception as _auc_err:
        # Don't crash the backtest pipeline on a metric-computation failure.
        _auc_print(f"[pooled-bt] OOS AUC computation failed: {_auc_err}")

    # ── Calibration stability diagnostic ─────────────────────────────────────
    # Checks whether the isotonic calibration curve stays consistent across
    # walk-forward blocks.  A high KS distance means the calibration is drifting
    # (likely overfitting to each block's training distribution).
    if len(_all_cal_probs) >= 50:
        _cal_stability = check_calibration_stability(
            pd.Series(_all_cal_probs),
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


def load_ticker_eligibility(tickers: list[str]) -> tuple[set[str], dict[str, str]]:
    """Legacy report-based eligibility helper.

    The main pooled research backtest does not call this because
    model_quality_report.csv may contain full-period results from a prior run.
    Causal eligibility is handled inside apply_cross_sectional_ranking().
    """
    def _metric(row: pd.Series, key: str, default: float = 0.0) -> float:
        val = row.get(key, default)
        try:
            if pd.isna(val):
                return default
            return float(val)
        except Exception:
            return default

    universe = {str(t).upper() for t in tickers}
    report = read_quality_report()
    if report.empty or "ticker" not in report.columns:
        return universe, {t: "no_quality_report_fail_open" for t in universe}

    report = report.copy()
    report["ticker"] = report["ticker"].astype(str).str.upper()
    latest = report.drop_duplicates("ticker", keep="first")
    reasons: dict[str, str] = {}
    eligible: set[str] = set()

    for ticker in sorted(universe):
        row = latest[latest["ticker"] == ticker]
        if row.empty:
            eligible.add(ticker)
            reasons[ticker] = "missing_quality_row_fail_open"
            continue

        r = row.iloc[0]
        trades = _metric(r, "walkforward_trades")
        pnl = _metric(r, "walkforward_total_pnl")
        win_rate = _metric(r, "walkforward_win_rate")
        sharpe = _metric(r, "walkforward_sharpe")
        profit_factor = _metric(r, "walkforward_profit_factor")
        dd = _metric(r, "walkforward_max_drawdown_pct")
        failures = []
        if trades < TICKER_ELIGIBILITY_MIN_TRADES:
            failures.append(f"trades {trades:.0f}<{TICKER_ELIGIBILITY_MIN_TRADES}")
        if pnl <= TICKER_ELIGIBILITY_MIN_TOTAL_PNL:
            failures.append(f"pnl {pnl:.2f}<={TICKER_ELIGIBILITY_MIN_TOTAL_PNL:.2f}")
        if win_rate < TICKER_ELIGIBILITY_MIN_WIN_RATE:
            failures.append(f"win_rate {win_rate:.3f}<{TICKER_ELIGIBILITY_MIN_WIN_RATE:.3f}")
        if sharpe < TICKER_ELIGIBILITY_MIN_SHARPE:
            failures.append(f"sharpe {sharpe:.3f}<{TICKER_ELIGIBILITY_MIN_SHARPE:.3f}")
        if profit_factor < TICKER_ELIGIBILITY_MIN_PROFIT_FACTOR:
            failures.append(f"profit_factor {profit_factor:.3f}<{TICKER_ELIGIBILITY_MIN_PROFIT_FACTOR:.3f}")
        if dd < TICKER_ELIGIBILITY_MAX_DRAWDOWN_PCT:
            failures.append(f"drawdown {dd:.2f}%<{TICKER_ELIGIBILITY_MAX_DRAWDOWN_PCT:.2f}%")

        if failures:
            reasons[ticker] = "; ".join(failures)
        else:
            eligible.add(ticker)
            reasons[ticker] = "eligible"

    if not eligible:
        return universe, {t: "eligibility_empty_fail_open" for t in universe}
    return eligible, reasons


def apply_cross_sectional_ranking(
    predictions_by_ticker: dict[str, pd.DataFrame],
    top_n: int,
    mode: str,
    eligible_tickers: set[str] | None = None,
    use_eligibility_filter: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Replace confidence-threshold filtering with cross-sectional ranking.

    At each date, rank all tickers by a NORMALISED score so the ranker picks
    "which ticker is most above its own baseline today" rather than "which
    ticker always has higher raw expected_return."

    Without normalisation the return model learns a static ticker-level bias
    (e.g. MU always scores 0.57%, V always scores 0.64%) so the same 4 tickers
    dominate every single day, rank Spearman → 0, and quality tiers become
    meaningless.

    Normalisation step:
      z_exp_ret   = (exp_ret   - rolling_mean_exp_ret)   / rolling_std_exp_ret
      z_dir_edge  = (dir_edge  - rolling_mean_dir_edge)  / rolling_std_dir_edge
      rank_score  = z_exp_ret + 0.5 * z_dir_edge
    Rolling window = 60 trading days (≈ 3 months).  Only applied once 20+
    observations are available; before that the ticker is ranked at 0.
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

    # ── Simple factor composite: preload factor values from parquets ─────────
    # When SIMPLE_FACTOR_RANKING is on, we bypass all XGBoost-based rank scores
    # and instead rank stocks by a cross-sectional z-score composite of raw
    # factor values (volatility, beta, etc.).  These factors showed OOS rank
    # IC = +0.055 (t=7.85) vs XGBoost's +0.007 (t=1.82).
    _factor_data: dict[str, pd.DataFrame] = {}  # ticker → DataFrame of factor cols
    if SIMPLE_FACTOR_RANKING:
        print("[rank] SIMPLE_FACTOR_RANKING enabled — using raw factor composite instead of XGBoost scores")
        for ticker in updated:
            path = os.path.join(DATA_DIR, f"{ticker}.parquet")
            try:
                pq = pd.read_parquet(path)
                # Extract only the factor columns we need
                available_cols = [c for c in SIMPLE_FACTOR_COLS if c in pq.columns]
                if available_cols:
                    _factor_data[ticker] = pq[available_cols].copy()
            except Exception:
                continue
        print(f"[rank] Loaded factor data for {len(_factor_data)}/{len(updated)} tickers, "
              f"factor cols: {SIMPLE_FACTOR_COLS}")

    # Per-ticker rolling history buffers for normalisation
    _NORM_WINDOW = 60   # days of history used to compute rolling mean/std
    _NORM_MIN    = 20   # minimum observations before normalisation kicks in
    ticker_exp_hist: dict[str, list[float]] = {t: [] for t in updated}
    ticker_dir_hist: dict[str, list[float]] = {t: [] for t in updated}

    # Causal ticker eligibility lives inside ranking, not in model_quality_report.
    # We only add a selected trade to a ticker's eligibility history after its
    # exit_date has passed, so a signal on date T cannot benefit from future PnL.
    ticker_pnl_hist: dict[str, list[float]] = {t: [] for t in updated}
    pending_pnls: dict[str, list[tuple[pd.Timestamp, float]]] = {t: [] for t in updated}
    rank_score_hist: list[float] = []
    rank_pnl_hist: list[float] = []
    pending_rank_outcomes: list[tuple[pd.Timestamp, float, float]] = []

    def _trade_pnl_pct(row: pd.Series, signal: str) -> float | None:
        entry = float(row.get("open_next", 0.0) or 0.0)
        exit_price = float(row.get("exit_future_close", 0.0) or 0.0)
        if entry <= 0 or exit_price <= 0:
            return None
        if signal == "SHORT":
            return (entry - exit_price) / entry * 100.0
        return (exit_price - entry) / entry * 100.0

    def _release_matured_pnls(now: pd.Timestamp) -> None:
        nonlocal pending_rank_outcomes
        for ticker in pending_pnls:
            matured = [p for exit_dt, p in pending_pnls[ticker] if exit_dt <= now]
            if matured:
                ticker_pnl_hist[ticker].extend(matured)
            pending_pnls[ticker] = [
                (exit_dt, p) for exit_dt, p in pending_pnls[ticker] if exit_dt > now
            ]
        matured_rank = [(s, p) for exit_dt, s, p in pending_rank_outcomes if exit_dt <= now]
        if matured_rank:
            rank_score_hist.extend(float(s) for s, _ in matured_rank)
            rank_pnl_hist.extend(float(p) for _, p in matured_rank)
        pending_rank_outcomes = [
            (exit_dt, s, p) for exit_dt, s, p in pending_rank_outcomes if exit_dt > now
        ]

    def _rank_orientation() -> tuple[float, float | None]:
        """Return +1 normally, -1 when recent causal rank evidence is inverted."""
        min_n = 40
        if len(rank_score_hist) < min_n:
            return 1.0, None
        scores = pd.Series(rank_score_hist[-120:], dtype=float)
        pnls = pd.Series(rank_pnl_hist[-120:], dtype=float)
        if scores.nunique(dropna=True) < 2 or pnls.nunique(dropna=True) < 2:
            return 1.0, None
        corr = scores.corr(pnls, method="spearman")
        if pd.isna(corr):
            return 1.0, None
        # A negative relationship means the ranker is backwards: lower scores
        # have been producing better matured PnL. Flip selection causally.
        return (-1.0 if float(corr) < -0.05 else 1.0), float(corr)

    def _ticker_is_eligible(ticker: str) -> bool:
        if eligible_tickers is not None and ticker.upper() not in eligible_tickers:
            return False
        if not (use_eligibility_filter and TICKER_ELIGIBILITY_FILTER_ENABLED):
            return True
        # Use a rolling window so a ticker can recover after a bad regime.
        pnls = ticker_pnl_hist.get(ticker, [])[-60:]
        if len(pnls) < TICKER_ELIGIBILITY_MIN_TRADES:
            return True
        pnl_arr = np.asarray(pnls, dtype=float)
        wins = pnl_arr[pnl_arr > 0]
        losses = np.abs(pnl_arr[pnl_arr <= 0])
        win_rate = float(len(wins) / len(pnl_arr))
        profit_factor = float(wins.sum() / (losses.sum() + 1e-9))
        sharpe = float((pnl_arr.mean() / (pnl_arr.std() + 1e-9)) * np.sqrt(252 / max(1, RETURN_HORIZON_DAYS)))
        equity_curve = np.cumsum(pnl_arr)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = float(np.min(equity_curve - peak)) if len(equity_curve) else 0.0
        return (
            float(pnl_arr.sum()) > TICKER_ELIGIBILITY_MIN_TOTAL_PNL
            and win_rate >= TICKER_ELIGIBILITY_MIN_WIN_RATE
            and sharpe >= TICKER_ELIGIBILITY_MIN_SHARPE
            and profit_factor >= TICKER_ELIGIBILITY_MIN_PROFIT_FACTOR
            and drawdown >= TICKER_ELIGIBILITY_MAX_DRAWDOWN_PCT
        )

    def _schedule_selected_trade(ticker: str, dt, signal: str, rank_score: float) -> None:
        row = updated[ticker].loc[dt]
        pnl = _trade_pnl_pct(row, signal)
        if pnl is None:
            return
        exit_dt = pd.Timestamp(row.get("exit_date", dt))
        pending_pnls[ticker].append((exit_dt, pnl))
        pending_rank_outcomes.append((exit_dt, float(rank_score), float(pnl)))

    for dt in sorted(all_dates):
        dt_ts = pd.Timestamp(dt)
        _release_matured_pnls(dt_ts)
        orientation, orientation_corr = _rank_orientation()
        day_rows: list[tuple[str, float, str]] = []

        for ticker, pdf in updated.items():
            if dt not in pdf.index:
                continue
            row = pdf.loc[dt]
            if META_LABELING_ENABLED and float(row.get("meta_confidence", 1.0) or 0.0) < META_LABEL_MIN_CONFIDENCE:
                continue
            dir_vote = str(row.get("direction_vote", "LONG"))
            exp_ret  = float(row.get("expected_return", 0.0))

            # ── Use RAW p_up for direction ───────────────────────────────────
            # p_up_raw is the blended multi-horizon probability (h20 + h5).
            # Prefer it over the calibrated value when available because it is
            # already linearly combined and centred around 0.5.
            if "p_up_raw" in row and pd.notna(row.get("p_up_raw")):
                p_up_raw = float(row.get("p_up_raw"))
            else:
                conf = float(row.get("confidence", 50.0)) / 100.0
                p_up_raw = conf if dir_vote == "LONG" else 1.0 - conf
            raw_direction = "LONG" if p_up_raw >= 0.5 else "SHORT"

            # ── Rank score: factor composite vs ranker vs classifier z-score ──
            # Priority order:
            #   1. SIMPLE_FACTOR_RANKING: use raw factor z-score composite
            #      (no ML, just volatility/beta factors with proven OOS IC)
            #   2. POOLED_RANKER_EXPERIMENT_ENABLED: use XGBRanker score
            #   3. Fallback: z-scored classifier (expected_return + direction_edge)
            rank_score = float("nan")

            if SIMPLE_FACTOR_RANKING and ticker in _factor_data:
                # Factor composite is computed per-date across all tickers below.
                # Store a sentinel here; actual score is filled in the
                # cross-sectional z-score block after all tickers are collected.
                # Keep the XGBoost raw_direction as the gate (hybrid mode):
                # XGBoost decides IF to go long, factors decide WHICH to prioritise.
                rank_score = float("nan")  # placeholder — filled below
            elif POOLED_RANKER_EXPERIMENT_ENABLED:
                _rs_raw = float(row.get("ranker_score", float("nan"))) if "ranker_score" in row.index else float("nan")
                if np.isfinite(_rs_raw):
                    rank_score = _rs_raw
                    raw_direction = "LONG"

            if not np.isfinite(rank_score) and not SIMPLE_FACTOR_RANKING:
                # Original z-score path (classifier fallback).
                hist_e = ticker_exp_hist[ticker][-_NORM_WINDOW:]
                hist_d = ticker_dir_hist[ticker][-_NORM_WINDOW:]
                if len(hist_e) >= _NORM_MIN:
                    mean_e = float(np.mean(hist_e));  std_e = max(float(np.std(hist_e)), 1e-6)
                    mean_d = float(np.mean(hist_d));  std_d = max(float(np.std(hist_d)), 1e-6)
                    z_e = (exp_ret  - mean_e) / std_e
                    z_d = (p_up_raw - mean_d) / std_d
                else:
                    z_e = 0.0
                    z_d = 0.0
                rank_score = z_e + 0.5 * z_d

            # Append AFTER computing score (no look-ahead)
            ticker_exp_hist[ticker].append(exp_ret)
            ticker_dir_hist[ticker].append(p_up_raw)

            if _ticker_is_eligible(ticker):
                day_rows.append((ticker, rank_score, raw_direction))

        if not day_rows:
            continue

        # ── Simple factor composite: compute cross-sectional z-scores ────────
        # For each factor column, z-score across all tickers present today,
        # then combine with the configured weights.  This replaces the
        # XGBoost-based rank_score with a transparent, stable signal.
        if SIMPLE_FACTOR_RANKING and _factor_data:
            dt_ts = pd.Timestamp(dt)
            factor_vals: dict[str, dict[str, float]] = {}  # factor_col → {ticker: val}
            day_tickers = [t for t, _, _ in day_rows]
            for col in SIMPLE_FACTOR_COLS:
                for ticker in day_tickers:
                    if ticker not in _factor_data:
                        continue
                    fdf = _factor_data[ticker]
                    if col not in fdf.columns:
                        continue
                    # Look up the value at this date (or nearest prior date)
                    if dt_ts in fdf.index:
                        val = float(fdf.loc[dt_ts, col])
                    else:
                        # Find the nearest prior date (forward-fill logic)
                        prior = fdf.index[fdf.index <= dt_ts]
                        if len(prior) > 0:
                            val = float(fdf.loc[prior[-1], col])
                        else:
                            val = float("nan")
                    if np.isfinite(val):
                        factor_vals.setdefault(col, {})[ticker] = val

            # Cross-sectional z-score each factor, then weighted sum.
            # Use adaptive weights from JSON if available (recomputed each
            # research run based on trailing IC); fall back to static settings.
            ticker_scores: dict[str, float] = {t: 0.0 for t in day_tickers}
            for col in SIMPLE_FACTOR_COLS:
                vals = factor_vals.get(col, {})
                if len(vals) < 3:
                    continue
                arr_vals = np.array(list(vals.values()))
                mean_v = float(arr_vals.mean())
                std_v = max(float(arr_vals.std()), 1e-9)
                w = _active_factor_weights.get(col, 0.0)
                for ticker, v in vals.items():
                    z = (v - mean_v) / std_v
                    ticker_scores[ticker] += w * z

            # Replace the placeholder rank_scores in day_rows
            day_rows = [
                (t, ticker_scores.get(t, 0.0), dv)
                for t, _, dv in day_rows
            ]

        # Sort descending by normalised score.
        # Only include tickers with a POSITIVE z-score (above their own rolling
        # average).  A negative z-score means the model thinks this ticker is
        # below-average bullish today — "least bearish" is not a LONG signal.
        # This prevents trading garbage during periods when all tickers look weak.
        # Sort descending by z-score.  Filter by raw_direction so we only go
        # LONG when the XGBoost model's raw output is > 0.5 (genuinely bullish).
        # Calibrated p_up is broken for this purpose (always < 0.5 due to label
        # prior), but raw p_up correctly captures which direction the model leans.
        sort_orientation = orientation if RANK_ORIENTATION_FLIP_ENABLED else 1.0
        day_rows.sort(key=lambda x: sort_orientation * x[1], reverse=True)
        longs  = [t for t, _, dv in day_rows if dv == "LONG"][:top_n]
        shorts = [t for t, _, dv in reversed(day_rows) if dv == "SHORT"][:top_n] if mode != "long_only" else []

        _quality_map = {0: "HIGH", 1: "MEDIUM"}
        for rank, ticker in enumerate(longs):
            quality = _quality_map.get(rank, "LOW")
            rank_score = float(updated[ticker].loc[dt].get("rank_score", np.nan))
            for candidate_ticker, candidate_score, _ in day_rows:
                if candidate_ticker == ticker:
                    rank_score = float(candidate_score)
                    break
            updated[ticker].loc[dt, "actionable"] = True
            updated[ticker].loc[dt, "signal"] = "LONG"
            updated[ticker].loc[dt, "signal_quality"] = quality
            updated[ticker].loc[dt, "rank_score"] = rank_score
            updated[ticker].loc[dt, "rank_orientation"] = orientation
            updated[ticker].loc[dt, "rank_orientation_corr"] = orientation_corr
            _schedule_selected_trade(ticker, dt, "LONG", rank_score)
        for rank, ticker in enumerate(shorts):
            quality = _quality_map.get(rank, "LOW")
            rank_score = float(updated[ticker].loc[dt].get("rank_score", np.nan))
            for candidate_ticker, candidate_score, _ in day_rows:
                if candidate_ticker == ticker:
                    rank_score = float(candidate_score)
                    break
            updated[ticker].loc[dt, "actionable"] = True
            updated[ticker].loc[dt, "signal"] = "SHORT"
            updated[ticker].loc[dt, "signal_quality"] = quality
            updated[ticker].loc[dt, "rank_score"] = rank_score
            updated[ticker].loc[dt, "rank_orientation"] = orientation
            updated[ticker].loc[dt, "rank_orientation_corr"] = orientation_corr
            _schedule_selected_trade(ticker, dt, "SHORT", rank_score)

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
                "horizon_future_close": float(df["Close"].iloc[exit_idx]),
                "exit_future_close": float(df["Close"].iloc[exit_idx]),
                "holding_days": int(exit_idx - entry_idx),
                "exit_reason": "time_exit",
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


def compute_rank_performance(trades_df: pd.DataFrame) -> dict:
    """Measure whether higher expected-return ranks produced better realized PnL."""
    if trades_df.empty or not {"date", "expected_return", "net_pnl"}.issubset(trades_df.columns):
        return {"rank_spearman": None, "top_rank_avg_pnl": None, "bottom_rank_avg_pnl": None}

    rows = []
    df = trades_df.copy()
    df["expected_return"] = pd.to_numeric(df["expected_return"], errors="coerce")
    df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce")
    for _, day in df.dropna(subset=["expected_return", "net_pnl"]).groupby("date"):
        if len(day) < 2:
            continue
        ranked = day.sort_values("expected_return", ascending=False).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        rows.append(ranked[["rank", "net_pnl"]])
    if not rows:
        return {"rank_spearman": None, "top_rank_avg_pnl": None, "bottom_rank_avg_pnl": None}

    ranked_all = pd.concat(rows, ignore_index=True)
    spearman = ranked_all["rank"].corr(ranked_all["net_pnl"], method="spearman")
    top = ranked_all[ranked_all["rank"] == 1]["net_pnl"]
    bottom_rank = ranked_all["rank"].max()
    bottom = ranked_all[ranked_all["rank"] == bottom_rank]["net_pnl"]
    return {
        "rank_spearman": round(float(spearman), 4) if pd.notna(spearman) else None,
        "top_rank_avg_pnl": round(float(top.mean()), 2) if len(top) else None,
        "bottom_rank_avg_pnl": round(float(bottom.mean()), 2) if len(bottom) else None,
    }


def _daily_ic_summary(daily_rows: pd.DataFrame, score_col: str) -> dict:
    subset = daily_rows[daily_rows["score_column"] == score_col].copy()
    if subset.empty:
        return {
            "score_column": score_col,
            "mean_ic": None,
            "t_stat": None,
            "median_ic": None,
            "std_ic": None,
            "positive_day_rate": None,
            "days": 0,
            "avg_names_per_day": None,
            "min_names_per_day": None,
            "max_names_per_day": None,
        }

    ic = pd.to_numeric(subset["rank_ic"], errors="coerce").dropna()
    counts = pd.to_numeric(subset["n_names"], errors="coerce").dropna()
    if ic.empty:
        return {
            "score_column": score_col,
            "mean_ic": None,
            "t_stat": None,
            "median_ic": None,
            "std_ic": None,
            "positive_day_rate": None,
            "days": 0,
            "avg_names_per_day": None,
            "min_names_per_day": None,
            "max_names_per_day": None,
        }

    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic))) if len(ic) > 1 and std_ic > 0 else None
    return {
        "score_column": score_col,
        "mean_ic": round(mean_ic, 4),
        "t_stat": round(float(t_stat), 3) if t_stat is not None and np.isfinite(t_stat) else None,
        "median_ic": round(float(ic.median()), 4),
        "std_ic": round(std_ic, 4),
        "positive_day_rate": round(float((ic > 0).mean()), 4),
        "days": int(len(ic)),
        "avg_names_per_day": round(float(counts.mean()), 1) if not counts.empty else None,
        "min_names_per_day": int(counts.min()) if not counts.empty else None,
        "max_names_per_day": int(counts.max()) if not counts.empty else None,
    }


def compute_daily_rank_ic(
    predictions_by_ticker: dict[str, pd.DataFrame],
    *,
    score_columns: tuple[str, ...] = ("ranker_score", "rank_score", "expected_return", "p_up_raw", "confidence"),
    min_names_per_day: int = 5,
) -> tuple[dict, pd.DataFrame]:
    """
    Cross-sectional rank IC: Spearman corr(prediction score, realized forward return)
    computed independently for each signal date.

    This is the right diagnostic for a daily stock ranker. AUC asks whether a
    score separates binary UP/DOWN labels across all dates; rank IC asks whether
    the model ranked the better forward returns higher within the same date.
    """
    empty_rows = pd.DataFrame(columns=["date", "score_column", "rank_ic", "n_names"])
    if not predictions_by_ticker:
        return {"mean_ic": None, "t_stat": None, "days": 0, "by_score": {}}, empty_rows

    panel_rows: list[pd.DataFrame] = []
    available_scores: set[str] = set()
    forward_price_col = "horizon_future_close"
    for ticker, pdf in predictions_by_ticker.items():
        if pdf is None or pdf.empty:
            continue
        df = pdf.copy()
        if "date" not in df.columns:
            df["date"] = df.index
        if "ticker" not in df.columns:
            df["ticker"] = ticker

        exit_col = "horizon_future_close" if "horizon_future_close" in df.columns else "exit_future_close"
        if exit_col != "horizon_future_close":
            forward_price_col = "exit_future_close"
        required = {"date", "ticker", "open_next", exit_col}
        if not required.issubset(df.columns):
            continue
        cols = [c for c in score_columns if c in df.columns]
        if not cols:
            continue
        available_scores.update(cols)
        sub = df[["date", "ticker", "open_next", exit_col, *cols]].copy()
        sub = sub.rename(columns={exit_col: "forward_close"})
        panel_rows.append(sub)

    if not panel_rows or not available_scores:
        return {"mean_ic": None, "t_stat": None, "days": 0, "by_score": {}}, empty_rows

    panel = pd.concat(panel_rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["open_next"] = pd.to_numeric(panel["open_next"], errors="coerce")
    panel["forward_close"] = pd.to_numeric(panel["forward_close"], errors="coerce")
    panel = panel.dropna(subset=["date", "open_next", "forward_close"])
    panel = panel[panel["open_next"] > 0]
    if panel.empty:
        return {"mean_ic": None, "t_stat": None, "days": 0, "by_score": {}}, empty_rows

    panel["forward_return"] = panel["forward_close"] / panel["open_next"] - 1.0
    daily_rows: list[dict] = []
    for dt, day in panel.groupby("date"):
        for score_col in score_columns:
            if score_col not in available_scores or score_col not in day.columns:
                continue
            valid = day[["forward_return", score_col]].copy()
            valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
            valid = valid.dropna()
            if len(valid) < min_names_per_day:
                continue
            if valid[score_col].nunique(dropna=True) < 2 or valid["forward_return"].nunique(dropna=True) < 2:
                continue
            ic = valid[score_col].corr(valid["forward_return"], method="spearman")
            if pd.isna(ic) or not np.isfinite(ic):
                continue
            daily_rows.append({
                "date": pd.Timestamp(dt).date().isoformat(),
                "score_column": score_col,
                "rank_ic": round(float(ic), 6),
                "n_names": int(len(valid)),
            })

    daily_ic = pd.DataFrame(daily_rows, columns=["date", "score_column", "rank_ic", "n_names"])
    if daily_ic.empty:
        return {
            "mean_ic": None,
            "t_stat": None,
            "days": 0,
            "min_names_per_day": int(min_names_per_day),
            "forward_price_col": forward_price_col,
            "by_score": {},
        }, daily_ic

    if POOLED_RANKER_EXPERIMENT_ENABLED and "ranker_score" in available_scores:
        primary_score = "ranker_score"
    elif POOLED_RANKER_EXPERIMENT_ENABLED and "rank_score" in available_scores:
        primary_score = "rank_score"
    elif "expected_return" in available_scores:
        primary_score = "expected_return"
    else:
        primary_score = sorted(available_scores)[0]
    by_score = {
        score_col: _daily_ic_summary(daily_ic, score_col)
        for score_col in score_columns
        if score_col in available_scores
    }
    primary = by_score.get(primary_score, _daily_ic_summary(daily_ic, primary_score))
    summary = {
        "primary_score": primary_score,
        "mean_ic": primary.get("mean_ic"),
        "t_stat": primary.get("t_stat"),
        "median_ic": primary.get("median_ic"),
        "std_ic": primary.get("std_ic"),
        "positive_day_rate": primary.get("positive_day_rate"),
        "days": primary.get("days", 0),
        "avg_names_per_day": primary.get("avg_names_per_day"),
        "min_names_per_day": int(min_names_per_day),
        "forward_price_col": forward_price_col,
        "by_score": by_score,
    }
    return summary, daily_ic.sort_values(["date", "score_column"]).reset_index(drop=True)


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


def _safe_decile_labels(series: pd.Series, label_prefix: str = "D") -> pd.Series:
    """Return stable decile labels when there are enough unique numeric values."""
    numeric = pd.to_numeric(series, errors="coerce")
    labels = pd.Series("missing", index=series.index, dtype=object)
    valid = numeric.dropna()
    if valid.empty:
        return labels
    try:
        deciles = pd.qcut(valid.rank(method="first"), q=10, labels=False, duplicates="drop")
    except ValueError:
        labels.loc[valid.index] = "all"
        return labels
    labels.loc[deciles.index] = [f"{label_prefix}{int(x) + 1}" for x in deciles.astype(int)]
    return labels


def _holding_bucket(days: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(days, errors="coerce").fillna(0)
    bins = [-np.inf, 1, 3, 5, 10, 20, np.inf]
    labels = ["1d", "2-3d", "4-5d", "6-10d", "11-20d", "20d+"]
    return pd.cut(numeric, bins=bins, labels=labels).astype(str)


def build_edge_audit_report(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Detailed completed-trade edge audit used before enabling new gates."""
    columns = [
        "group_type", "group", "trades", "win_rate", "avg_pnl", "median_pnl",
        "total_pnl", "profit_factor", "raw_sharpe_proxy", "avg_confidence",
        "avg_expected_return", "avg_rank_score", "avg_holding_days",
    ]
    if trades_df.empty:
        return pd.DataFrame(columns=columns)

    df = trades_df.copy()
    df["net_pnl"] = pd.to_numeric(df.get("net_pnl"), errors="coerce").fillna(0.0)
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce")
    df["expected_return"] = pd.to_numeric(df.get("expected_return"), errors="coerce")
    df["rank_score"] = pd.to_numeric(df.get("rank_score"), errors="coerce")
    df["holding_days"] = pd.to_numeric(df.get("holding_days"), errors="coerce")
    if "entry_date" in df.columns:
        df["entry_year"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.year.astype("Int64").astype(str)
    df["confidence_decile"] = _safe_decile_labels(df["confidence"], "C")
    df["expected_return_decile"] = _safe_decile_labels(df["expected_return"], "E")
    df["rank_score_decile"] = _safe_decile_labels(df["rank_score"], "R")
    df["holding_days_bucket"] = _holding_bucket(df["holding_days"])

    group_specs = [
        ("regime_state", "regime_state"),
        ("ticker", "ticker"),
        ("sector", "sector"),
        ("signal_quality", "signal_quality"),
        ("exit_reason", "exit_reason"),
        ("entry_year", "entry_year"),
        ("confidence_decile", "confidence_decile"),
        ("expected_return_decile", "expected_return_decile"),
        ("rank_score_decile", "rank_score_decile"),
        ("holding_days_bucket", "holding_days_bucket"),
    ]
    rows = []
    for group_type, col in group_specs:
        if col not in df.columns:
            continue
        for group, g in df.groupby(col, dropna=False):
            pnl = pd.to_numeric(g["net_pnl"], errors="coerce").fillna(0.0)
            wins = float(pnl[pnl > 0].sum())
            losses = float(abs(pnl[pnl < 0].sum()))
            std = float(pnl.std(ddof=0))
            rows.append({
                "group_type": group_type,
                "group": str(group),
                "trades": int(len(g)),
                "win_rate": round(float((pnl > 0).mean()), 4),
                "avg_pnl": round(float(pnl.mean()), 2),
                "median_pnl": round(float(pnl.median()), 2),
                "total_pnl": round(float(pnl.sum()), 2),
                "profit_factor": round(wins / losses, 3) if losses > 0 else None,
                "raw_sharpe_proxy": round(float(pnl.mean() / (std + 1e-9)), 3) if len(pnl) else 0.0,
                "avg_confidence": round(float(g["confidence"].mean()), 2) if g["confidence"].notna().any() else None,
                "avg_expected_return": round(float(g["expected_return"].mean()), 4) if g["expected_return"].notna().any() else None,
                "avg_rank_score": round(float(g["rank_score"].mean()), 4) if g["rank_score"].notna().any() else None,
                "avg_holding_days": round(float(g["holding_days"].mean()), 2) if g["holding_days"].notna().any() else None,
            })
    return pd.DataFrame(rows, columns=columns).sort_values(["group_type", "total_pnl"], ascending=[True, False])


def summarize_edge_audit(edge_audit: pd.DataFrame) -> dict:
    if edge_audit.empty:
        return {"enabled": bool(EDGE_AUDIT_ENABLED), "groups": 0}
    useful = edge_audit[pd.to_numeric(edge_audit["trades"], errors="coerce").fillna(0) >= 5].copy()
    if useful.empty:
        useful = edge_audit.copy()
    worst = useful.sort_values("total_pnl", ascending=True).head(5)
    best = useful.sort_values("total_pnl", ascending=False).head(5)
    return {
        "enabled": bool(EDGE_AUDIT_ENABLED),
        "groups": int(len(edge_audit)),
        "best_groups": best[["group_type", "group", "trades", "total_pnl", "profit_factor"]].to_dict("records"),
        "worst_groups": worst[["group_type", "group", "trades", "total_pnl", "profit_factor"]].to_dict("records"),
    }


def single_name_crash_filter_reason(hist: pd.DataFrame, dt, signal: str) -> str | None:
    """Return the long-entry veto reason for crash/distress regimes, if any."""
    if not SINGLE_NAME_CRASH_FILTER_ENABLED or signal != "LONG" or "Close" not in hist.columns:
        return None

    close = pd.to_numeric(hist["Close"].loc[:pd.Timestamp(dt)], errors="coerce").dropna()
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


def passes_single_name_crash_filter(hist: pd.DataFrame, dt, signal: str) -> bool:
    return single_name_crash_filter_reason(hist, dt, signal) is None


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
    equity_curve, equity_index, gross_exposure_curve = [], [], []
    single_name_crash_filter_skips = 0
    single_name_crash_filter_skips_by_reason: dict[str, int] = {}
    single_name_crash_filter_skips_by_ticker: dict[str, int] = {}
    conformal_gate_skips = 0
    regime_edge_gate_skips = 0
    rank_decile_gate_skips = 0
    regime_edge_hist: dict[str, list[float]] = {}
    rank_decile_edge_hist: dict[int, list[float]] = {i: [] for i in range(10)}
    rank_edge_scores: list[float] = []

    def _edge_profit_factor(pnls: list[float]) -> float:
        arr = np.asarray(pnls, dtype=float)
        wins = float(arr[arr > 0].sum())
        losses = float(abs(arr[arr < 0].sum()))
        if losses <= 0:
            return float("inf") if wins > 0 else 0.0
        return wins / losses

    def _edge_history_passes(pnls: list[float]) -> bool:
        if len(pnls) < EDGE_GATE_MIN_TRADES:
            return True
        recent = [float(x) for x in pnls[-120:]]
        avg_pnl = float(np.mean(recent)) if recent else 0.0
        profit_factor = _edge_profit_factor(recent)
        return (
            avg_pnl >= EDGE_GATE_MIN_AVG_PNL
            and profit_factor >= EDGE_GATE_MIN_PROFIT_FACTOR
        )

    def _rank_score_decile(score: float) -> int | None:
        if not np.isfinite(score) or len(rank_edge_scores) < EDGE_GATE_MIN_TRADES:
            return None
        hist = np.asarray(rank_edge_scores[-250:], dtype=float)
        hist = hist[np.isfinite(hist)]
        if len(hist) < EDGE_GATE_MIN_TRADES:
            return None
        pct = float(np.mean(hist <= float(score)))
        return int(np.clip(np.floor(pct * 10.0), 0, 9))

    def _record_edge_outcome(closed_trade: dict) -> None:
        pnl = float(closed_trade.get("net_pnl", 0.0) or 0.0)
        regime = str(closed_trade.get("regime_state", "unknown"))
        regime_edge_hist.setdefault(regime, []).append(pnl)
        rank_score = closed_trade.get("rank_score")
        try:
            score = float(rank_score)
        except Exception:
            return
        if not np.isfinite(score):
            return
        decile = _rank_score_decile(score)
        if decile is not None:
            rank_decile_edge_hist[decile].append(pnl)
        rank_edge_scores.append(score)

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
                closed_trade = {
                    "date": pos["signal_date"], "entry_date": pos["entry_date"], "exit_date": pos["exit_date"],
                    "ticker": pos["ticker"], "signal": pos["signal"], "signal_quality": pos["signal_quality"],
                    "confidence": round(pos["confidence"], 2), "expected_return": round(pos["expected_return"], 2),
                    "rank_score": (
                        round(float(pos.get("rank_score")), 4)
                        if pd.notna(pos.get("rank_score", np.nan))
                        else None
                    ),
                    "rank_orientation": round(float(pos.get("rank_orientation", 1.0)), 1),
                    "rank_orientation_corr": (
                        round(float(pos.get("rank_orientation_corr")), 4)
                        if pos.get("rank_orientation_corr") is not None
                        else None
                    ),
                    "conformal_singleton": pos.get("conformal_singleton"),
                    "conformal_set_size": pos.get("conformal_set_size"),
                    "conformal_qhat": pos.get("conformal_qhat"),
                    "position_pct": round(pos["requested_position_pct"] * 100, 2), "regime_state": pos["regime_state"],
                    "entry_price": round(pos["entry_price"], 4), "exit_price": round(exit_px, 4),
                    "holding_days": int(pos["holding_days"]), "net_pnl": round(net, 2),
                    "exit_reason": pos.get("exit_reason", "time_exit"),
                    "sector": SECTOR_MAP.get(pos["ticker"], "OTHER"),
                    "capacity_warning": bool(pos.get("capacity_warning", False)),
                }
                trade_rows.append(closed_trade)
                _record_edge_outcome(closed_trade)
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
            if CONFORMAL_TRADE_GATE_ENABLED and not bool(row.get("conformal_singleton", True)):
                conformal_gate_skips += 1
                continue
            # Signal quality gate removed — apply_cross_sectional_ranking already
            # limits candidates to top-N by expected_return.  Filtering here by
            # confidence-bucket quality labels was redundant noise (probabilities
            # collapse around 0.5 so bucket boundaries were arbitrary).
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
            ranker_trade = (
                POOLED_RANKER_EXPERIMENT_ENABLED
                and np.isfinite(float(row.get("ranker_score", np.nan)))
            )
            if not ranker_trade:
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
            if REGIME_EDGE_GATE_ENABLED and not _edge_history_passes(regime_edge_hist.get(reg_state, [])):
                regime_edge_gate_skips += 1
                continue
            if RANK_DECILE_GATE_ENABLED:
                try:
                    rank_score_now = float(row.get("rank_score", np.nan))
                except Exception:
                    rank_score_now = np.nan
                rank_decile = _rank_score_decile(rank_score_now)
                if rank_decile is not None and not _edge_history_passes(rank_decile_edge_hist.get(rank_decile, [])):
                    rank_decile_gate_skips += 1
                    continue
            crash_filter_reason = single_name_crash_filter_reason(price_history[ticker], dt, signal)
            if crash_filter_reason is not None:
                single_name_crash_filter_skips += 1
                single_name_crash_filter_skips_by_reason[crash_filter_reason] = (
                    single_name_crash_filter_skips_by_reason.get(crash_filter_reason, 0) + 1
                )
                single_name_crash_filter_skips_by_ticker[ticker] = (
                    single_name_crash_filter_skips_by_ticker.get(ticker, 0) + 1
                )
                continue
            if POSITION_SIZING_MODE == "vol_kelly":
                # Vol-target base × fractional-Kelly confidence multiplier.
                # Uses only prices known on this signal date (no look-ahead).
                asset_vol = historical_annual_vol(price_history, ticker, dt)
                requested_pct = compute_position_size(
                    float(row["confidence"]),
                    expected_return,
                    str(row["signal_quality"]),
                    current_equity,
                    asset_vol,
                    base_pct=BASE_POSITION_SIZE_PCT,
                    max_pct=MAX_SINGLE_NAME_EXPOSURE,
                    vol_target=VOL_TARGET_PCT,
                    kelly_fraction=KELLY_SAFETY_FRACTION,
                ) * reg_mult
            else:
                # ATR-based risk sizing: risk ATR_RISK_BUDGET_PCT of capital per
                # trade, using ATR_SL_MULT × 20-day ATR as the stop estimate.
                # High-vol tickers (NVDA, MU) get smaller positions than calm
                # tickers (MSFT, UNH), equalising dollar risk across the book.
                requested_pct = atr_position_pct(price_history, ticker, dt) * reg_mult
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
        # Compute the gross/net exposure already committed in open positions.
        # This tells approve_day how much room is left under the cap before
        # opening any new trades today.
        open_gross = sum(abs(p["requested_position_pct"]) for p in open_positions)
        open_net   = sum(
            p["requested_position_pct"] if p.get("signal", "LONG") == "LONG"
            else -p["requested_position_pct"]
            for p in open_positions
        )
        open_ticker_set = {p["ticker"] for p in open_positions}
        approved = manager.approve_day(
            candidates,
            {k: v["Close"] for k, v in price_history.items()},
            eq_series,
            current_gross=open_gross,
            current_net=open_net,
            open_tickers=open_ticker_set,
        )
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
                    str(row.get("exit_reason", "time_exit")),
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
                "rank_score": float(row.get("rank_score", np.nan)),
                "rank_orientation": float(row.get("rank_orientation", 1.0)),
                "rank_orientation_corr": (
                    float(row.get("rank_orientation_corr"))
                    if pd.notna(row.get("rank_orientation_corr", np.nan))
                    else None
                ),
                "conformal_singleton": row.get("conformal_singleton"),
                "conformal_set_size": row.get("conformal_set_size"),
                "conformal_qhat": row.get("conformal_qhat"),
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
        gross_exposure_curve.append(sum(abs(float(p["requested_position_pct"])) for p in open_positions))

    raw_equity = pd.Series(index=pd.DatetimeIndex(equity_index), data=equity_curve, dtype=float)
    gross_exposure = pd.Series(index=pd.DatetimeIndex(equity_index), data=gross_exposure_curve, dtype=float)
    benchmarks = build_benchmarks(raw_equity.index)
    equity = raw_equity
    overlay_alloc = pd.Series(index=raw_equity.index, data=0.0, dtype=float)
    if CASH_OVERLAY_ENABLED:
        equity, overlay_alloc = apply_cash_overlay(raw_equity, benchmarks, gross_exposure)

    benchmark = benchmarks["BLEND"] if "BLEND" in benchmarks.columns else benchmarks["SPY"]
    daily_ret = equity.pct_change().fillna(0.0)
    dd = equity / equity.cummax() - 1.0
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0
    years = max((equity.index.max() - equity.index.min()).days / 365.25, 1 / 365.25)
    if len(equity) and (1.0 + total_ret) > 0.0:
        ann_ret = (1.0 + total_ret) ** (1.0 / years) - 1.0
    else:
        # Crash/audit runs can drive simulated equity below zero. Fractional
        # annualization of a negative base becomes complex; report total loss
        # instead so the metrics file is still written and the gates can fail.
        ann_ret = -1.0 if len(equity) else 0.0
    bench_ret = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0) if len(benchmark) else 0.0
    sharpe = float((daily_ret.mean() / (daily_ret.std() + 1e-12)) * np.sqrt(252)) if len(daily_ret) else 0.0
    # Compute Newey-West t-stat vs 0 (risk-free / cash benchmark).
    # A selective strategy that deliberately holds cash cannot be tested vs a
    # fully-invested SPY — every cash day would register as a "loss" against SPY's
    # positive drift, making the t-stat negative even when all trades are profitable.
    # The correct null hypothesis for a sparse trading strategy is:
    #   H0: mean daily return == 0  (is there any return above cash?)
    # We still log benchmark-relative t-stats separately for information.
    primary_daily = benchmark.pct_change().reindex(equity.index).fillna(0.0)
    spy_daily = benchmarks["SPY"].reindex(equity.index).ffill().bfill().pct_change().fillna(0.0)
    nw_tstat = _newey_west_tstat(daily_ret)          # vs 0 (cash) — primary gate
    nw_tstat_vs_primary = _newey_west_tstat(daily_ret - primary_daily)  # informational only
    nw_tstat_vs_spy = _newey_west_tstat(daily_ret - spy_daily)          # informational only
    benchmark_comparisons = compute_benchmark_comparisons(equity, benchmarks)
    raw_daily_ret = raw_equity.pct_change().fillna(0.0)
    raw_total_ret = float(raw_equity.iloc[-1] / raw_equity.iloc[0] - 1.0) if len(raw_equity) else 0.0
    raw_sharpe = float((raw_daily_ret.mean() / (raw_daily_ret.std() + 1e-12)) * np.sqrt(252)) if len(raw_daily_ret) else 0.0
    overlay_contribution = total_ret - raw_total_ret
    avg_gross_exposure = float(gross_exposure.reindex(equity.index).fillna(0.0).mean()) if len(gross_exposure) else 0.0
    in_market_pct = float((gross_exposure.reindex(equity.index).fillna(0.0) > 0).mean() * 100.0) if len(gross_exposure) else 0.0
    trades_for_metrics = pd.DataFrame(trade_rows)
    rank_performance = compute_rank_performance(trades_for_metrics)
    edge_audit = build_edge_audit_report(trades_for_metrics) if EDGE_AUDIT_ENABLED else pd.DataFrame()

    # Evaluate every kill-criteria gate and record which passed / failed.
    gate_results = {
        "sharpe_pass": bool(sharpe >= KILL_CRITERIA["min_sharpe"]),
        "nw_tstat_pass": bool(nw_tstat >= KILL_CRITERIA["min_nw_tstat"]),
        "drawdown_pass": bool((float(dd.min()) if len(dd) else 0.0) >= KILL_CRITERIA["max_drawdown"]),
        "raw_stock_sharpe_pass": bool(raw_sharpe >= KILL_CRITERIA["min_raw_stock_sharpe"]),
        "raw_stock_return_pass": bool(raw_total_ret >= KILL_CRITERIA["min_raw_stock_return"]),
    }
    gate_results["all_pass"] = all(gate_results.values())
    raw_stock_gates_pass = bool(gate_results["raw_stock_sharpe_pass"] and gate_results["raw_stock_return_pass"])

    metrics = {
        "trades": int(len(trade_rows)),
        "total_return_pct": round(total_ret * 100, 2),
        "annual_return_pct": round(ann_ret * 100, 2),
        "benchmark_return_pct": round(bench_ret * 100, 2),
        "alpha_pct": round((total_ret - bench_ret) * 100, 2),
        "cagr_pct": round(ann_ret * 100, 2),
        "benchmark_comparisons": benchmark_comparisons,
        "primary_benchmark": "BLEND",
        "primary_benchmark_weights": _normalise_weights(PRIMARY_BENCHMARK_WEIGHTS),
        "raw_strategy_total_return_pct": round(raw_total_ret * 100, 2),
        "raw_strategy_sharpe_ratio_daily": round(raw_sharpe, 3),
        "raw_stock_gates_pass": raw_stock_gates_pass,
        "paper_mode_strategy": PAPER_MODE_STRATEGY,
        "single_name_paper_trading_enabled": bool(SINGLE_NAME_PAPER_TRADING_ENABLED and raw_stock_gates_pass),
        "cash_overlay_enabled": bool(CASH_OVERLAY_ENABLED),
        "cash_overlay_max_alloc": float(CASH_OVERLAY_MAX_ALLOC),
        "cash_overlay_avg_alloc": round(float(overlay_alloc.mean()), 4) if len(overlay_alloc) else 0.0,
        "cash_overlay_days_active": int((overlay_alloc > 0).sum()) if len(overlay_alloc) else 0,
        "overlay_contribution_pct": round(overlay_contribution * 100, 2),
        "stock_sleeve_metrics": {
            "total_return_pct": round(raw_total_ret * 100, 2),
            "sharpe_ratio_daily": round(raw_sharpe, 3),
            "gates_pass": raw_stock_gates_pass,
            "paper_ready": raw_stock_gates_pass,
        },
        "cash_overlay_metrics": {
            "enabled": bool(CASH_OVERLAY_ENABLED),
            "avg_alloc": round(float(overlay_alloc.mean()), 4) if len(overlay_alloc) else 0.0,
            "days_active": int((overlay_alloc > 0).sum()) if len(overlay_alloc) else 0,
            "contribution_pct": round(overlay_contribution * 100, 2),
        },
        "combined_portfolio_metrics": {
            "total_return_pct": round(total_ret * 100, 2),
            "sharpe_ratio_daily": round(sharpe, 3),
            "max_drawdown_pct": round(float(dd.min()) * 100, 2) if len(dd) else 0.0,
            "alpha_pct": round((total_ret - bench_ret) * 100, 2),
        },
        "single_name_crash_filter_enabled": bool(SINGLE_NAME_CRASH_FILTER_ENABLED),
        "single_name_distress_filter_enabled": bool(SINGLE_NAME_DISTRESS_FILTER_ENABLED),
        "single_name_crash_filter_skips": int(single_name_crash_filter_skips),
        "single_name_crash_filter_skips_by_reason": dict(sorted(single_name_crash_filter_skips_by_reason.items())),
        "single_name_crash_filter_skips_by_ticker": dict(
            sorted(single_name_crash_filter_skips_by_ticker.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "conformal_trade_gate_enabled": bool(CONFORMAL_TRADE_GATE_ENABLED),
        "conformal_alpha": float(CONFORMAL_ALPHA),
        "conformal_gate_skips": int(conformal_gate_skips),
        "edge_audit_enabled": bool(EDGE_AUDIT_ENABLED),
        "regime_edge_gate_enabled": bool(REGIME_EDGE_GATE_ENABLED),
        "rank_decile_gate_enabled": bool(RANK_DECILE_GATE_ENABLED),
        "edge_gate_min_trades": int(EDGE_GATE_MIN_TRADES),
        "edge_gate_min_profit_factor": float(EDGE_GATE_MIN_PROFIT_FACTOR),
        "edge_gate_min_avg_pnl": float(EDGE_GATE_MIN_AVG_PNL),
        "regime_edge_gate_skips": int(regime_edge_gate_skips),
        "rank_decile_gate_skips": int(rank_decile_gate_skips),
        "edge_audit": summarize_edge_audit(edge_audit),
        "avg_gross_exposure_pct": round(avg_gross_exposure * 100.0, 2),
        "in_market_pct": round(in_market_pct, 2),
        "rank_performance": rank_performance,
        "sharpe_ratio_daily": round(sharpe, 3),
        "nw_tstat_vs_cash": round(nw_tstat, 3),
        "nw_tstat_vs_primary_benchmark": round(nw_tstat_vs_primary, 3),
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
    return trades_for_metrics, equity, benchmarks, metrics


def default_backtest_tickers(approved_only: bool = False) -> list[str]:
    available = {f.replace(".parquet", "") for f in os.listdir(DATA_DIR) if f.endswith(".parquet")}
    if approved_only:
        report = read_quality_report()
        if not report.empty and {"ticker", "approved_for_live"}.issubset(report.columns):
            approved = report[report["approved_for_live"].astype(str).str.lower().isin({"true", "1"})]
            tickers = [str(t).upper() for t in approved["ticker"].tolist() if str(t).upper() in available]
            if tickers:
                return tickers
        return []

    # Research backtests should use the full available watchlist by default.
    # The live-approved subset is opt-in via --approved-only.
    return [t for t in WATCHLIST if t in available]


def write_quality_outputs(tickers: list[str], trades_df: pd.DataFrame, metrics: dict, quality_source: str) -> pd.DataFrame:
    rows = []
    tickers = [str(t).upper() for t in tickers]
    if not tickers:
        report = read_quality_report()
        approved = (
            report[report["approved_for_live"].astype(str).str.lower().isin({"true", "1"})].copy()
            if not report.empty and "approved_for_live" in report.columns
            else pd.DataFrame(columns=["ticker", "approval_reason"])
        )
        approved_path = os.path.join(SIGNAL_DIR, "approved_live_tickers.csv")
        approved[["ticker", "approval_reason"]].to_csv(approved_path, index=False)
        print("Saved ->", approved_path)
        return report

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
                "portfolio_cagr_pct": metrics.get("cagr_pct"),
                "portfolio_avg_gross_exposure_pct": metrics.get("avg_gross_exposure_pct"),
                "portfolio_in_market_pct": metrics.get("in_market_pct"),
                "rank_spearman": (metrics.get("rank_performance") or {}).get("rank_spearman"),
                "daily_rank_ic_mean": (metrics.get("daily_rank_ic") or {}).get("mean_ic"),
                "daily_rank_ic_tstat": (metrics.get("daily_rank_ic") or {}).get("t_stat"),
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


def run_spy_timing_backtest(
    raw_predictions: dict[str, pd.DataFrame],
    threshold: float = SPY_TIMING_BULL_THRESHOLD,
    invest_pct: float = SPY_TIMING_INVEST_PCT,
    instruments: dict[str, float] | None = None,
    output_prefix: str = "spy_timing",
    write_outputs: bool = True,
    quiet: bool = False,
) -> dict:
    """Backtest the SPY/QQQ timing strategy using walk-forward predictions.

    PLAIN ENGLISH: Instead of picking individual stocks, we count how many
    tickers the model says "LONG" on each day.  When the majority are
    bullish, we buy SPY/QQQ.  When not, we sit in cash.

    This uses the SAME walk-forward predictions already computed — no extra
    model training.  It just changes the EXECUTION from "pick top-4 stocks"
    to "buy index when model is bullish."

    Returns a dict with performance metrics for the timing strategy.
    """
    instruments = instruments or SPY_TIMING_INSTRUMENTS
    # ── 1. Build daily bull-fraction time series ─────────────────────────────
    # For each date, count how many tickers voted LONG vs total available.
    date_votes: dict[pd.Timestamp, dict] = {}
    for ticker, pdf in raw_predictions.items():
        if pdf.empty:
            continue
        for dt in pdf.index:
            row = pdf.loc[dt]
            dv = str(row.get("direction_vote", "LONG"))
            if dt not in date_votes:
                date_votes[dt] = {"n_long": 0, "n_total": 0}
            date_votes[dt]["n_total"] += 1
            if dv == "LONG":
                date_votes[dt]["n_long"] += 1

    if not date_votes:
        if not quiet:
            print("[spy-timing] No predictions to build timing signal from")
        return {}

    timeline = sorted(date_votes.keys())
    bull_fractions = pd.Series(
        {dt: v["n_long"] / max(v["n_total"], 1) for dt, v in date_votes.items()},
        dtype=float,
    ).sort_index()

    # ── 2. Download benchmark prices ─────────────────────────────────────────
    start_str = str(timeline[0].date() - pd.Timedelta(days=5))
    end_str   = str(timeline[-1].date() + pd.Timedelta(days=RETURN_HORIZON_DAYS + 5))
    bench_prices: dict[str, pd.Series] = {}
    benchmark_symbols = sorted(set(instruments) | {"SPY", "QQQ"})
    for sym in benchmark_symbols:
        try:
            raw = _download_yfinance(sym, start=start_str, end=end_str, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            bench_prices[sym] = raw["Close"]
        except Exception as e:
            if not quiet:
                print(f"[spy-timing] WARNING: could not download {sym}: {e}")

    if not bench_prices:
        if not quiet:
            print("[spy-timing] No benchmark prices downloaded")
        return {}

    # ── 3. Simulate the strategy ─────────────────────────────────────────────
    # Smoothing: rolling average of bull_fraction reduces daily whipsaw.
    # Hysteresis: separate enter/exit thresholds create a dead zone so small
    #   fluctuations don't trigger trades.
    # Min hold: once invested, hold at least N days before considering exit.
    smooth_w = SPY_TIMING_SMOOTH_WINDOW
    enter_thr = SPY_TIMING_ENTER_THRESHOLD
    exit_thr = SPY_TIMING_EXIT_THRESHOLD
    min_hold = SPY_TIMING_MIN_HOLD_DAYS

    # Smooth the bull_fraction series with a rolling average.
    smoothed_bf = bull_fractions.rolling(window=smooth_w, min_periods=1).mean()

    equity = INITIAL_CAPITAL
    equity_curve = []
    signals_log = []
    n_invested = 0
    n_cash = 0
    prev_weights = {sym: 0.0 for sym in benchmark_symbols}
    total_turnover = 0.0
    is_invested = False          # current position state
    days_since_last_rebal = 999  # force first rebalance immediately

    for dt in timeline:
        bf_raw = float(bull_fractions.get(dt, 0.0))
        bf = float(smoothed_bf.get(dt, bf_raw))
        days_since_last_rebal += 1

        # Only evaluate signal at rebalance points (every min_hold days).
        # Between rebalances, hold the current position regardless of signal.
        # This dramatically reduces turnover while still capturing regime shifts.
        if days_since_last_rebal >= max(min_hold, 1):
            # Hysteresis logic at rebalance point:
            if not is_invested:
                if bf >= enter_thr:
                    is_invested = True
                    days_since_last_rebal = 0
            else:
                if bf <= exit_thr:
                    is_invested = False
                    days_since_last_rebal = 0

        # Fixed allocation when invested — binary in/out.
        if is_invested:
            day_invest = invest_pct
        else:
            day_invest = 0.0

        current_weights = {sym: 0.0 for sym in benchmark_symbols}
        for sym, weight in instruments.items():
            current_weights[sym] = day_invest * float(weight)
        day_turnover = sum(
            abs(float(current_weights.get(sym, 0.0)) - float(prev_weights.get(sym, 0.0)))
            for sym in benchmark_symbols
        )
        total_turnover += day_turnover
        prev_weights = current_weights

        # Transaction cost: deducted from equity each day based on turnover.
        # SPY/QQQ ETFs have ~1-2bps round-trip cost, much lower than stocks.
        cost = day_turnover * SPY_TIMING_ETF_COST_BPS / 10_000.0

        # Compute blended benchmark return for this day
        day_ret = 0.0
        for sym, weight in instruments.items():
            if sym in bench_prices:
                px = bench_prices[sym]
                if dt in px.index:
                    idx = px.index.get_loc(dt)
                    if idx + 1 < len(px):
                        day_ret += weight * (float(px.iloc[idx + 1]) / float(px.iloc[idx]) - 1.0)

        # Apply position-sized return minus transaction cost
        portfolio_ret = day_invest * day_ret - cost
        equity *= (1.0 + portfolio_ret)
        equity_curve.append({"date": dt, "equity": equity, "bull_fraction": bf,
                             "bull_fraction_raw": bf_raw,
                             "invested": day_invest, "day_ret": day_ret,
                             "strategy_ret": portfolio_ret})
        signals_log.append({
            "date": dt, "bull_fraction_raw": round(bf_raw, 3),
            "bull_fraction_smoothed": round(bf, 3),
            "signal": "BUY_SPY" if is_invested else "STAY_CASH",
            "invest_pct": round(day_invest, 3),
            "n_long": date_votes[dt]["n_long"],
            "n_total": date_votes[dt]["n_total"],
            "days_held": days_since_last_rebal if is_invested else 0,
        })
        if is_invested:
            n_invested += 1
        else:
            n_cash += 1

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    signals_df = pd.DataFrame(signals_log)

    # ── 4. Compute metrics ───────────────────────────────────────────────────
    total_return = (equity / INITIAL_CAPITAL - 1.0) * 100.0
    n_days = len(eq_df)
    years = max(n_days / 252.0, 0.01)
    cagr = ((equity / INITIAL_CAPITAL) ** (1.0 / years) - 1.0) * 100.0

    daily_rets = eq_df["equity"].pct_change().dropna()
    sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(252))
    downside = daily_rets[daily_rets < 0]
    sortino = float(daily_rets.mean() / (downside.std() + 1e-9) * np.sqrt(252)) if len(downside) else 0.0
    nw_tstat = _newey_west_tstat(eq_df["equity"].pct_change().fillna(0.0))

    # Max drawdown
    peak = eq_df["equity"].cummax()
    dd = (eq_df["equity"] - peak) / peak
    max_dd = float(dd.min()) * 100.0

    def _buy_hold_return(symbol: str) -> float | None:
        if symbol not in bench_prices:
            return None
        px = bench_prices[symbol].dropna()
        if px.empty:
            return None
        first_slice = px.loc[px.index >= timeline[0]]
        last_slice = px.loc[px.index <= timeline[-1]]
        if first_slice.empty or last_slice.empty:
            return None
        first = float(first_slice.iloc[0])
        last = float(last_slice.iloc[-1])
        if first <= 0:
            return None
        return (last / first - 1.0) * 100.0

    def _weighted_benchmark_return(weights: dict[str, float]) -> float:
        ret = 0.0
        total_w = 0.0
        for sym, weight in _normalise_weights(weights).items():
            sym_ret = _buy_hold_return(sym)
            if sym_ret is None:
                continue
            ret += float(weight) * sym_ret
            total_w += float(weight)
        return ret if total_w > 0 else 0.0

    # Compute buy-and-hold benchmark returns over the same period.
    bh_return = _weighted_benchmark_return(instruments)
    primary_bh_return = _weighted_benchmark_return(PRIMARY_BENCHMARK_WEIGHTS)
    benchmark_comparisons: dict[str, dict] = {}
    for sym in ("SPY", "QQQ"):
        sym_ret = _buy_hold_return(sym)
        benchmark_comparisons[sym] = {
            "data_available": sym_ret is not None,
            "benchmark_return_pct": round(float(sym_ret), 2) if sym_ret is not None else None,
            "alpha_pct": round(total_return - float(sym_ret), 2) if sym_ret is not None else None,
        }
    benchmark_comparisons["BLEND"] = {
        "data_available": True,
        "benchmark_return_pct": round(primary_bh_return, 2),
        "alpha_pct": round(total_return - primary_bh_return, 2),
        "weights": _normalise_weights(PRIMARY_BENCHMARK_WEIGHTS),
    }
    benchmark_comparisons["ACTIVE_INSTRUMENTS"] = {
        "data_available": True,
        "benchmark_return_pct": round(bh_return, 2),
        "alpha_pct": round(total_return - bh_return, 2),
        "weights": _normalise_weights(instruments),
    }

    # This is a cost diagnostic only; the legacy timing equity curve is left
    # unchanged so reporting upgrades are directly comparable to the baseline.
    # Use the ETF-specific cost (1.5bps default), not the stock slippage (10bps).
    estimated_cost_pct = total_turnover * (SPY_TIMING_ETF_COST_BPS / 100.0)
    avg_daily_turnover_pct = (total_turnover / max(n_days, 1)) * 100.0

    timing_gate_results = {
        "sharpe_pass": bool(sharpe >= MIN_TIMING_SHARPE),
        "drawdown_pass": bool(max_dd >= MAX_TIMING_DRAWDOWN_PCT),
        "nw_tstat_pass": bool(nw_tstat >= MIN_TIMING_NW_TSTAT_VS_CASH),
        "exposure_days_pass": bool(n_invested >= MIN_TIMING_TRADES_OR_EXPOSURE_DAYS),
    }
    timing_gate_results["all_pass"] = all(timing_gate_results.values())
    timing_gate_thresholds = {
        "min_timing_sharpe": MIN_TIMING_SHARPE,
        "max_timing_drawdown_pct": MAX_TIMING_DRAWDOWN_PCT,
        "min_timing_nw_tstat_vs_cash": MIN_TIMING_NW_TSTAT_VS_CASH,
        "min_timing_exposure_days": MIN_TIMING_TRADES_OR_EXPOSURE_DAYS,
    }

    in_market_pct = n_invested / max(n_invested + n_cash, 1) * 100.0
    avg_invest = float(eq_df["invested"].mean()) * 100.0

    metrics = {
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "nw_tstat_vs_cash": round(nw_tstat, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "benchmark_return_pct": round(bh_return, 2),
        "primary_benchmark_return_pct": round(primary_bh_return, 2),
        "benchmark_comparisons": benchmark_comparisons,
        "alpha_pct": round(total_return - bh_return, 2),
        "n_days_invested": n_invested,
        "n_days_cash": n_cash,
        "in_market_pct": round(in_market_pct, 1),
        "avg_allocation_pct": round(avg_invest, 1),
        "turnover_pct": round(total_turnover * 100.0, 2),
        "avg_daily_turnover_pct": round(avg_daily_turnover_pct, 3),
        "estimated_cost_pct": round(estimated_cost_pct, 4),
        "threshold": threshold,
        "invest_pct": invest_pct,
        "instruments": _normalise_weights(instruments),
        "years": round(years, 1),
        "timing_gate_thresholds": timing_gate_thresholds,
        "timing_gate_results": timing_gate_results,
        "paper_ready": bool(timing_gate_results["all_pass"]),
    }

    # ── 5. Print results ─────────────────────────────────────────────────────
    if not quiet:
        print(f"\nSPY TIMING BACKTEST")
        print("-" * 48)
        print(f"Period          : {timeline[0].date()} to {timeline[-1].date()} ({metrics['years']:.1f} years)")
        print(f"Total return    : {total_return:.2f}%")
        print(f"CAGR            : {cagr:.2f}%")
        print(f"Benchmark (B&H) : {bh_return:.2f}%")
        print(f"Alpha           : {total_return - bh_return:+.2f}%")
        print(f"Sharpe          : {sharpe:.3f}")
        print(f"Sortino         : {sortino:.3f}")
        print(f"NW t-stat(0)    : {nw_tstat:.3f}")
        print(f"Max drawdown    : {max_dd:.2f}%")
        print(f"In market       : {in_market_pct:.0f}% of days")
        print(f"Avg allocation  : {avg_invest:.0f}%")
        print(f"Turnover        : {metrics['turnover_pct']:.2f}% total")
        print(f"Est. cost       : {metrics['estimated_cost_pct']:.4f}% of capital")
        print(f"Threshold       : {threshold:.0%} bull fraction")
        print(f"Timing gates    : {'PASS' if timing_gate_results['all_pass'] else 'FAIL'}")
        print("-" * 48)

    if write_outputs:
        eq_out = os.path.join(SIGNAL_DIR, f"{output_prefix}_equity.csv")
        sig_out = os.path.join(SIGNAL_DIR, f"{output_prefix}_signals.csv")
        met_out = os.path.join(SIGNAL_DIR, f"{output_prefix}_metrics.json")
        eq_df.to_csv(eq_out)
        signals_df.to_csv(sig_out, index=False)
        with open(met_out, "w") as f:
            json.dump(metrics, f, indent=2)
        if not quiet:
            print(f"Equity file     : {eq_out}")
            print(f"Signals file    : {sig_out}")
            print(f"Metrics file    : {met_out}")

    return metrics


def _build_daily_vote_fraction(raw_predictions: dict[str, pd.DataFrame]) -> tuple[pd.Series, dict[pd.Timestamp, dict]]:
    cache_key = id(raw_predictions)
    if cache_key in _ETF_VOTE_CACHE:
        cached_bf, cached_votes = _ETF_VOTE_CACHE[cache_key]
        return cached_bf.copy(), {k: dict(v) for k, v in cached_votes.items()}

    long_parts: list[pd.Series] = []
    total_parts: list[pd.Series] = []
    for _ticker, pdf in raw_predictions.items():
        if pdf is None or pdf.empty or "direction_vote" not in pdf.columns:
            continue
        idx = pd.DatetimeIndex(pdf.index)
        direction = pdf["direction_vote"].astype(str)
        long_parts.append(pd.Series((direction == "LONG").astype(int).to_numpy(), index=idx))
        total_parts.append(pd.Series(1, index=idx))
    if not long_parts:
        return pd.Series(dtype=float), {}
    n_long = pd.concat(long_parts).groupby(level=0).sum().sort_index()
    n_total = pd.concat(total_parts).groupby(level=0).sum().reindex(n_long.index).fillna(0).astype(int)
    bull_fractions = (n_long / n_total.clip(lower=1)).astype(float).sort_index()
    date_votes = {
        dt: {"n_long": int(n_long.loc[dt]), "n_total": int(n_total.loc[dt])}
        for dt in bull_fractions.index
    }
    _ETF_VOTE_CACHE[cache_key] = (bull_fractions.copy(), {k: dict(v) for k, v in date_votes.items()})
    return bull_fractions, date_votes


def _load_etf_price_frame(index: pd.DatetimeIndex, symbols: tuple[str, ...] | list[str]) -> pd.DataFrame:
    if index.empty:
        return pd.DataFrame()
    clean_symbols = tuple(sorted({str(sym).upper() for sym in symbols}))
    cache_key = (clean_symbols, str(index.min().date()), str(index.max().date()))
    if cache_key in _ETF_PRICE_FRAME_CACHE:
        return _ETF_PRICE_FRAME_CACHE[cache_key].copy()
    frame = pd.DataFrame(index=index)
    missing: list[str] = []
    for symbol in clean_symbols:
        local_path = os.path.join(DATA_DIR, f"{symbol}.parquet")
        if os.path.exists(local_path):
            try:
                bench_df = pd.read_parquet(local_path)
                bench_df.index = pd.DatetimeIndex(bench_df.index)
                close = bench_df["Close"].reindex(index).ffill().bfill()
                if not close.empty and float(close.iloc[0]) != 0.0:
                    frame[symbol] = close / close.iloc[0]
                    continue
            except Exception:
                pass
        if symbol in _ETF_DOWNLOAD_FAILED_SYMBOLS:
            frame[symbol] = 1.0
        else:
            missing.append(symbol)

    if missing:
        start = index.min().strftime("%Y-%m-%d")
        end = (index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            raw = _download_yfinance(
                missing,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=False,
                timeout=20,
            )
        except Exception:
            raw = pd.DataFrame()
        for symbol in missing:
            close = pd.Series(index=index, data=np.nan, dtype=float)
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if symbol in raw.columns.get_level_values(0):
                        close = raw[(symbol, "Close")]
                    elif "Close" in raw.columns.get_level_values(0):
                        close = raw["Close"][symbol]
                elif "Close" in raw.columns:
                    close = raw["Close"]
            except Exception:
                close = pd.Series(index=index, data=np.nan, dtype=float)
            close = close.reindex(index).ffill().bfill()
            first_close = float(close.iloc[0]) if not close.empty and pd.notna(close.iloc[0]) else 0.0
            if close.empty or close.isna().all() or not np.isfinite(first_close) or first_close == 0.0:
                _ETF_DOWNLOAD_FAILED_SYMBOLS.add(symbol)
                frame[symbol] = 1.0
            else:
                frame[symbol] = close / first_close
    frame = frame.ffill().bfill()
    _ETF_PRICE_FRAME_CACHE[cache_key] = frame.copy()
    return frame


def _portfolio_stats_from_equity(equity: pd.Series) -> dict:
    if equity.empty:
        return {"total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0}
    daily = equity.pct_change().dropna()
    years = max(len(equity) / 252.0, 0.01)
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = (float(equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if equity.iloc[0] else 0.0
    sharpe = float(daily.mean() / (daily.std() + 1e-9) * np.sqrt(252)) if len(daily) else 0.0
    downside = daily[daily < 0]
    sortino = float(daily.mean() / (downside.std() + 1e-9) * np.sqrt(252)) if len(downside) else 0.0
    dd = equity / equity.cummax() - 1.0
    return {
        "total_return_pct": round(total_ret * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(float(dd.min()) * 100.0, 2),
    }


def _target_weights_for_regime(
    regime: str,
    preset: dict,
    *,
    defensive_mix: dict[str, float] | None,
    realized_vol: float,
    vol_target: float,
) -> dict[str, float]:
    if regime == "risk_off" and defensive_mix:
        base = {str(k).upper(): float(v) for k, v in defensive_mix.items() if float(v) > 0}
    else:
        base = {str(k).upper(): float(v) for k, v in preset.get(regime, {}).items() if float(v) > 0}
    if not base:
        return {}

    max_gross = float(preset.get("max_gross", 1.0))
    if regime != "risk_on":
        max_gross = min(max_gross, 1.0)
    else:
        max_gross = min(max_gross, float(ETF_ROTATION_MAX_GROSS_EXPOSURE))

    gross = sum(abs(v) for v in base.values())
    if gross <= 0:
        return {}
    cap_scale = min(1.0, max_gross / gross)
    vol_scale = 1.0
    if realized_vol > 0 and vol_target > 0 and regime in {"risk_on", "neutral"}:
        vol_scale = min(1.0, float(vol_target) / float(realized_vol))
    scale = cap_scale * vol_scale
    return {sym: round(weight * scale, 6) for sym, weight in base.items()}


def _classify_etf_regime(
    *,
    bull_fraction: float,
    prev_regime: str,
    days_in_regime: int,
    spy_trend_ok: bool,
    qqq_trend_ok: bool,
    realized_vol: float,
    vol_target: float,
    enter_threshold: float,
    exit_threshold: float,
    neutral_threshold: float,
    min_hold_days: int,
) -> str:
    high_vol = bool(realized_vol > max(0.20, float(vol_target) * 1.75))
    if not spy_trend_ok or bull_fraction < exit_threshold:
        desired = "risk_off"
    elif bull_fraction >= enter_threshold and qqq_trend_ok and not high_vol:
        desired = "risk_on"
    elif bull_fraction >= neutral_threshold and spy_trend_ok:
        desired = "neutral"
    else:
        desired = "risk_off"

    if prev_regime == "risk_on" and spy_trend_ok and bull_fraction >= exit_threshold and days_in_regime < min_hold_days:
        return "risk_on"
    if prev_regime != desired and days_in_regime < min_hold_days and prev_regime in {"risk_on", "neutral", "risk_off"}:
        if desired != "risk_off" or spy_trend_ok:
            return prev_regime
    return desired


def run_etf_rotation_backtest(
    raw_predictions: dict[str, pd.DataFrame],
    *,
    preset_name: str = "limited_leverage",
    preset: dict | None = None,
    enter_threshold: float = ETF_ROTATION_RISK_ON_ENTER,
    exit_threshold: float = ETF_ROTATION_RISK_ON_EXIT,
    neutral_threshold: float = ETF_ROTATION_NEUTRAL_THRESHOLD,
    vol_target: float = ETF_ROTATION_DEFAULT_VOL_TARGET,
    defensive_mix_name: str = "bil_ief_gld",
    defensive_mix: dict[str, float] | None = None,
    output_prefix: str = "etf_rotation",
    write_outputs: bool = True,
    quiet: bool = False,
) -> dict:
    """Backtest benchmark-relative ETF rotation from the model's market votes."""
    bull_fractions, date_votes = _build_daily_vote_fraction(raw_predictions)
    if bull_fractions.empty:
        if not quiet:
            print("[etf-rotation] No predictions to build rotation signal from")
        return {}

    preset = preset or ETF_ROTATION_PRESETS.get(preset_name) or ETF_ROTATION_PRESETS["limited_leverage"]
    defensive_mix = defensive_mix or ETF_ROTATION_DEFENSIVE_MIXES.get(defensive_mix_name) or ETF_ROTATION_DEFENSIVE_MIXES["bil_ief_gld"]
    timeline = pd.DatetimeIndex(sorted(bull_fractions.index.unique()))
    symbols = tuple(sorted(set(ETF_ROTATION_ASSETS) | {"SPY", "QQQ"}))
    prices = _load_etf_price_frame(timeline, symbols)
    if prices.empty or prices.nunique(dropna=True).max() <= 1:
        if not quiet:
            print("[etf-rotation] No usable ETF price data")
        return {}

    daily_asset_rets = prices.pct_change().shift(-1).fillna(0.0)
    spy_ma200 = prices["SPY"].rolling(200, min_periods=50).mean()
    qqq_ma100 = prices["QQQ"].rolling(100, min_periods=40).mean()
    qqq_realized_vol = prices["QQQ"].pct_change().rolling(20, min_periods=10).std().mul(np.sqrt(252)).fillna(0.0)
    symbols_list = list(symbols)
    symbol_idx = {sym: i for i, sym in enumerate(symbols_list)}
    bf_values = bull_fractions.reindex(timeline).fillna(0.0).to_numpy(dtype=float)
    ret_values = daily_asset_rets[symbols_list].reindex(timeline).fillna(0.0).to_numpy(dtype=float)
    spy_values = prices["SPY"].reindex(timeline).ffill().bfill().to_numpy(dtype=float)
    qqq_values = prices["QQQ"].reindex(timeline).ffill().bfill().to_numpy(dtype=float)
    spy_ma_values = spy_ma200.reindex(timeline).ffill().bfill().to_numpy(dtype=float)
    qqq_ma_values = qqq_ma100.reindex(timeline).ffill().bfill().to_numpy(dtype=float)
    vol_values = qqq_realized_vol.reindex(timeline).fillna(0.0).to_numpy(dtype=float)

    equity = INITIAL_CAPITAL
    equity_rows: list[dict] = []
    signal_rows: list[dict] = []
    prev_weights = np.zeros(len(symbols_list), dtype=float)
    current_weights = np.zeros(len(symbols_list), dtype=float)
    prev_regime = "risk_off"
    days_in_regime = 999
    total_turnover = 0.0
    rebalances = 0

    for i, dt in enumerate(timeline):
        bf = float(bf_values[i])
        realized_vol = float(vol_values[i])
        spy_trend_ok = bool(spy_values[i] >= spy_ma_values[i])
        qqq_trend_ok = bool(qqq_values[i] >= qqq_ma_values[i])
        regime = _classify_etf_regime(
            bull_fraction=bf,
            prev_regime=prev_regime,
            days_in_regime=days_in_regime,
            spy_trend_ok=spy_trend_ok,
            qqq_trend_ok=qqq_trend_ok,
            realized_vol=realized_vol,
            vol_target=vol_target,
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold,
            neutral_threshold=neutral_threshold,
            min_hold_days=ETF_ROTATION_MIN_HOLD_DAYS,
        )
        if regime != prev_regime:
            days_in_regime = 0
        else:
            days_in_regime += 1
        prev_regime = regime

        desired_weights = np.zeros(len(symbols_list), dtype=float)
        for sym, weight in _target_weights_for_regime(
            regime,
            preset,
            defensive_mix=defensive_mix,
            realized_vol=realized_vol,
            vol_target=vol_target,
        ).items():
            if sym in symbol_idx:
                desired_weights[symbol_idx[sym]] = float(weight)

        weight_delta = float(np.abs(desired_weights - current_weights).sum())
        if weight_delta >= ETF_ROTATION_REBALANCE_BAND or not equity_rows:
            current_weights = desired_weights
            rebalances += 1
        turnover = float(np.abs(current_weights - prev_weights).sum())
        total_turnover += turnover
        cost = turnover * SLIPPAGE_BASE_PCT
        prev_weights = current_weights.copy()

        gross = float(np.abs(current_weights).sum())
        invested_ret = float(np.dot(current_weights, ret_values[i]))
        strategy_ret = invested_ret - cost
        equity *= (1.0 + strategy_ret)
        cash_weight = max(0.0, 1.0 - float(current_weights.sum()))

        row = {
            "date": dt,
            "equity": equity,
            "strategy_ret": strategy_ret,
            "bull_fraction": bf,
            "regime": regime,
            "gross_exposure": gross,
            "cash_weight": cash_weight,
            "realized_vol": realized_vol,
            "spy_trend_ok": spy_trend_ok,
            "qqq_trend_ok": qqq_trend_ok,
            "turnover": turnover,
            "cost": cost,
        }
        for sym, j in symbol_idx.items():
            row[f"weight_{sym.lower()}"] = float(current_weights[j])
        equity_rows.append(row)
        signal_rows.append({
            "date": dt,
            "regime": regime,
            "bull_fraction": round(bf, 3),
            "n_long": date_votes[dt]["n_long"],
            "n_total": date_votes[dt]["n_total"],
            "gross_exposure": round(gross, 4),
            "target_spy_weight": round(float(current_weights[symbol_idx.get("SPY", 0)]), 4) if "SPY" in symbol_idx else 0.0,
            "target_qqq_weight": round(float(current_weights[symbol_idx.get("QQQ", 0)]), 4) if "QQQ" in symbol_idx else 0.0,
            "target_bil_weight": round(float(current_weights[symbol_idx.get("BIL", 0)]), 4) if "BIL" in symbol_idx else 0.0,
            "target_ief_weight": round(float(current_weights[symbol_idx.get("IEF", 0)]), 4) if "IEF" in symbol_idx else 0.0,
            "target_gld_weight": round(float(current_weights[symbol_idx.get("GLD", 0)]), 4) if "GLD" in symbol_idx else 0.0,
            "target_cash_weight": round(cash_weight, 4),
        })

    eq_df = pd.DataFrame(equity_rows).set_index("date")
    signals_df = pd.DataFrame(signal_rows)
    equity_curve = eq_df["equity"].astype(float)
    stats = _portfolio_stats_from_equity(equity_curve)
    benchmarks = pd.DataFrame(
        {
            "SPY": prices["SPY"],
            "QQQ": prices["QQQ"],
            "BLEND": prices["SPY"] * 0.60 + prices["QQQ"] * 0.40,
        },
        index=timeline,
    ).ffill().bfill()
    benchmark_comparisons = compute_benchmark_comparisons(equity_curve, benchmarks)
    benchmark_stats = {
        symbol: _portfolio_stats_from_equity((INITIAL_CAPITAL * benchmarks[symbol] / benchmarks[symbol].iloc[0]).astype(float))
        for symbol in benchmarks.columns
    }
    best_benchmark_sharpe = max(float(v.get("sharpe", 0.0)) for v in benchmark_stats.values())
    qqq_drawdown = float(benchmark_stats.get("QQQ", {}).get("max_drawdown_pct", 0.0))
    daily_ret = equity_curve.pct_change().fillna(0.0)
    blend_daily = benchmarks["BLEND"].pct_change().fillna(0.0)
    nw_vs_blend = _newey_west_tstat(daily_ret - blend_daily)

    alpha_spy = float(benchmark_comparisons.get("SPY", {}).get("alpha_pct", -9999.0))
    alpha_qqq = float(benchmark_comparisons.get("QQQ", {}).get("alpha_pct", -9999.0))
    alpha_blend = float(benchmark_comparisons.get("BLEND", {}).get("alpha_pct", -9999.0))
    gate_results = {
        "alpha_vs_spy_pass": bool(alpha_spy > MIN_ETF_ROTATION_ALPHA_VS_SPY_PCT),
        "alpha_vs_qqq_pass": bool(alpha_qqq > MIN_ETF_ROTATION_ALPHA_VS_QQQ_PCT),
        "alpha_vs_blend_pass": bool(alpha_blend > MIN_ETF_ROTATION_ALPHA_VS_BLEND_PCT),
        "sharpe_vs_best_benchmark_pass": bool(float(stats["sharpe"]) >= best_benchmark_sharpe),
        "drawdown_vs_qqq_pass": bool(float(stats["max_drawdown_pct"]) >= qqq_drawdown),
        "nw_tstat_vs_blend_pass": bool(nw_vs_blend > MIN_ETF_ROTATION_NW_TSTAT_VS_BLEND),
    }
    gate_results["all_pass"] = all(gate_results.values())
    metrics = {
        **stats,
        "period_start": str(timeline[0].date()),
        "period_end": str(timeline[-1].date()),
        "years": round(max(len(equity_curve) / 252.0, 0.01), 1),
        "preset": preset_name,
        "defensive_mix": defensive_mix_name,
        "enter_threshold": float(enter_threshold),
        "exit_threshold": float(exit_threshold),
        "neutral_threshold": float(neutral_threshold),
        "vol_target": float(vol_target),
        "max_gross_exposure_allowed": float(min(float(preset.get("max_gross", 1.0)), ETF_ROTATION_MAX_GROSS_EXPOSURE)),
        "max_gross_exposure_used": round(float(eq_df["gross_exposure"].max()), 3),
        "avg_gross_exposure": round(float(eq_df["gross_exposure"].mean()), 3),
        "turnover_pct": round(total_turnover * 100.0, 2),
        "avg_daily_turnover_pct": round(total_turnover / max(len(eq_df), 1) * 100.0, 3),
        "estimated_cost_pct": round(total_turnover * SLIPPAGE_BASE_PCT * 100.0, 4),
        "rebalances": int(rebalances),
        "regime_counts": {str(k): int(v) for k, v in eq_df["regime"].value_counts().to_dict().items()},
        "benchmark_comparisons": benchmark_comparisons,
        "benchmark_stats": benchmark_stats,
        "best_benchmark_sharpe": round(best_benchmark_sharpe, 3),
        "nw_tstat_vs_blend": round(nw_vs_blend, 3),
        "etf_rotation_gate_thresholds": {
            "min_alpha_vs_spy_pct": MIN_ETF_ROTATION_ALPHA_VS_SPY_PCT,
            "min_alpha_vs_qqq_pct": MIN_ETF_ROTATION_ALPHA_VS_QQQ_PCT,
            "min_alpha_vs_blend_pct": MIN_ETF_ROTATION_ALPHA_VS_BLEND_PCT,
            "min_nw_tstat_vs_blend": MIN_ETF_ROTATION_NW_TSTAT_VS_BLEND,
            "sharpe_must_match_best_benchmark": True,
            "drawdown_must_be_no_worse_than_qqq": True,
        },
        "etf_rotation_gate_results": gate_results,
        "paper_ready": bool(gate_results["all_pass"]),
        "single_name_signals_enabled": False,
        "single_name_paper_trading_enabled": False,
    }

    if not quiet:
        print("\nETF ROTATION BACKTEST")
        print("-" * 48)
        print(f"Period          : {metrics['period_start']} to {metrics['period_end']} ({metrics['years']:.1f} years)")
        print(f"Preset          : {preset_name} / defensive={defensive_mix_name}")
        print(f"Total return    : {metrics['total_return_pct']:.2f}%")
        print(f"CAGR            : {metrics['cagr_pct']:.2f}%")
        print(f"Sharpe          : {metrics['sharpe']:.3f} (best benchmark {best_benchmark_sharpe:.3f})")
        print(f"Max drawdown    : {metrics['max_drawdown_pct']:.2f}% (QQQ {qqq_drawdown:.2f}%)")
        print(f"Alpha vs SPY    : {alpha_spy:+.2f}%")
        print(f"Alpha vs QQQ    : {alpha_qqq:+.2f}%")
        print(f"Alpha vs BLEND  : {alpha_blend:+.2f}%")
        print(f"NW t-stat blend : {nw_vs_blend:.3f}")
        print(f"Max gross used  : {metrics['max_gross_exposure_used']:.2f}x")
        print(f"Turnover        : {metrics['turnover_pct']:.2f}% total")
        print(f"ETF gates       : {'PASS' if gate_results['all_pass'] else 'FAIL'}")
        print("-" * 48)

    if write_outputs:
        eq_out = os.path.join(SIGNAL_DIR, f"{output_prefix}_equity.csv")
        sig_out = os.path.join(SIGNAL_DIR, f"{output_prefix}_signals.csv")
        met_out = os.path.join(SIGNAL_DIR, f"{output_prefix}_metrics.json")
        eq_df.to_csv(eq_out)
        signals_df.to_csv(sig_out, index=False)
        with open(met_out, "w") as f:
            json.dump(metrics, f, indent=2)
        if not quiet:
            print(f"Equity file     : {eq_out}")
            print(f"Signals file    : {sig_out}")
            print(f"Metrics file    : {met_out}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict walk-forward backtest aligned to train.py")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--mode", type=str, default="long_only", choices=["long_short", "long_only", "long_only_bear_cash"])
    parser.add_argument("--allow-shorts", action="store_true", help="Backward-compatible alias for --mode long_short.")
    parser.add_argument("--approved-only", action="store_true", help="Backtest only tickers currently approved for live trading.")
    parser.add_argument("--no-eligibility-filter", action="store_true", help="Disable prior walk-forward ticker eligibility filtering.")
    parser.add_argument("--use-trade-rules", action="store_true", help="Apply optimized per-ticker trade rules. Off by default for pooled research backtests.")
    parser.add_argument("--ignore-trade-rules", action="store_true", help="Deprecated; trade rules are ignored unless --use-trade-rules is passed.")
    parser.add_argument("--skip-quality-write", action="store_true", help="Do not update model_quality_report.csv or the experiment ledger.")
    parser.add_argument("--output", choices=["summary", "verbose"], default="summary", help="Console output level. summary prints only model results.")
    parser.add_argument("--stress", type=float, default=1.0)
    parser.add_argument("--spy-timing", action="store_true",
                        help="Run SPY/QQQ market-timing backtest using walk-forward direction votes.")
    parser.add_argument("--etf-rotation", action="store_true",
                        help="Run benchmark-relative ETF rotation using walk-forward direction votes.")
    parser.add_argument(
        "--ablate-blend",
        action="store_true",
        help=(
            "After the main backtest, sweep h20/h5 blend weights and compare Sharpe ratios. "
            "Tested combos: (h20=1.0, h5=0.0), (0.80, 0.20), (0.65, 0.35), (0.50, 0.50). "
            "Only valid for pooled mode — per-ticker path has no h5 model."
        ),
    )
    args = parser.parse_args()
    verbose_output = args.output == "verbose"

    os.makedirs(SIGNAL_DIR, exist_ok=True)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = default_backtest_tickers(approved_only=args.approved_only)

    if not tickers:
        raise SystemExit("No backtest tickers selected")

    # Compute effective_mode early so pooled path can use it for ranking.
    # Keep the default long-only, but make an explicit --mode long_short do
    # what it says. --allow-shorts remains for older runbooks/scripts.
    effective_mode = "long_short" if args.allow_shorts else args.mode
    use_trade_rules = bool(args.use_trade_rules and not args.ignore_trade_rules)
    use_pooled = POOLED_TRAINING and len(tickers) > 1
    use_eligibility_filter = bool(TICKER_ELIGIBILITY_FILTER_ENABLED and not args.no_eligibility_filter)
    raw_pooled_predictions: dict[str, pd.DataFrame] | None = None
    pooled_min_train_rows: int | None = None
    # Do not pre-filter from model_quality_report.csv inside a research backtest:
    # that report may have been written by a prior full-period run, which creates
    # future-informed ticker selection.  Pooled ranking applies a causal rolling
    # eligibility filter using only selected trades whose exits are already known.
    eligible_tickers: set[str] | None = None
    if verbose_output and use_eligibility_filter:
        print("[eligibility] Using causal rolling eligibility inside cross-sectional ranking")

    if use_pooled:
        # ── Pooled cross-sectional walk-forward ───────────────────────────────
        if verbose_output:
            print(f"[backtest] POOLED mode: training one model across {len(tickers)} tickers")
        # Rolling-retrain walk-forward: retrain every 126 trading days (~6 months).
        # min_total_train_rows scales with ticker count so the first block starts
        # after roughly five years of pooled history.
        _min_train = max(20_000, len(tickers) * 1_250)
        pooled_min_train_rows = _min_train
        # Survivorship tickers are intentionally excluded from the walk-forward
        # training pool. Their parquet files contain post-failure data (OTC stubs,
        # successor-company relisting) that produces contradictory labels in recent
        # blocks. They remain in train.py's static pooled model where the single
        # global split isolates the pre-crash period more cleanly.
        if verbose_output:
            raw_predictions = walk_forward_predictions_pooled(
                tickers,
                min_total_train_rows=_min_train,
                test_block=126,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                raw_predictions = walk_forward_predictions_pooled(
                    tickers,
                    min_total_train_rows=_min_train,
                    test_block=126,
                )
        raw_pooled_predictions = {
            t: pdf.copy()
            for t, pdf in raw_predictions.items()
            if pdf is not None and not pdf.empty
        }
        # Apply cross-sectional ranking instead of confidence-threshold filtering
        predictions = apply_cross_sectional_ranking(
            raw_predictions,
            top_n=CROSS_SECTIONAL_TOP_N,
            mode=effective_mode,
            eligible_tickers=eligible_tickers if use_eligibility_filter else None,
            use_eligibility_filter=use_eligibility_filter,
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
                        if verbose_output:
                            print(f"SKIP {ticker}: dir_acc={_acc:.1f}% vs baseline={_baseline:.1f}% (>5pp below, model harmful)")
                        continue
                if eligible_tickers is not None and ticker.upper() not in eligible_tickers:
                    if verbose_output:
                        print(f"SKIP {ticker}: ineligible by prior walk-forward contribution")
                    continue
                predictions[ticker] = walk_forward_predictions_for_ticker(ticker)
            except Exception as e:
                if verbose_output:
                    print(f"ERROR - {ticker}: {e}")

    # Save raw per-ticker prediction DataFrames so we can diagnose whether
    # predictions continue through later years and whether they stop being
    # actionable versus disappearing entirely.
    for ticker, pdf in predictions.items():
        if pdf is None or pdf.empty:
            continue
        pred_out = os.path.join(SIGNAL_DIR, f"{ticker}_walkforward_predictions.csv")
        pdf.reset_index().to_csv(pred_out, index=False)
        if verbose_output:
            print("Saved ->", pred_out)

    predictions = {k: v for k, v in predictions.items() if not v.empty}
    if not predictions:
        raise SystemExit("No walk-forward predictions generated")

    # ── SPY timing backtest (optional) ───────────────────────────────────────
    # Uses the raw walk-forward predictions (before cross-sectional ranking)
    # to compute a daily bull/bear signal and simulate buying SPY/QQQ.
    if args.spy_timing:
        spy_source = raw_pooled_predictions if raw_pooled_predictions is not None else predictions
        run_spy_timing_backtest(spy_source)
    if args.etf_rotation or ETF_ROTATION_MODE:
        etf_source = raw_pooled_predictions if raw_pooled_predictions is not None else predictions
        run_etf_rotation_backtest(etf_source)

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
    edge_audit_out = os.path.join(SIGNAL_DIR, f"{suffix}_edge_audit.csv")
    daily_rank_ic_out = os.path.join(SIGNAL_DIR, f"{suffix}_daily_rank_ic.csv")

    rank_ic_source = raw_pooled_predictions if raw_pooled_predictions is not None else predictions
    daily_rank_ic, daily_rank_ic_df = compute_daily_rank_ic(rank_ic_source)
    daily_rank_ic["source"] = "raw_pooled_predictions" if raw_pooled_predictions is not None else "predictions"
    metrics["daily_rank_ic"] = daily_rank_ic
    daily_rank_ic_df.to_csv(daily_rank_ic_out, index=False)
    metrics["daily_rank_ic_path"] = daily_rank_ic_out

    trades_df.to_csv(trades_out, index=False)
    equity_frame = pd.DataFrame({"equity": equity})
    for symbol in benchmarks.columns:
        equity_frame[f"benchmark_{symbol.lower()}_norm"] = benchmarks[symbol].reindex(equity.index)
    equity_frame.to_csv(equity_out)
    summarize_ticker_breakdown(trades_df).to_csv(breakdown_out, index=False)
    build_trade_attribution_report(trades_df).to_csv(attribution_out, index=False)
    if EDGE_AUDIT_ENABLED:
        build_edge_audit_report(trades_df).to_csv(edge_audit_out, index=False)
        metrics["edge_audit_path"] = edge_audit_out
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    if not args.skip_quality_write:
        quality_tickers = list(predictions.keys())
        if use_eligibility_filter and eligible_tickers is not None:
            # Do not overwrite excluded tickers with zero-trade rejection rows.  The
            # eligibility filter intentionally prevents those names from entering the
            # ranking, so a filtered portfolio run is not fresh evidence that they
            # failed.  Use --no-eligibility-filter for a full universe re-evaluation.
            quality_tickers = [t for t in quality_tickers if t.upper() in eligible_tickers]
            skipped_quality = sorted(set(predictions.keys()) - set(quality_tickers))
            if skipped_quality:
                if verbose_output:
                    print(
                        "[quality] Preserving prior quality rows for eligibility-filtered tickers: "
                        + ", ".join(skipped_quality)
                    )
        if verbose_output:
            quality_report = write_quality_outputs(quality_tickers, trades_df, metrics, "backtest_walkforward")
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                quality_report = write_quality_outputs(quality_tickers, trades_df, metrics, "backtest_walkforward")
        approved_models = int(
            quality_report["approved_for_live"].astype(str).str.lower().isin({"true", "1"}).sum()
        ) if "approved_for_live" in quality_report.columns else 0
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
                "approved_models": approved_models,
                "kill_all_pass": bool(metrics.get("kill_criteria", {}).get("results", {}).get("all_pass", False)),
                "daily_rank_ic_mean": (metrics.get("daily_rank_ic") or {}).get("mean_ic"),
                "daily_rank_ic_tstat": (metrics.get("daily_rank_ic") or {}).get("t_stat"),
            },
            artifacts={
                "trades": trades_out,
                "equity": equity_out,
                "metrics": metrics_out,
                "breakdown": breakdown_out,
                "attribution": attribution_out,
                "edge_audit": edge_audit_out if EDGE_AUDIT_ENABLED else None,
                "daily_rank_ic": daily_rank_ic_out,
            },
        )
    if verbose_output:
        print("Saved ->", trades_out)
        print("Saved ->", equity_out)
        print("Saved ->", metrics_out)
        print("Saved ->", breakdown_out)
        print("Saved ->", attribution_out)
        print("Saved ->", daily_rank_ic_out)
        if EDGE_AUDIT_ENABLED:
            print("Saved ->", edge_audit_out)
        print(metrics)
    else:
        print_model_summary(
            metrics,
            mode=effective_mode,
            tickers=list(predictions.keys()),
            trades_path=trades_out,
            equity_path=equity_out,
            metrics_path=metrics_out,
        )

    # ── Kill-criteria gate ────────────────────────────────────────────────────
    # Print a clear PASS/FAIL summary so the result is impossible to miss.
    # Exit with code 1 so CI / shell scripts can detect a failed backtest.
    kc = metrics["kill_criteria"]
    res = kc["results"]
    thr = kc["thresholds"]
    if verbose_output:
        print("\n" + "=" * 60)
        print("KILL-CRITERIA GATE")
        print("=" * 60)
        print(f"  Sharpe        : {metrics['sharpe_ratio_daily']:.3f}  (need >= {thr['min_sharpe']})  "
              f"{'PASS' if res['sharpe_pass'] else 'FAIL'}")
        print(f"  NW t-stat(0)  : {metrics['nw_tstat_vs_cash']:.3f}  (need >= {thr['min_nw_tstat']})  "
              f"{'PASS' if res['nw_tstat_pass'] else 'FAIL'}  "
              f"[vs {metrics.get('primary_benchmark', 'benchmark')}: {metrics.get('nw_tstat_vs_primary_benchmark', 0.0):.3f}; "
              f"vs SPY: {metrics['nw_tstat_vs_spy']:.3f}]")
        print(f"  Max drawdown  : {metrics['max_drawdown_pct']:.1f}%  (need >= {thr['max_drawdown']*100:.0f}%)  "
              f"{'PASS' if res['drawdown_pass'] else 'FAIL'}")
        print(f"  Raw stk Sharpe: {metrics['raw_strategy_sharpe_ratio_daily']:.3f}  "
              f"(need >= {thr['min_raw_stock_sharpe']})  "
              f"{'PASS' if res['raw_stock_sharpe_pass'] else 'FAIL'}")
        print(f"  Raw stk return: {metrics['raw_strategy_total_return_pct']:.2f}%  "
              f"(need >= {thr['min_raw_stock_return']*100:.1f}%)  "
              f"{'PASS' if res['raw_stock_return_pass'] else 'FAIL'}")
        print("-" * 60)
        if res["all_pass"]:
            print("RESULT: ALL GATES PASSED - strategy may advance to paper trading")
        else:
            failed = [k.replace("_pass", "") for k, v in res.items() if k != "all_pass" and not v]
            print(f"RESULT: FAILED - gates not cleared: {', '.join(failed)}")
            print("Strategy should NOT advance to paper trading.")
        print("=" * 60 + "\n")

    gate_failed = not bool(res["all_pass"])

    # ── Blend-weight ablation ─────────────────────────────────────────────────
    # Run only when the user passes --ablate-blend AND we are in pooled mode
    # (per-ticker path has no h5 model, so p_up_h20 / p_up_h5 won't be in the
    # prediction DataFrames).
    #
    # PLAIN ENGLISH: We want to know if the default h20/h5 blend is optimal, or
    # whether a different mix gives a better Sharpe. Non-default combos rerun
    # pooled walk-forward so meta-labelling and ranking see the same blend that
    # would be used in production.
    if args.ablate_blend and use_pooled:
        _COMBOS = [
            (1.00, 0.00, "h20-only"),
            (0.80, 0.20, "80/20"),
            (0.65, 0.35, "65/35 (default)"),
            (0.50, 0.50, "50/50"),
        ]
        _ablation_rows: list[dict] = []
        _current_weights = (float(MULTI_HORIZON_H20_WEIGHT), float(MULTI_HORIZON_H5_WEIGHT))

        print("\n" + "=" * 60)
        print("BLEND-WEIGHT ABLATION  (pooled walk-forward)")
        print("=" * 60)
        print(f"{'Combo':<22}  {'w20':>6}  {'w5':>6}  {'Sharpe':>8}  {'Ret%':>8}  {'Trades':>7}")
        print("-" * 60)

        for w20, w5, label in _COMBOS:
            try:
                if (
                    raw_pooled_predictions is not None
                    and abs(w20 - _current_weights[0]) < 1e-12
                    and abs(w5 - _current_weights[1]) < 1e-12
                ):
                    _abl_preds = {t: pdf.copy() for t, pdf in raw_pooled_predictions.items()}
                else:
                    if pooled_min_train_rows is None:
                        pooled_min_train_rows = max(20_000, len(tickers) * 1_250)
                    with contextlib.redirect_stdout(io.StringIO()):
                        _abl_preds = walk_forward_predictions_pooled(
                            tickers,
                            min_total_train_rows=pooled_min_train_rows,
                            test_block=126,
                            blend_weights=(w20, w5),
                            training_only_tickers=SURVIVORSHIP_TRAINING_TICKERS,
                        )

                _abl_ranked = apply_cross_sectional_ranking(
                    _abl_preds,
                    top_n=CROSS_SECTIONAL_TOP_N,
                    mode=effective_mode,
                    eligible_tickers=None,
                    use_eligibility_filter=use_eligibility_filter,
                )
                _abl_ranked = {k: v for k, v in _abl_ranked.items() if not v.empty}
                if not _abl_ranked:
                    print(f"  {label:<22}  {w20:>6.2f}  {w5:>6.2f}   NO TRADES")
                    continue
                # Suppress all internal prints during the portfolio simulation.
                with contextlib.redirect_stdout(io.StringIO()):
                    _abl_trades, _abl_equity, _, _abl_metrics = run_portfolio_backtest(
                        _abl_ranked,
                        effective_mode,
                        args.stress,
                        use_trade_rules=use_trade_rules,
                    )
                _sharpe = float(_abl_metrics.get("sharpe_ratio_daily", float("nan")))
                _ret    = float(_abl_metrics.get("total_return_pct", float("nan")))
                _ntrades = int(len(_abl_trades))
                _ablation_rows.append({
                    "combo": label, "w20": w20, "w5": w5,
                    "sharpe": round(_sharpe, 4),
                    "total_return_pct": round(_ret, 2),
                    "n_trades": _ntrades,
                })
                print(f"  {label:<22}  {w20:>6.2f}  {w5:>6.2f}  {_sharpe:>8.4f}  {_ret:>8.2f}%  {_ntrades:>7d}")
            except Exception as _exc:
                print(f"  {label:<22}  ERROR: {_exc}")

        _valid_rows = [r for r in _ablation_rows if np.isfinite(float(r.get("sharpe", float("nan"))))]
        _best_row = max(_valid_rows, key=lambda r: float(r["sharpe"])) if _valid_rows else None
        if _best_row is not None:
            print("-" * 60)
            print(
                f"BEST: {_best_row['combo']}  "
                f"Sharpe={_best_row['sharpe']}  Ret%={_best_row['total_return_pct']}  "
                f"Trades={_best_row['n_trades']}"
            )
        print("=" * 60 + "\n")

        # Save ablation results alongside the main backtest outputs.
        _abl_out = os.path.join(SIGNAL_DIR, "blend_weight_ablation.json")
        with open(_abl_out, "w") as _f:
            json.dump(
                {
                    "ablation": _ablation_rows,
                    "best": _best_row,
                    "run_at": pd.Timestamp.now().isoformat(),
                    "use_eligibility_filter": use_eligibility_filter,
                    "mode": effective_mode,
                },
                _f,
                indent=2,
            )
        print(f"Blend ablation results saved → {_abl_out}")

    if gate_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
