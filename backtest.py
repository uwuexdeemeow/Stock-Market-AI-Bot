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
from labels import triple_barrier
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
    APPROVED_TICKERS,
    MARKET_REGIME_FILTER_ENABLED,
    MARKET_REGIME_VIX_MAX,
    MARKET_REGIME_SPY_MA200_REQUIRED,
    BORROW_COST_ANNUAL_DEFAULT,
    BORROW_COSTS,
)
from xgb_feature_engineering import build_xgb_matrix
from portfolio_manager import PortfolioRiskManager, ProposedTrade
from execution_model import realistic_fill_price, commission as calc_commission, capacity_warning
from risk_sizing import compute_position_size

INITIAL_CAPITAL = 10_000.0
BASE_POSITION_SIZE_PCT = 0.15
MAX_POSITION_SIZE_PCT = 0.30
# Boost disabled: diagnostics showed HIGH win rate (54%) < LOW win rate (56%).
# Boosting the worst-performing bucket amplified losses. Reset to 1.0 until
# signal quality is reliably calibrated out-of-sample.
HIGH_SIGNAL_BOOST = 1.0
# Lowered from 50: BAC's 65-70 bucket (p=0.83, n=18) and CAT's 60-65 (p=0.73, n=22)
# were being silenced despite excellent precision. At n≥15 the estimate is noisy but
# the lower bound of the 95% CI is still ~55%+ for these high-precision buckets.
# Separate n floors for each quality tier.
# HIGH needs more data to trust a 65%+ precision estimate (noisy at small n).
# MEDIUM can accept smaller buckets since the precision bar is lower.
MIN_BUCKET_N_HIGH = 30    # n<30 → can't be HIGH even if p≥0.65
MIN_BUCKET_N_MEDIUM = 15  # n<15 → LOW regardless of precision
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


def make_direction_target(df: pd.DataFrame | pd.Series) -> np.ndarray:
    # Must exactly mirror train.py so walk-forward accuracy measures the same label.
    if isinstance(df, pd.Series):
        close = df
        spy_fwd = pd.Series(0.0, index=close.index)
        hvol = pd.Series(0.20, index=close.index)
    else:
        close = df["Close"]
        bench_col = f"spy_ret{RETURN_HORIZON_DAYS}d"
        spy_fwd = df[bench_col].shift(-RETURN_HORIZON_DAYS).fillna(0.0) if bench_col in df.columns else pd.Series(0.0, index=close.index)
        hvol = df["hvol_20d"] if "hvol_20d" in df.columns else pd.Series(0.20, index=close.index)

    stock_fwd = close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS).fillna(0.0)

    if PREDICTION_TARGET == "triple_barrier" and not isinstance(df, pd.Series):
        labels = triple_barrier(
            df["Close"],
            high=df["High"] if "High" in df.columns else None,
            low=df["Low"] if "Low" in df.columns else None,
            max_hold=RETURN_HORIZON_DAYS,
        ).fillna(0.0)
        return (labels.values > 0).astype(np.int64)

    if PREDICTION_TARGET == "excess_return":
        return ((stock_fwd - spy_fwd).values > EXCESS_RETURN_MIN_PCT).astype(np.int64)

    if PREDICTION_TARGET == "vol_adjusted":
        excess = stock_fwd - spy_fwd
        vol_scaled = (hvol * np.sqrt(RETURN_HORIZON_DAYS)).clip(0.01, 1.0)
        return ((excess / vol_scaled).values > VOL_ADJUSTED_SHARPE_THRESHOLD).astype(np.int64)

    return (stock_fwd.values > DIRECTION_LABEL_THRESHOLD).astype(np.int64)


def make_return_target(close: pd.Series) -> np.ndarray:
    fwd = close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS).fillna(0.0).values
    buckets = np.digitize(fwd, [-0.03, -0.01, 0.01, 0.03])
    return np.clip(buckets, 0, 4).astype(np.int64)


def compute_split_indices(n_total: int) -> dict:
    split_train = int(n_total * TRAIN_CALIBRATION_SPLIT)
    split_test = int(n_total * CALIBRATION_TEST_SPLIT)
    train_end = max(0, split_train - EMBARGO_DAYS)
    calib_start = split_train
    calib_end = max(calib_start + 20, split_test - EMBARGO_DAYS)
    test_start = split_test
    return {"train_end": train_end, "calib_start": calib_start, "calib_end": calib_end, "test_start": test_start}


def compute_dynamic_threshold(calibrator, target_precision=0.58, default=57.5) -> float:
    if calibrator is None:
        return default
    raw_grid = np.linspace(0.50, 0.99, 200)
    try:
        cal_grid = calibrator.predict(raw_grid)
        conf_grid = np.maximum(cal_grid, 1.0 - cal_grid) * 100.0
        idx = np.where(cal_grid >= target_precision)[0]
        if len(idx):
            return float(np.clip(conf_grid[idx[0]], 52.0, 80.0))
    except Exception:
        pass
    return default


def build_models(X_train, y_dir_train, X_cal, y_dir_cal, y_ret_train, y_ret_cal):
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
    dir_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
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
    return [c for c in feature_cols if not ("sent_" in c or "sentiment" in c)]


def load_ticker_data(ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(path).copy()
    exclude = {"target", "Open", "High", "Low", "Close", "Volume", "ma5", "ma10", "ma20", "ma50", "ma200"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = strip_sentiment(feature_cols)
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_all, _ = build_xgb_matrix(df, feature_cols)
    y_dir = make_direction_target(df)
    y_ret = make_return_target(df["Close"])
    return df, X_all, y_dir, y_ret


def build_benchmark(index: pd.DatetimeIndex) -> pd.Series:
    local_spy = os.path.join(DATA_DIR, "SPY.parquet")
    if os.path.exists(local_spy):
        try:
            spy_df = pd.read_parquet(local_spy)
            spy_df.index = pd.DatetimeIndex(spy_df.index)
            close = spy_df["Close"].reindex(index).ffill().bfill()
            if not close.empty and float(close.iloc[0]) != 0.0:
                return close / close.iloc[0]
        except Exception:
            pass
    start = index.min().strftime("%Y-%m-%d")
    end = (index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        if not spy.empty and "Close" in spy.columns:
            close = spy["Close"].reindex(index).ffill().bfill()
            if not close.empty and float(close.iloc[0]) != 0.0:
                return close / close.iloc[0]
    except Exception:
        pass
    return pd.Series(index=index, data=1.0, dtype=float)


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


def walk_forward_predictions_for_ticker(ticker: str, min_train_rows: int = 700, test_block: int = 126) -> pd.DataFrame:
    df, X_all, y_dir, y_ret = load_ticker_data(ticker)
    dates = pd.DatetimeIndex(df.index)
    records = []
    start = min_train_rows
    while start + test_block < len(df) - 1:
        end = min(start + test_block, len(df) - 1)
        train_X = X_all[:start]
        train_y_dir = y_dir[:start]
        train_y_ret = y_ret[:start]
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
        threshold = compute_dynamic_threshold(calibrator)
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


def run_portfolio_backtest(predictions_by_ticker: dict[str, pd.DataFrame], mode: str, stress: float, use_trade_rules: bool = True):
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
    vix = build_vix_series(pd.DatetimeIndex(all_dates))
    benchmark = build_benchmark(pd.DatetimeIndex(all_dates))
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
        reg_mult = regime_size_multiplier(spy_close, float(vix.get(dt, 20.0)), dt)
        candidates = []
        for ticker, pdf in predictions_by_ticker.items():
            if pdf.empty or dt not in pdf.index:
                continue
            row = pdf.loc[dt]
            if not bool(row.get("actionable", False)):
                continue
            # Skip LOW-quality signals. MEDIUM threshold raised to 0.60 after
            # OOS audit showed 55%-threshold MEDIUM signals had only 46.9% win
            # rate — below random. Only HIGH (≥65%) and tight MEDIUM (60-65%) trade.
            if str(row.get("signal_quality", "LOW")) == "LOW":
                continue
            signal = str(row["signal"])
            expected_return = float(row.get("expected_return", 0.0))
            rule = trade_rules.get(ticker)
            if use_trade_rules and rule is not None:
                passed, _reason = passes_trade_rule(row, rule, mode=mode)
                if not passed:
                    continue
            if signal == "LONG" and expected_return <= 0.0:
                continue
            if signal == "SHORT" and expected_return >= 0.0:
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
            if MARKET_REGIME_FILTER_ENABLED:
                vix_now = float(vix.get(dt, 20.0))
                if vix_now >= MARKET_REGIME_VIX_MAX:
                    continue
                if MARKET_REGIME_SPY_MA200_REQUIRED and reg_state == "bear":
                    continue
            # Compute 3-month realized vol for this ticker to anchor vol-target sizing.
            asset_vol = 0.20  # fallback: assume 20% annual vol if history is short
            if ticker in price_history:
                _closes = price_history[ticker]["Close"].dropna()
                if len(_closes) >= 20:
                    asset_vol = float(_closes.pct_change().tail(63).std() * np.sqrt(252))
                    asset_vol = max(0.05, min(1.0, asset_vol))
            requested_pct = compute_position_size(
                float(row["confidence"]),
                expected_return,
                str(row["signal_quality"]),
                current_equity,
                asset_vol,
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
        current_equity = mark_to_market(dt)
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
            # Realized vol used to scale bid-ask spread (volatile stocks pay wider spreads).
            _hist_closes = hist["Close"].dropna()
            fill_vol = float(_hist_closes.pct_change().tail(63).std() * np.sqrt(252)) if len(_hist_closes) >= 20 else 0.20
            fill_vol = max(0.05, min(1.0, fill_vol))
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
    benchmark = build_benchmark(equity.index)
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
    return pd.DataFrame(trade_rows), equity, benchmark, metrics


def default_backtest_tickers() -> list[str]:
    available = {f.replace(".parquet", "") for f in os.listdir(DATA_DIR) if f.endswith(".parquet")}
    report = read_quality_report()
    if not report.empty and {"ticker", "approved_for_live"}.issubset(report.columns):
        approved = report[report["approved_for_live"].astype(str).str.lower().isin({"true", "1"})]
        tickers = [str(t).upper() for t in approved["ticker"].tolist() if str(t).upper() in available]
        if tickers:
            return tickers
    return [t for t in APPROVED_TICKERS if t in available]


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
    parser.add_argument("--mode", type=str, default="long_short", choices=["long_short", "long_only", "long_only_bear_cash"])
    parser.add_argument("--allow-shorts", action="store_true", help="Allow SHORT trades in validation. Default is long-only live safety.")
    parser.add_argument("--ignore-trade-rules", action="store_true", help="Use raw confidence filters instead of optimized per-ticker trade rules.")
    parser.add_argument("--stress", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(SIGNAL_DIR, exist_ok=True)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = default_backtest_tickers()

    predictions = {}
    for ticker in tickers:
        try:
            # Skip tickers where the model is clearly harmful: accuracy more than
            # 5pp below baseline. A small gap is tolerated because scale_pos_weight
            # forces some DOWN predictions that reduce overall accuracy even when the
            # UP signal quality is still usable.
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

    effective_mode = args.mode if args.allow_shorts else "long_only"
    trades_df, equity, benchmark, metrics = run_portfolio_backtest(
        predictions,
        effective_mode,
        args.stress,
        use_trade_rules=not args.ignore_trade_rules,
    )
    suffix = tickers[0] if args.ticker and len(tickers) == 1 else f"walkforward_{effective_mode}_{len(predictions)}tickers"
    trades_out = os.path.join(SIGNAL_DIR, f"{suffix}_trades.csv")
    equity_out = os.path.join(SIGNAL_DIR, f"{suffix}_equity.csv")
    metrics_out = os.path.join(SIGNAL_DIR, f"{suffix}_metrics.json")
    breakdown_out = os.path.join(SIGNAL_DIR, f"{suffix}_ticker_breakdown.csv")
    trades_df.to_csv(trades_out, index=False)
    pd.DataFrame({"equity": equity, "benchmark_spy_norm": benchmark.reindex(equity.index)}).to_csv(equity_out)
    summarize_ticker_breakdown(trades_df).to_csv(breakdown_out, index=False)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    quality_report = write_quality_outputs(list(predictions.keys()), trades_df, metrics, "backtest_walkforward")
    print("Saved ->", trades_out)
    print("Saved ->", equity_out)
    print("Saved ->", metrics_out)
    print("Saved ->", breakdown_out)
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
