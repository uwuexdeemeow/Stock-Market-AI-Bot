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
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from confidence_calibration import fit_direction_calibrator, calibrate_p_up
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
)
from xgb_feature_engineering import build_xgb_matrix
from portfolio_manager import PortfolioRiskManager, ProposedTrade

INITIAL_CAPITAL = 10_000.0
BASE_POSITION_SIZE_PCT = 0.15
MAX_POSITION_SIZE_PCT = 0.30
HIGH_SIGNAL_BOOST = 1.25
RETURN_BIN_CENTRES = np.array([-0.04, -0.02, 0.00, 0.02, 0.04], dtype=float)
N_RETURN_BINS = len(RETURN_BIN_CENTRES)


def make_direction_target(close: pd.Series) -> np.ndarray:
    return (close.pct_change(RETURN_HORIZON_DAYS).shift(-RETURN_HORIZON_DAYS).fillna(0.0).values > 0).astype(np.int64)


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


def assign_signal_quality(confidence: float, bucket_report: list[dict]) -> str:
    for row in bucket_report:
        lo, hi = row["bucket"].split("-")
        lo = float(lo); hi = float(hi)
        if confidence >= lo and (confidence < hi or hi >= 100):
            precision = row.get("precision")
            if precision is None:
                return "MEDIUM"
            if precision >= 0.65:
                return "HIGH"
            if precision >= 0.55:
                return "MEDIUM"
            return "LOW"
    return "MEDIUM"


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
    y_dir = make_direction_target(df["Close"])
    y_ret = make_return_target(df["Close"])
    return df, X_all, y_dir, y_ret


def build_benchmark(index: pd.DatetimeIndex) -> pd.Series:
    start = index.min().strftime("%Y-%m-%d")
    end = (index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    close = spy["Close"].reindex(index).ffill().bfill()
    return close / close.iloc[0]


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
        bucket_report = confidence_buckets(cal_probs_raw, train_y_dir[calib_start:calib_end])
        X_oos = scaler.transform(X_all[start:end])
        dir_probs = dir_model.predict_proba(X_oos)
        ret_probs = ret_model.predict_proba(X_oos)
        for j, idx in enumerate(range(start, end)):
            p_up = float(calibrate_p_up(calibrator, float(dir_probs[j][1]))) if calibrator is not None else float(dir_probs[j][1])
            p_down = 1.0 - p_up
            confidence = min(max(max(p_up, p_down) * 100.0, 50.0), 99.0)
            expected_return = float((ret_probs[j][:N_RETURN_BINS] * np.array([-0.04,-0.02,0.0,0.02,0.04])).sum()) * 100.0
            signal = "LONG" if expected_return >= 0 else "SHORT"
            direction_vote = "LONG" if p_up >= p_down else "SHORT"
            signal_quality = assign_signal_quality(confidence, bucket_report)
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
                "actionable": bool(confidence >= threshold and signal == "LONG"),
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


def run_portfolio_backtest(predictions_by_ticker: dict[str, pd.DataFrame], mode: str, stress: float):
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
                net = gross - pos["commission_entry"] - pos["commission_exit"]
                trade_rows.append({
                    "date": pos["signal_date"], "entry_date": pos["entry_date"], "exit_date": pos["exit_date"],
                    "ticker": pos["ticker"], "signal": pos["signal"], "signal_quality": pos["signal_quality"],
                    "confidence": round(pos["confidence"], 2), "expected_return": round(pos["expected_return"], 2),
                    "position_pct": round(pos["requested_position_pct"] * 100, 2), "regime_state": pos["regime_state"],
                    "entry_price": round(pos["entry_price"], 4), "exit_price": round(exit_px, 4),
                    "holding_days": int(pos["holding_days"]), "net_pnl": round(net, 2),
                    "sector": SECTOR_MAP.get(pos["ticker"], "OTHER"),
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
            signal = str(row["signal"])
            if mode == "long_only" and signal != "LONG":
                continue
            if mode == "long_only_bear_cash":
                if reg_state == "bear":
                    continue
                if signal != "LONG":
                    continue
            candidates.append(ProposedTrade(
                ticker=ticker,
                date=dt,
                signal=signal,
                confidence=float(row["confidence"]),
                expected_return=float(row["expected_return"]),
                requested_position_pct=requested_position_pct(float(row["confidence"]), float(row["expected_return"]), str(row["signal_quality"])) * reg_mult,
            ))
        current_equity = mark_to_market(dt)
        eq_series = pd.Series(index=pd.DatetimeIndex(equity_index), data=equity_curve, dtype=float) if equity_curve else pd.Series(dtype=float)
        approved = manager.approve_day(candidates, {k: v["Close"] for k, v in price_history.items()}, eq_series)
        eff_slip = SLIPPAGE_BASE_PCT * stress
        for tr in approved:
            row = predictions_by_ticker[tr.ticker].loc[dt]
            entry = float(row["open_next"])
            exit_px = float(row["exit_future_close"])
            if tr.signal == "LONG":
                entry *= (1 + eff_slip)
                exit_px *= (1 - eff_slip)
            else:
                entry *= (1 - eff_slip)
                exit_px *= (1 + eff_slip)
            trade_value = current_equity * tr.requested_position_pct
            shares = trade_value / max(entry, 1e-9)
            commission_entry = shares * COMMISSION_PER_SHARE
            commission_exit = shares * COMMISSION_PER_SHARE
            if tr.signal == "LONG":
                cash -= shares * entry + commission_entry
                margin_reserved = 0.0
            else:
                cash -= trade_value + commission_entry
                margin_reserved = trade_value
            open_positions.append({
                "signal_date": dt,
                "entry_date": pd.Timestamp(row["entry_date"]),
                "exit_date": pd.Timestamp(row["exit_date"]),
                "ticker": tr.ticker,
                "signal": tr.signal,
                "signal_quality": str(row["signal_quality"]),
                "confidence": tr.confidence,
                "expected_return": tr.expected_return,
                "requested_position_pct": tr.requested_position_pct,
                "regime_state": reg_state,
                "entry_price": entry,
                "exit_price": exit_px,
                "holding_days": int(row["holding_days"]),
                "shares": shares,
                "commission_entry": commission_entry,
                "commission_exit": commission_exit,
                "margin_reserved": margin_reserved,
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
    metrics = {
        "total_return_pct": round(total_ret * 100, 2),
        "annual_return_pct": round(ann_ret * 100, 2),
        "benchmark_return_pct": round(bench_ret * 100, 2),
        "alpha_pct": round((total_ret - bench_ret) * 100, 2),
        "sharpe_ratio_daily": round(sharpe, 3),
        "max_drawdown_pct": round(float(dd.min()) * 100, 2) if len(dd) else 0.0,
        "slippage_assumption_bps": round(SLIPPAGE_BASE_PCT * stress * 10_000, 1),
        "mode": mode,
        "diagnostics_2022": compute_period_metrics(equity, benchmark, "2022-01-01", "2022-12-31"),
        "signal_quality": summarize_signal_quality(pd.DataFrame(trade_rows)),
    }
    return pd.DataFrame(trade_rows), equity, benchmark, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict walk-forward backtest aligned to train.py")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--mode", type=str, default="long_only", choices=["long_short", "long_only", "long_only_bear_cash"])
    parser.add_argument("--stress", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(SIGNAL_DIR, exist_ok=True)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = [f.replace(".parquet", "") for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".parquet")]

    predictions = {}
    for ticker in tickers:
        try:
            predictions[ticker] = walk_forward_predictions_for_ticker(ticker)
        except Exception as e:
            print(f"ERROR - {ticker}: {e}")
    predictions = {k: v for k, v in predictions.items() if not v.empty}
    if not predictions:
        raise SystemExit("No walk-forward predictions generated")

    trades_df, equity, benchmark, metrics = run_portfolio_backtest(predictions, args.mode, args.stress)
    suffix = tickers[0] if args.ticker and len(tickers) == 1 else f"walkforward_{args.mode}_{len(predictions)}tickers"
    trades_out = os.path.join(SIGNAL_DIR, f"{suffix}_trades.csv")
    equity_out = os.path.join(SIGNAL_DIR, f"{suffix}_equity.csv")
    metrics_out = os.path.join(SIGNAL_DIR, f"{suffix}_metrics.json")
    breakdown_out = os.path.join(SIGNAL_DIR, f"{suffix}_ticker_breakdown.csv")
    trades_df.to_csv(trades_out, index=False)
    pd.DataFrame({"equity": equity, "benchmark_spy_norm": benchmark.reindex(equity.index)}).to_csv(equity_out)
    summarize_ticker_breakdown(trades_df).to_csv(breakdown_out, index=False)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved →", trades_out)
    print("Saved →", equity_out)
    print("Saved →", metrics_out)
    print("Saved →", breakdown_out)
    print(metrics)


if __name__ == "__main__":
    main()
