"""
scanner.py — Production scanner with market-regime awareness and model-status tagging.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

from settings import (
    WATCHLIST,
    TOP_N_STOCKS,
    SHORTLIST_FILE,
    LOG_DIR,
    MODEL_DIR,
    MIN_PRICE,
    LIVE_SHORTS_ENABLED,
)

os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "scanner.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("scanner")


def flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def score_volume_spike(df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df.columns or len(df) < 21:
        return 0.0
    recent = df["Volume"].iloc[-21:]
    avg_volume = recent.iloc[:-1].mean()
    today_volume = recent.iloc[-1]
    if avg_volume <= 0:
        return 0.0
    return float(min((today_volume / avg_volume) / 3.0, 1.0))


def score_rsi_extreme(close: pd.Series) -> float:
    if len(close) < 15:
        return 0.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))
    current_rsi = rsi.iloc[-1]
    if pd.isna(current_rsi):
        return 0.0
    return float(min(abs(current_rsi - 50) / 45.0, 1.0))


def score_recent_volatility(close: pd.Series) -> float:
    if len(close) < 6:
        return 0.0
    avg_move_pct = close.pct_change().abs().iloc[-5:].mean() * 100
    return float(min(avg_move_pct / 3.0, 1.0))


def score_price_momentum(close: pd.Series) -> float:
    if len(close) < 6:
        return 0.0
    five_day_return = close.iloc[-1] / close.iloc[-6] - 1
    daily_returns = close.pct_change().iloc[-5:]
    consistent_days = (daily_returns > 0).sum() if five_day_return > 0 else (daily_returns < 0).sum()
    consistency_bonus = consistent_days / 5.0
    magnitude = min(abs(five_day_return) / 0.10, 1.0)
    return float(magnitude * consistency_bonus)


def score_market_regime() -> float:
    try:
        spy = yf.download("SPY", period="3mo", progress=False, auto_adjust=True)
        spy = flatten_yf(spy)
        if spy.empty or len(spy) < 50:
            return 1.0
        close = spy["Close"]
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        current = close.iloc[-1]
        if current < ma20.iloc[-1] < ma50.iloc[-1]:
            return 0.3
        if current < ma20.iloc[-1]:
            return 0.6
        return 1.0
    except Exception:
        return 1.0


def model_status(ticker: str) -> str:
    scaler = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    dir_json = os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.json")
    ret_json = os.path.join(MODEL_DIR, f"{ticker}_xgb_ret.json")
    dir_pkl = os.path.join(MODEL_DIR, f"{ticker}_xgb_dir.pkl")
    ret_pkl = os.path.join(MODEL_DIR, f"{ticker}_xgb_ret.pkl")
    calibrator = os.path.join(MODEL_DIR, f"{ticker}_direction_calibrator.pkl")

    has_scaler = os.path.exists(scaler)
    has_dir = os.path.exists(dir_json) or os.path.exists(dir_pkl)
    has_ret = os.path.exists(ret_json) or os.path.exists(ret_pkl)
    has_cal = os.path.exists(calibrator)

    if has_scaler and has_dir and has_ret and has_cal:
        return "ready"
    if has_scaler and has_dir and has_ret:
        return "ready_no_calibrator"
    if has_scaler or has_dir or has_ret:
        return "partial_model"
    return "needs_research"


def scan_ticker(ticker: str) -> dict | None:
    try:
        df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
        df = flatten_yf(df)
        if df.empty or len(df) < 20:
            return None

        current_price = float(df["Close"].iloc[-1])
        if current_price < MIN_PRICE:
            return None

        close = df["Close"]
        vol_score = score_volume_spike(df)
        rsi_score = score_rsi_extreme(close)
        vola_score = score_recent_volatility(close)
        mom_score = score_price_momentum(close)
        regime = score_market_regime()
        combined = (vol_score * 0.35 + vola_score * 0.25 + rsi_score * 0.25 + mom_score * 0.15) * regime
        status = model_status(ticker)

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "score_volume": round(vol_score, 3),
            "score_rsi": round(rsi_score, 3),
            "score_vola": round(vola_score, 3),
            "score_mom": round(mom_score, 3),
            "market_regime_mult": round(regime, 3),
            "score_total": round(combined, 3),
            "model_status": status,
            "live_shorts_enabled": bool(LIVE_SHORTS_ENABLED),
            "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except Exception as e:
        log.error("scan failed for %s: %s", ticker, e)
        return None


def run_scanner(watchlist: list[str], top_n: int) -> pd.DataFrame:
    results = [r for r in (scan_ticker(t) for t in watchlist) if r is not None]
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("score_total", ascending=False).reset_index(drop=True)
    shortlist = df.head(top_n).copy()

    os.makedirs(os.path.dirname(SHORTLIST_FILE), exist_ok=True)
    shortlist.to_csv(SHORTLIST_FILE, index=False)

    shortlist[shortlist["model_status"] == "needs_research"].to_csv(
        os.path.join(os.path.dirname(SHORTLIST_FILE), "shortlist_needs_research.csv"),
        index=False,
    )
    shortlist[shortlist["model_status"].isin(["ready", "ready_no_calibrator", "partial_model"])].to_csv(
        os.path.join(os.path.dirname(SHORTLIST_FILE), "shortlist_model_ready.csv"),
        index=False,
    )
    return shortlist


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan watchlist and rank interesting stocks")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--top", type=int, default=TOP_N_STOCKS)
    args = parser.parse_args()

    watchlist = ["AAPL", "MSFT", "NVDA"] if args.test else WATCHLIST
    df = run_scanner(watchlist, args.top)
    if df.empty:
        print("No results.")
        sys.exit(1)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
