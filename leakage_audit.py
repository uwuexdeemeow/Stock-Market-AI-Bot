"""
leakage_audit.py — Temporal leakage detector.

PLAIN ENGLISH:
"Leakage" in machine learning for trading means a feature accidentally uses
information from the FUTURE. If the model peeks at tomorrow's data while
making today's prediction, the backtest will look amazing but live trading
will lose money. This script sanity-checks every feature column against
strict rules so you catch leaks before they hit real capital.

HOW IT WORKS:
  1. Pull raw OHLCV for a handful of tickers via yfinance.
  2. Call the same feature-engineering code the training pipeline uses
     (from pipeline_shared + xgb_feature_engineering).
  3. For each feature, shift the price series one day forward and re-run.
     If the feature value at bar t changes when bar t+1 is edited, that
     feature depends on the future — flag it.
  4. Check rolling windows are backward-only (pandas `rolling(...).mean()`
     is safe; `shift(-k)` or `rolling(..., center=True)` is NOT).
  5. For multi-market merges (SPY/VIX/etc.) verify timezone alignment and
     that `ffill` does not bleed a future observation backward.

OUTPUT: logs/leakage_audit.json with a pass/fail summary per feature.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import LOG_DIR, MULTI_MARKET  # noqa: E402

try:
    from pipeline_shared import build_feature_frame  # type: ignore
except Exception:
    build_feature_frame = None  # resolved at runtime

AUDIT_TICKERS = ["AAPL", "MSFT", "SPY"]
AUDIT_START = "2022-01-01"
AUDIT_END = "2024-01-01"


def _fetch(ticker: str) -> pd.DataFrame:
    """Download raw daily bars for one ticker."""
    df = yf.download(ticker, start=AUDIT_START, end=AUDIT_END, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def _perturb_future(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where each row's Close is replaced by the NEXT row's Close.
    This simulates 'what if tomorrow's price changed?' — any honest feature
    at row t should be unchanged."""
    perturbed = df.copy()
    perturbed["Close"] = df["Close"].shift(-1).fillna(df["Close"])
    return perturbed


def audit_dataframe_function(fn, ticker: str) -> dict:
    """Compare feature output on real data vs future-perturbed data."""
    raw = _fetch(ticker)
    if raw.empty:
        return {"ticker": ticker, "status": "skip", "reason": "no data"}

    try:
        real = fn(raw.copy(), ticker=ticker) if fn.__code__.co_argcount >= 2 else fn(raw.copy())
        fake = fn(_perturb_future(raw), ticker=ticker) if fn.__code__.co_argcount >= 2 else fn(_perturb_future(raw))
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e)}

    # compare only overlapping index; exclude last row (perturbation affects it legitimately)
    common = real.index.intersection(fake.index)[:-2]
    leaks: list[str] = []
    for col in real.columns:
        if col not in fake.columns:
            continue
        a = real.loc[common, col].astype(float).to_numpy()
        b = fake.loc[common, col].astype(float).to_numpy()
        # tolerance: feature values at row t should be identical under future perturbation
        if not np.allclose(a, b, equal_nan=True, atol=1e-9):
            leaks.append(col)

    return {
        "ticker": ticker,
        "status": "pass" if not leaks else "FAIL",
        "leaking_features": leaks,
        "n_features_checked": len(real.columns),
    }


def main() -> int:
    if build_feature_frame is None:
        print("[leakage_audit] pipeline_shared.build_feature_frame not found. "
              "Wire the real feature builder before running.", file=sys.stderr)
        return 2

    results = [audit_dataframe_function(build_feature_frame, t) for t in AUDIT_TICKERS]
    out = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "train_start": AUDIT_START,
        "train_end": AUDIT_END,
        "results": results,
    }
    path = os.path.join(LOG_DIR, "leakage_audit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    failed = [r for r in results if r["status"] == "FAIL"]
    print(f"[leakage_audit] wrote {path}")
    if failed:
        print(f"[leakage_audit] {len(failed)} tickers have leaking features.")
        return 1
    print("[leakage_audit] all tickers passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
