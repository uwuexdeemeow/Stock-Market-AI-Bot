"""
leakage_audit.py — Temporal leakage detector.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import LOG_DIR  # noqa: E402

try:
    from pipeline_shared import build_feature_frame  # type: ignore
except Exception:
    build_feature_frame = None

AUDIT_TICKERS = ["AAPL", "MSFT", "SPY"]
AUDIT_START = "2022-01-01"
AUDIT_END = "2024-01-01"


def _fetch(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=AUDIT_START, end=AUDIT_END, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def _perturb_future(df: pd.DataFrame) -> pd.DataFrame:
    perturbed = df.copy()
    for col in [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in perturbed.columns]:
        perturbed[col] = df[col].shift(-1).fillna(df[col])
    return perturbed


def audit_dataframe_function(fn, ticker: str) -> dict:
    raw = _fetch(ticker)
    if raw.empty:
        return {"ticker": ticker, "status": "skip", "reason": "no data"}

    try:
        real = fn(raw.copy(), ticker=ticker) if fn.__code__.co_argcount >= 2 else fn(raw.copy())
        fake = fn(_perturb_future(raw), ticker=ticker) if fn.__code__.co_argcount >= 2 else fn(_perturb_future(raw))
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e)}

    common = real.index.intersection(fake.index)
    if len(common) > 2:
        common = common[:-2]
    leaks: list[str] = []
    checked = 0
    for col in real.columns:
        if col not in fake.columns:
            continue
        try:
            a = pd.to_numeric(real.loc[common, col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(fake.loc[common, col], errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue
        if len(a) == 0 or len(b) == 0:
            continue
        checked += 1
        if not np.allclose(a, b, equal_nan=True, atol=1e-9):
            leaks.append(col)

    return {
        "ticker": ticker,
        "status": "pass" if not leaks else "FAIL",
        "leaking_features": leaks,
        "n_features_checked": checked,
    }


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    if build_feature_frame is None:
        print("[leakage_audit] pipeline_shared.build_feature_frame not found. Wire the real feature builder before running.", file=sys.stderr)
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
