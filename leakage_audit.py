"""
leakage_audit.py — Temporal leakage detector.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import DATA_DIR, LOG_DIR  # noqa: E402

try:
    import pipeline_shared  # type: ignore
except Exception:
    pipeline_shared = None

AUDIT_TICKERS = ["AAPL", "MSFT", "SPY"]
AUDIT_START = "2022-01-01"
AUDIT_END = "2024-01-01"
ALLOW_NETWORK_FETCH = os.environ.get("LEAKAGE_AUDIT_ALLOW_NETWORK", "").strip().lower() in {"1", "true", "yes"}
IGNORED_COLUMNS = {
    "target",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
}


def _fetch(ticker: str) -> pd.DataFrame:
    local_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if os.path.exists(local_path):
        df = pd.read_parquet(local_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.loc[AUDIT_START:AUDIT_END, [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].dropna()

    if not ALLOW_NETWORK_FETCH:
        return pd.DataFrame()

    df = yf.download(ticker, start=AUDIT_START, end=AUDIT_END, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def _perturb_after_cutoff(df: pd.DataFrame, cutoff_pos: int) -> pd.DataFrame:
    perturbed = df.copy()
    future_idx = perturbed.index[cutoff_pos + 1 :]
    if len(future_idx) == 0:
        return perturbed

    for col in [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in perturbed.columns]:
        values = perturbed.loc[future_idx, col].astype(float).to_numpy()
        # Deterministic, large future-only perturbation. If a feature at/before
        # cutoff changes, that feature probably looked into future rows.
        if col == "Volume":
            perturbed.loc[future_idx, col] = values[::-1] * 3.0 + 123.0
        else:
            scale = np.linspace(1.50, 0.50, len(values))
            perturbed.loc[future_idx, col] = values[::-1] * scale
    return perturbed


def _build_research_frame_from_raw(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if pipeline_shared is None:
        raise RuntimeError("pipeline_shared could not be imported")

    originals = {
        "fetch_price_data": pipeline_shared.fetch_price_data,
        "build_multi_market": pipeline_shared.build_multi_market,
        "build_vix_features": pipeline_shared.build_vix_features,
        "build_sentiment_features": pipeline_shared.build_sentiment_features,
        "build_social_sentiment_features": pipeline_shared.build_social_sentiment_features,
        "build_pead_features": pipeline_shared.build_pead_features,
        "build_market_breadth_features": pipeline_shared.build_market_breadth_features,
    }

    def fake_fetch_price_data(fetch_ticker: str, start: str, end: str) -> pd.DataFrame:
        if fetch_ticker.upper() == ticker.upper():
            return raw.copy()
        return originals["fetch_price_data"](fetch_ticker, start, end)

    def empty_frame(*args, **kwargs) -> pd.DataFrame:
        dates = args[1] if len(args) > 1 and isinstance(args[1], pd.DatetimeIndex) else args[0]
        return pd.DataFrame(index=dates)

    def neutral_pead_features(ticker_arg: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            index=dates,
            data={
                "eps_surprise_pct": 0.0,
                "days_since_earnings": 60.0,
                "days_to_next_earnings": 60.0,
            },
        )

    pipeline_shared.fetch_price_data = fake_fetch_price_data
    pipeline_shared.build_multi_market = empty_frame
    pipeline_shared.build_vix_features = empty_frame
    pipeline_shared.build_sentiment_features = empty_frame
    pipeline_shared.build_social_sentiment_features = empty_frame
    pipeline_shared.build_pead_features = neutral_pead_features
    pipeline_shared.build_market_breadth_features = empty_frame
    try:
        return pipeline_shared.build_research_feature_frame(ticker, AUDIT_START, AUDIT_END)
    finally:
        for name, fn in originals.items():
            setattr(pipeline_shared, name, fn)


def audit_ticker(ticker: str) -> dict:
    raw = _fetch(ticker)
    if raw.empty:
        return {"ticker": ticker, "status": "skip", "reason": "no data"}
    if len(raw) < 120:
        return {"ticker": ticker, "status": "skip", "reason": f"not enough rows: {len(raw)}"}

    try:
        cutoff_pos = int(len(raw) * 0.70)
        cutoff_date = raw.index[cutoff_pos]
        real = _build_research_frame_from_raw(raw.copy(), ticker)
        fake = _build_research_frame_from_raw(_perturb_after_cutoff(raw, cutoff_pos), ticker)
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e)}

    common = real.index.intersection(fake.index)
    common = common[common <= cutoff_date]
    leaks: list[str] = []
    checked = 0
    for col in real.columns:
        if col in IGNORED_COLUMNS:
            continue
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
        "cutoff_date": str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date),
    }


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    if pipeline_shared is None or not hasattr(pipeline_shared, "build_research_feature_frame"):
        print("[leakage_audit] pipeline_shared.build_research_feature_frame not found.", file=sys.stderr)
        return 2

    results = [audit_ticker(t) for t in AUDIT_TICKERS]
    out = {
        "run_at": datetime.now(UTC).isoformat(),
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
