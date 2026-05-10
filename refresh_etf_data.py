from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from settings import DATA_DIR, LOG_DIR
from data_provider import download_single, flatten_yf


DEFAULT_ETFS = ("SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD")
DATA = Path(DATA_DIR)
LOGS = Path(LOG_DIR)
MIN_ROWS = 252
MAX_AGE_BUSINESS_DAYS = 5


def _completed_day() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def _validate_etf_frame(frame: pd.DataFrame, *, symbol: str, max_age_business_days: int = MAX_AGE_BUSINESS_DAYS) -> dict:
    issues: list[str] = []
    if frame.empty:
        return {"symbol": symbol, "ok": False, "issues": ["empty_frame"], "rows": 0, "latest_date": None}
    if "Close" not in frame.columns:
        issues.append("missing_close_column")
    close_raw = frame.get("Close", pd.Series(dtype=float))
    if isinstance(close_raw, pd.DataFrame):
        close_raw = close_raw.iloc[:, 0] if close_raw.shape[1] else pd.Series(dtype=float)
    close = pd.to_numeric(close_raw, errors="coerce").dropna()
    if len(close) < MIN_ROWS:
        issues.append(f"rows_{len(close)}_lt_{MIN_ROWS}")
    if not close.empty and (close <= 0).any():
        issues.append("nonpositive_close")
    if len(close) > 20 and float(close.tail(20).std()) == 0.0:
        issues.append("flat_recent_close")
    idx = pd.to_datetime(frame.index, errors="coerce")
    latest = pd.Timestamp(idx.max()).normalize() if len(idx) and not pd.isna(idx.max()) else pd.NaT
    age = None
    if pd.isna(latest):
        issues.append("missing_latest_date")
    else:
        age = int(len(pd.bdate_range(latest + pd.tseries.offsets.BDay(1), _completed_day())))
        if age > int(max_age_business_days):
            issues.append(f"stale_{age}_bdays")
    return {
        "symbol": symbol,
        "ok": not issues,
        "issues": issues,
        "rows": int(len(close)),
        "latest_date": None if pd.isna(latest) else str(latest.date()),
        "age_business_days": age,
    }


def _read_local(symbol: str) -> pd.DataFrame:
    path = DATA / f"{symbol.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _download(symbol: str) -> pd.DataFrame:
    """
    Download ETF price data with automatic fallback across providers.

    PLAIN ENGLISH: Tries yfinance first, falls back to yahooquery, then Stooq.
    This way the pipeline doesn't break when one provider is down.
    """
    try:
        frame = download_single(symbol, period="max", auto_adjust=False)
    except RuntimeError:
        return pd.DataFrame()
    if frame.empty:
        return frame
    frame = flatten_yf(frame)
    return frame


def validate_etfs(symbols: list[str], *, refresh: bool = False) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for symbol in symbols:
        symbol = symbol.upper().strip()
        frame = _read_local(symbol)
        local = _validate_etf_frame(frame, symbol=symbol)
        refreshed = False
        if refresh and not local["ok"]:
            downloaded = _download(symbol)
            downloaded_check = _validate_etf_frame(downloaded, symbol=symbol)
            if downloaded_check["ok"]:
                downloaded.to_parquet(DATA / f"{symbol}.parquet")
                local = downloaded_check
                refreshed = True
            else:
                local = {**local, "download_issues": downloaded_check.get("issues", [])}
        results.append({**local, "refreshed": refreshed})
    ok = all(item["ok"] for item in results)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": ok,
        "refresh": bool(refresh),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or refresh ETF parquet data used by core alpha.")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_ETFS), help="ETF symbols to validate.")
    parser.add_argument("--refresh", action="store_true", help="Download and replace stale/missing ETF parquet data.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    report = validate_etfs([str(s).upper() for s in args.symbols], refresh=bool(args.refresh))
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / "etf_data_health.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("ETF Data Health")
        print("-" * 72)
        print(f"OK: {report['ok']} refresh={report['refresh']}")
        for row in report["results"]:
            issues = ",".join(row.get("issues", [])) or "none"
            print(f"{row['symbol']:5s} ok={row['ok']} rows={row['rows']} latest={row.get('latest_date')} age_bdays={row.get('age_business_days')} issues={issues}")
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
