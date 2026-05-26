from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from settings import DATA_DIR, LOG_DIR
from data_provider import download_single, flatten_yf


DEFAULT_ETFS = ("SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD")
DATA = Path(DATA_DIR)
LOGS = Path(LOG_DIR)
MIN_ROWS = 252
MAX_AGE_BUSINESS_DAYS = 5


def _nyse_calendar():
    """Load the NYSE calendar when available."""
    try:
        import exchange_calendars as xcals

        return xcals.get_calendar("XNYS")
    except Exception:
        return None


def _latest_weekday_on_or_before(day: object) -> pd.Timestamp:
    ts = pd.Timestamp(day).normalize()
    while ts.weekday() >= 5:
        ts -= pd.Timedelta(days=1)
    return ts


def _latest_nyse_session_on_or_before(day: object) -> pd.Timestamp:
    ts = pd.Timestamp(day).normalize()
    calendar = _nyse_calendar()
    if calendar is not None:
        for _ in range(14):
            if calendar.is_session(ts):
                return ts
            ts -= pd.Timedelta(days=1)
        return ts
    return _latest_weekday_on_or_before(ts)


def _completed_day(now: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    eastern_ts = now_ts.to_pydatetime().astimezone(ZoneInfo("America/New_York"))
    current_day = pd.Timestamp(eastern_ts.date())
    calendar = _nyse_calendar()
    if calendar is not None:
        if not calendar.is_session(current_day):
            return _latest_nyse_session_on_or_before(current_day - pd.Timedelta(days=1))
        if now_ts < calendar.session_close(current_day):
            return _latest_nyse_session_on_or_before(current_day - pd.Timedelta(days=1))
        return current_day
    if eastern_ts.weekday() >= 5:
        return _latest_weekday_on_or_before(current_day - pd.Timedelta(days=1))
    close_ts = eastern_ts.replace(hour=16, minute=0, second=0, microsecond=0)
    if eastern_ts < close_ts:
        return _latest_weekday_on_or_before(current_day - pd.Timedelta(days=1))
    return current_day


def _count_trading_sessions(start: object, end: object) -> int:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        return 0
    calendar = _nyse_calendar()
    if calendar is not None:
        return int(len(calendar.sessions_in_range(start_ts, end_ts)))
    return int(sum(1 for day in pd.date_range(start_ts, end_ts, freq="D") if day.weekday() < 5))


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
        age = _count_trading_sessions(latest + pd.Timedelta(days=1), _completed_day())
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


def validate_etfs(symbols: list[str], *, refresh: bool = False, force: bool = False) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for symbol in symbols:
        symbol = symbol.upper().strip()
        frame = _read_local(symbol)
        local = _validate_etf_frame(frame, symbol=symbol)
        refreshed = False
        if refresh and (force or not local["ok"]):
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
        "force": bool(force),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or refresh ETF parquet data used by core alpha.")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_ETFS), help="ETF symbols to validate.")
    parser.add_argument("--refresh", action="store_true", help="Download and replace stale/missing ETF parquet data.")
    parser.add_argument("--force", action="store_true", help="Download and replace ETF parquet data even if local data passes validation.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    report = validate_etfs([str(s).upper() for s in args.symbols], refresh=bool(args.refresh), force=bool(args.force))
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / "etf_data_health.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("ETF Data Health")
        print("-" * 72)
        print(f"OK: {report['ok']} refresh={report['refresh']} force={report['force']}")
        for row in report["results"]:
            issues = ",".join(row.get("issues", [])) or "none"
            print(f"{row['symbol']:5s} ok={row['ok']} rows={row['rows']} latest={row.get('latest_date')} age_bdays={row.get('age_business_days')} issues={issues}")
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
