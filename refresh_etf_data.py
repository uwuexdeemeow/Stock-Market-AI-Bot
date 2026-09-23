from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests

from settings import DATA_DIR, LOG_DIR
import data_provider
from data_provider import download_single, flatten_yf
from data_manifest import (
    read_parquet_manifest,
    validate_provider_transition,
    write_parquet_manifest,
)
from safe_io import atomic_write_json, atomic_write_parquet


DEFAULT_ETFS = ("SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD")
DATA = Path(DATA_DIR)
LOGS = Path(LOG_DIR)
MIN_ROWS = 252
MAX_AGE_BUSINESS_DAYS = 5
REQUIRED_ETF_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


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
    missing_columns = [column for column in REQUIRED_ETF_COLUMNS if column not in frame.columns]
    for column in missing_columns:
        issues.append(f"missing_{column.lower()}_column")
    close_raw = frame.get("Close", pd.Series(dtype=float))
    if isinstance(close_raw, pd.DataFrame):
        close_raw = close_raw.iloc[:, 0] if close_raw.shape[1] else pd.Series(dtype=float)
    # Keep missing values visible: dropping them here made a dated but blank
    # QQQ bar look healthy even though the strategy cannot use that bar.
    close_values = pd.to_numeric(close_raw, errors="coerce")
    close = close_values.dropna()
    if len(close_values) and bool((~np.isfinite(close_values)).any()):
        issues.append("missing_or_nonfinite_close")
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


def _repair_recent_bar_from_alpaca(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Use a checked, adjusted Alpaca bar only for a missing completed session."""
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret or frame.empty:
        return pd.DataFrame()
    target = _completed_day()
    candidate = frame.copy()
    candidate.index = pd.DatetimeIndex(candidate.index).tz_localize(None).normalize()
    if not candidate.index.is_unique:
        return pd.DataFrame()
    # Repair only a missing latest bar. Earlier gaps need a full source refresh,
    # not a one-day patch that could hide a broken history.
    older = candidate.loc[candidate.index < target]
    if not _validate_etf_frame(older, symbol=symbol, max_age_business_days=5)["ok"]:
        return pd.DataFrame()
    try:
        response = requests.get(
            f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            params={
                "timeframe": "1Day",
                "start": (target - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
                "end": (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "adjustment": "all",
                "feed": "iex",
            },
            timeout=20,
        )
        response.raise_for_status()
        bars = response.json().get("bars") or []
    except (requests.RequestException, ValueError) as exc:
        print(f"  WARNING: Alpaca ETF backup unavailable for {symbol}: {type(exc).__name__}")
        return pd.DataFrame()
    if not bars:
        return pd.DataFrame()
    backup = pd.DataFrame(bars)
    backup.index = pd.to_datetime(backup["t"], utc=True).dt.tz_localize(None).dt.normalize()
    backup = backup.loc[~backup.index.duplicated(keep="last")]
    if target not in backup.index:
        return pd.DataFrame()
    # IEX is one exchange, so compare its adjusted closes with the primary
    # source on several shared days before trusting its missing-day price.
    primary_close = pd.to_numeric(older["Close"], errors="coerce")
    backup_close = pd.to_numeric(backup["c"], errors="coerce")
    common = primary_close.index.intersection(backup.index)
    common = common[common < target][-5:]
    if len(common) < 3:
        return pd.DataFrame()
    disagreement = (primary_close.loc[common] - backup_close.loc[common]).abs() / primary_close.loc[common]
    if not bool(np.isfinite(disagreement).all()) or bool((disagreement > 0.005).any()):
        print(f"  WARNING: Alpaca and primary ETF prices disagree for {symbol}; refusing backup")
        return pd.DataFrame()
    bar = backup.loc[target]
    prices = pd.to_numeric(bar[["o", "h", "l", "c"]], errors="coerce")
    if not bool(np.isfinite(prices).all()) or not (0 < prices["l"] <= min(prices["o"], prices["c"]) <= max(prices["o"], prices["c"]) <= prices["h"]):
        return pd.DataFrame()
    # Preserve the primary source's full-market volume when it is present;
    # IEX volume represents only one exchange.
    old_volume = pd.to_numeric(candidate.loc[target, "Volume"], errors="coerce") if target in candidate.index else np.nan
    volume = old_volume if np.isfinite(old_volume) and old_volume > 0 else float(bar["v"])
    if not np.isfinite(volume) or volume <= 0:
        return pd.DataFrame()
    candidate.loc[target, ["Open", "High", "Low", "Close", "Volume"]] = [
        float(prices["o"]), float(prices["h"]), float(prices["l"]), float(prices["c"]), float(volume),
    ]
    candidate = candidate.sort_index()
    if not _validate_etf_frame(candidate, symbol=symbol, max_age_business_days=0)["ok"]:
        return pd.DataFrame()
    print(f"  INFO: Restored {symbol} {target.date()} bar from cross-checked Alpaca IEX data")
    return candidate


def _download(symbol: str) -> pd.DataFrame:
    """
    Download ETF price data with automatic fallback across providers.

    PLAIN ENGLISH: Tries yfinance first, falls back to yahooquery, then Stooq.
    This way the pipeline doesn't break when one provider is down.
    """
    incomplete: list[tuple[str, pd.DataFrame]] = []

    def accept(candidate: pd.DataFrame) -> bool:
        simple = flatten_yf(candidate.copy())
        valid = _validate_etf_frame(simple, symbol=symbol, max_age_business_days=0)["ok"]
        if not valid:
            incomplete.append((str(candidate.attrs.get("price_provider", "unknown")), simple))
        return valid

    try:
        # Reject incomplete provider results before the provider layer settles
        # on a source. A second provider can then supply a usable completed bar.
        frame = download_single(
            symbol, period="max", auto_adjust=True,
            accept_frame=accept,
        )
    except RuntimeError as exc:
        # When both Yahoo paths lack the latest price, a checked Alpaca bar can
        # complete the otherwise-good history without rewriting older prices.
        for provider, partial in incomplete:
            repaired = _repair_recent_bar_from_alpaca(symbol, partial)
            if not repaired.empty:
                data_provider.provider_for_ticker[symbol] = f"{provider}+alpaca_iex"
                return repaired
        # Show which sources were rejected so a future data outage is clear in
        # the workflow log, even though the saved-file health check runs next.
        print(f"  WARNING: No complete adjusted price history for {symbol}: {exc}")
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
        # A refresh before the US market opens must already contain the last
        # completed session; older prices cannot stand in for yesterday's bar.
        age_limit = 0 if refresh else MAX_AGE_BUSINESS_DAYS
        local = _validate_etf_frame(frame, symbol=symbol, max_age_business_days=age_limit)
        refreshed = False
        if refresh and (force or not local["ok"]):
            downloaded = _download(symbol)
            downloaded_check = _validate_etf_frame(
                downloaded, symbol=symbol, max_age_business_days=age_limit,
            )
            if downloaded_check["ok"]:
                parquet_path = DATA / f"{symbol}.parquet"
                previous = read_parquet_manifest(parquet_path)
                provider = data_provider.provider_for_ticker.get(symbol, "unknown")
                transition = validate_provider_transition(
                    frame,
                    downloaded,
                    previous_provider=str(previous.get("provider", "")),
                    new_provider=provider,
                )
                if transition.get("ok", False):
                    atomic_write_parquet(downloaded, parquet_path, index=True)
                    write_parquet_manifest(
                        parquet_path,
                        ticker=symbol,
                        provider=provider,
                        adjustment_mode="adjusted_ohlcv",
                        frame=downloaded,
                        provider_transition=transition,
                    )
                    local = downloaded_check
                    refreshed = True
                else:
                    local = {**local, "download_issues": [transition.get("reason", "provider_transition_failed")]}
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


def write_etf_data_health(report: dict, output_path: Path | None = None) -> Path:
    """Write the ETF data-health report through a crash-safe JSON helper.

    PLAIN ENGLISH: Daily gates read this report before trading. Atomic writing
    keeps the report from being half-written if the refresh is interrupted.
    """
    out = output_path or (LOGS / "etf_data_health.json")
    atomic_write_json(report, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or refresh ETF parquet data used by core alpha.")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_ETFS), help="ETF symbols to validate.")
    parser.add_argument("--refresh", action="store_true", help="Download and replace stale/missing ETF parquet data.")
    parser.add_argument("--force", action="store_true", help="Download and replace ETF parquet data even if local data passes validation.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any ETF data check fails.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    report = validate_etfs([str(s).upper() for s in args.symbols], refresh=bool(args.refresh), force=bool(args.force))
    out = write_etf_data_health(report)
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
    if args.strict and not bool(report.get("ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
