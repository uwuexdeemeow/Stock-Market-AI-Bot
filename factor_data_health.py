"""
factor_data_health.py - Validate the factor-data cache used by live trading.

The daily trading workflow can now skip the slow research.py refresh and use
the latest successful cached factor data.  This script makes that safe by
writing a small manifest and failing early when the restored cache is stale or
incomplete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ranker_utils import load_adaptive_factor_weights
from safe_io import atomic_write_json
from data_manifest import read_parquet_manifest
from settings import (
    ADAPTIVE_WEIGHTS_FILE,
    ADAPTIVE_WEIGHTS_MAX_AGE_DAYS,
    DATA_DIR,
    SIGNAL_DIR,
    SIMPLE_FACTOR_COLS,
    SIMPLE_FACTOR_WEIGHTS,
    SURVIVORSHIP_TRAINING_TICKERS,
    WATCHLIST,
)


DEFAULT_WARN_TRADING_DAYS = 1
DEFAULT_BLOCK_TRADING_DAYS = 2
MANIFEST_NAME = "factor_data_health.json"
REQUIRED_FACTOR_COLUMNS = ["Close", *SIMPLE_FACTOR_COLS]


def _as_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _count_nyse_sessions(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count real NYSE trading sessions in an inclusive date window."""
    start = _as_timestamp(start)
    end = _as_timestamp(end)
    if start > end:
        return 0
    try:
        import exchange_calendars as xcals

        nyse = xcals.get_calendar("XNYS")
        return int(len(nyse.sessions_in_range(start, end)))
    except Exception:
        # If the calendar package is unavailable, fall back to weekdays. This
        # keeps the health check usable, but the normal path handles holidays.
        return int(sum(1 for day in pd.date_range(start, end, freq="D") if day.weekday() < 5))


def trading_day_age(latest: pd.Timestamp, *, now: pd.Timestamp | None = None) -> int:
    """Count NYSE sessions since latest factor data."""
    latest = _as_timestamp(latest)
    today = _as_timestamp(pd.Timestamp.now() if now is None else now)
    if latest >= today:
        return 0
    return _count_nyse_sessions(latest + pd.Timedelta(days=1), today)


def _latest_parquet_date(path: Path) -> tuple[pd.Timestamp, int, int, list[str]]:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("empty parquet")

    if isinstance(df.index, pd.DatetimeIndex):
        latest = df.index.max()
    elif "date" in df.columns:
        latest = pd.to_datetime(df["date"], errors="coerce").max()
    else:
        raise ValueError("no DatetimeIndex or date column")

    if pd.isna(latest):
        raise ValueError("no valid latest date")
    return _as_timestamp(latest), int(len(df)), int(len(df.columns)), [str(col) for col in df.columns]


def _ticker_list(values: Iterable[str]) -> list[str]:
    return sorted({str(v).upper().strip() for v in values if str(v).strip()})


def _feature_quality_status(
    *,
    signal_dir: Path,
    factor_paths: list[Path],
) -> dict:
    report_path = signal_dir / "feature_quality_report.json"
    summary_path = signal_dir / "feature_quality_summary.csv"
    status = {
        "path": str(report_path),
        "summary_path": str(summary_path),
        "exists": report_path.exists(),
        "summary_exists": summary_path.exists(),
        "ready": False,
        "reason": "missing_report",
        "feature_count": 0,
        "mtime": None,
        "stale_vs_factor_data": False,
    }
    if not report_path.exists():
        return status

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        status["reason"] = f"invalid_json:{exc}"
        return status

    features = payload.get("features", [])
    status["feature_count"] = len(features) if isinstance(features, list) else 0
    status["mtime"] = pd.Timestamp.fromtimestamp(report_path.stat().st_mtime).isoformat()
    if status["feature_count"] <= 0:
        status["reason"] = "no_feature_grades"
        return status

    report_mtime = report_path.stat().st_mtime
    newer_factor_files = [
        str(path)
        for path in factor_paths
        if path.exists() and path.stat().st_mtime > report_mtime
    ]
    if newer_factor_files:
        status["stale_vs_factor_data"] = True
        status["reason"] = "report_older_than_factor_data"
        status["newer_factor_files"] = newer_factor_files[:10]
        status["newer_factor_file_count"] = len(newer_factor_files)
        return status

    status["ready"] = True
    status["reason"] = "ok"
    return status


def _feature_health_status(
    *,
    signal_dir: Path,
    factor_paths: list[Path],
) -> dict:
    profile_path = signal_dir / "feature_health_profile.json"
    summary_path = signal_dir / "feature_health_profile.csv"
    quality_path = signal_dir / "feature_quality_report.json"
    status = {
        "path": str(profile_path),
        "summary_path": str(summary_path),
        "exists": profile_path.exists(),
        "summary_exists": summary_path.exists(),
        "ready": False,
        "reason": "missing_profile",
        "feature_health_gate_pass": False,
        "feature_health_gate_reasons": [],
        "mtime": None,
        "stale_vs_feature_quality": False,
        "stale_vs_factor_data": False,
    }
    if not profile_path.exists():
        return status
    if not summary_path.exists():
        status["reason"] = "missing_summary_csv"
        return status

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        status["reason"] = f"invalid_json:{exc}"
        return status

    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    features = payload.get("features", []) if isinstance(payload, dict) else []
    gate_pass = bool(summary.get("feature_health_gate_pass", False))
    status["feature_health_gate_pass"] = gate_pass
    status["feature_health_gate_reasons"] = list(summary.get("feature_health_gate_reasons") or [])
    status["active_cluster_count"] = summary.get("active_cluster_count")
    status["max_cluster_weight"] = summary.get("max_cluster_weight")
    status["feature_count"] = len(features) if isinstance(features, list) else 0
    status["mtime"] = pd.Timestamp.fromtimestamp(profile_path.stat().st_mtime).isoformat()
    profile_mtime = profile_path.stat().st_mtime

    if quality_path.exists() and quality_path.stat().st_mtime > profile_mtime:
        status["stale_vs_feature_quality"] = True
        status["reason"] = "profile_older_than_feature_quality"
        return status

    newer_factor_files = [
        str(path)
        for path in factor_paths
        if path.exists() and path.stat().st_mtime > profile_mtime
    ]
    if newer_factor_files:
        status["stale_vs_factor_data"] = True
        status["reason"] = "profile_older_than_factor_data"
        status["newer_factor_files"] = newer_factor_files[:10]
        status["newer_factor_file_count"] = len(newer_factor_files)
        return status

    if not gate_pass:
        status["reason"] = "feature_health_gate_failed"
        return status

    status["ready"] = True
    status["reason"] = "ok"
    return status


def _adaptive_weight_status(*, now: pd.Timestamp | None = None) -> dict:
    _, meta = load_adaptive_factor_weights(
        ADAPTIVE_WEIGHTS_FILE,
        SIMPLE_FACTOR_WEIGHTS,
        SIMPLE_FACTOR_COLS,
        max_age_days=ADAPTIVE_WEIGHTS_MAX_AGE_DAYS,
        now=now,
        return_metadata=True,
    )
    return meta


def build_factor_data_health(
    *,
    data_dir: str | Path = DATA_DIR,
    signal_dir: str | Path = SIGNAL_DIR,
    tickers: Iterable[str] = WATCHLIST,
    optional_tickers: Iterable[str] = SURVIVORSHIP_TRAINING_TICKERS,
    warn_days: int = DEFAULT_WARN_TRADING_DAYS,
    block_days: int = DEFAULT_BLOCK_TRADING_DAYS,
    now: pd.Timestamp | None = None,
) -> dict:
    data_path = Path(data_dir)
    signal_path = Path(signal_dir)
    required = _ticker_list(tickers)
    optional = [ticker for ticker in _ticker_list(optional_tickers) if ticker not in set(required)]
    today = _as_timestamp(pd.Timestamp.now() if now is None else now)

    per_ticker: dict[str, dict] = {}
    missing: list[str] = []
    unreadable: list[dict] = []
    missing_required_columns: list[dict] = []
    stale: list[dict] = []
    blocked: list[dict] = []
    latest_dates: list[pd.Timestamp] = []
    existing_required_paths: list[Path] = []
    manifest_errors: list[dict] = []

    for ticker in required:
        path = data_path / f"{ticker}.parquet"
        if not path.exists():
            missing.append(ticker)
            per_ticker[ticker] = {"path": str(path), "status": "missing"}
            continue
        existing_required_paths.append(path)
        try:
            latest, rows, cols, column_names = _latest_parquet_date(path)
        except Exception as exc:
            unreadable.append({"ticker": ticker, "error": str(exc)})
            per_ticker[ticker] = {"path": str(path), "status": "unreadable", "error": str(exc)}
            continue

        missing_cols = sorted(set(REQUIRED_FACTOR_COLUMNS) - set(column_names))
        age = trading_day_age(latest, now=today)
        record = {
            "path": str(path),
            "status": "missing_required_columns" if missing_cols else "ok",
            "latest_date": str(latest.date()),
            "age_trading_days": age,
            "rows": rows,
            "columns": cols,
            "missing_required_columns": missing_cols,
        }
        per_ticker[ticker] = record
        sidecar = read_parquet_manifest(path)
        sidecar_issues: list[str] = []
        if not sidecar:
            sidecar_issues.append("manifest_missing_or_invalid")
        else:
            if str(sidecar.get("ticker", "")).upper() != ticker:
                sidecar_issues.append("manifest_ticker_mismatch")
            if int(sidecar.get("row_count", -1) or -1) != rows:
                sidecar_issues.append("manifest_row_count_mismatch")
            if str(sidecar.get("last_date", "")) != str(latest.date()):
                sidecar_issues.append("manifest_last_date_mismatch")
            if sidecar.get("quality_issues") and sidecar.get("provider") != "legacy_unknown":
                sidecar_issues.append("manifest_quality_issues")
            if str(sidecar.get("adjustment_mode", "")) != "adjusted_ohlcv":
                sidecar_issues.append("manifest_adjustment_mode_invalid")
        record["data_manifest"] = {
            "provider": sidecar.get("provider") if sidecar else None,
            "adjustment_mode": sidecar.get("adjustment_mode") if sidecar else None,
            "issues": sidecar_issues,
        }
        if sidecar_issues:
            manifest_errors.append({"ticker": ticker, "issues": sidecar_issues})
        latest_dates.append(latest)
        if missing_cols:
            missing_required_columns.append({
                "ticker": ticker,
                "missing_columns": missing_cols,
            })
        if age > warn_days:
            stale.append({"ticker": ticker, "latest_date": str(latest.date()), "age_trading_days": age})
        if age > block_days:
            blocked.append({"ticker": ticker, "latest_date": str(latest.date()), "age_trading_days": age})

    optional_missing: list[str] = []
    optional_unreadable: list[dict] = []
    optional_present = 0
    for ticker in optional:
        path = data_path / f"{ticker}.parquet"
        if not path.exists():
            optional_missing.append(ticker)
            continue
        try:
            _latest_parquet_date(path)
            optional_present += 1
        except Exception as exc:
            optional_unreadable.append({"ticker": ticker, "error": str(exc)})

    oldest_latest = min(latest_dates) if latest_dates else None
    newest_latest = max(latest_dates) if latest_dates else None
    max_age = max((int(item.get("age_trading_days", 0)) for item in per_ticker.values()), default=None)
    factor_data_ready = (
        not missing
        and not unreadable
        and not missing_required_columns
        and not manifest_errors
        and not blocked
        and bool(latest_dates)
    )
    factor_data_fresh = factor_data_ready and not stale
    feature_quality = _feature_quality_status(
        signal_dir=signal_path,
        factor_paths=existing_required_paths,
    )
    feature_health = _feature_health_status(
        signal_dir=signal_path,
        factor_paths=existing_required_paths,
    )
    adaptive_weights = _adaptive_weight_status(now=today)
    trade_ready = bool(
        factor_data_fresh
        and feature_quality.get("ready", False)
        and feature_health.get("ready", False)
    )
    signal_ready = bool(
        factor_data_ready
        and feature_quality.get("ready", False)
        and feature_health.get("ready", False)
    )

    reasons: list[str] = []
    if missing:
        reasons.append("missing_factor_parquets")
    if unreadable:
        reasons.append("unreadable_factor_parquets")
    if missing_required_columns:
        reasons.append("factor_data_missing_required_columns")
    if manifest_errors:
        reasons.append("factor_data_manifest_errors")
    if blocked:
        reasons.append("factor_data_too_stale")
    elif stale:
        reasons.append("factor_data_warn_stale")
    if not feature_quality.get("ready", False):
        reasons.append(f"feature_quality_{feature_quality.get('reason', 'not_ready')}")
    if not feature_health.get("ready", False):
        reasons.append(f"feature_health_{feature_health.get('reason', 'not_ready')}")
    if not latest_dates:
        reasons.append("no_factor_dates")

    return {
        "computed_at": pd.Timestamp.now().isoformat(),
        "as_of_date": str(today.date()),
        "warn_trading_days": int(warn_days),
        "block_trading_days": int(block_days),
        "required_ticker_count": len(required),
        "optional_ticker_count": len(optional),
        "required_present_count": len(required) - len(missing) - len(unreadable),
        "missing_tickers": missing,
        "unreadable_tickers": unreadable,
        "missing_required_column_tickers": missing_required_columns,
        "manifest_error_tickers": manifest_errors,
        "required_factor_columns": list(REQUIRED_FACTOR_COLUMNS),
        "stale_tickers": stale,
        "blocked_tickers": blocked,
        "optional_present_count": optional_present,
        "optional_missing_tickers": optional_missing,
        "optional_unreadable_tickers": optional_unreadable,
        "oldest_latest_factor_date": str(oldest_latest.date()) if oldest_latest is not None else None,
        "newest_latest_factor_date": str(newest_latest.date()) if newest_latest is not None else None,
        "max_age_trading_days": max_age,
        "factor_data_ready": bool(factor_data_ready),
        "factor_data_fresh": bool(factor_data_fresh),
        "feature_quality": feature_quality,
        "feature_health": feature_health,
        "adaptive_weights": adaptive_weights,
        "signal_ready": bool(signal_ready),
        "trade_ready": bool(trade_ready),
        "reasons": reasons,
        "tickers": per_ticker,
    }


def _print_summary(manifest: dict) -> None:
    status = "OK" if manifest.get("trade_ready") else "NOT READY"
    print(f"Factor data health: {status}")
    print(f"  Required tickers: {manifest.get('required_present_count')}/{manifest.get('required_ticker_count')}")
    print(f"  Oldest factor date: {manifest.get('oldest_latest_factor_date')}")
    print(f"  Newest factor date: {manifest.get('newest_latest_factor_date')}")
    print(f"  Max age: {manifest.get('max_age_trading_days')} trading days")
    print(f"  Stale tickers: {len(manifest.get('stale_tickers', []))}")
    print(f"  Missing tickers: {len(manifest.get('missing_tickers', []))}")
    print(f"  Missing required columns: {len(manifest.get('missing_required_column_tickers', []))}")
    print(f"  Feature quality: {manifest.get('feature_quality', {}).get('reason')}")
    print(f"  Feature health: {manifest.get('feature_health', {}).get('reason')}")
    print(
        "  Adaptive weights: "
        f"{manifest.get('adaptive_weights', {}).get('adaptive_weight_status')} "
        f"({manifest.get('adaptive_weights', {}).get('adaptive_weight_reason')})"
    )
    if manifest.get("reasons"):
        print("  Reasons: " + ", ".join(str(x) for x in manifest["reasons"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cached factor data for live trading")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless data is trade-ready")
    parser.add_argument("--ready-only", action="store_true", help="Exit non-zero unless data is signal-ready")
    parser.add_argument("--no-write", action="store_true", help="Do not write signals/factor_data_health.json")
    parser.add_argument("--output", default=str(Path(SIGNAL_DIR) / MANIFEST_NAME))
    parser.add_argument("--warn-days", type=int, default=DEFAULT_WARN_TRADING_DAYS)
    parser.add_argument("--block-days", type=int, default=DEFAULT_BLOCK_TRADING_DAYS)
    args = parser.parse_args()

    manifest = build_factor_data_health(
        warn_days=int(args.warn_days),
        block_days=int(args.block_days),
    )
    _print_summary(manifest)
    if not args.no_write:
        atomic_write_json(manifest, args.output)
        print(f"  Manifest: {args.output}")

    if args.strict and not manifest.get("trade_ready", False):
        raise SystemExit(1)
    if args.ready_only and not manifest.get("signal_ready", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
