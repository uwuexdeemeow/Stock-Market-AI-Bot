"""Create and validate provenance manifests for stored market data.

PLAIN ENGLISH: A parquet file contains numbers, but not enough information to
prove where those numbers came from. This module writes a small JSON sidecar
that records the provider, adjustment rule, dates, schema, and checksum.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from safe_io import atomic_write_json


MAX_PROVIDER_MEDIAN_CLOSE_DIFF_PCT = 0.5
MAX_PROVIDER_CLOSE_DIFF_PCT = 2.0
# Prices arrive as decimal numbers, but computers store them as tiny binary
# approximations. This tolerance ignores microscopic rounding dust while still
# catching a real high/low price contradiction.
OHLC_ABSOLUTE_TOLERANCE = 1e-8
OHLC_RELATIVE_TOLERANCE = 1e-10
# Large earnings and turnaround moves are real. Treat only a 100% single-day
# adjusted-close jump as suspicious; the old 50% cutoff falsely blocked AMD.
MAX_SUSPICIOUS_DAILY_CLOSE_MOVE_PCT = 100.0


def parquet_manifest_path(parquet_path: str | Path) -> Path:
    """Return the sidecar path for one ticker parquet."""
    path = Path(parquet_path)
    return path.parent / "manifests" / f"{path.stem}.json"


def _file_sha256(path: Path) -> str:
    """Hash one stored file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_quality_issues(frame: pd.DataFrame) -> list[str]:
    """Find structural OHLCV problems before data is trusted."""
    issues: list[str] = []
    if frame.empty:
        return ["empty_frame"]
    if not isinstance(frame.index, pd.DatetimeIndex):
        issues.append("index_not_datetime")
    if frame.index.has_duplicates:
        issues.append("duplicate_sessions")
    if not frame.index.is_monotonic_increasing:
        issues.append("sessions_not_sorted")
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        issues.append("missing_columns:" + ",".join(missing))
        return issues
    numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    if numeric[["Open", "High", "Low", "Close"]].isna().all(axis=1).any():
        issues.append("session_missing_all_prices")
    high_floor = numeric[["Open", "Close", "Low"]].max(axis=1)
    low_ceiling = numeric[["Open", "Close", "High"]].min(axis=1)
    price_scale = numeric[["Open", "High", "Low", "Close"]].abs().max(axis=1)
    tolerance = OHLC_ABSOLUTE_TOLERANCE + OHLC_RELATIVE_TOLERANCE * price_scale
    invalid_ohlc = (
        (numeric["High"] + tolerance < high_floor)
        | (numeric["Low"] - tolerance > low_ceiling)
        | (numeric[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    )
    if bool(invalid_ohlc.fillna(False).any()):
        issues.append("invalid_ohlc_relationship")
    if bool((numeric["Volume"].dropna() < 0).any()):
        issues.append("negative_volume")
    close_returns = numeric["Close"].pct_change(fill_method=None).abs()
    if bool((close_returns > MAX_SUSPICIOUS_DAILY_CLOSE_MOVE_PCT / 100.0).any()):
        issues.append("unexplained_move_over_100pct")
    if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index) >= 5:
        try:
            import exchange_calendars as xcals

            recent = pd.DatetimeIndex(frame.index[-252:]).tz_localize(None).normalize()
            expected = xcals.get_calendar("XNYS").sessions_in_range(recent.min(), recent.max())
            expected = pd.DatetimeIndex(expected).tz_localize(None).normalize()
            missing_sessions = expected.difference(recent)
            if len(missing_sessions) > 2:
                issues.append(f"missing_recent_sessions:{len(missing_sessions)}")
        except Exception:
            pass
    return issues


def frame_integrity_diagnostics(frame: pd.DataFrame) -> dict:
    """Describe calendar, adjustment, and possible corporate-action evidence.

    PLAIN ENGLISH: A large move may be a real split, dividend adjustment, or a
    bad bar. We record candidates for review instead of silently guessing and
    rewriting prices. This report does not alter the active data.
    """
    if frame.empty:
        return {"status": "unavailable", "reason": "empty_frame"}
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce")).dropna()
    close = pd.to_numeric(frame.get("Close", pd.Series(dtype=float)), errors="coerce")
    moves = close.pct_change(fill_method=None).abs()
    candidate_positions = list(moves[moves >= 0.20].index)
    adjusted_columns = [
        str(column) for column in frame.columns
        if str(column).lower().replace("_", " ") in {"adj close", "adjusted close"}
    ]
    duplicate_calendar_days = 0
    if len(index):
        normalized = index.tz_localize(None).normalize() if index.tz is not None else index.normalize()
        duplicate_calendar_days = int(normalized.duplicated().sum())
    return {
        "status": "review" if candidate_positions or duplicate_calendar_days else "ok",
        "calendar": "XNYS",
        "index_timezone": str(index.tz) if len(index) and index.tz is not None else "naive",
        "duplicate_calendar_days": duplicate_calendar_days,
        "corporate_action_candidate_count": len(candidate_positions),
        "corporate_action_candidate_dates": [str(pd.Timestamp(value).date()) for value in candidate_positions[-10:]],
        "candidate_move_threshold_pct": 20.0,
        "adjusted_price_columns": adjusted_columns,
        "note": "Candidates require provider/corporate-action review; prices are not changed here.",
    }


def compare_provider_overlap(existing: pd.DataFrame, fresh: pd.DataFrame) -> dict:
    """Measure adjusted-close agreement across the shared date range."""
    if "Close" not in existing or "Close" not in fresh:
        return {"ok": False, "reason": "close_column_missing", "overlap_rows": 0}
    left = pd.to_numeric(existing["Close"], errors="coerce").rename("old")
    right = pd.to_numeric(fresh["Close"], errors="coerce").rename("new")
    overlap = pd.concat([left, right], axis=1, join="inner").dropna().tail(60)
    if len(overlap) < 5:
        return {"ok": False, "reason": "provider_overlap_too_short", "overlap_rows": int(len(overlap))}
    denominator = overlap["old"].abs().replace(0.0, np.nan)
    diff_pct = ((overlap["new"] - overlap["old"]).abs() / denominator * 100.0).dropna()
    median = float(diff_pct.median()) if len(diff_pct) else float("inf")
    maximum = float(diff_pct.max()) if len(diff_pct) else float("inf")
    ok = median <= MAX_PROVIDER_MEDIAN_CLOSE_DIFF_PCT and maximum <= MAX_PROVIDER_CLOSE_DIFF_PCT
    return {
        "ok": bool(ok),
        "reason": "ok" if ok else "provider_overlap_price_mismatch",
        "overlap_rows": int(len(diff_pct)),
        "median_close_diff_pct": round(median, 6),
        "max_close_diff_pct": round(maximum, 6),
        "median_limit_pct": MAX_PROVIDER_MEDIAN_CLOSE_DIFF_PCT,
        "max_limit_pct": MAX_PROVIDER_CLOSE_DIFF_PCT,
    }


def read_parquet_manifest(parquet_path: str | Path) -> dict:
    """Read one sidecar, returning an empty dictionary when unavailable."""
    path = parquet_manifest_path(parquet_path)
    if not path.exists():
        return {}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_provider_transition(
    existing: pd.DataFrame,
    fresh: pd.DataFrame,
    *,
    previous_provider: str,
    new_provider: str,
) -> dict:
    """Require overlap agreement only when an incremental update changes source."""
    if not previous_provider or previous_provider == new_provider:
        return {"ok": True, "reason": "same_or_unknown_provider"}
    return compare_provider_overlap(existing, fresh)


def write_parquet_manifest(
    parquet_path: str | Path,
    *,
    ticker: str,
    provider: str,
    adjustment_mode: str,
    frame: pd.DataFrame,
    provider_transition: dict | None = None,
) -> Path:
    """Write provenance after the parquet has been atomically saved."""
    path = Path(parquet_path)
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce")).dropna()
    schema_text = "|".join(f"{column}:{frame[column].dtype}" for column in frame.columns)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": str(ticker).upper(),
        "parquet_path": str(path),
        "provider": str(provider or "unknown"),
        "adjustment_mode": str(adjustment_mode),
        "first_date": str(index.min().date()) if len(index) else None,
        "last_date": str(index.max().date()) if len(index) else None,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "parquet_sha256": _file_sha256(path),
        "quality_issues": frame_quality_issues(frame),
        "integrity_diagnostics": frame_integrity_diagnostics(frame),
        "provider_transition": provider_transition or {"ok": True, "reason": "not_applicable"},
    }
    out = parquet_manifest_path(path)
    atomic_write_json(manifest, out)
    return out
