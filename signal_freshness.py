"""Shared freshness and safety checks for broker signal files."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


DEFAULT_SIGNAL_TIMEZONE = os.environ.get("PAPER_SIGNAL_TIMEZONE", os.environ.get("TZ", "Asia/Singapore"))
CORE_ETFS = {"SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD"}
MAX_SIGNAL_FUTURE_MINUTES = float(os.environ.get("MAX_SIGNAL_FUTURE_MINUTES", "5"))


def _signal_timezone(default_timezone: str | None = None):
    try:
        return ZoneInfo(str(default_timezone or DEFAULT_SIGNAL_TIMEZONE))
    except Exception:
        return datetime.now().astimezone().tzinfo or timezone.utc


def parse_signal_timestamp(value: object, *, default_timezone: str | None = None) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize(_signal_timezone(default_timezone))
    return out.tz_convert("UTC")


def _nyse_calendar():
    """Load the NYSE calendar when the optional calendar package is available."""
    try:
        import exchange_calendars as xcals

        return xcals.get_calendar("XNYS")
    except Exception:
        return None


def _latest_weekday_on_or_before(day: object) -> pd.Timestamp:
    """Fallback calendar: find the latest Monday-Friday date."""
    ts = pd.Timestamp(day).normalize()
    while ts.weekday() >= 5:
        ts -= pd.Timedelta(days=1)
    return ts


def _latest_nyse_session_on_or_before(day: object) -> pd.Timestamp:
    """Find the latest real NYSE session on or before `day`."""
    ts = pd.Timestamp(day).normalize()
    calendar = _nyse_calendar()
    if calendar is not None:
        for _ in range(14):
            if calendar.is_session(ts):
                return ts
            ts -= pd.Timedelta(days=1)
        return ts
    return _latest_weekday_on_or_before(ts)


def _count_us_trading_sessions(start: object, end: object) -> int:
    """Count real NYSE sessions in an inclusive window."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        return 0
    calendar = _nyse_calendar()
    if calendar is not None:
        return int(len(calendar.sessions_in_range(start_ts, end_ts)))
    return int(sum(1 for day in pd.date_range(start_ts, end_ts, freq="D") if day.weekday() < 5))


def latest_completed_us_trading_day(now: datetime | None = None) -> pd.Timestamp:
    eastern = ZoneInfo("America/New_York")
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    ts = now_ts.to_pydatetime().astimezone(eastern)
    current_day = pd.Timestamp(ts.date())
    calendar = _nyse_calendar()
    if calendar is not None:
        if not calendar.is_session(current_day):
            return _latest_nyse_session_on_or_before(current_day - pd.Timedelta(days=1))
        close_ts = calendar.session_close(current_day)
        if now_ts < close_ts:
            return _latest_nyse_session_on_or_before(current_day - pd.Timedelta(days=1))
        return current_day
    if ts.weekday() >= 5:
        return _latest_weekday_on_or_before(current_day - pd.Timedelta(days=1))
    close_ts = ts.replace(hour=16, minute=0, second=0, microsecond=0)
    if ts < close_ts:
        return _latest_weekday_on_or_before(current_day - pd.Timedelta(days=1))
    return current_day


def validate_signal_freshness(
    signal,
    *,
    max_signal_age_hours: float,
    max_factor_age_trading_days: int,
    now: datetime | None = None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")

    predicted_at = signal.get("predicted_at", "")
    predicted_ts = parse_signal_timestamp(predicted_at)
    if pd.isna(predicted_ts):
        issues.append("missing_predicted_at")
    else:
        age_hours = (now_ts - predicted_ts).total_seconds() / 3600.0
        if age_hours < -(MAX_SIGNAL_FUTURE_MINUTES / 60.0):
            future_minutes = abs(age_hours) * 60.0
            issues.append(f"signal_from_future_{future_minutes:.1f}m_gt_{MAX_SIGNAL_FUTURE_MINUTES:.1f}m")
        elif age_hours > float(max_signal_age_hours):
            issues.append(f"signal_age_{age_hours:.1f}h_gt_{float(max_signal_age_hours):.1f}h")

    latest_factor_date = signal.get("latest_factor_date", "")
    factor_ts = pd.to_datetime(latest_factor_date, errors="coerce")
    if pd.isna(factor_ts):
        issues.append("missing_latest_factor_date")
    else:
        completed_us_day = latest_completed_us_trading_day(now=now)
        age_days = _count_us_trading_sessions(pd.Timestamp(factor_ts) + pd.Timedelta(days=1), completed_us_day)
        if age_days > int(max_factor_age_trading_days):
            issues.append(f"factor_age_{age_days}_bdays_gt_{int(max_factor_age_trading_days)}")
    return not issues, issues


def _coerce_float(value: object, default: float = 0.0) -> float:
    """Turn a CSV/JSON value into a number, falling back safely.

    PLAIN ENGLISH: Broker signals are read from CSV files, so a number may
    arrive as text, blank text, NaN, or None.  This helper keeps those odd
    values from crashing the safety checks.
    """
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_signal_weights(signal) -> dict[str, float]:
    """Read all target weights from the live signal into one dict.

    PLAIN ENGLISH: The signal can store core ETF weights in columns like
    `target_spy_weight` and stock weights in `overlay_weights_json`.  Older
    code also used `target_weight_SPY`.  This one parser understands all of
    those formats so every safety check sees the same portfolio.
    """
    weights: dict[str, float] = {}

    if hasattr(signal, "items"):
        items = list(signal.items())
    else:
        items = []

    # Old format: target_weight_AAPL, target_weight_SPY, etc.
    for key, value in items:
        key_str = str(key)
        if key_str.startswith("target_weight_"):
            ticker = key_str.replace("target_weight_", "", 1).upper().strip()
            weight = _coerce_float(value)
            if ticker and abs(weight) > 1e-9:
                weights[ticker] = weights.get(ticker, 0.0) + weight

    # Current core ETF columns produced by core_satellite_alpha.py.
    core_columns = {
        "SPY": "target_spy_weight",
        "QQQ": "target_qqq_weight",
        "TQQQ": "target_tqqq_weight",
    }
    for ticker, column in core_columns.items():
        weight = _coerce_float(signal.get(column, 0.0) if hasattr(signal, "get") else 0.0)
        if abs(weight) > 1e-9:
            weights[ticker] = weights.get(ticker, 0.0) + weight

    # Dict format used by a few tests/tools.
    if hasattr(signal, "get") and isinstance(signal.get("target_weights"), dict):
        for ticker, value in signal["target_weights"].items():
            weight = _coerce_float(value)
            ticker_str = str(ticker).upper().strip()
            if ticker_str and abs(weight) > 1e-9:
                weights[ticker_str] = weights.get(ticker_str, 0.0) + weight

    # Stock overlay JSON from the unified core-satellite signal.
    overlay_raw = signal.get("overlay_weights_json", "{}") if hasattr(signal, "get") else "{}"
    if isinstance(overlay_raw, dict):
        overlay = overlay_raw
    else:
        try:
            overlay = json.loads(str(overlay_raw)) if pd.notna(overlay_raw) else {}
        except Exception:
            overlay = {}
    if isinstance(overlay, dict):
        for ticker, value in overlay.items():
            ticker_str = str(ticker).upper().strip()
            weight = _coerce_float(value)
            if ticker_str and abs(weight) > 1e-9:
                weights[ticker_str] = weights.get(ticker_str, 0.0) + weight

    return weights


def live_config_identity(payload: dict, strategy: str = "core-alpha") -> dict:
    """Return the fields that identify the approved live config.

    PLAIN ENGLISH: If you publish a new walkforward config, old signals should
    not trade.  These fields are the "ID card" of the config that created a
    signal.
    """
    approved = (payload.get("approved_live_configs", {}) or {}).get(strategy, {}) or {}
    return {
        "created_at": str(payload.get("created_at", "") or ""),
        "source_json": str(payload.get("source_json", "") or ""),
        "approved_config_family": str(approved.get("approved_config_family", "") or ""),
        "approved_family_signature": str(approved.get("approved_family_signature", "") or ""),
        "approved_exact_config": str(approved.get("approved_exact_config", "") or ""),
        "config": approved.get("config", {}) if isinstance(approved.get("config", {}), dict) else {},
    }


def live_config_fingerprint(payload: dict, strategy: str = "core-alpha") -> str:
    """Hash the approved live config identity for cheap CSV comparison."""
    blob = json.dumps(
        live_config_identity(payload, strategy=strategy),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def validate_live_config_match(
    signal,
    *,
    live_config_path: str | Path,
    strategy: str = "core-alpha",
) -> tuple[bool, list[str]]:
    """Check that a signal was generated from the current live config.

    PLAIN ENGLISH: This catches the bad case where you publish a new
    walkforward config but the broker is still holding yesterday's signal.
    """
    issues: list[str] = []
    path = Path(live_config_path)
    if not path.exists():
        return False, [f"missing_live_config_file:{path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"invalid_live_config_file:{exc.__class__.__name__}"]

    expected_hash = live_config_fingerprint(payload, strategy=strategy)
    signal_hash = str(signal.get("live_config_hash", "") if hasattr(signal, "get") else "").strip()
    if not signal_hash:
        issues.append("missing_live_config_hash")
    elif signal_hash != expected_hash:
        issues.append(f"live_config_hash_mismatch:{signal_hash}!={expected_hash}")

    expected_created_at = str(payload.get("created_at", "") or "")
    signal_created_at = str(signal.get("live_config_created_at", "") if hasattr(signal, "get") else "").strip()
    if expected_created_at and signal_created_at and signal_created_at != expected_created_at:
        issues.append("live_config_created_at_mismatch")

    return not issues, issues


def validate_signal_sanity(
    signal,
    *,
    max_gross_exposure: float = 1.5,
    max_single_weight: float = 0.30,
    max_core_etf_weight: float = 1.0,
    required_core_etfs: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Check if signal weights are sane before submitting orders.

    PLAIN ENGLISH: Even if a signal is fresh, it could be WRONG — a bug in
    signal generation might produce weights that sum to 500% equity, or put
    50% into a single stock, or forget SPY entirely.  This catches those
    bugs at submission time so you don't blindly execute a broken signal.

    Checks:
      1. Total gross exposure doesn't exceed max_gross_exposure (default 150%)
      2. No single ticker exceeds max_single_weight (default 30%)
      3. Required core ETFs (SPY or QQQ) are present with non-zero weight
      4. No negative weights (we don't short in this strategy)

    Returns (is_sane, list_of_issues).
    """
    issues: list[str] = []
    if required_core_etfs is None:
        required_core_etfs = {"SPY", "QQQ"}

    weights = extract_signal_weights(signal)

    if not weights:
        return False, ["missing_target_weights"]

    # Check gross exposure
    gross = sum(abs(w) for w in weights.values())
    if gross > max_gross_exposure:
        issues.append(f"gross_exposure_{gross:.2f}_exceeds_{max_gross_exposure:.2f}")

    # Check single-name concentration
    for ticker, w in weights.items():
        cap = max_core_etf_weight if ticker in CORE_ETFS else max_single_weight
        if abs(w) > cap:
            issues.append(f"{ticker}_weight_{w:.3f}_exceeds_{cap:.2f}")

    # Check for negative weights (no shorting in this strategy)
    neg_tickers = [t for t, w in weights.items() if w < -0.001]
    if neg_tickers:
        issues.append(f"negative_weights: {', '.join(neg_tickers)}")

    # Check required core ETFs present
    present_core = required_core_etfs.intersection(weights.keys())
    if not present_core and weights:
        issues.append(f"missing_core_etfs: need at least one of {required_core_etfs}")

    return not issues, issues
