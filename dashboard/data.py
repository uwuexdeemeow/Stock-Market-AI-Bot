"""
dashboard/data.py — Centralized data loaders for the dashboard

PLAIN ENGLISH: Every page in the dashboard reads its data through
functions in THIS file.  When you change where the backend writes
state (e.g. swap CSV → database, add S3, etc.), you only edit the
functions here — the page UIs don't need to change.

The cache decorator (@st.cache_data(ttl=30)) means Streamlit re-reads
the underlying files every 30 seconds.  That's the "dynamic" part —
the dashboard auto-refreshes without you reloading.
"""

# ── Imports ────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

# ── Load .env at module import ─────────────────────────────────────────
# PLAIN ENGLISH: Streamlit doesn't auto-read .env files like the backend
# scripts do (those call load_dotenv() explicitly).  We do it here once
# at import so EVERY page that imports from dashboard.* picks up keys
# like GITHUB_TOKEN, ALPACA_API_KEY, etc.  Silent no-op if .env missing.
try:
    from dotenv import load_dotenv
    # override=False — anything already set in the OS env wins over .env.
    # This way "GITHUB_TOKEN=... streamlit run dashboard.py" still works.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — env vars must come from the OS


# ── Paths — single source of truth for "where backend data lives" ──────
# When you reorganize files, change ONLY these constants.  Pages stay clean.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_DIR = PROJECT_ROOT / "signals"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Specific files we read
LIVE_CONFIG_FILE = SIGNALS_DIR / "core_satellite_live_configs.json"
WALKFORWARD_CSV = SIGNALS_DIR / "core_satellite_nested_walkforward.csv"
WALKFORWARD_JSON = SIGNALS_DIR / "core_satellite_nested_walkforward.json"
SIGNAL_CSV = SIGNALS_DIR / "core_satellite_alpha_signal.csv"
METRICS_JSON = SIGNALS_DIR / "core_satellite_alpha_metrics.json"
ORDERS_CSV = SIGNALS_DIR / "core_satellite_alpha_orders.csv"
ALPACA_EQUITY = SIGNALS_DIR / "alpaca_paper_equity.csv"
ALPACA_LOG = SIGNALS_DIR / "alpaca_paper_log.csv"
ALPACA_STATUS = SIGNALS_DIR / "alpaca_daily_status.json"
ALPACA_HEALTH = SIGNALS_DIR / "alpaca_paper_health.json"
FEATURE_HEALTH = SIGNALS_DIR / "feature_health_profile.json"
FEATURE_QUALITY = SIGNALS_DIR / "feature_quality_summary.csv"
BROKER_HEALTH = SIGNALS_DIR / "broker_health.json"
REGIME_HEARTBEAT = SIGNALS_DIR / "monitor_heartbeat.json"


# ── Helpers — graceful fallbacks for missing files ────────────────────
def _read_json_safe(path: Path) -> dict | list | None:
    """Return parsed JSON or None if file is missing/malformed.
    Never raise — the dashboard should handle missing files gracefully.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Return DataFrame or empty DataFrame on missing/error."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _file_age_minutes(path: Path) -> Optional[float]:
    """Minutes since file was last modified (None if missing)."""
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 60.0


# ── LIVE CONFIG (which config is currently approved for live trading) ─
@st.cache_data(ttl=60)
def load_live_config() -> dict | None:
    """Return the currently-approved live config payload, or None."""
    return _read_json_safe(LIVE_CONFIG_FILE)


# ── WALKFORWARD RESULTS (the historical OOS validation) ───────────────
@st.cache_data(ttl=60)
def load_walkforward_results() -> pd.DataFrame:
    """Return the per-fold walkforward CSV (year-by-year OOS metrics)."""
    return _read_csv_safe(WALKFORWARD_CSV)


@st.cache_data(ttl=60)
def load_walkforward_summary() -> dict | None:
    """Return the aggregate walkforward stats from the JSON."""
    return _read_json_safe(WALKFORWARD_JSON)


# ── TODAY'S SIGNAL (what the bot decided to trade) ────────────────────
@st.cache_data(ttl=30)
def load_current_signal() -> dict | None:
    """Return today's signal as a dict (first row of the CSV)."""
    df = _read_csv_safe(SIGNAL_CSV)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


@st.cache_data(ttl=30)
def load_signal_metrics() -> dict | None:
    """Return the metrics JSON associated with today's signal."""
    return _read_json_safe(METRICS_JSON)


@st.cache_data(ttl=30)
def load_order_plan() -> pd.DataFrame:
    """Return today's planned orders (BUY/SELL list)."""
    return _read_csv_safe(ORDERS_CSV)


# ── ALPACA STATE (live broker positions, equity, fills) ───────────────
@st.cache_data(ttl=30)
def load_alpaca_status() -> dict | None:
    """Return the latest Alpaca account snapshot.

    Primary source is `signals/alpaca_daily_status.json` written by
    `alpaca_paper_trading.py --status`.  That file is committed by the
    daily workflow, so a fresh `pull_daily` will land it.

    Fallback: if the status file is missing (older `signals/latest`
    snapshots before the workflow started committing it, or a
    workflow that crashed before the status step), borrow the equity
    fields from `signals/alpaca_paper_health.json` which the workflow
    has been committing for longer.  paper_health.json includes
    `account_equity` and a `concentration.position_count` so we can
    populate enough of the snapshot to keep the dashboard's account
    summary card from showing zeros.
    """
    status = _read_json_safe(ALPACA_STATUS)
    if status and float(status.get("account_equity") or 0) > 0:
        return status

    # ── Fallback path ───────────────────────────────────────────────
    # paper_health.json's account_equity is the same dollar value the
    # status file would have written, written by the same broker
    # snapshot call earlier in the daily pipeline.  Cash and invested
    # aren't tracked there, so we leave them None (the summary card
    # already degrades gracefully when those are missing).
    health = _read_json_safe(ALPACA_HEALTH) or {}
    health_equity = float(health.get("account_equity") or 0)
    if health_equity <= 0:
        # Neither source has data — return whatever (possibly empty)
        # status dict we got so the caller can decide what to display.
        return status
    concentration = health.get("concentration") or {}
    fallback = dict(status or {})
    fallback["account_equity"] = health_equity
    # Best-effort position count; concentration.position_count is the
    # one paper_health publishes.  Default to 0 if absent.
    fallback.setdefault("position_count", int(concentration.get("position_count") or 0))
    fallback.setdefault("broker", "alpaca")
    fallback.setdefault(
        "generated_at",
        health.get("generated_at") or health.get("as_of") or "",
    )
    # Mark the source so the dashboard can show a small "fallback"
    # badge if it wants to.
    fallback["_source"] = "alpaca_paper_health.json (fallback)"
    return fallback


@st.cache_data(ttl=60)
def load_alpaca_equity_history() -> pd.DataFrame:
    """Return the equity-over-time CSV for charting."""
    df = _read_csv_safe(ALPACA_EQUITY)
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    return df


@st.cache_data(ttl=60)
def load_alpaca_orders() -> pd.DataFrame:
    """Return the order log with fill status."""
    df = _read_csv_safe(ALPACA_LOG)
    if df.empty:
        return df
    # Normalize the date column for filtering
    for col in ("submitted_at", "timestamp", "date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.sort_values(col, ascending=False)
            break
    return df


# ── HEALTH (per-broker drift + risk summary) ──────────────────────────
@st.cache_data(ttl=60)
def load_health_report() -> dict | None:
    """Return the latest health summary JSON (drift, slippage, risk)."""
    return _read_json_safe(ALPACA_HEALTH)


@st.cache_data(ttl=60)
def load_broker_health() -> dict | None:
    """Return the broker connectivity ping result."""
    return _read_json_safe(BROKER_HEALTH)


# ── FEATURE HEALTH (which factors are active, decay state) ────────────
@st.cache_data(ttl=120)
def load_feature_health() -> dict | None:
    return _read_json_safe(FEATURE_HEALTH)


@st.cache_data(ttl=120)
def load_feature_quality_summary() -> pd.DataFrame:
    return _read_csv_safe(FEATURE_QUALITY)


# ── SYSTEM HEALTH (last daily_run results, freshness, etc.) ───────────
@st.cache_data(ttl=30)
def load_latest_daily_run() -> dict | None:
    """Find and return the most recent daily_run_YYYYMMDD.json log."""
    if not LOGS_DIR.exists():
        return None
    runs = sorted(LOGS_DIR.glob("daily_run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return None
    return _read_json_safe(runs[0])


@st.cache_data(ttl=60)
def load_etf_data(ticker: str = "QQQ", days: int = 365) -> pd.DataFrame:
    """Return the last N days of an ETF's price history.
    Used to plot benchmark vs strategy equity.
    """
    f = DATA_DIR / f"{ticker}.parquet"
    if not f.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(f, columns=["Close"])
        df.index = pd.to_datetime(df.index)
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]
        df.columns = [ticker.lower() + "_close"]
        return df.reset_index().rename(columns={"index": "date", "Date": "date"})
    except Exception:
        return pd.DataFrame()


# ── FILE FRESHNESS (for system health page) ──────────────────────────
def file_status_table() -> pd.DataFrame:
    """Return a DataFrame of every dashboard-tracked file with its age."""
    files = {
        "Live config": LIVE_CONFIG_FILE,
        "Walkforward CSV": WALKFORWARD_CSV,
        "Walkforward JSON": WALKFORWARD_JSON,
        "Today's signal": SIGNAL_CSV,
        "Order plan": ORDERS_CSV,
        "Alpaca equity": ALPACA_EQUITY,
        "Alpaca status": ALPACA_STATUS,
        "Alpaca health": ALPACA_HEALTH,
        "Alpaca orders log": ALPACA_LOG,
        "Feature health": FEATURE_HEALTH,
        "Feature quality": FEATURE_QUALITY,
        "Broker health": BROKER_HEALTH,
    }
    rows = []
    for label, path in files.items():
        if path.exists():
            age_min = _file_age_minutes(path) or 0
            size_kb = path.stat().st_size / 1024
            rows.append({
                "File": label,
                "Path": str(path.relative_to(PROJECT_ROOT)),
                "Age (min)": round(age_min, 1),
                "Size (KB)": round(size_kb, 1),
                "Status": "🟢 fresh" if age_min < 60 else "🟡 stale" if age_min < 1440 else "🔴 old",
            })
        else:
            rows.append({
                "File": label,
                "Path": str(path.relative_to(PROJECT_ROOT)),
                "Age (min)": None,
                "Size (KB)": None,
                "Status": "⚫ missing",
            })
    return pd.DataFrame(rows)


# ── Derived computations (used by multiple pages) ──────────────────────
def compute_account_summary() -> dict:
    """Return a single-glance account snapshot for the top of pages.
    Pulls from multiple sources and degrades gracefully if any are missing.
    """
    status = load_alpaca_status() or {}
    history = load_alpaca_equity_history()

    equity = float(status.get("account_equity", 0) or 0)
    cash = float(status.get("account_cash", 0) or 0)
    invested = float(status.get("account_invested", equity - cash) or 0)
    position_count = int(status.get("position_count", 0) or 0)

    # Today vs yesterday change
    if len(history) >= 2 and "equity" in history.columns:
        latest = float(history["equity"].iloc[-1])
        prior = float(history["equity"].iloc[-2])
        change_pct = (latest - prior) / prior * 100 if prior > 0 else 0
        change_abs = latest - prior
    else:
        change_pct = 0
        change_abs = 0

    # All-time peak and current drawdown
    if len(history) > 0 and "equity" in history.columns:
        peak = float(history["equity"].cummax().iloc[-1])
        current_dd_pct = (equity - peak) / peak * 100 if peak > 0 else 0
    else:
        peak = equity
        current_dd_pct = 0

    return {
        "equity": equity,
        "cash": cash,
        "invested": invested,
        "position_count": position_count,
        "change_abs_today": change_abs,
        "change_pct_today": change_pct,
        "peak_equity": peak,
        "current_drawdown_pct": current_dd_pct,
        "generated_at": status.get("generated_at"),
    }
