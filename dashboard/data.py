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
from datetime import datetime, timedelta, timezone
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
ALPACA_SLIPPAGE_REPORT = SIGNALS_DIR / "alpaca_slippage_reversal_report.json"
ALPACA_EXECUTION_SCORECARD = SIGNALS_DIR / "alpaca_execution_scorecard.json"
ETF_DATA_HEALTH = LOGS_DIR / "etf_data_health.json"
BROKER_TRUTH_JSON = SIGNALS_DIR / "broker_truth.json"
BROKER_TRUTH_CSV = SIGNALS_DIR / "broker_truth.csv"
SHADOW_EQUITY = SIGNALS_DIR / "shadow_paper_equity.csv"
PAPER_SHADOW_COMPARE_JSON = SIGNALS_DIR / "paper_shadow_compare.json"
PAPER_SHADOW_COMPARE_CSV = SIGNALS_DIR / "paper_shadow_compare.csv"
FEATURE_HEALTH = SIGNALS_DIR / "feature_health_profile.json"
FEATURE_QUALITY = SIGNALS_DIR / "feature_quality_summary.csv"
BROKER_HEALTH = SIGNALS_DIR / "broker_health.json"
REGIME_HEARTBEAT = SIGNALS_DIR / "monitor_heartbeat.json"
WORKFLOW_HEARTBEATS = {
    "Daily paper": SIGNALS_DIR / "workflow_heartbeat_daily.json",
    "Shadow journal": SIGNALS_DIR / "workflow_heartbeat_shadow.json",
}

def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back if the value is blank or invalid."""
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


DASHBOARD_LIVE_ALPACA = os.environ.get("DASHBOARD_LIVE_ALPACA", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DASHBOARD_LIVE_ALPACA_TTL_SECONDS = _env_int("DASHBOARD_LIVE_ALPACA_TTL_SECONDS", 30)


# ── Helpers — graceful fallbacks for missing files ────────────────────
def _read_json_safe(path: Path) -> dict | list | None:
    """Return parsed JSON or None if file is missing/malformed.
    Never raise — the dashboard should handle missing files gracefully.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Return DataFrame or empty DataFrame on missing/error."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _first_nonempty(row: pd.Series | dict, *keys: str, default: object = None) -> object:
    """Return the first non-blank value from a row with old or new column names."""
    for key in keys:
        value = row.get(key, None)
        if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}:
            return value
    return default


def _to_float(value: object, default: float = 0.0) -> float:
    """Convert a CSV cell into a float, using default for blanks/bad text."""
    try:
        return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0])
    except Exception:
        return float(default)


def _to_bool(value: object) -> bool:
    """Convert common CSV/JSON truthy values into a real boolean."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def signal_trade_gate_state(signal: dict | None) -> dict:
    """Return the same broker-submit gate verdict used by status.py.

    PLAIN ENGLISH: The dashboard should only show paper trading as ready if the
    real Alpaca submission script would also be allowed to submit orders.
    """
    signal = signal or {}
    paper_ready = _to_bool(signal.get("paper_ready"))
    gates_all_pass = _to_bool(signal.get("gates_all_pass"))
    medium_risk_review_pass = _to_bool(signal.get("medium_risk_review_pass"))
    block_reasons: list[str] = []
    if not paper_ready:
        block_reasons.append("paper_ready=false")
    if not gates_all_pass:
        block_reasons.append("gates_all_pass=false")
    if not medium_risk_review_pass:
        block_reasons.append("medium_risk_review_pass=false")
    return {
        "paper_ready": paper_ready,
        "gates_all_pass": gates_all_pass,
        "medium_risk_review_pass": medium_risk_review_pass,
        "trade_ready": paper_ready and gates_all_pass and medium_risk_review_pass,
        "block_reasons": block_reasons,
    }


def _signed_slippage_bps(side: str, fill_price: float, reference_price: float) -> float | None:
    """Positive slippage means the fill was worse than the planned price."""
    if fill_price <= 0 or reference_price <= 0:
        return None
    if str(side).lower().strip() == "buy":
        return (fill_price - reference_price) / reference_price * 10_000
    return (reference_price - fill_price) / reference_price * 10_000


def _safe_mean(values: list[float]) -> float | None:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return None if clean.empty else float(clean.mean())


def _safe_median(values: list[float]) -> float | None:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return None if clean.empty else float(clean.median())


def _execution_report_from_order_log(log: pd.DataFrame) -> dict | None:
    """Build a dashboard-friendly fill report from alpaca_paper_log.csv.

    PLAIN ENGLISH: The richer Alpaca API report has minute-bar reversal data,
    but it may be missing locally until `--slippage-report` runs.  The paper
    log still has enough information to show actual fills and basic slippage.
    """
    if log.empty:
        return None

    report_rows: list[dict] = []
    for row in log.to_dict("records"):
        side = str(_first_nonempty(row, "side", "action", default="")).lower().strip()
        status = str(row.get("fill_status", row.get("status", ""))).lower().strip()
        filled_qty = _to_float(_first_nonempty(row, "filled_qty", "broker_dealt_qty"))
        if filled_qty <= 0 and status not in {"filled", "partial", "partially_filled"}:
            continue

        fill_price = _to_float(_first_nonempty(row, "filled_avg_price", "broker_dealt_avg_price", "fill_price"))
        reference_price = _to_float(_first_nonempty(row, "price", "limit_price"))
        slip = _signed_slippage_bps(side, fill_price, reference_price)
        submitted_at = _first_nonempty(row, "filled_at", "submitted_at", "timestamp", default="")
        report_rows.append({
            "filled_at": str(submitted_at),
            "symbol": str(row.get("ticker", row.get("symbol", ""))).upper(),
            "side": side,
            "order_type": str(_first_nonempty(row, "order_type", "submitted_order_type", default="paper_log")),
            "filled_qty": int(filled_qty) if filled_qty.is_integer() else round(filled_qty, 6),
            "fill_price": round(fill_price, 4) if fill_price > 0 else None,
            "fill_minute_vwap": None,
            "slippage_bps": round(float(slip), 2) if slip is not None else None,
            "adverse_5m_bps": None,
            "adverse_15m_bps": None,
            "adverse_30m_bps": None,
            "adverse_60m_bps": None,
            "worst_adverse_60m_bps": None,
            "best_favorable_60m_bps": None,
            "fill_status": status,
        })

    if not report_rows:
        return None

    def _segment_summary(rows: list[dict]) -> dict:
        """Summarize one group of fills for the dashboard segment table."""
        slip_values = [
            float(row["slippage_bps"])
            for row in rows
            if row.get("slippage_bps") is not None
        ]
        return {
            "orders_analyzed": len(rows),
            "avg_slippage_bps": round(_safe_mean(slip_values), 2) if _safe_mean(slip_values) is not None else None,
            "median_slippage_bps": round(_safe_median(slip_values), 2) if _safe_median(slip_values) is not None else None,
            "slippage_bad_count": int(sum(1 for value in slip_values if float(value) > 0)),
            "adverse_5m_count": 0,
            "adverse_15m_count": 0,
            "adverse_30m_count": 0,
            "adverse_60m_count": 0,
            "avg_worst_adverse_60m_bps": None,
            "max_worst_adverse_60m_bps": None,
        }

    slip_values = [
        float(row["slippage_bps"])
        for row in report_rows
        if row.get("slippage_bps") is not None
    ]
    limit_rows = [r for r in report_rows if str(r.get("order_type", "")).lower() == "limit"]
    market_rows = [r for r in report_rows if str(r.get("order_type", "")).lower() == "market"]
    by_symbol: list[dict] = []
    report_df = pd.DataFrame(report_rows)
    for symbol, grp in report_df.groupby("symbol", sort=False):
        slippage = pd.to_numeric(grp.get("slippage_bps"), errors="coerce").dropna()
        bad_rate = float((slippage > 0).mean()) if len(slippage) else None
        avg_slip = float(slippage.mean()) if len(slippage) else None
        by_symbol.append({
            "symbol": str(symbol),
            "orders": int(len(grp)),
            "avg_slippage_bps": round(avg_slip, 2) if avg_slip is not None else None,
            "bad_slippage_rate": round(bad_rate, 3) if bad_rate is not None else None,
            "adverse_15m_rate": None,
            "avg_worst_adverse_60m_bps": None,
            "execution_risk_score": round(max(0.0, avg_slip or 0.0) * 0.5, 2),
        })
    by_symbol.sort(key=lambda row: float(row.get("execution_risk_score") or 0), reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "alpaca_paper_log.csv fallback",
        "summary": {
            "orders_analyzed": len(report_rows),
            "avg_slippage_bps": round(_safe_mean(slip_values), 2) if _safe_mean(slip_values) is not None else None,
            "median_slippage_bps": round(_safe_median(slip_values), 2) if _safe_median(slip_values) is not None else None,
            "slippage_bad_count": int(sum(1 for value in slip_values if float(value) > 0)),
            "adverse_5m_count": 0,
            "adverse_15m_count": 0,
            "adverse_30m_count": 0,
            "adverse_60m_count": 0,
            "avg_worst_adverse_60m_bps": None,
            "max_worst_adverse_60m_bps": None,
        },
        "segments": {
            "all_orders": _segment_summary(report_rows),
            "limit_orders": _segment_summary(limit_rows),
            "market_orders": _segment_summary(market_rows),
            "trailing_stops": _segment_summary([]),
        },
        "by_symbol": by_symbol,
        "orders": report_rows,
        "errors": ["alpaca_slippage_reversal_report_missing_using_paper_log"],
    }


def _file_age_minutes(path: Path) -> Optional[float]:
    """Minutes since file was last modified (None if missing)."""
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 60.0


def _display_path(path: Path) -> str:
    """Show repo-relative paths, falling back to absolute paths for tests."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@st.cache_data(ttl=DASHBOARD_LIVE_ALPACA_TTL_SECONDS, show_spinner=False)
def refresh_live_alpaca_snapshot() -> dict:
    """Refresh local Alpaca snapshot files directly from the paper broker.

    PLAIN ENGLISH: Most dashboard widgets read local CSV/JSON files.  This
    helper asks Alpaca for the current paper-account equity and positions, then
    rewrites those local files.  That makes the dashboard "live" without
    waiting for `alpaca_paper_trading.py --reconcile`.
    """
    if not DASHBOARD_LIVE_ALPACA:
        return {"ok": False, "skipped": True, "reason": "DASHBOARD_LIVE_ALPACA=0"}
    try:
        from alpaca_paper_trading import AlpacaBroker, snapshot_equity, snapshot_status

        broker = AlpacaBroker()
        snapshot_equity(broker)
        snapshot_status(broker)
        return {
            "ok": True,
            "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "alpaca_api",
        }
    except Exception as exc:
        # Never break the dashboard if credentials are missing or Alpaca is down.
        # The callers will fall back to the most recent files on disk.
        return {
            "ok": False,
            "error": str(exc),
            "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


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
    live_refresh = refresh_live_alpaca_snapshot()
    status = _read_json_safe(ALPACA_STATUS)
    if status and float(status.get("account_equity") or 0) > 0:
        if live_refresh:
            status["_live_refresh"] = live_refresh
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
        if status is not None and live_refresh:
            status["_live_refresh"] = live_refresh
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
    fallback["_live_refresh"] = live_refresh
    return fallback


@st.cache_data(ttl=30)
def load_alpaca_equity_history() -> pd.DataFrame:
    """Return the equity-over-time CSV for charting."""
    refresh_live_alpaca_snapshot()
    df = _read_csv_safe(ALPACA_EQUITY)
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    return df


@st.cache_data(ttl=60)
def load_shadow_equity_history() -> pd.DataFrame:
    """Return the shadow paper equity curve."""
    df = _read_csv_safe(SHADOW_EQUITY)
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    return df


@st.cache_data(ttl=60)
def load_paper_shadow_compare() -> dict:
    """Return the Alpaca-vs-shadow comparison summary and table.

    PLAIN ENGLISH: Prefer the JSON/CSV written by paper_shadow_compare.py.  If
    they are not present yet, compute the same numbers in memory so the
    dashboard can still show something useful.
    """
    summary = _read_json_safe(PAPER_SHADOW_COMPARE_JSON)
    table = _read_csv_safe(PAPER_SHADOW_COMPARE_CSV)
    if isinstance(summary, dict) and summary.get("status") == "ok":
        return {"summary": summary, "table": table}
    try:
        from paper_shadow_compare import build_comparison_payload

        computed_summary, computed_table = build_comparison_payload(
            alpaca_path=ALPACA_EQUITY,
            shadow_path=SHADOW_EQUITY,
        )
        return {"summary": computed_summary, "table": computed_table}
    except Exception as exc:
        return {
            "summary": {"status": "error", "reason": str(exc)},
            "table": pd.DataFrame(),
        }


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
def load_slippage_reversal_report() -> dict | None:
    """Return the recent Alpaca fill slippage/reversal report."""
    report = _read_json_safe(ALPACA_SLIPPAGE_REPORT)
    if isinstance(report, dict) and report.get("orders"):
        return report
    return _execution_report_from_order_log(_read_csv_safe(ALPACA_LOG)) or report


@st.cache_data(ttl=60)
def load_execution_scorecard() -> dict | None:
    """Return the latest execution scorecard payload."""
    payload = _read_json_safe(ALPACA_EXECUTION_SCORECARD)
    return payload if isinstance(payload, dict) else None


@st.cache_data(ttl=60)
def load_etf_data_health() -> dict | None:
    """Return the latest ETF data-health payload."""
    payload = _read_json_safe(ETF_DATA_HEALTH)
    return payload if isinstance(payload, dict) else None


@st.cache_data(ttl=30)
def load_broker_truth() -> dict:
    """Return the broker truth report plus its per-ticker table.

    PLAIN ENGLISH: broker_truth.py compares what the strategy wants, what the
    local log says happened, and what Alpaca actually holds.  The dashboard
    keeps the JSON summary and CSV table together so pages can show both the
    traffic-light status and the detailed ticker gaps.
    """
    payload = _read_json_safe(BROKER_TRUTH_JSON)
    table = _read_csv_safe(BROKER_TRUTH_CSV)
    if not isinstance(payload, dict):
        payload = {
            "status": "missing",
            "score": None,
            "summary": {},
            "global_issues": [],
            "rows": [],
        }
    return {"payload": payload, "table": table}


def broker_truth_chip_status(payload: dict | None) -> str:
    """Map broker truth status into the dashboard chip statuses."""
    status = str((payload or {}).get("status") or "missing").strip().lower()
    if status == "pass":
        return "ok"
    if status == "warning":
        return "warn"
    if status == "fail":
        return "fail"
    return "unknown"


def broker_truth_issue_table(table: pd.DataFrame, *, limit: int = 10) -> pd.DataFrame:
    """Return the most important broker truth rows for display."""
    if table.empty or "issue_severity" not in table.columns:
        return pd.DataFrame()
    work = table.copy()
    work["issue_severity"] = work["issue_severity"].astype(str).str.lower().str.strip()
    work = work[work["issue_severity"].isin({"fail", "warning"})]
    if work.empty:
        return pd.DataFrame()
    severity_order = {"fail": 0, "warning": 1}
    work["_severity_rank"] = work["issue_severity"].map(severity_order).fillna(2)
    work = work.sort_values(["_severity_rank", "ticker"]).head(limit)
    display_cols = [
        col
        for col in [
            "ticker",
            "issue_severity",
            "target_weight",
            "broker_weight",
            "broker_qty",
            "open_sell_qty",
            "trailing_stop_qty",
            "issues",
        ]
        if col in work.columns
    ]
    return work[display_cols].reset_index(drop=True)


def _brief_join(values: list[object], *, limit: int = 4) -> str:
    """Join short values for dashboard action reasons."""
    clean = [str(value).strip() for value in values if str(value).strip()]
    if not clean:
        return ""
    text = ", ".join(clean[:limit])
    if len(clean) > limit:
        text += f", +{len(clean) - limit} more"
    return text


def _add_action(
    actions: list[dict],
    *,
    severity: str,
    area: str,
    action: str,
    why: str,
    command: str,
) -> None:
    """Append one normalized dashboard action."""
    priority = {"fail": 0, "warn": 1, "info": 2}.get(severity, 2)
    actions.append({
        "priority": priority,
        "severity": severity,
        "area": area,
        "action": action,
        "why": why,
        "command": command,
    })


def build_action_checklist() -> pd.DataFrame:
    """Return the dashboard's operator action checklist.

    PLAIN ENGLISH: The dashboard has many health files.  This function turns
    them into a short "do this next" list so the operator does not need to
    manually compare broker truth, signal gates, daily logs, and freshness.
    """
    actions: list[dict] = []

    broker_truth = load_broker_truth()
    truth_payload = broker_truth.get("payload") or {}
    truth_status = str(truth_payload.get("status") or "missing").lower()
    truth_summary = truth_payload.get("summary") or {}
    if truth_status == "fail":
        issues = [
            str(item.get("issue", ""))
            for item in truth_payload.get("global_issues", []) or []
            if isinstance(item, dict)
        ]
        table_issues = broker_truth_issue_table(broker_truth.get("table", pd.DataFrame()), limit=3)
        if not table_issues.empty and "issues" in table_issues.columns:
            issues.extend(table_issues["issues"].astype(str).tolist())
        _add_action(
            actions,
            severity="fail",
            area="Broker truth",
            action="Reconcile broker truth before new buys",
            why=_brief_join(issues) or "broker truth status is FAIL",
            command="python broker_truth.py --strict",
        )
    elif truth_status == "warning":
        _add_action(
            actions,
            severity="warn",
            area="Broker truth",
            action="Review broker truth warnings",
            why=(
                f"fail={int(truth_summary.get('fail_count') or 0)}, "
                f"warn={int(truth_summary.get('warning_count') or 0)}"
            ),
            command="python broker_truth.py",
        )
    elif truth_status in {"missing", "unknown", ""}:
        _add_action(
            actions,
            severity="warn",
            area="Broker truth",
            action="Generate broker truth report",
            why="broker truth JSON is missing",
            command="python broker_truth.py",
        )

    signal = load_current_signal() or {}
    if not signal:
        _add_action(
            actions,
            severity="fail",
            area="Signal",
            action="Generate today's trading signal",
            why="core_satellite_alpha_signal.csv is missing or unreadable",
            command="python core_satellite_alpha.py",
        )
    else:
        gate_state = signal_trade_gate_state(signal)
        if not gate_state["trade_ready"]:
            _add_action(
                actions,
                severity="fail",
                area="Signal gate",
                action="Fix blocked signal gates before submit",
                why=_brief_join(gate_state["block_reasons"]) or "submit gates are not all true",
                command="python daily_run.py --alpaca --dry-run",
            )

    last_run = load_latest_daily_run() or {}
    total = int(last_run.get("steps_total") or 0)
    passed = int(last_run.get("steps_ok") or 0)
    failed = int(last_run.get("steps_failed") or 0)
    if not last_run:
        _add_action(
            actions,
            severity="warn",
            area="Daily run",
            action="Run or pull the latest daily workflow output",
            why="no daily_run_*.json log found",
            command="python daily_run.py --alpaca --dry-run",
        )
    elif total and passed < total:
        bad_steps = [
            row.get("name", "")
            for row in last_run.get("results", []) or []
            if str(row.get("status", "")).lower() not in {"ok", "pass", "success"}
        ]
        _add_action(
            actions,
            severity="fail" if failed else "warn",
            area="Daily run",
            action="Inspect incomplete daily run steps",
            why=f"{passed}/{total} steps ok; {_brief_join(bad_steps)}",
            command="python daily_run.py --alpaca --dry-run",
        )

    scorecard = load_execution_scorecard()
    if not scorecard:
        _add_action(
            actions,
            severity="warn",
            area="Execution",
            action="Build execution scorecard",
            why="alpaca_execution_scorecard.json is missing",
            command="python execution_scorecard.py",
        )
    else:
        scorecard_status = str(scorecard.get("status") or "unknown").lower()
        if scorecard_status == "fail":
            failed_checks = [
                check.get("name", "")
                for check in scorecard.get("checks", []) or []
                if str(check.get("status", "")).lower() == "fail"
            ]
            _add_action(
                actions,
                severity="fail",
                area="Execution",
                action="Review failed execution checks",
                why=_brief_join(failed_checks) or "execution scorecard failed",
                command="python execution_scorecard.py --strict",
            )
        elif scorecard_status == "collecting":
            _add_action(
                actions,
                severity="info",
                area="Execution",
                action="Collect more fill samples",
                why="execution scorecard is still collecting",
                command="python execution_scorecard.py",
            )

    etf_health = load_etf_data_health()
    if not etf_health:
        _add_action(
            actions,
            severity="warn",
            area="ETF data",
            action="Build ETF data-health report",
            why="logs/etf_data_health.json is missing",
            command="python refresh_etf_data.py",
        )
    elif not bool(etf_health.get("ok", False)):
        bad_rows = []
        for row in etf_health.get("results", []) or []:
            if isinstance(row, dict) and not bool(row.get("ok", False)):
                issues = ",".join(str(item) for item in row.get("issues", [])[:3])
                bad_rows.append(f"{row.get('symbol', '?')}:{issues or 'not_ok'}")
        _add_action(
            actions,
            severity="fail",
            area="ETF data",
            action="Refresh stale or partial ETF data",
            why=_brief_join(bad_rows) or "ETF data health failed",
            command="python refresh_etf_data.py --refresh --force",
        )

    file_table = file_status_table()
    if not file_table.empty:
        status_text = file_table["Status"].astype(str).str.lower()
        critical = {
            "Today's signal",
            "Order plan",
            "Alpaca status",
            "Broker truth",
            "ETF data health",
            "Daily workflow",
        }
        critical_problem = file_table[
            file_table["File"].isin(critical)
            & status_text.str.contains("old|missing", regex=True)
        ]
        stale_problem = file_table[
            file_table["File"].isin(critical)
            & status_text.str.contains("stale", regex=False)
        ]
        if not critical_problem.empty:
            _add_action(
                actions,
                severity="fail",
                area="Freshness",
                action="Refresh missing or old critical files",
                why=_brief_join(critical_problem["File"].tolist()),
                command="pull_daily.bat",
            )
        elif not stale_problem.empty:
            _add_action(
                actions,
                severity="warn",
                area="Freshness",
                action="Watch stale critical files",
                why=_brief_join(stale_problem["File"].tolist()),
                command="pull_daily.bat",
            )

    workflow_df = load_workflow_heartbeats()
    if not workflow_df.empty:
        bad_workflows = []
        for row in workflow_df.to_dict("records"):
            status = str(row.get("status") or row.get("conclusion") or "unknown").lower()
            if status not in {"success", "completed", "ok"}:
                bad_workflows.append(row.get("label") or row.get("workflow") or "workflow")
        if bad_workflows:
            _add_action(
                actions,
                severity="warn",
                area="Workflow",
                action="Check latest workflow heartbeat",
                why=_brief_join(bad_workflows),
                command="gh run list --limit 5",
            )

    if not actions:
        return pd.DataFrame(columns=["priority", "severity", "area", "action", "why", "command"])
    return (
        pd.DataFrame(actions)
        .sort_values(["priority", "area", "action"])
        .reset_index(drop=True)
    )


@st.cache_data(ttl=60)
def load_broker_health() -> dict | None:
    """Return the broker connectivity ping result."""
    return _read_json_safe(BROKER_HEALTH)


@st.cache_data(ttl=60)
def load_workflow_heartbeats() -> pd.DataFrame:
    """Return workflow heartbeat JSON files as a small status table."""
    rows: list[dict] = []
    for label, path in WORKFLOW_HEARTBEATS.items():
        payload = _read_json_safe(path)
        if isinstance(payload, dict):
            row = dict(payload)
            row["label"] = label
            row["path"] = str(path.relative_to(PROJECT_ROOT))
            row["age_minutes"] = _file_age_minutes(path)
            rows.append(row)
        else:
            rows.append({
                "label": label,
                "workflow": label,
                "status": "missing",
                "event": None,
                "completed_at": None,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "age_minutes": None,
            })
    return pd.DataFrame(rows)


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
        "Alpaca execution": ALPACA_SLIPPAGE_REPORT,
        "Execution scorecard": ALPACA_EXECUTION_SCORECARD,
        "ETF data health": ETF_DATA_HEALTH,
        "Broker truth": BROKER_TRUTH_JSON,
        "Alpaca orders log": ALPACA_LOG,
        "Shadow equity": SHADOW_EQUITY,
        "Paper vs shadow": PAPER_SHADOW_COMPARE_JSON,
        "Daily workflow": WORKFLOW_HEARTBEATS["Daily paper"],
        "Shadow workflow": WORKFLOW_HEARTBEATS["Shadow journal"],
        "Feature health": FEATURE_HEALTH,
        "Feature quality": FEATURE_QUALITY,
        "Broker health": BROKER_HEALTH,
    }

    # ── Per-file expected-cadence freshness ──────────────────────────────
    # PLAIN ENGLISH: Different files update on different schedules.  The
    # Alpaca paper log fires once per trading day (~24h cycle), so flagging
    # it "🔴 old" at 24 hours guarantees that every dashboard view right
    # before the next morning's cron paints the whole panel red even when
    # everything is healthy.  Walkforward outputs only update monthly, so
    # the old 24-hour rule kept them permanently red.
    #
    # Each entry below carries (fresh_minutes, stale_minutes) thresholds:
    #   age < fresh           → 🟢 fresh
    #   fresh ≤ age < stale   → 🟡 stale (expected window — about to refresh)
    #   age ≥ stale           → 🔴 old (missed expected refresh)
    #
    # Defaults reflect the actual update cadence of each file.  Add or
    # adjust entries here when a new file is plumbed into the panel.
    FRESHNESS_THRESHOLDS = {
        # Walkforward outputs — monthly cadence.  Give 35 days fresh,
        # 60 days stale before flagging red.
        "Walkforward CSV":   (45 * 1440, 60 * 1440),
        "Walkforward JSON":  (45 * 1440, 60 * 1440),
        # Today's signal + orders + Alpaca log all update once per
        # weekday cron.  Fresh window covers cron-to-cron (~26 h
        # buffer for weekends).  Red only after a full missed day.
        "Today's signal":    (26 * 60, 72 * 60),
        "Order plan":        (26 * 60, 72 * 60),
        "Alpaca orders log": (26 * 60, 72 * 60),
        "Alpaca equity":     (26 * 60, 72 * 60),
        "Alpaca status":     (26 * 60, 72 * 60),
        "Alpaca health":     (26 * 60, 72 * 60),
        "Alpaca execution":  (26 * 60, 72 * 60),
        "Execution scorecard": (26 * 60, 72 * 60),
        "ETF data health":   (26 * 60, 72 * 60),
        "Broker truth":      (26 * 60, 72 * 60),
        "Broker health":     (26 * 60, 72 * 60),
        "Shadow equity":     (26 * 60, 72 * 60),
        "Paper vs shadow":   (26 * 60, 72 * 60),
        "Daily workflow":    (26 * 60, 72 * 60),
        "Shadow workflow":   (26 * 60, 72 * 60),
        # Feature health + quality update on the factor-data-refresh
        # cron (also daily) but commits to signals/latest only after
        # the trade workflow finishes.  Same 26h/72h works.
        "Feature health":    (26 * 60, 72 * 60),
        "Feature quality":   (26 * 60, 72 * 60),
    }
    # Default for any file not explicitly listed above — sensible 24h/72h.
    DEFAULT_FRESH_MINUTES = 26 * 60
    DEFAULT_STALE_MINUTES = 72 * 60

    rows = []
    for label, path in files.items():
        if path.exists():
            age_min = _file_age_minutes(path) or 0
            size_kb = path.stat().st_size / 1024
            fresh_thresh, stale_thresh = FRESHNESS_THRESHOLDS.get(
                label, (DEFAULT_FRESH_MINUTES, DEFAULT_STALE_MINUTES),
            )
            if age_min < fresh_thresh:
                status = "🟢 fresh"
            elif age_min < stale_thresh:
                status = "🟡 stale"
            else:
                status = "🔴 old"
            rows.append({
                "File": label,
                "Path": _display_path(path),
                "Age (min)": round(age_min, 1),
                "Size (KB)": round(size_kb, 1),
                "Status": status,
            })
        else:
            rows.append({
                "File": label,
                "Path": _display_path(path),
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
    history_latest: dict = {}
    if not history.empty:
        history_latest = history.iloc[-1].to_dict()

    # PLAIN ENGLISH: Sometimes the live status JSON is missing but the equity
    # CSV exists.  In that case, use the latest equity row so the top dashboard
    # card never shows a fake $0 while the equity chart shows real account value.
    source = status.get("_source", "alpaca_daily_status.json" if status else "alpaca_paper_equity.csv")
    generated_at = status.get("generated_at")
    if equity <= 0 and history_latest:
        equity = _to_float(history_latest.get("equity"), 0.0)
        source = "alpaca_paper_equity.csv"
        generated_at = str(
            history_latest.get("timestamp")
            or history_latest.get("date")
            or generated_at
            or ""
        )

    if "account_cash" in status:
        cash = _to_float(status.get("account_cash"), 0.0)
    else:
        cash = _to_float(history_latest.get("cash"), 0.0) if history_latest else 0.0

    if "account_invested" in status:
        invested = _to_float(status.get("account_invested"), 0.0)
    elif history_latest and "invested" in history_latest:
        invested = _to_float(history_latest.get("invested"), max(equity - cash, 0.0))
    else:
        invested = max(equity - cash, 0.0)

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
        "generated_at": generated_at,
        "source": source,
        "live_refresh": status.get("_live_refresh"),
    }
