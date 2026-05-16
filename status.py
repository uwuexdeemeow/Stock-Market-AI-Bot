#!/usr/bin/env python3
"""
status.py — Single-screen terminal dashboard for the Stock Market AI Bot.

PLAIN ENGLISH: This script reads all signal files, logs, and configs from disk
and prints a colour-coded summary of your entire trading system's state.  No
API calls needed — it just reads what's already on disk from the last daily run.

HOW TO RUN:
    python3 status.py              # Full status
    python3 status.py --json       # Machine-readable JSON output
    python3 status.py --short      # One-line summary (for cron/motd)

KEY CONCEPTS:
  - "paper_ready" = the system has an approved signal and is actively trading
  - "regime" = the market's current state (risk_on, neutral, risk_off)
  - "live config" = the walkforward-approved parameter set used for signals
  - "factor decay" = whether the model's predictive features are still working
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SIGNAL_DIR = PROJECT_ROOT / "signals"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Signal/config files
LIVE_CONFIG_PATH = SIGNAL_DIR / "core_satellite_live_configs.json"
SIGNAL_PATH = SIGNAL_DIR / "core_satellite_alpha_signal.csv"
METRICS_PATH = SIGNAL_DIR / "core_satellite_alpha_metrics.json"
ORDERS_PATH = SIGNAL_DIR / "core_satellite_alpha_orders.csv"
PAPER_STATUS_PATH = SIGNAL_DIR / "paper_daily_status.json"
EQUITY_PATH = SIGNAL_DIR / "alpaca_paper_equity.csv"
TRADE_LOG_PATH = SIGNAL_DIR / "alpaca_paper_log.csv"
FACTOR_DECAY_PATH = SIGNAL_DIR / "factor_decay_monitor.csv"
BROKER_HEALTH_PATH = SIGNAL_DIR / "broker_health.json"
FEATURE_HEALTH_PATH = SIGNAL_DIR / "feature_health_profile.json"


# ── ANSI colours ─────────────────────────────────────────────────────────────
# PLAIN ENGLISH: These are escape codes that make terminal text colourful.
# If the output is piped to a file, colours are disabled automatically.
USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap text in ANSI colour code."""
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t: str) -> str: return _c("32", t)
def red(t: str) -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def cyan(t: str) -> str: return _c("36", t)
def bold(t: str) -> str: return _c("1", t)
def dim(t: str) -> str: return _c("2", t)


def status_icon(ok: bool) -> str:
    """Return ✓ or ✗ with colour."""
    return green("✓") if ok else red("✗")


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    """Load a JSON file, return None if missing/broken."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_csv_last_row(path: Path) -> dict | None:
    """Load the last row of a CSV as a dict."""
    if not path.exists():
        return None
    try:
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-1] if rows else None
    except Exception:
        return None


def _load_csv_rows(path: Path) -> list[dict]:
    """Load all rows of a CSV."""
    if not path.exists():
        return []
    try:
        import csv
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _file_age_str(path: Path) -> str:
    """Human-readable file age."""
    if not path.exists():
        return "missing"
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - mtime
    if age.total_seconds() < 60:
        return "just now"
    elif age.total_seconds() < 3600:
        return f"{int(age.total_seconds() / 60)}m ago"
    elif age.total_seconds() < 86400:
        return f"{int(age.total_seconds() / 3600)}h ago"
    else:
        return f"{age.days}d ago"


# ── Section renderers ────────────────────────────────────────────────────────

def render_live_config() -> dict:
    """Live config status: approved?, age, expiry countdown."""
    info = {"section": "Live Config"}
    payload = _load_json(LIVE_CONFIG_PATH)
    if not payload:
        info["status"] = "MISSING"
        info["line"] = f"Live Config:     {red('MISSING')} — run nested walkforward"
        return info

    # Parse config age
    created_str = payload.get("created_at", "")
    age_days = None
    if created_str:
        try:
            created = datetime.fromisoformat(created_str)
            age_days = (datetime.now(timezone.utc) - created).days
        except (ValueError, TypeError):
            pass

    # Check approval
    approvals = payload.get("approvals", {})
    core_approval = approvals.get("core-alpha", {})
    approved = bool(core_approval.get("approved", False))

    # Config family (short name)
    config_family = core_approval.get("approved_config_family", "unknown")
    # Shorten for display
    short_config = config_family.split(",score=")[-1].split(",")[0] if "score=" in config_family else config_family[:40]

    # Expiry
    max_age = int(os.environ.get("LIVE_CONFIG_MAX_AGE_DAYS", "45"))
    days_left = (max_age - age_days) if age_days is not None else None

    status_str = green("APPROVED") if approved else red("REJECTED")
    age_str = f"{age_days}d old" if age_days is not None else "unknown age"
    expiry_str = ""
    if days_left is not None:
        if days_left <= 7:
            expiry_str = f", {red(f'expires in {days_left}d')}"
        elif days_left <= 14:
            expiry_str = f", {yellow(f'expires in {days_left}d')}"
        else:
            expiry_str = f", expires in {days_left}d"

    info["approved"] = approved
    info["age_days"] = age_days
    info["days_until_expiry"] = days_left
    info["line"] = f"Live Config:     {status_str} ({age_str}{expiry_str}) [{short_config}]"
    return info


def render_signal() -> dict:
    """Latest signal: paper_ready, positions, regime."""
    info = {"section": "Signal"}
    row = _load_csv_last_row(SIGNAL_PATH)
    if not row:
        info["line"] = f"Latest Signal:   {red('NO SIGNAL FILE')}"
        return info

    paper_ready = str(row.get("paper_ready", "")).lower() == "true"
    regime = row.get("current_regime", "?")
    score_source = row.get("score_source", "?")
    predicted_at = row.get("predicted_at", "?")

    # Parse overlay tickers
    overlay_json = row.get("overlay_weights_json", "{}")
    try:
        overlay = json.loads(overlay_json)
        overlay_tickers = list(overlay.keys())
        n_positions = len(overlay_tickers)
    except (json.JSONDecodeError, TypeError):
        overlay_tickers = []
        n_positions = 0

    # Core weight
    core_gross = row.get("core_gross", "?")
    overlay_gross = row.get("overlay_gross", "?")

    ready_str = green("READY") if paper_ready else red("BLOCKED")
    regime_color = {"risk_on": green, "neutral": yellow, "risk_off": red}.get(regime, dim)
    regime_str = regime_color(regime)

    # Predicted at — how fresh?
    age_str = ""
    try:
        pred_dt = datetime.fromisoformat(predicted_at)
        pred_age = datetime.now(timezone.utc) - pred_dt.astimezone(timezone.utc)
        if pred_age.total_seconds() < 86400:
            age_str = f" ({int(pred_age.total_seconds() / 3600)}h ago)"
        else:
            age_str = f" ({pred_age.days}d ago)"
    except (ValueError, TypeError):
        pass

    info["paper_ready"] = paper_ready
    info["regime"] = regime
    info["n_positions"] = n_positions
    info["line"] = (
        f"Latest Signal:   {ready_str} | regime={regime_str} | "
        f"{n_positions} overlay positions | core={core_gross} overlay={overlay_gross}{age_str}"
    )
    return info


def render_equity() -> dict:
    """Alpaca paper trading equity."""
    info = {"section": "Equity"}
    row = _load_csv_last_row(EQUITY_PATH)
    if not row:
        # Try paper_daily_status.json as fallback
        status = _load_json(PAPER_STATUS_PATH)
        if status and "account_equity" in status:
            equity = float(status["account_equity"])
            info["equity"] = equity
            info["line"] = f"Paper Equity:    ${equity:,.2f} (from moomoo status)"
            return info
        info["line"] = f"Paper Equity:    {dim('no equity data')}"
        return info

    try:
        equity = float(row.get("equity", 0))
        cash = float(row.get("cash", 0))
        invested = float(row.get("invested", 0))
        date = row.get("date", "?")
    except (ValueError, TypeError):
        info["line"] = f"Paper Equity:    {dim('parse error')}"
        return info

    # Simple P&L (assuming 100k start)
    starting = 100000.0
    pnl = equity - starting
    pnl_pct = (pnl / starting) * 100 if starting > 0 else 0
    pnl_str = green(f"+{pnl_pct:.2f}%") if pnl >= 0 else red(f"{pnl_pct:.2f}%")

    info["equity"] = equity
    info["pnl_pct"] = pnl_pct
    info["line"] = f"Paper Equity:    ${equity:,.2f} ({pnl_str} since inception) | cash=${cash:,.0f} | as of {date}"
    return info


def render_positions() -> dict:
    """Current positions from paper_daily_status."""
    info = {"section": "Positions"}
    status = _load_json(PAPER_STATUS_PATH)
    if not status or "positions" not in status:
        info["line"] = f"Positions:       {dim('no position data')}"
        return info

    positions = status.get("positions", {})
    position_values = status.get("position_values", {})
    total_equity = float(status.get("account_equity", 1))

    # Sort by value descending
    sorted_pos = sorted(
        positions.items(),
        key=lambda x: float(position_values.get(x[0], 0)),
        reverse=True,
    )

    lines = []
    for ticker, qty in sorted_pos[:10]:
        value = float(position_values.get(ticker, 0))
        weight = (value / total_equity) * 100 if total_equity > 0 else 0
        lines.append(f"  {ticker:5s} {int(qty):>5d} sh  ${value:>10,.0f}  ({weight:.1f}%)")

    exposure = float(status.get("current_gross_exposure", 0)) * 100
    info["n_positions"] = len(positions)
    info["exposure"] = exposure
    info["lines"] = [
        f"Positions:       {len(positions)} holdings | gross exposure={exposure:.1f}%",
        *lines,
    ]
    return info


def render_orders() -> dict:
    """Recent orders."""
    info = {"section": "Orders"}
    orders = _load_csv_rows(ORDERS_PATH)
    if not orders:
        info["line"] = f"Today's Orders:  {dim('none')}"
        return info

    buys = sum(1 for o in orders if o.get("side") == "buy")
    sells = sum(1 for o in orders if o.get("side") == "sell")
    total_value = sum(float(o.get("trade_value", 0)) for o in orders)

    info["buys"] = buys
    info["sells"] = sells
    info["line"] = f"Today's Orders:  {buys} buys, {sells} sells (${total_value:,.0f} notional)"
    return info


def render_factor_health() -> dict:
    """Factor decay monitor status."""
    info = {"section": "Factor Health"}
    rows = _load_csv_rows(FACTOR_DECAY_PATH)
    if not rows:
        info["line"] = f"Factor Health:   {dim('no decay monitor data')}"
        return info

    # Use the most recent row (longest lookback usually last)
    latest = rows[-1]
    status = latest.get("edge_health_status", "unknown")
    ic_mean = latest.get("daily_ic_mean", "?")
    warning = str(latest.get("warning", "")).lower() == "true"

    status_color = {
        "pass": green, "advisory": yellow, "warning": red, "block": red
    }.get(status, dim)

    info["status"] = status
    info["warning"] = warning
    info["line"] = f"Factor Health:   {status_color(status)} | IC mean={ic_mean} | as of {latest.get('as_of', '?')}"
    return info


def render_feature_health() -> dict:
    """Feature health profile."""
    info = {"section": "Feature Health"}
    data = _load_json(FEATURE_HEALTH_PATH)
    if not data:
        info["line"] = f"Feature Health:  {dim('no profile data')}"
        return info

    n_features = data.get("n_features") or data.get("total_features", "?")
    quarantined = data.get("quarantined", []) or data.get("quarantined_features", [])
    n_quarantined = len(quarantined) if isinstance(quarantined, list) else 0

    q_str = green("0 quarantined") if n_quarantined == 0 else yellow(f"{n_quarantined} quarantined")
    info["n_features"] = n_features
    info["n_quarantined"] = n_quarantined
    info["line"] = f"Feature Health:  {n_features} features | {q_str}"
    return info


def render_last_daily_run() -> dict:
    """Last daily run log."""
    info = {"section": "Last Daily Run"}
    # Find most recent daily_run log
    log_files = sorted(LOG_DIR.glob("daily_run_*.json"), reverse=True)
    if not log_files:
        info["line"] = f"Last Daily Run:  {dim('no logs found')}"
        return info

    log_path = log_files[0]
    data = _load_json(log_path)
    if not data:
        info["line"] = f"Last Daily Run:  {dim('unreadable log')}"
        return info

    timestamp = data.get("timestamp", "?")
    steps_ok = int(data.get("steps_ok", 0))
    steps_failed = int(data.get("steps_failed", 0))
    elapsed = float(data.get("total_elapsed_seconds", 0))
    steps_total = int(data.get("steps_total", 0))

    # Count skipped
    results = data.get("results", [])
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    if steps_failed > 0:
        status_str = red(f"FAILED ({steps_failed}/{steps_total} steps)")
    elif skipped == steps_total:
        status_str = yellow("ALL SKIPPED")
    else:
        status_str = green(f"OK ({steps_ok}/{steps_total} steps)")

    elapsed_str = f"{elapsed:.0f}s" if elapsed < 120 else f"{elapsed/60:.1f}m"
    age_str = _file_age_str(log_path)

    info["ok"] = steps_failed == 0
    info["line"] = f"Last Daily Run:  {status_str} | {elapsed_str} | {age_str} ({log_path.name})"
    return info


def render_walkforward_due() -> dict:
    """When the next walkforward should run (based on config expiry)."""
    info = {"section": "Walkforward Due"}
    payload = _load_json(LIVE_CONFIG_PATH)
    if not payload:
        info["line"] = f"Next Walkforward: {yellow('NOW')} (no live config exists)"
        return info

    created_str = payload.get("created_at", "")
    max_age = int(os.environ.get("LIVE_CONFIG_MAX_AGE_DAYS", "45"))
    if created_str:
        try:
            created = datetime.fromisoformat(created_str)
            expiry_date = created + timedelta(days=max_age)
            days_left = (expiry_date - datetime.now(timezone.utc)).days
            if days_left <= 0:
                info["line"] = f"Next Walkforward: {red('OVERDUE')} (expired {abs(days_left)}d ago)"
            elif days_left <= 7:
                info["line"] = f"Next Walkforward: {yellow(f'due in {days_left}d')} (~{expiry_date.strftime('%Y-%m-%d')})"
            else:
                info["line"] = f"Next Walkforward: due in {days_left}d (~{expiry_date.strftime('%Y-%m-%d')})"
            info["days_left"] = days_left
            return info
        except (ValueError, TypeError):
            pass

    info["line"] = f"Next Walkforward: {dim('unknown (no created_at in config)')}"
    return info


def render_broker_health() -> dict:
    """Broker connectivity status."""
    info = {"section": "Broker Health"}
    data = _load_json(BROKER_HEALTH_PATH)
    if not data:
        info["line"] = f"Broker Health:   {dim('no health check data')}"
        return info

    # broker_health.json structure varies — try common fields
    connected = data.get("connected", data.get("alpaca_connected", None))
    status = data.get("status", "unknown")
    checked_at = data.get("checked_at", data.get("timestamp", ""))

    if connected is True or status == "ok":
        info["line"] = f"Broker Health:   {green('connected')} ({_file_age_str(BROKER_HEALTH_PATH)})"
    elif connected is False:
        info["line"] = f"Broker Health:   {red('DISCONNECTED')} ({_file_age_str(BROKER_HEALTH_PATH)})"
    else:
        info["line"] = f"Broker Health:   {dim(status)} ({_file_age_str(BROKER_HEALTH_PATH)})"
    return info


def render_data_freshness() -> dict:
    """How fresh are the parquet data files."""
    info = {"section": "Data Freshness"}
    if not DATA_DIR.exists():
        info["line"] = f"Data Freshness:  {red('no data/ directory')}"
        return info

    parquets = list(DATA_DIR.glob("*.parquet"))
    if not parquets:
        info["line"] = f"Data Freshness:  {red('no parquet files')}"
        return info

    # Check most recent and oldest modification times
    mtimes = [(p, p.stat().st_mtime) for p in parquets]
    newest = max(mtimes, key=lambda x: x[1])
    oldest = min(mtimes, key=lambda x: x[1])

    newest_age = datetime.now(timezone.utc) - datetime.fromtimestamp(newest[1], tz=timezone.utc)
    oldest_age = datetime.now(timezone.utc) - datetime.fromtimestamp(oldest[1], tz=timezone.utc)

    n_files = len(parquets)
    newest_str = f"{newest_age.days}d" if newest_age.days > 0 else f"{int(newest_age.total_seconds()/3600)}h"
    oldest_str = f"{oldest_age.days}d" if oldest_age.days > 0 else f"{int(oldest_age.total_seconds()/3600)}h"

    fresh = newest_age.days <= 2
    info["n_files"] = n_files
    info["newest_age_days"] = newest_age.days
    info["line"] = (
        f"Data Freshness:  {n_files} parquets | "
        f"newest={status_icon(fresh)} {newest_str} ({newest[0].name}) | "
        f"oldest={oldest_str} ({oldest[0].name})"
    )
    return info


# ── Main output ──────────────────────────────────────────────────────────────

def render_full_status() -> list[dict]:
    """Render all sections."""
    sections = [
        render_live_config(),
        render_signal(),
        render_equity(),
        render_positions(),
        render_orders(),
        render_factor_health(),
        render_feature_health(),
        render_broker_health(),
        render_data_freshness(),
        render_last_daily_run(),
        render_walkforward_due(),
    ]
    return sections


def print_status():
    """Print the full terminal dashboard."""
    print()
    print(bold("═══════════════════════════════════════════════════════"))
    print(bold("          📈  STOCK BOT STATUS  📈"))
    print(bold("═══════════════════════════════════════════════════════"))
    print()

    sections = render_full_status()
    for section in sections:
        if "lines" in section:
            for line in section["lines"]:
                print(f"  {line}")
        elif "line" in section:
            print(f"  {section['line']}")

    print()
    print(dim(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    print(dim(f"  Project:   {PROJECT_ROOT}"))
    print()


def print_short():
    """One-line summary for cron/motd."""
    config = render_live_config()
    signal = render_signal()
    equity = render_equity()

    ready = "READY" if signal.get("paper_ready") else "BLOCKED"
    eq = f"${equity.get('equity', 0):,.0f}" if equity.get("equity") else "?"
    regime = signal.get("regime", "?")
    approved = "approved" if config.get("approved") else "REJECTED"

    print(f"StockBot: {ready} | {eq} | regime={regime} | config={approved}")


def print_json():
    """Machine-readable JSON."""
    sections = render_full_status()
    # Strip ANSI from lines for JSON
    import re
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    output = {}
    for s in sections:
        key = s["section"].lower().replace(" ", "_")
        clean = dict(s)
        if "line" in clean:
            clean["line"] = ansi_escape.sub("", clean["line"])
        if "lines" in clean:
            clean["lines"] = [ansi_escape.sub("", l) for l in clean["lines"]]
        output[key] = clean
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stock Bot status dashboard — shows system health at a glance."
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--short", action="store_true", help="One-line summary")
    args = parser.parse_args()

    if args.json:
        print_json()
    elif args.short:
        print_short()
    else:
        print_status()
