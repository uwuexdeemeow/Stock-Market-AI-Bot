"""
Run Scripts — control panel for backend operations

PLAIN ENGLISH: Click a button → that backend script runs.  Output
streams live into the page.  Full logs saved to logs/dashboard_*.log
for later review.

The buttons are grouped by what they do:
  1. Data refresh    — pull fresh prices + recompute features
  2. Signal pipeline — full daily run OR individual steps
  3. Trading actions — submit orders, reconcile fills (destructive!)
  4. Diagnostics     — health checks, drift, gauntlet, analyzer
  5. Heavy ops       — walkforward (hours), publish live config

Numbered "0_" so this page sits at the TOP of the sidebar.
"""

import sys
import streamlit as st

from dashboard.components import sidebar_refresh, script_button


st.set_page_config(page_title="Run Scripts", page_icon="⚙️", layout="wide")
sidebar_refresh()
st.title("⚙️ Run Backend Scripts")
st.caption(
    "Click a button to run a backend script.  Output streams live into "
    "this page.  Full logs saved to `logs/dashboard_*.log`."
)


# The Python interpreter currently running the dashboard.  Using the
# same one for child processes guarantees we hit the same venv/site-packages.
PY = sys.executable


# ─────────────────────────────────────────────────────────────────────
# Section 1 — DATA REFRESH
# ─────────────────────────────────────────────────────────────────────
st.markdown("### 📥 Data Refresh")
st.caption("Pull fresh prices + recompute features.  Run these first if data is stale.")

col1, col2, col3 = st.columns(3)

with col1:
    script_button(
        "Refresh ETF data",
        [PY, "refresh_etf_data.py", "--refresh", "--force"],
        key="refresh_etf",
        help_text="Downloads SPY/QQQ/TQQQ/etc.  ~30 seconds.",
        expected_runtime="~30s",
    )

with col2:
    script_button(
        "Refresh research data (incremental)",
        [PY, "research.py", "--incremental"],
        key="refresh_research_inc",
        help_text="Updates the factor panel with new days only.  Skips historical recomputation.",
        expected_runtime="2-5 min",
    )

with col3:
    script_button(
        "Refresh feature quality report",
        [PY, "feature_quality_diagnostic.py", "--top", "48"],
        key="refresh_fq",
        help_text="Re-ranks features by predictive power.",
        expected_runtime="1-2 min",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────
# Section 2 — SIGNAL PIPELINE
# ─────────────────────────────────────────────────────────────────────
st.markdown("### 🎯 Signal Pipeline")
st.caption("Full daily pipeline OR just the signal-generation step.")

col1, col2 = st.columns(2)

with col1:
    script_button(
        "Run full daily pipeline (--alpaca)",
        [PY, "daily_run.py", "--alpaca", "--skip-refresh", "--timeout", "600"],
        key="daily_run",
        help_text=(
            "Full daily pipeline: signal → submit → reconcile → health "
            "→ gauntlet.  Skips data refresh (use buttons above for that)."
        ),
        expected_runtime="3-5 min",
    )

with col2:
    script_button(
        "Generate signal only (no orders)",
        [PY, "core_satellite_alpha.py"],
        key="signal_only",
        help_text="Just regenerates today's signal CSV.  Doesn't submit any orders.",
        expected_runtime="~30s",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────
# Section 3 — TRADING ACTIONS (destructive)
# ─────────────────────────────────────────────────────────────────────
st.markdown("### 💸 Trading Actions")
st.warning(
    "⚠️  These send orders to Alpaca paper.  Paper money = no real risk, "
    "but still produces fill events and equity changes.",
    icon="⚠️",
)

col1, col2, col3 = st.columns(3)

with col1:
    script_button(
        "Submit orders to Alpaca",
        [PY, "alpaca_paper_trading.py", "--submit"],
        key="submit",
        help_text="Submit today's rebalance orders.  Will skip if already submitted.",
        expected_runtime="~1 min",
        destructive=True,
    )

with col2:
    script_button(
        "Reconcile fills",
        [PY, "alpaca_paper_trading.py", "--reconcile"],
        key="reconcile",
        help_text="Check pending orders for fills/cancellations.",
        expected_runtime="~30s",
    )

with col3:
    script_button(
        "Show account status (no orders)",
        [PY, "alpaca_paper_trading.py", "--status"],
        key="status",
        help_text="Print current positions, cash, equity.  Read-only.",
        expected_runtime="~10s",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────
# Section 4 — DIAGNOSTICS / HEALTH
# ─────────────────────────────────────────────────────────────────────
st.markdown("### 🩺 Diagnostics & Health")
st.caption("Read-only checks.  Safe to run anytime.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    script_button(
        "Broker health check",
        [PY, "broker_health.py", "--alpaca"],
        key="broker_health",
        help_text="Ping Alpaca API, return equity + latency.",
        expected_runtime="~10s",
    )

with col2:
    script_button(
        "Paper health summary",
        [PY, "paper_health.py", "--broker", "alpaca"],
        key="paper_health",
        help_text="Build deep health summary: drift, slippage, risk, scorecard.",
        expected_runtime="~30s",
    )

with col3:
    script_button(
        "Alpaca paper gauntlet",
        [PY, "alpaca_paper_gauntlet.py"],
        key="gauntlet",
        help_text="Go-live readiness check (Sharpe, drawdown, fill rate, etc.).",
        expected_runtime="~20s",
    )

with col4:
    script_button(
        "Walkforward analyzer",
        [PY, "walkforward_analyzer.py"],
        key="wf_analyzer",
        help_text="Checks the latest walkforward CSV for calibration issues.",
        expected_runtime="~5s",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────
# Section 5 — HEAVY OPERATIONS
# ─────────────────────────────────────────────────────────────────────
st.markdown("### 🏋️ Heavy Operations")
st.caption(
    "These take a long time — only run when you actually need to "
    "re-validate or re-publish."
)

col1, col2 = st.columns(2)

with col1:
    confirm_wf = st.checkbox(
        "I understand this takes hours",
        key="wf_confirm",
        help="The nested walkforward runs 384 configs × 14 folds.  Takes hours on a laptop.",
    )
    if confirm_wf:
        script_button(
            "Run nested walkforward (hours)",
            [PY, "core_satellite_nested_walkforward.py", "--strategy", "core-alpha", "--full"],
            key="walkforward",
            help_text="The full nested walkforward.  Takes hours.  Don't close the tab.",
            expected_runtime="2-6 hours",
            destructive=True,
        )
    else:
        st.button(
            "Run nested walkforward (hours)  ·  ⏱ 2-6 hours",
            disabled=True,
            use_container_width=True,
            help="Check the confirmation box above to enable.",
        )

with col2:
    script_button(
        "Publish live config from CSV",
        [PY, "publish_live_config_from_csv.py"],
        key="publish",
        help_text=(
            "Re-publishes the live config from the most recent "
            "walkforward CSV.  Useful when you tweaked thresholds or "
            "want to repromote a config without re-running the WF."
        ),
        expected_runtime="~5s",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────
# Section 6 — RECENT DASHBOARD-INITIATED RUNS
# ─────────────────────────────────────────────────────────────────────
st.markdown("### 📜 Recent dashboard logs")
from pathlib import Path

logs_dir = Path("logs")
if logs_dir.exists():
    dashboard_logs = sorted(
        logs_dir.glob("dashboard_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:10]
    if dashboard_logs:
        import datetime as _dt
        for log_path in dashboard_logs:
            mtime = _dt.datetime.fromtimestamp(log_path.stat().st_mtime)
            size_kb = log_path.stat().st_size / 1024
            with st.expander(f"📄 {log_path.name}  ·  {mtime.strftime('%H:%M:%S')}  ·  {size_kb:.1f} KB"):
                try:
                    content = log_path.read_text(errors="replace")
                    # Show last 100 lines
                    lines = content.splitlines()[-100:]
                    st.code("\n".join(lines), language="text")
                except Exception as e:
                    st.error(f"Couldn't read log: {e}")
    else:
        st.caption("No dashboard runs yet.")
else:
    st.caption("`logs/` folder will be created on first run.")
