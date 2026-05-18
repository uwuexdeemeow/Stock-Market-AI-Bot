"""
dashboard.py — Stock Bot Dashboard (main entry point)

PLAIN ENGLISH: This is the home page when you run `streamlit run
dashboard.py`.  It shows a quick health summary, then directs you to
the detail pages in the sidebar (Overview, Performance, Signal, etc.).

The dashboard is DYNAMIC — it reads from the bot's signal/log files
and auto-refreshes every 30-60 seconds so you see fresh data without
reloading.  When you change where data lives, edit dashboard/data.py
(single source of truth for paths) — the pages stay clean.

Run with:
    streamlit run dashboard.py

Then open the URL it prints (usually http://localhost:8501).

Files this dashboard reads (all optional — missing files degrade
gracefully):
  - signals/core_satellite_live_configs.json   (live config)
  - signals/core_satellite_nested_walkforward.csv (WF results)
  - signals/core_satellite_alpha_signal.csv    (today's signal)
  - signals/alpaca_paper_equity.csv            (equity history)
  - signals/alpaca_daily_status.json           (live positions)
  - signals/alpaca_paper_health.json           (drift detection)
  - signals/feature_health_profile.json        (active features)
  - logs/daily_run_YYYYMMDD.json               (latest pipeline run)
"""

# ── Imports ────────────────────────────────────────────────────────────
import streamlit as st

from dashboard import data
from dashboard.components import (
    account_summary_card,
    equity_curve_chart,
    regime_indicator,
    status_chip,
    freshness_badge,
    sidebar_refresh,
)


# ── Page config (always first) ─────────────────────────────────────────
st.set_page_config(
    page_title="Stock Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar — minimal refresh control (page nav is auto-injected) ─────
sidebar_refresh()


# ── Main page — top-line health snapshot ───────────────────────────────
st.title("📊 Stock Bot Dashboard")
st.caption("Welcome. This home page is a one-glance health view. Use the sidebar for detail.")

# ── Row 1: account summary ────────────────────────────────────────────
summary = data.compute_account_summary()
account_summary_card(summary)
st.divider()

# ── Row 2: regime + key health chips ─────────────────────────────────
col_a, col_b = st.columns([2, 3])

with col_a:
    signal = data.load_current_signal() or {}
    regime = str(signal.get("current_regime", "unknown"))
    regime_indicator(regime)
    st.caption(f"Predicted at: {signal.get('predicted_at', 'n/a')}")

with col_b:
    st.markdown("### Quick health")
    h1, h2, h3 = st.columns(3)
    with h1:
        wf = data.load_walkforward_summary() or {}
        approval = (wf.get("live_config_approval") or {}).get("approved", False)
        status_chip("Walkforward approved" if approval else "WF not approved",
                    "ok" if approval else "fail")
    with h2:
        ready = bool(signal.get("paper_ready", False))
        gates = bool(signal.get("gates_all_pass", False))
        status_chip("Paper ready" if ready and gates else "Gates blocked",
                    "ok" if ready and gates else "warn")
    with h3:
        last_run = data.load_latest_daily_run() or {}
        passed = last_run.get("steps_ok", 0)
        total = last_run.get("steps_total", 0)
        if total:
            ratio = passed / total
            status = "ok" if ratio >= 0.9 else "warn" if ratio >= 0.6 else "fail"
            status_chip(f"Daily run {passed}/{total}", status)
        else:
            status_chip("No daily run log", "unknown")

st.divider()

# ── Row 3: mini equity curve ──────────────────────────────────────────
st.markdown("### Equity (last 90 days)")
equity_df = data.load_alpaca_equity_history()
qqq_df = data.load_etf_data("QQQ", days=90)

if equity_df.empty:
    st.info("No equity history yet. After a few days of trading, the curve will populate.")
else:
    st.plotly_chart(
        equity_curve_chart(equity_df.tail(90), qqq_df),
        use_container_width=True,
    )

st.divider()

# ── Row 4: data freshness summary ─────────────────────────────────────
with st.expander("📁 Data freshness (which files are current?)", expanded=False):
    st.dataframe(data.file_status_table(), use_container_width=True, hide_index=True)


# ── Auto-refresh notice ───────────────────────────────────────────────
st.caption(
    "🔄 Data cached for 30–60s.  Use **Refresh** in the sidebar to force a "
    "reread, or wait for the cache to expire.  When the daily run completes "
    "(~9:35 AM ET), the files update and the dashboard reflects it on the "
    "next auto-refresh."
)
