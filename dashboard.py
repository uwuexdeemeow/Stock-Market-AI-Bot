"""
dashboard.py — Stock Bot Dashboard (main entry point)

PLAIN ENGLISH: This is the home page when you run `streamlit run
dashboard.py`.  It shows a quick health summary and equity curve;
detail lives in the sidebar pages.

Run with:
    streamlit run dashboard.py
"""

# ── Imports ────────────────────────────────────────────────────────────
import streamlit as st

from dashboard import data
from dashboard.components import (
    account_summary_card,
    equity_curve_chart,
    regime_indicator,
    status_chip,
    sidebar_refresh,
)


# ── Page config (always first) ─────────────────────────────────────────
st.set_page_config(
    page_title="Stock Bot",
    page_icon="•",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar — refresh only (page nav is auto-injected) ────────────────
sidebar_refresh()


# ── Main page ─────────────────────────────────────────────────────────
st.title("Stock Bot")
st.caption("Live paper-trading dashboard")

# Account summary at top
summary = data.compute_account_summary()
account_summary_card(summary)

st.divider()

# Regime + health chips
col_a, col_b = st.columns([2, 3])

with col_a:
    signal = data.load_current_signal() or {}
    regime = str(signal.get("current_regime", "unknown"))
    regime_indicator(regime)
    if signal.get("predicted_at"):
        st.caption(f"Signal generated: {signal.get('predicted_at')}")

with col_b:
    st.markdown("##### System status")
    h1, h2, h3 = st.columns(3)
    with h1:
        wf = data.load_walkforward_summary() or {}
        approval = (wf.get("live_config_approval") or {}).get("approved", False)
        status_chip("Walkforward approved" if approval else "Not approved",
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

# Equity curve
st.markdown("##### Equity (last 90 days)")
equity_df = data.load_alpaca_equity_history()
qqq_df = data.load_etf_data("QQQ", days=90)

if equity_df.empty:
    st.info("No equity history yet. After a few days of trading the curve will populate.")
else:
    st.plotly_chart(
        equity_curve_chart(equity_df.tail(90), qqq_df),
        use_container_width=True,
    )

st.divider()

# Quick file freshness check
with st.expander("Data freshness"):
    st.dataframe(data.file_status_table(), use_container_width=True, hide_index=True)

st.caption(
    "Data cached for 30–60s. Use **Refresh** in the sidebar to force a "
    "reread. When the daily run completes (~9:35 AM ET) files update and "
    "the dashboard reflects it on the next auto-refresh."
)
