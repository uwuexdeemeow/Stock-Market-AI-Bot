"""
dashboard.py — Stock Bot Dashboard (main entry point)

PLAIN ENGLISH: Run with `streamlit run dashboard.py`.  This is the home
page — quick health summary, equity curve, regime indicator, freshness
checks.  Detailed views live in the sidebar pages.
"""

# ── Imports ────────────────────────────────────────────────────────────
from datetime import datetime

import streamlit as st

from dashboard import data
from dashboard.components import (
    account_summary_card,
    equity_curve_chart,
    live_fragment_decorator,
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


# ── Shared CSS + sidebar refresh button (applied to every page) ────────
sidebar_refresh()


# ── Hero header ────────────────────────────────────────────────────────
hero_left, hero_right = st.columns([5, 2])
with hero_left:
    st.title("Stock Bot")
    st.caption("Live paper-trading dashboard · core-satellite strategy")
with hero_right:
    # Top-right clock so it's clear which timezone the data is in.
    # Use `opacity` instead of hardcoded color so it adapts to dark mode.
    now = datetime.now()
    st.markdown(
        f"<div style='text-align:right;padding-top:1rem;font-size:0.875rem;opacity:0.6'>"
        f"As of <strong style='opacity:1.4'>{now.strftime('%a %b %d')}</strong>"
        f" · {now.strftime('%H:%M')} local"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Account summary ────────────────────────────────────────────────────
def _render_live_account_summary() -> None:
    """Render live account metrics in a refreshable fragment."""
    summary = data.compute_account_summary()
    account_summary_card(summary)
    live_refresh = summary.get("live_refresh") or {}
    if live_refresh.get("ok"):
        st.caption(f"Live Alpaca refresh · {live_refresh.get('refreshed_at', '')}")
    elif live_refresh.get("error"):
        st.caption(f"Live Alpaca refresh unavailable · using cached files · {live_refresh.get('error')}")


_live_decorator = live_fragment_decorator()
if _live_decorator is not None:
    @_live_decorator
    def _live_account_fragment() -> None:
        _render_live_account_summary()

    _live_account_fragment()
else:
    _render_live_account_summary()

st.divider()


# ── Paper vs shadow + workflow heartbeat ──────────────────────────────
st.markdown("##### Paper vs shadow")
compare_payload = data.load_paper_shadow_compare()
compare_summary = compare_payload.get("summary") or {}
compare_status = compare_summary.get("status")
if compare_status == "ok":
    alpaca_compare = compare_summary.get("alpaca") or {}
    shadow_compare = compare_summary.get("shadow") or {}
    spread = compare_summary.get("spread") or {}
    latest_dates_match = bool(compare_summary.get("latest_dates_match"))
    comp1, comp2, comp3, comp4 = st.columns(4)
    with comp1:
        st.metric(
            "Alpaca return",
            f"{float(alpaca_compare.get('return_pct_since_common_start') or 0):+.2f}%",
            delta=alpaca_compare.get("latest_date"),
            delta_color="off",
        )
    with comp2:
        st.metric(
            "Shadow return",
            f"{float(shadow_compare.get('return_pct_since_common_start') or 0):+.2f}%",
            delta=shadow_compare.get("latest_date"),
            delta_color="off",
        )
    with comp3:
        st.metric(
            "Alpaca minus shadow",
            f"{float(spread.get('alpaca_minus_shadow_return_pct') or 0):+.2f}%",
            delta=str(spread.get("leader", "unknown")),
            delta_color="normal",
        )
    with comp4:
        label = f"{int(compare_summary.get('aligned_days') or 0)} aligned days"
        status_chip(label, "ok" if latest_dates_match else "warn")
        if not latest_dates_match:
            st.caption("Latest dates differ")
else:
    st.info("Paper-vs-shadow comparison will appear after Alpaca and shadow equity files exist.")

workflow_df = data.load_workflow_heartbeats()
if not workflow_df.empty:
    hb_cols = st.columns(min(2, len(workflow_df)))
    for idx, row in workflow_df.head(2).iterrows():
        with hb_cols[idx % len(hb_cols)]:
            status = str(row.get("status") or row.get("conclusion") or "unknown").lower()
            event = str(row.get("event") or "missing")
            age = row.get("age_minutes")
            if status in {"success", "completed", "ok"}:
                chip_status = "ok"
            elif status in {"missing", "unknown"}:
                chip_status = "unknown"
            else:
                chip_status = "warn"
            age_text = "" if age is None else f" · {float(age):.0f}m old"
            status_chip(f"{row.get('label')}: {event}{age_text}", chip_status)

st.divider()


# ── Regime + health chips ──────────────────────────────────────────────
col_a, col_b = st.columns([2, 3])

with col_a:
    signal = data.load_current_signal() or {}
    regime = str(signal.get("current_regime", "unknown"))
    regime_indicator(regime)
    if signal.get("predicted_at"):
        st.caption(f"Signal generated · {signal.get('predicted_at')}")

with col_b:
    st.markdown("##### System status")
    h1, h2, h3 = st.columns(3)
    with h1:
        wf = data.load_walkforward_summary() or {}
        approval = (wf.get("live_config_approval") or {}).get("approved", False)
        status_chip(
            "Walkforward approved" if approval else "Approval blocked",
            "ok" if approval else "fail",
        )
    with h2:
        ready = bool(signal.get("paper_ready", False))
        gates = bool(signal.get("gates_all_pass", False))
        status_chip(
            "Paper-trading ready" if ready and gates else "Gates blocked",
            "ok" if ready and gates else "warn",
        )
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


# ── Equity curve ───────────────────────────────────────────────────────
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


# ── Data freshness check ───────────────────────────────────────────────
with st.expander("Data freshness · click to expand"):
    st.dataframe(data.file_status_table(), use_container_width=True, hide_index=True)

st.caption(
    "Dashboard refreshes the live Alpaca account tiles without reloading the whole page. "
    "Use the sidebar controls to change the interval or force a full reread."
)
