"""
Signal & Orders page — today's picks, order log, drift from targets

PLAIN ENGLISH: "What did the bot decide to do today?"  Shows the current
signal (which stocks at what weights), the order plan that came out of
it, and the recent order log with fill statuses.
"""

import json
import streamlit as st
import pandas as pd

from dashboard import data
from dashboard.components import (
    positions_pie,
    regime_indicator,
    status_chip,
    sidebar_refresh,
)


st.set_page_config(page_title="Signal & Orders", page_icon="•", layout="wide")
sidebar_refresh()
st.title("Signal & Orders")

signal = data.load_current_signal()
if signal is None:
    st.warning("No signal CSV found. Run `core_satellite_alpha.py` to generate one.")
    st.stop()

# ── Top row: regime + readiness ─────────────────────────────────────
col_a, col_b = st.columns([2, 3])
with col_a:
    regime_indicator(str(signal.get("current_regime", "unknown")))
    st.caption(f"Predicted at: {signal.get('predicted_at', 'n/a')}")
with col_b:
    st.markdown("### Signal gates")
    g1, g2, g3 = st.columns(3)
    with g1:
        ready = bool(signal.get("paper_ready", False))
        status_chip("paper_ready" if ready else "NOT paper_ready",
                    "ok" if ready else "fail")
    with g2:
        gates = bool(signal.get("gates_all_pass", False))
        status_chip("gates_all_pass" if gates else "gates blocked",
                    "ok" if gates else "fail")
    with g3:
        approved = bool(signal.get("walkforward_approval_pass", False))
        status_chip("WF approved" if approved else "WF not approved",
                    "ok" if approved else "fail")

st.divider()

# ── Target allocation ─────────────────────────────────────────────
st.markdown("### Target allocation")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("**Core (ETFs)**")
    core_rows = [
        ("SPY", float(signal.get("target_spy_weight", 0) or 0)),
        ("QQQ", float(signal.get("target_qqq_weight", 0) or 0)),
        ("TQQQ", float(signal.get("target_tqqq_weight", 0) or 0)),
        ("Cash", float(signal.get("target_cash_weight", 0) or 0)),
    ]
    core_df = pd.DataFrame(core_rows, columns=["Ticker", "Weight"])
    core_df["Weight %"] = (core_df["Weight"] * 100).round(2).astype(str) + "%"
    st.dataframe(core_df[["Ticker", "Weight %"]], hide_index=True, use_container_width=True)
    st.caption(
        f"core_gross: {float(signal.get('core_gross', 0)):.2f} · "
        f"overlay_gross: {float(signal.get('overlay_gross', 0)):.2f} · "
        f"max_single: {float(signal.get('max_single_name_weight', 0))*100:.1f}%"
    )

with col2:
    overlay_weights_str = signal.get("overlay_weights_json", "{}")
    try:
        overlay_weights = json.loads(overlay_weights_str) if isinstance(overlay_weights_str, str) else {}
    except Exception:
        overlay_weights = {}
    st.markdown("**Overlay (stock picks)**")
    if overlay_weights:
        st.plotly_chart(positions_pie(overlay_weights, "Overlay weights"), use_container_width=True)
    else:
        st.info("No overlay positions in today's signal.")

# ── Sticky holdings (carry-forward from prior session) ─────────────
st.markdown("### Sticky holdings (carried from last session)")
sticky_source = str(signal.get("sticky_holdings_source", "none"))
sticky_used = bool(signal.get("sticky_holdings_used", False))
status_chip(f"source: {sticky_source}", "ok" if sticky_used else "warn")
sticky_tickers = str(signal.get("sticky_held_tickers", "") or "")
if sticky_tickers:
    st.markdown(f"**Held tickers:** `{sticky_tickers}`")
else:
    st.caption("No sticky holdings carried.")

st.divider()

# ── Order plan ─────────────────────────────────────────────────────
st.markdown("### Order plan for today")
orders_df = data.load_order_plan()
if orders_df.empty:
    st.info("No order plan file yet — submit hasn't run today.")
else:
    # Highlight BUYs and SELLs differently
    display = orders_df.copy()
    # If standard columns present, format them
    cols = ["ticker", "action", "current_qty", "target_qty", "delta_qty", "price", "order_value"]
    cols = [c for c in cols if c in display.columns]
    st.dataframe(display[cols] if cols else display, hide_index=True, use_container_width=True)

st.divider()

# ── Recent Alpaca order log ─────────────────────────────────────
st.markdown("### Recent Alpaca orders (last 50)")
orders_log = data.load_alpaca_orders()
if orders_log.empty:
    st.info("No Alpaca order log yet. After the first --submit run, this will populate.")
else:
    interesting_cols = [c for c in ["submitted_at", "ticker", "side", "qty", "status", "fill_price", "filled_qty"]
                        if c in orders_log.columns]
    if not interesting_cols:
        interesting_cols = list(orders_log.columns)[:8]
    st.dataframe(orders_log[interesting_cols].head(50),
                 hide_index=True, use_container_width=True)
