"""
Performance page — equity curve, drawdown, return distribution

PLAIN ENGLISH: "Is the bot actually making money?"  This page shows the
account's growth over time, drawdown from peak, and how it compares
to a buy-and-hold QQQ benchmark.
"""

import streamlit as st
import pandas as pd
import numpy as np

from dashboard import data
from dashboard.components import (
    account_summary_card,
    equity_curve_chart,
    drawdown_chart,
    sidebar_refresh,
)


st.set_page_config(page_title="Performance", page_icon="📈", layout="wide")
sidebar_refresh()
st.title("📈 Performance")

summary = data.compute_account_summary()
account_summary_card(summary)
st.divider()

# Pull both data sources
equity_df = data.load_alpaca_equity_history()

if equity_df.empty:
    st.warning("No equity history yet — the bot hasn't accumulated enough trading days.")
    st.stop()

# ── Date range selector ─────────────────────────────────────────────
st.markdown("### Equity history")
col1, col2 = st.columns(2)
with col1:
    days = st.selectbox("Window", [30, 60, 90, 180, 365, 9999],
                       index=2, format_func=lambda x: "All" if x == 9999 else f"{x} days")
with col2:
    show_qqq = st.checkbox("Overlay QQQ benchmark", value=True)

# Filter to window
cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
df_window = equity_df[equity_df["date"] >= cutoff] if days < 9999 else equity_df

if df_window.empty:
    st.info("No data in selected window.")
    st.stop()

# Equity curve
qqq_df = data.load_etf_data("QQQ", days=days) if show_qqq else None
st.plotly_chart(equity_curve_chart(df_window, qqq_df), use_container_width=True)

# Drawdown chart
st.plotly_chart(drawdown_chart(df_window), use_container_width=True)

# ── Performance metrics ─────────────────────────────────────────────
st.markdown("### Metrics")
m1, m2, m3, m4 = st.columns(4)

if len(df_window) >= 2:
    eq = df_window["equity"].astype(float).values
    rets = np.diff(np.log(eq))  # log returns

    total_return_pct = (eq[-1] / eq[0] - 1) * 100
    days_elapsed = (df_window["date"].iloc[-1] - df_window["date"].iloc[0]).days
    cagr = ((eq[-1] / eq[0]) ** (365 / max(1, days_elapsed)) - 1) * 100 if days_elapsed > 0 else 0
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    peak = pd.Series(eq).cummax().values
    max_dd = ((eq - peak) / peak * 100).min()

    with m1:
        st.metric("Total return", f"{total_return_pct:+.2f}%")
    with m2:
        st.metric("CAGR (annualized)", f"{cagr:+.2f}%")
    with m3:
        st.metric("Sharpe (daily)", f"{sharpe:.2f}")
    with m4:
        st.metric("Max drawdown", f"{max_dd:.2f}%")

# ── vs QQQ comparison ────────────────────────────────────────────────
if qqq_df is not None and not qqq_df.empty:
    st.markdown("### vs. QQQ")
    qqq_aligned = qqq_df.copy()
    qqq_aligned["qqq_return_pct"] = (qqq_aligned["qqq_close"] / qqq_aligned["qqq_close"].iloc[0] - 1) * 100
    qqq_total = qqq_aligned["qqq_return_pct"].iloc[-1] if not qqq_aligned.empty else 0
    alpha = total_return_pct - qqq_total
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Strategy return", f"{total_return_pct:+.2f}%")
    with c2:
        st.metric("QQQ return", f"{qqq_total:+.2f}%")
    with c3:
        st.metric("Alpha vs QQQ", f"{alpha:+.2f}%",
                  delta="beating" if alpha > 0 else "trailing")

st.divider()
st.caption("Tip: live metrics are noisy with <20 trading days. The walkforward (a longer-horizon backtest) is the real expected performance — see the Walkforward page.")
