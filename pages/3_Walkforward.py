"""
Walkforward page — year-by-year OOS results + approval status

PLAIN ENGLISH: "Is the strategy still valid?"  The walkforward is the
backtest that simulates years of out-of-sample trading.  This page
shows what it found, the approval verdict, and the trend over time.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from dashboard import data
from dashboard.components import status_chip, sidebar_refresh


st.set_page_config(page_title="Walkforward", page_icon="•", layout="wide")
sidebar_refresh()
st.title("Walkforward Validation")

wf_df = data.load_walkforward_results()
wf_summary = data.load_walkforward_summary() or {}
quant_audit = data.load_quant_performance_audit()
quant_payload = quant_audit.get("payload") or {}

if wf_df.empty:
    st.warning("No walkforward results found. Run `core_satellite_nested_walkforward.py` first.")
    st.stop()

# The periodic fold summary remains visible for continuity, but the independent
# daily audit is the truth reference for risk and after-cost comparison.
st.markdown("### Independent daily audit")
if not quant_payload:
    st.warning("Daily mark-to-market audit not built yet. Periodic headline results are provisional.")
else:
    audited = (quant_payload.get("selected_walkforward_audit") or {}).get("net") or {}
    audit_status = str(quant_payload.get("reference_audit_status") or "unknown")
    status_chip(
        f"Reference audit {audit_status}",
        "ok" if audit_status == "passed" else "warn" if audit_status == "blocked" else "fail",
    )
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Audited net CAGR", f"{float(audited.get('strategy_cagr_pct') or 0):.1f}%")
    with a2:
        st.metric("Net alpha vs QQQ", f"{float(audited.get('net_alpha_vs_qqq_pct') or 0):+.1f}%")
    with a3:
        st.metric("Daily information ratio", f"{float(audited.get('information_ratio_vs_qqq') or 0):.2f}")
    with a4:
        st.metric("Daily max drawdown", f"{float(audited.get('max_drawdown_pct') or 0):.1f}%")
    blockers = quant_payload.get("promotion_blockers") or []
    if blockers:
        st.caption("Promotion blocked: " + "; ".join(str(item) for item in blockers))
    recommendation = quant_payload.get("ranked_shadow_recommendation") or {}
    st.caption(
        f"Shadow recommendation: {recommendation.get('candidate', 'none')} · "
        f"{recommendation.get('reason', 'not available')}"
    )
    experiments = quant_audit.get("experiments")
    if isinstance(experiments, pd.DataFrame) and not experiments.empty:
        st.dataframe(experiments, hide_index=True, use_container_width=True)

st.divider()

# ── Top-line stats ──────────────────────────────────────────────────
st.markdown("### Aggregate stats")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Folds", wf_summary.get("fold_count", len(wf_df)))
with c2:
    st.metric("Compound return", f"{wf_summary.get('compound_oos_return_pct', 0):.0f}%")
with c3:
    st.metric("CAGR", f"{wf_summary.get('mean_oos_cagr_pct', 0):.1f}%")
with c4:
    st.metric("Mean Sharpe", f"{wf_summary.get('mean_oos_sharpe', 0):.2f}")

c5, c6, c7, c8 = st.columns(4)
with c5:
    st.metric("Mean alpha vs QQQ", f"{wf_summary.get('mean_oos_alpha_vs_qqq_pct', 0):+.1f}%")
with c6:
    hit_rate = wf_summary.get("oos_positive_alpha_hit_rate", 0)
    st.metric("Beat QQQ rate", f"{hit_rate*100:.0f}%")
with c7:
    st.metric("Worst drawdown", f"{wf_summary.get('worst_oos_max_drawdown_pct', 0):.1f}%")
with c8:
    st.metric("Top config freq", f"{wf_summary.get('best_config_frequency', 0)*100:.0f}%")

# ── Approval status ─────────────────────────────────────────────────
st.markdown("### Approval gate")
approval = wf_summary.get("live_config_approval", {}) or {}
approved = bool(approval.get("approved", False))
status_chip("APPROVED for live" if approved else "NOT approved",
            "ok" if approved else "fail")
if approval.get("reasons"):
    st.error("Rejection reasons: " + "; ".join(approval.get("reasons", [])))
if "approved_config_family" in approval:
    st.caption(f"Config family: `{approval['approved_config_family']}`")

st.divider()

# ── Per-fold table ──────────────────────────────────────────────────
st.markdown("### Year-by-year OOS results")
show_cols = [c for c in ["fold_year", "selected_config", "oos_return_pct",
                          "oos_sharpe", "oos_max_drawdown_pct",
                          "oos_alpha_vs_qqq_pct", "oos_alpha_vs_spy_pct",
                          "oos_turnover_pct"]
             if c in wf_df.columns]
display_df = wf_df[show_cols].copy() if show_cols else wf_df.copy()
display_df = display_df.sort_values("fold_year", ascending=False) if "fold_year" in display_df.columns else display_df
st.dataframe(display_df, hide_index=True, use_container_width=True)

# ── Per-fold return bar chart ────────────────────────────────────────
if "oos_return_pct" in wf_df.columns and "fold_year" in wf_df.columns:
    st.markdown("### Per-fold OOS return")
    chart_df = wf_df.copy()
    chart_df["color"] = chart_df["oos_return_pct"].apply(lambda x: "positive" if x >= 0 else "negative")
    fig = px.bar(
        chart_df,
        x="fold_year",
        y="oos_return_pct",
        color="color",
        color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"},
        labels={"oos_return_pct": "OOS return (%)", "fold_year": "Year"},
        title="OOS return per fold (year by year)",
    )
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── Alpha vs QQQ chart ────────────────────────────────────────────────
if "oos_alpha_vs_qqq_pct" in wf_df.columns and "fold_year" in wf_df.columns:
    st.markdown("### Alpha vs QQQ per fold")
    chart_df = wf_df.copy()
    chart_df["color"] = chart_df["oos_alpha_vs_qqq_pct"].apply(lambda x: "beats" if x >= 0 else "trails")
    fig = px.bar(
        chart_df,
        x="fold_year",
        y="oos_alpha_vs_qqq_pct",
        color="color",
        color_discrete_map={"beats": "#22c55e", "trails": "#94a3b8"},
        labels={"oos_alpha_vs_qqq_pct": "Alpha vs QQQ (%)", "fold_year": "Year"},
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

# ── Config evolution ──────────────────────────────────────────────────
if "selected_config" in wf_df.columns:
    st.markdown("### Config selection frequency")
    cfg_counts = wf_df["selected_config"].value_counts().reset_index()
    cfg_counts.columns = ["Config", "Folds selected"]
    cfg_counts["Frequency"] = (cfg_counts["Folds selected"] / len(wf_df) * 100).round(1).astype(str) + "%"
    st.dataframe(cfg_counts, hide_index=True, use_container_width=True)
