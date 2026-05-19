"""
Features page — which factors the model is currently using

PLAIN ENGLISH: The strategy combines several "factors" (momentum,
volume, etc.) to score stocks.  This page shows which ones are
currently active, how they cluster, and whether any are decaying.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard import data
from dashboard.components import status_chip, sidebar_refresh


st.set_page_config(page_title="Features", page_icon="•", layout="wide")
sidebar_refresh()
st.title("Feature Health")

profile = data.load_feature_health()
quality = data.load_feature_quality_summary()

if profile is None and quality.empty:
    st.warning("No feature health files found. Run `feature_quality_diagnostic.py --top 48`.")
    st.stop()

# ── Top stats ───────────────────────────────────────────────────────
if profile:
    summary = profile.get("summary", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Raw features", summary.get("raw_feature_count", "?"))
    with c2:
        st.metric("Active clusters", f"{summary.get('active_cluster_count', '?')}/{summary.get('cluster_count', '?')}")
    with c3:
        st.metric("Max cluster weight", f"{summary.get('max_cluster_weight', 0):.1%}")
    with c4:
        gate = bool(summary.get("feature_health_gate_pass", False))
        status_chip("Gate pass" if gate else "Gate FAIL", "ok" if gate else "fail")

    st.divider()

    # Quarantined / watchlist
    quarantined = summary.get("quarantined_features", []) or []
    watchlist = summary.get("watchlist_features", []) or []
    col_q, col_w = st.columns(2)
    with col_q:
        st.markdown("**Quarantined features** (decay too severe — excluded from scoring)")
        if quarantined:
            st.error("\n".join(f"• {f}" for f in quarantined))
        else:
            st.success("None — all features active.")
    with col_w:
        st.markdown("**Watchlist features** (decay warning — still in use)")
        if watchlist:
            st.warning("\n".join(f"• {f}" for f in watchlist))
        else:
            st.success("None on watch.")

    st.divider()

    # Cluster breakdown
    clusters = profile.get("clusters", []) or []
    if clusters:
        st.markdown("### Cluster breakdown")
        cluster_rows = []
        for c in clusters:
            cluster_rows.append({
                "Cluster": c.get("cluster_id"),
                "State": c.get("health_state"),
                "Effective weight": f"{c.get('effective_weight', 0):.1%}",
                "Members": ", ".join(c.get("features", [])),
            })
        st.dataframe(pd.DataFrame(cluster_rows), hide_index=True, use_container_width=True)

# ── Quality summary table ─────────────────────────────────────────
if not quality.empty:
    st.divider()
    st.markdown("### Feature quality summary")

    # Optional filter by grade
    if "grade" in quality.columns:
        grades = sorted(quality["grade"].dropna().unique().tolist())
        selected_grades = st.multiselect("Grades to show", grades, default=grades)
        filtered = quality[quality["grade"].isin(selected_grades)] if selected_grades else quality
    else:
        filtered = quality

    st.dataframe(filtered, hide_index=True, use_container_width=True)

    # IC distribution chart
    if "ic" in filtered.columns:
        st.markdown("### Information Coefficient (IC) distribution")
        fig = px.histogram(filtered, x="ic", nbins=20, title="IC distribution",
                          labels={"ic": "Information Coefficient"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
