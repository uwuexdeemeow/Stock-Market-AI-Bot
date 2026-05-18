"""
dashboard/components.py — Reusable UI widgets

PLAIN ENGLISH: These are little Streamlit "widgets" — health-status
chips, account summary cards, equity charts.  Pages compose them
instead of duplicating the same layout code.
"""

# ── Imports ────────────────────────────────────────────────────────────
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Color palette (consistent across pages) ────────────────────────────
COLOR_OK = "#22c55e"      # green
COLOR_WARN = "#f59e0b"    # amber
COLOR_FAIL = "#ef4444"    # red
COLOR_NEUTRAL = "#94a3b8" # slate


def status_chip(label: str, status: str) -> None:
    """Render a colored status chip.

    status: 'ok' | 'warn' | 'fail' | 'unknown'
    """
    colors = {
        "ok": COLOR_OK,
        "warn": COLOR_WARN,
        "fail": COLOR_FAIL,
        "unknown": COLOR_NEUTRAL,
    }
    icons = {"ok": "✅", "warn": "⚠️", "fail": "❌", "unknown": "❔"}
    color = colors.get(status, COLOR_NEUTRAL)
    icon = icons.get(status, "❔")
    st.markdown(
        f"<span style='background-color:{color}33;color:{color};"
        f"padding:4px 10px;border-radius:12px;font-weight:600;"
        f"font-size:14px'>{icon} {label}</span>",
        unsafe_allow_html=True,
    )


def account_summary_card(summary: dict) -> None:
    """Render the top-of-page account summary (equity, P&L, drawdown).

    summary is the dict returned by data.compute_account_summary().
    Designed to fit in a 4-column row.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        equity = summary.get("equity", 0)
        change_pct = summary.get("change_pct_today", 0)
        st.metric(
            "Account Equity",
            f"${equity:,.2f}",
            delta=f"{change_pct:+.2f}%" if change_pct else None,
        )

    with col2:
        cash = summary.get("cash", 0)
        invested = summary.get("invested", 0)
        cash_pct = (cash / equity * 100) if equity > 0 else 0
        st.metric(
            "Cash",
            f"${cash:,.2f}",
            delta=f"{cash_pct:.1f}% of equity",
            delta_color="off",
        )

    with col3:
        dd_pct = summary.get("current_drawdown_pct", 0)
        peak = summary.get("peak_equity", 0)
        dd_color = "inverse" if dd_pct < -5 else "normal"
        st.metric(
            "Drawdown from peak",
            f"{dd_pct:.2f}%",
            delta=f"peak ${peak:,.2f}",
            delta_color="off",
        )

    with col4:
        n_positions = summary.get("position_count", 0)
        gen = summary.get("generated_at", "")
        # Show how stale the snapshot is
        if gen:
            try:
                gen_dt = datetime.fromisoformat(gen.replace("Z", ""))
                age_min = (datetime.now() - gen_dt).total_seconds() / 60
                age_label = f"{int(age_min)} min ago"
            except Exception:
                age_label = gen
        else:
            age_label = "no snapshot"
        st.metric("Positions", n_positions, delta=age_label, delta_color="off")


def equity_curve_chart(equity_df: pd.DataFrame, qqq_df: Optional[pd.DataFrame] = None) -> go.Figure:
    """Return a plotly equity-curve figure with optional QQQ benchmark overlay.

    equity_df: must have 'date' and 'equity' columns.
    qqq_df: optional, with 'date' and 'qqq_close' columns.  If provided, we
            normalize QQQ to the same starting equity for an apples-to-apples comparison.
    """
    fig = go.Figure()

    if not equity_df.empty and "equity" in equity_df.columns:
        fig.add_trace(go.Scatter(
            x=equity_df["date"],
            y=equity_df["equity"],
            name="Strategy",
            mode="lines",
            line=dict(color="#3b82f6", width=2.5),
        ))

        # Benchmark overlay (only if both have data)
        if qqq_df is not None and not qqq_df.empty and "qqq_close" in qqq_df.columns:
            # Normalize QQQ to start at the strategy's starting equity
            start_equity = float(equity_df["equity"].iloc[0])
            qqq_aligned = qqq_df.copy()
            qqq_start = float(qqq_aligned["qqq_close"].iloc[0])
            qqq_aligned["qqq_normalized"] = qqq_aligned["qqq_close"] / qqq_start * start_equity
            fig.add_trace(go.Scatter(
                x=qqq_aligned["date"],
                y=qqq_aligned["qqq_normalized"],
                name="QQQ (normalized)",
                mode="lines",
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
            ))

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Account value ($)",
        hovermode="x unified",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def drawdown_chart(equity_df: pd.DataFrame) -> go.Figure:
    """Return a plotly drawdown chart (underwater chart)."""
    fig = go.Figure()

    if not equity_df.empty and "equity" in equity_df.columns:
        df = equity_df.copy()
        df["peak"] = df["equity"].cummax()
        df["drawdown_pct"] = (df["equity"] / df["peak"] - 1) * 100

        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["drawdown_pct"],
            name="Drawdown",
            mode="lines",
            fill="tozeroy",
            line=dict(color="#ef4444", width=1.5),
            fillcolor="rgba(239,68,68,0.2)",
        ))

    fig.update_layout(
        title="Drawdown from Peak",
        xaxis_title="Date",
        yaxis_title="% from peak",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def positions_pie(weights: dict, title: str = "Position Weights") -> go.Figure:
    """Render a pie chart of ticker → weight."""
    if not weights:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no positions)", height=300)
        return fig

    fig = go.Figure(data=[
        go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.4,
            textposition="auto",
        )
    ])
    fig.update_layout(title=title, height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def regime_indicator(regime: str) -> None:
    """Render a big regime banner: risk_on / neutral / risk_off."""
    colors = {
        "risk_on": "#22c55e",
        "neutral": "#f59e0b",
        "risk_off": "#ef4444",
    }
    icons = {"risk_on": "🟢", "neutral": "🟡", "risk_off": "🔴"}
    color = colors.get(regime, COLOR_NEUTRAL)
    icon = icons.get(regime, "❔")
    label = regime.replace("_", " ").upper() if regime else "UNKNOWN"
    st.markdown(
        f"<div style='background:{color}22;border-left:5px solid {color};"
        f"padding:12px 16px;border-radius:8px;font-size:18px;font-weight:700'>"
        f"{icon} REGIME: {label}</div>",
        unsafe_allow_html=True,
    )


def freshness_badge(age_minutes: Optional[float]) -> None:
    """Show a small freshness indicator for the page top-right."""
    if age_minutes is None:
        status_chip("data missing", "fail")
    elif age_minutes < 30:
        status_chip(f"fresh ({age_minutes:.0f}m)", "ok")
    elif age_minutes < 24 * 60:
        status_chip(f"stale ({age_minutes:.0f}m)", "warn")
    else:
        hours = age_minutes / 60
        status_chip(f"old ({hours:.1f}h)", "fail")


def sidebar_refresh() -> None:
    """One-button sidebar shown on every page.

    Streamlit auto-injects the page navigation at the top of the
    sidebar, so we only add the refresh control below it.  Anything
    else (titles, captions, page lists) is redundant and removed.
    """
    with st.sidebar:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
