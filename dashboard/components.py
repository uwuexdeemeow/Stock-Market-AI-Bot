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

from safe_io import popen_utf8


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
    # Minimal indicators — colored dots instead of emoji
    color = colors.get(status, COLOR_NEUTRAL)
    st.markdown(
        f"<span style='background-color:{color}1A;color:{color};"
        f"padding:4px 12px;border-radius:14px;font-weight:600;"
        f"font-size:13px;letter-spacing:0.2px;"
        f"border:1px solid {color}55'>"
        f"<span style='display:inline-block;width:8px;height:8px;"
        f"border-radius:50%;background:{color};margin-right:6px;"
        f"vertical-align:middle'></span>{label}</span>",
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
    """Render a regime banner: risk_on / neutral / risk_off."""
    colors = {
        "risk_on": "#22c55e",
        "neutral": "#f59e0b",
        "risk_off": "#ef4444",
    }
    color = colors.get(regime, COLOR_NEUTRAL)
    label = regime.replace("_", " ").upper() if regime else "UNKNOWN"
    st.markdown(
        f"<div style='background:{color}14;border-left:4px solid {color};"
        f"padding:14px 18px;border-radius:6px;font-size:15px;font-weight:600;"
        f"letter-spacing:0.3px;color:{color}'>"
        f"<span style='font-size:11px;font-weight:500;color:{COLOR_NEUTRAL};"
        f"text-transform:uppercase;letter-spacing:0.8px'>Regime</span><br/>"
        f"{label}</div>",
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


def apply_page_style() -> None:
    """Inject the shared CSS used on every page.

    Theme-aware: uses Streamlit's CSS variables (--background-color,
    --text-color, --secondary-background-color) so the look adapts to
    both light and dark mode without hardcoding colors.  Idempotent —
    safe to call from every page header.
    """
    st.markdown(
        """
        <style>
          /* ── Typography — system font stack, crisp on every OS ───── */
          html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Inter",
              "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
          }

          /* ── Layout — tighter container, more room to breathe ───── */
          .block-container {
            padding-top: 2rem !important;
            padding-bottom: 4rem !important;
            max-width: 1400px;
          }

          /* ── Sidebar — adapts to theme via Streamlit's own vars ──── */
          /* Use --secondary-background-color so it's slightly off the
             main background in BOTH themes (light gray on light theme,
             slightly lifted dark gray on dark theme).                 */
          [data-testid="stSidebar"] > div:first-child {
            background-color: var(--secondary-background-color);
          }

          /* ── Headings — weight + tracking, color from theme ─────── */
          h1 {
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            font-size: 2rem !important;
          }
          h2, h3 {
            font-weight: 600 !important;
            letter-spacing: -0.015em !important;
          }
          h4, h5 {
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            margin-bottom: 0.5rem !important;
          }

          /* ── Metric cards — bigger numbers, subtle labels ───────── */
          [data-testid="stMetricValue"] {
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            font-size: 1.75rem !important;
          }
          [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.05em;
          }
          [data-testid="stMetricDelta"] {
            font-size: 0.8rem !important;
            font-weight: 500 !important;
          }

          /* ── Buttons — flatter, with hover lift ─────────────────── */
          .stButton button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.15s ease !important;
          }
          .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
          }
          .stButton button[kind="primary"] {
            background-color: #3b82f6 !important;
            border-color: #3b82f6 !important;
            color: white !important;
          }

          /* ── Dividers — softer, theme-adaptive via currentColor ── */
          hr {
            margin: 1.75rem 0 !important;
            opacity: 0.2;
          }

          /* ── Expanders ─────────────────────────────────────────── */
          .streamlit-expanderHeader {
            font-weight: 500 !important;
            border-radius: 8px !important;
          }

          /* ── Tabs — modern blue underline ──────────────────────── */
          .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid;
            border-color: rgba(128, 128, 128, 0.2);
          }
          .stTabs [data-baseweb="tab"] {
            padding: 8px 16px !important;
            font-weight: 500 !important;
            border-radius: 8px 8px 0 0 !important;
          }
          .stTabs [aria-selected="true"] {
            color: #3b82f6 !important;
            border-bottom: 2px solid #3b82f6 !important;
          }

          /* ── Code blocks — use Streamlit's own bg vars ─────────── */
          pre {
            border-radius: 8px !important;
            border: 1px solid rgba(128, 128, 128, 0.2) !important;
          }
          pre code { font-size: 0.85rem !important; }

          /* ── Dataframes ───────────────────────────────────────── */
          [data-testid="stDataFrame"] {
            border-radius: 8px;
            border: 1px solid rgba(128, 128, 128, 0.2);
          }

          /* ── Alerts ───────────────────────────────────────────── */
          .stAlert {
            border-radius: 8px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_refresh() -> None:
    """One-button sidebar shown on every page.

    Also applies the shared page CSS so each page picks up the styling
    without each one repeating the markdown block.
    """
    apply_page_style()
    with st.sidebar:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


# ── Subprocess runner with live streaming + logging ────────────────────
import subprocess
import sys as _sys
from pathlib import Path as _Path
from datetime import datetime as _datetime


def run_script(cmd: list[str], label: str, *,
               cwd: str | _Path | None = None,
               max_lines_shown: int = 200) -> dict:
    """Run a subprocess and stream its output into the Streamlit UI.

    PLAIN ENGLISH: Click a button → this function fires the command,
    captures its output line by line, and shows it live in the page.
    Also tees the full output to logs/dashboard_*.log for later review.

    Returns a dict with:
        success    : bool
        returncode : int
        log_file   : Path
        line_count : int

    The page calls this synchronously — Streamlit blocks until the
    command exits.  For a script that takes hours (nested walkforward)
    use run_script_async() instead (TODO).
    """
    # Where to save the live log file
    log_dir = _Path(cwd or ".") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label.lower())
    log_file = log_dir / f"dashboard_{stamp}_{safe_label}.log"

    # Header box showing the command being executed
    st.info(f"▶ **{label}**\n```\n{' '.join(cmd)}\n```")
    # Live output area
    output_box = st.empty()
    lines: list[str] = []

    # Run the subprocess with line-buffered output
    try:
        proc = popen_utf8(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            cwd=str(cwd) if cwd else None,
            env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        )
    except FileNotFoundError as e:
        st.error(f"✗ Command not found: {e}")
        return {"success": False, "returncode": -1, "log_file": log_file, "line_count": 0}

    with open(log_file, "w", encoding="utf-8") as f:
        if proc.stdout is None:
            proc.wait()
        else:
            for raw in iter(proc.stdout.readline, ""):
                line = raw.rstrip("\n")
                lines.append(line)
                f.write(raw)
                f.flush()
                # Throttle UI updates — only redraw every 5 lines (or last lines)
                # to avoid Streamlit slowdown on bursty output.
                if len(lines) % 5 == 0 or len(line) > 80:
                    output_box.code("\n".join(lines[-max_lines_shown:]), language="text")
            proc.stdout.close()
        rc = proc.wait()

    # Final redraw + verdict
    output_box.code("\n".join(lines[-max_lines_shown:]) or "(no output)", language="text")
    if rc == 0:
        st.success(f"✓ **{label}** finished successfully  ·  {len(lines)} lines  ·  log: `{log_file.name}`")
    else:
        st.error(f"✗ **{label}** failed with exit code {rc}  ·  log: `{log_file.name}`")

    return {
        "success": rc == 0,
        "returncode": rc,
        "log_file": log_file,
        "line_count": len(lines),
    }


def script_button(label: str, cmd: list[str], *,
                  key: str | None = None,
                  help_text: str | None = None,
                  expected_runtime: str = "fast",
                  destructive: bool = False) -> None:
    """Render a button that runs a script when clicked.

    expected_runtime: free-text label ("fast", "1-3 min", "hours") —
        shown as a small caption so the user knows what to expect.
    destructive: if True, the button uses a warning style.  Use for
        submit/order operations that send real orders.
    """
    btn_label = label
    if expected_runtime and expected_runtime != "fast":
        btn_label = f"{label}  ·  ⏱ {expected_runtime}"
    if destructive:
        btn_label = f"⚠️  {btn_label}"

    if st.button(btn_label, key=key, help=help_text, use_container_width=True):
        run_script(cmd, label, cwd=_Path(__file__).resolve().parent.parent)
        # Invalidate cached data after any script run — chances are some
        # signal/log file just changed and the dashboard should reflect it.
        st.cache_data.clear()
