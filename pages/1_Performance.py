"""
Performance — equity, drawdown, returns, benchmarks

Broker-app-style filters: preset periods (1D…ALL), custom date range,
multi-benchmark comparison, rolling metrics, return distribution.
"""

from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard import data
from dashboard.components import (
    account_summary_card,
    COLOR_FAIL,
    COLOR_WARN,
    sidebar_refresh,
)


st.set_page_config(page_title="Performance", page_icon="•", layout="wide")
sidebar_refresh()
st.title("Performance")

summary = data.compute_account_summary()
account_summary_card(summary)
st.divider()

equity_df = data.load_alpaca_equity_history()
if equity_df.empty:
    st.warning("No equity history yet.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Period filter bar — broker-app style preset buttons
# ─────────────────────────────────────────────────────────────────────
PERIOD_DAYS = {
    "1D": 1, "1W": 7, "1M": 30, "3M": 90,
    "6M": 180, "YTD": "ytd", "1Y": 365, "ALL": 99999,
}

if "perf_period" not in st.session_state:
    st.session_state.perf_period = "3M"

cols = st.columns([1] * len(PERIOD_DAYS) + [2, 2])

for i, (label, _days) in enumerate(PERIOD_DAYS.items()):
    with cols[i]:
        is_active = st.session_state.perf_period == label
        if st.button(label,
                     type="primary" if is_active else "secondary",
                     use_container_width=True,
                     key=f"period_{label}"):
            st.session_state.perf_period = label
            st.rerun()

selected = st.session_state.perf_period
if selected == "YTD":
    cutoff = datetime(datetime.now().year, 1, 1)
elif selected == "ALL":
    cutoff = pd.Timestamp(equity_df["date"].min()).to_pydatetime()
else:
    cutoff = datetime.now() - timedelta(days=PERIOD_DAYS[selected])

with cols[-2]:
    start_date = st.date_input("From", value=cutoff.date(), key="perf_start")
with cols[-1]:
    end_date = st.date_input("To", value=datetime.now().date(), key="perf_end")

# Display options
bc1, bc2, bc3 = st.columns([2, 2, 2])
with bc1:
    benchmarks = st.multiselect(
        "Compare against",
        ["QQQ", "SPY", "TQQQ"],
        default=["QQQ"],
        help="Normalized to same starting equity",
    )
with bc2:
    chart_mode = st.radio("Chart mode", ["Equity", "Cumulative return %"],
                           horizontal=True, key="perf_chart_mode")
with bc3:
    show_drawdown = st.checkbox("Show drawdown panel", value=True, key="perf_show_dd")

# Filter to window
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
df_window = equity_df[(equity_df["date"] >= start_ts) & (equity_df["date"] < end_ts)].copy()

if df_window.empty:
    st.info("No data in selected window.")
    st.stop()

# Pull benchmark data
benchmark_data = {}
period_days_value = PERIOD_DAYS.get(selected, 365)
if isinstance(period_days_value, int):
    days_fetch = max(period_days_value, 30)
else:
    days_fetch = max((datetime.now() - cutoff).days, 30)

for bench in benchmarks:
    bdf = data.load_etf_data(bench, days=days_fetch + 30)
    if not bdf.empty:
        bdf["date"] = pd.to_datetime(bdf["date"])
        bdf = bdf[(bdf["date"] >= start_ts) & (bdf["date"] < end_ts)]
        if not bdf.empty:
            benchmark_data[bench] = bdf

# ─────────────────────────────────────────────────────────────────────
# Metrics — two rows, broker-style
# ─────────────────────────────────────────────────────────────────────
def _compute_metrics(equity_series: np.ndarray, dates: pd.Series) -> dict:
    if len(equity_series) < 2:
        return {}
    rets = np.diff(np.log(equity_series))
    total_return_pct = (equity_series[-1] / equity_series[0] - 1) * 100
    days_elapsed = max(1, (dates.iloc[-1] - dates.iloc[0]).days)
    cagr = ((equity_series[-1] / equity_series[0]) ** (365 / days_elapsed) - 1) * 100
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    sortino = (
        rets.mean() / rets[rets < 0].std() * np.sqrt(252)
        if (rets < 0).any() and rets[rets < 0].std() > 0 else 0
    )
    peak = pd.Series(equity_series).cummax().values
    drawdowns = (equity_series - peak) / peak
    max_dd = drawdowns.min() * 100
    win_rate = (rets > 0).mean() * 100 if len(rets) > 0 else 0
    avg_win = rets[rets > 0].mean() * 100 if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() * 100 if (rets < 0).any() else 0
    profit_factor = (
        rets[rets > 0].sum() / -rets[rets < 0].sum()
        if (rets < 0).any() and rets[rets < 0].sum() != 0 else 0
    )
    volatility_annual = rets.std() * np.sqrt(252) * 100
    return {
        "total_return": total_return_pct,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "volatility": volatility_annual,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "n_periods": len(rets),
    }


eq_array = df_window["equity"].astype(float).values
metrics = _compute_metrics(eq_array, df_window["date"])

mr1, mr2, mr3, mr4, mr5 = st.columns(5)
with mr1:
    st.metric("Total return", f"{metrics.get('total_return', 0):+.2f}%")
with mr2:
    st.metric("CAGR", f"{metrics.get('cagr', 0):+.2f}%")
with mr3:
    st.metric("Sharpe", f"{metrics.get('sharpe', 0):.2f}")
with mr4:
    st.metric("Sortino", f"{metrics.get('sortino', 0):.2f}")
with mr5:
    st.metric("Volatility (ann.)", f"{metrics.get('volatility', 0):.2f}%")

mr6, mr7, mr8, mr9, mr10 = st.columns(5)
with mr6:
    st.metric("Max drawdown", f"{metrics.get('max_dd', 0):.2f}%")
with mr7:
    st.metric("Win rate", f"{metrics.get('win_rate', 0):.1f}%")
with mr8:
    st.metric("Avg win", f"{metrics.get('avg_win', 0):+.2f}%")
with mr9:
    st.metric("Avg loss", f"{metrics.get('avg_loss', 0):+.2f}%")
with mr10:
    pf = metrics.get('profit_factor', 0)
    st.metric("Profit factor", f"{pf:.2f}" if pf else "—")

st.divider()


# ─────────────────────────────────────────────────────────────────────
# Main chart — equity OR cumulative return %, optional drawdown panel
# ─────────────────────────────────────────────────────────────────────
def build_chart(equity_df: pd.DataFrame,
                bench_data: dict,
                mode: str,
                with_drawdown: bool) -> go.Figure:
    if with_drawdown:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("", "Drawdown from peak"),
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    start_equity = float(equity_df["equity"].iloc[0])

    if mode == "Equity":
        fig.add_trace(go.Scatter(
            x=equity_df["date"], y=equity_df["equity"],
            name="Strategy", mode="lines",
            line=dict(color="#3b82f6", width=2.5),
            hovertemplate="$%{y:,.2f}<extra>Strategy</extra>",
        ), row=1, col=1)
        for name, bdf in bench_data.items():
            if bdf.empty: continue
            col = next((c for c in bdf.columns if c.endswith("_close")), None)
            if col is None: continue
            normalized = bdf[col] / float(bdf[col].iloc[0]) * start_equity
            fig.add_trace(go.Scatter(
                x=bdf["date"], y=normalized,
                name=name, mode="lines",
                line=dict(width=1.5, dash="dash"),
                hovertemplate=f"$%{{y:,.2f}}<extra>{name}</extra>",
            ), row=1, col=1)
        fig.update_yaxes(title_text="Account value ($)", row=1, col=1)
    else:
        strat_return = (equity_df["equity"] / start_equity - 1) * 100
        fig.add_trace(go.Scatter(
            x=equity_df["date"], y=strat_return,
            name="Strategy", mode="lines",
            line=dict(color="#3b82f6", width=2.5),
            hovertemplate="%{y:+.2f}%<extra>Strategy</extra>",
        ), row=1, col=1)
        for name, bdf in bench_data.items():
            if bdf.empty: continue
            col = next((c for c in bdf.columns if c.endswith("_close")), None)
            if col is None: continue
            bench_ret = (bdf[col] / float(bdf[col].iloc[0]) - 1) * 100
            fig.add_trace(go.Scatter(
                x=bdf["date"], y=bench_ret,
                name=name, mode="lines",
                line=dict(width=1.5, dash="dash"),
                hovertemplate=f"%{{y:+.2f}}%<extra>{name}</extra>",
            ), row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative return (%)", row=1, col=1)

    if with_drawdown:
        peak = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] / peak - 1) * 100
        fig.add_trace(go.Scatter(
            x=equity_df["date"], y=drawdown,
            name="Drawdown", mode="lines",
            fill="tozeroy",
            line=dict(color="#ef4444", width=1.5),
            fillcolor="rgba(239,68,68,0.15)",
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="DD (%)", row=2, col=1)

    fig.update_layout(
        hovermode="x unified",
        height=560 if with_drawdown else 420,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


st.plotly_chart(
    build_chart(df_window, benchmark_data, chart_mode, show_drawdown),
    use_container_width=True,
)


# ─────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────
if benchmark_data:
    st.markdown("##### Strategy vs benchmarks")
    rows = []
    strat_total = metrics.get("total_return", 0)
    rows.append({
        "Asset": "Strategy",
        "Return %": f"{strat_total:+.2f}%",
        "Alpha vs strategy": "—",
    })
    for name, bdf in benchmark_data.items():
        if bdf.empty: continue
        col = next((c for c in bdf.columns if c.endswith("_close")), None)
        if col is None: continue
        bench_total = (bdf[col].iloc[-1] / bdf[col].iloc[0] - 1) * 100
        rows.append({
            "Asset": name,
            "Return %": f"{bench_total:+.2f}%",
            "Alpha vs strategy": f"{strat_total - bench_total:+.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Execution quality — recent fill slippage and reversals
# ─────────────────────────────────────────────────────────────────────
def _fmt_bps(value) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):+.1f} bps"
    except Exception:
        return "—"


def _fmt_count(value, total: int) -> str:
    try:
        return f"{int(value)}/{int(total)}"
    except Exception:
        return f"0/{int(total)}"


def _execution_quality_chart(exec_df: pd.DataFrame) -> go.Figure:
    chart_df = exec_df.tail(20).iloc[::-1].copy()
    chart_df["label"] = chart_df.apply(
        lambda r: f"{r.get('symbol', '')} {str(r.get('side', ''))[:1].upper()}",
        axis=1,
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=chart_df["label"],
        y=pd.to_numeric(chart_df["slippage_bps"], errors="coerce"),
        name="Slippage vs VWAP",
        marker_color=COLOR_WARN,
        hovertemplate="%{x}<br>%{y:+.1f} bps<extra>Slippage</extra>",
    ))
    fig.add_trace(go.Bar(
        x=chart_df["label"],
        y=pd.to_numeric(chart_df["adverse_15m_bps"], errors="coerce"),
        name="15m reversal",
        marker_color=COLOR_FAIL,
        hovertemplate="%{x}<br>%{y:+.1f} bps<extra>15m reversal</extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        barmode="group",
        hovermode="x unified",
        height=320,
        margin=dict(l=40, r=10, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Basis points",
        xaxis_title="Recent fills",
    )
    return fig


execution_report = data.load_slippage_reversal_report()
if execution_report:
    st.divider()
    st.markdown("##### Execution quality")
    if execution_report.get("source") != "alpaca_api":
        st.caption("Showing basic fill data from `alpaca_paper_log.csv`; run `alpaca_paper_trading.py --slippage-report` for reversal metrics.")
    execution_summary = execution_report.get("summary") or {}
    execution_rows = execution_report.get("orders") or []
    execution_total = int(execution_summary.get("orders_analyzed") or len(execution_rows) or 0)

    ex1, ex2, ex3, ex4, ex5 = st.columns(5)
    with ex1:
        st.metric(
            "Avg slippage",
            _fmt_bps(execution_summary.get("avg_slippage_bps")),
            delta=f"bad {_fmt_count(execution_summary.get('slippage_bad_count', 0), execution_total)}",
            delta_color="inverse",
        )
    with ex2:
        st.metric("Median slippage", _fmt_bps(execution_summary.get("median_slippage_bps")))
    with ex3:
        st.metric(
            "5m reversals",
            _fmt_count(execution_summary.get("adverse_5m_count", 0), execution_total),
            delta_color="inverse",
        )
    with ex4:
        st.metric(
            "15m reversals",
            _fmt_count(execution_summary.get("adverse_15m_count", 0), execution_total),
            delta_color="inverse",
        )
    with ex5:
        st.metric(
            "Worst 60m avg",
            _fmt_bps(execution_summary.get("avg_worst_adverse_60m_bps")),
            delta=_fmt_bps(execution_summary.get("max_worst_adverse_60m_bps")),
            delta_color="inverse",
        )

    exec_df = pd.DataFrame(execution_rows)
    if not exec_df.empty:
        st.plotly_chart(_execution_quality_chart(exec_df), use_container_width=True)

        symbol_df = pd.DataFrame(execution_report.get("by_symbol") or [])
        if not symbol_df.empty:
            symbol_table = symbol_df.head(8).copy()
            symbol_table["Symbol"] = symbol_table["symbol"]
            symbol_table["Orders"] = symbol_table["orders"]
            symbol_table["Avg slip"] = pd.to_numeric(symbol_table["avg_slippage_bps"], errors="coerce").round(1)
            symbol_table["Bad slip %"] = (pd.to_numeric(symbol_table["bad_slippage_rate"], errors="coerce") * 100).round(0)
            symbol_table["15m rev %"] = (pd.to_numeric(symbol_table["adverse_15m_rate"], errors="coerce") * 100).round(0)
            symbol_table["Risk score"] = pd.to_numeric(symbol_table["execution_risk_score"], errors="coerce").round(1)
            st.dataframe(
                symbol_table[["Symbol", "Orders", "Avg slip", "Bad slip %", "15m rev %", "Risk score"]],
                hide_index=True,
                use_container_width=True,
            )

        table = exec_df.copy()
        table["filled_at"] = pd.to_datetime(table["filled_at"], errors="coerce", utc=True)
        table["Time"] = table["filled_at"].dt.strftime("%m-%d %H:%M")
        table["Symbol"] = table["symbol"]
        table["Side"] = table["side"].astype(str).str.upper()
        table["Type"] = table["order_type"]
        table["Qty"] = table["filled_qty"]
        table["Fill"] = pd.to_numeric(table["fill_price"], errors="coerce").round(4)
        table["Slip bps"] = pd.to_numeric(table["slippage_bps"], errors="coerce").round(1)
        table["Rev 5m"] = pd.to_numeric(table["adverse_5m_bps"], errors="coerce").round(1)
        table["Rev 15m"] = pd.to_numeric(table["adverse_15m_bps"], errors="coerce").round(1)
        table["Rev 60m"] = pd.to_numeric(table["adverse_60m_bps"], errors="coerce").round(1)
        st.dataframe(
            table[["Time", "Symbol", "Side", "Type", "Qty", "Fill", "Slip bps", "Rev 5m", "Rev 15m", "Rev 60m"]].head(25),
            hide_index=True,
            use_container_width=True,
        )
    elif execution_report.get("errors"):
        st.caption(f"Execution report issue · {execution_report['errors'][0]}")
else:
    st.divider()
    st.markdown("##### Execution quality")
    st.info("No execution-quality report yet.")


# ─────────────────────────────────────────────────────────────────────
# Rolling metrics + return distribution
# ─────────────────────────────────────────────────────────────────────
adv1, adv2 = st.columns(2)

with adv1:
    st.markdown("##### Rolling Sharpe (30 periods)")
    if len(df_window) >= 30:
        rets = df_window["equity"].pct_change().dropna()
        rolling_sharpe = rets.rolling(30).mean() / rets.rolling(30).std() * np.sqrt(252)
        rs_fig = go.Figure()
        rs_fig.add_trace(go.Scatter(
            x=df_window["date"].iloc[-len(rolling_sharpe):],
            y=rolling_sharpe.values,
            mode="lines",
            line=dict(color="#8b5cf6", width=2),
            fill="tozeroy",
            fillcolor="rgba(139,92,246,0.1)",
            hovertemplate="%{y:.2f}<extra>Rolling Sharpe</extra>",
        ))
        rs_fig.add_hline(y=0, line_dash="dot", line_color="gray")
        rs_fig.update_layout(
            height=260, margin=dict(l=40, r=10, t=10, b=30),
            xaxis_title="", yaxis_title="Sharpe",
        )
        st.plotly_chart(rs_fig, use_container_width=True)
    else:
        st.info(f"Need ≥30 days of data ({len(df_window)} so far).")

with adv2:
    st.markdown("##### Return distribution")
    if len(df_window) >= 5:
        rets_pct = df_window["equity"].pct_change().dropna() * 100
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=rets_pct,
            nbinsx=min(30, max(5, len(rets_pct) // 3)),
            marker_color="#3b82f6",
            opacity=0.75,
        ))
        hist_fig.add_vline(x=0, line_dash="dot", line_color="gray")
        hist_fig.add_vline(x=rets_pct.mean(), line_dash="dash",
                           line_color="#22c55e", annotation_text="mean")
        hist_fig.update_layout(
            height=260, margin=dict(l=40, r=10, t=10, b=30),
            xaxis_title="Period return (%)", yaxis_title="Count",
            showlegend=False,
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.info(f"Need ≥5 days of data ({len(df_window)} so far).")


# ─────────────────────────────────────────────────────────────────────
# Raw equity table
# ─────────────────────────────────────────────────────────────────────
with st.expander("Raw equity history"):
    show_df = df_window.copy()
    show_df["change_pct"] = (show_df["equity"].pct_change() * 100).round(3)
    show_df["date"] = show_df["date"].dt.strftime("%Y-%m-%d %H:%M")
    cols_to_show = ["date", "equity"]
    if "cash" in show_df.columns:
        cols_to_show += ["cash", "invested"]
    cols_to_show.append("change_pct")
    st.dataframe(
        show_df[cols_to_show], hide_index=True, use_container_width=True,
    )
