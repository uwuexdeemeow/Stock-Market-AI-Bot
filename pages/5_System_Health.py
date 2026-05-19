"""
System Health page — pipeline status, broker connectivity, data freshness

PLAIN ENGLISH: "Is the infrastructure working?"  This page tells you
whether the daily run succeeded, whether Alpaca is reachable, whether
data files are fresh, and whether the drift detector is happy.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from dashboard import data
from dashboard.components import status_chip, sidebar_refresh


st.set_page_config(page_title="System Health", page_icon="•", layout="wide")
sidebar_refresh()
st.title("System Health")

# ── 1. Latest daily run ──────────────────────────────────────────
st.markdown("### Latest daily pipeline run")
last_run = data.load_latest_daily_run()
if last_run is None:
    st.error("No daily_run log found. The pipeline hasn't executed yet.")
else:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ts = last_run.get("timestamp", "")
        st.metric("Timestamp", ts.split("T")[0] if "T" in ts else ts)
    with c2:
        total = last_run.get("steps_total", 0)
        passed = last_run.get("steps_ok", 0)
        ratio = passed / total if total else 0
        status = "ok" if ratio >= 0.9 else "warn" if ratio >= 0.6 else "fail"
        status_chip(f"{passed}/{total} steps", status)
    with c3:
        elapsed = last_run.get("total_elapsed_seconds", 0)
        st.metric("Elapsed", f"{elapsed:.0f}s")
    with c4:
        failed = last_run.get("steps_failed", 0) or sum(
            1 for s in (last_run.get("results", []) or [])
            if s.get("status") not in ("ok", "skipped")
        )
        st.metric("Failed steps", failed)

    # Per-step list
    results = last_run.get("results", []) or []
    if results:
        st.markdown("**Steps:**")
        steps_df = pd.DataFrame(results)
        if "status" in steps_df.columns:
            # Tag with icons
            steps_df["status_icon"] = steps_df["status"].map({
                "ok": "✅", "failed": "❌", "skipped": "⏭", "blocked": "🚫", "timeout": "⏱"
            }).fillna("❔")
        show_cols = [c for c in ["status_icon", "name", "status", "elapsed", "error"] if c in steps_df.columns]
        st.dataframe(steps_df[show_cols] if show_cols else steps_df,
                     hide_index=True, use_container_width=True)

st.divider()

# ── 2. Broker connectivity ───────────────────────────────────────
st.markdown("### Broker connectivity")
broker = data.load_broker_health()
if broker is None:
    st.info("No broker health snapshot. The broker_health step hasn't run.")
else:
    brokers = broker.get("brokers", broker)  # support both flat + nested
    cols = st.columns(len(brokers)) if isinstance(brokers, dict) and brokers else None
    if isinstance(brokers, dict) and cols:
        for i, (name, info) in enumerate(brokers.items()):
            if not isinstance(info, dict):
                continue
            with cols[i]:
                healthy = info.get("healthy", info.get("connected", False))
                status_chip(name, "ok" if healthy else "fail")
                if "latency_ms" in info:
                    st.caption(f"Latency: {info['latency_ms']}ms")
                if "equity" in info:
                    st.caption(f"Equity: ${float(info['equity']):,.2f}")
    else:
        st.json(broker)

st.divider()

# ── 3. Drift detection (live vs walkforward) ────────────────────
st.markdown("### Drift detection (live vs walkforward expectations)")
health = data.load_health_report()
if health is None:
    st.info("No drift report yet. After ~10 trading days, drift comparison becomes meaningful.")
else:
    drift = health.get("drift_vs_walkforward", health.get("backtest_drift", {})) or {}
    warnings = drift.get("warnings", []) or []
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        status_chip("No drift warnings", "ok")

    # Show comparison table if present
    bt_metrics = {k: v for k, v in drift.items() if k.startswith("backtest_")}
    live_metrics = {k: v for k, v in drift.items() if k.startswith("live_")}
    if bt_metrics or live_metrics:
        rows = []
        for key in ("sharpe", "cagr_pct", "max_drawdown_pct", "alpha_vs_qqq_pct"):
            rows.append({
                "Metric": key,
                "Backtest": drift.get(f"backtest_{key}", "—"),
                "Live": drift.get(f"live_{key}", "—"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()

# ── 4. Data freshness ──────────────────────────────────────────
st.markdown("### Data file freshness")
st.dataframe(data.file_status_table(), use_container_width=True, hide_index=True)

st.divider()

# ── 5. Live config snapshot ──────────────────────────────────────
st.markdown("### Active live config")
live = data.load_live_config()
if live is None:
    st.warning("No live config approved. Run publish_live_config_from_csv.py.")
else:
    cfg_block = (live.get("approved_live_configs", {}) or {}).get("core-alpha", {})
    cfg = cfg_block.get("config", {}) or {}
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Strategy", cfg.get("strategy", "—"))
        st.metric("Holding days", cfg.get("holding_days", "—"))
        st.metric("Shape", cfg.get("shape", "—"))
        st.metric("Weighting", cfg.get("weighting", "—"))
    with c2:
        st.metric("Overlay gross", f"{float(cfg.get('overlay_gross', 0)):.2f}")
        st.metric("Core gross", f"{float(cfg.get('core_gross', 0)):.2f}")
        st.metric("TQQQ weight", f"{float(cfg.get('tqqq_weight', 0)):.2f}")
        cb = float(cfg.get("drawdown_circuit_breaker", 0))
        if cb > 0:
            st.metric("Drawdown CB", f"{cb*100:.0f}%", delta="armed", delta_color="off")
        else:
            st.metric("Drawdown CB", "off", delta="DISABLED", delta_color="off")
    if "created_at" in live:
        st.caption(f"Config approved at: {live['created_at']}")
