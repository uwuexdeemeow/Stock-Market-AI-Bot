"""
Run Scripts — control panel for backend operations

PLAIN ENGLISH: Click into a section, set parameters, click Run.  Output
streams live below.  Full logs saved to logs/dashboard_*.log.

Each script is in a collapsible expander.  Defaults match the most-common
usage (what GitHub Actions would run).  Tweak the widgets to customize.
"""

import sys
from pathlib import Path

import streamlit as st

from dashboard.components import sidebar_refresh, run_script


st.set_page_config(page_title="Run Scripts", page_icon="⚙️", layout="wide")
sidebar_refresh()
st.title("⚙️ Run Backend Scripts")
st.caption(
    "Click into a section → set parameters → click Run.  Output streams "
    "live and is saved to `logs/dashboard_*.log`."
)

# Same Python interpreter as the dashboard — guarantees we hit the same venv
PY = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Helper — build a run-block inside an expander
# ─────────────────────────────────────────────────────────────────────
def _run_block(label: str, cmd: list[str], *,
               key: str,
               warning: str | None = None,
               help_text: str | None = None) -> None:
    """Show the resolved command + a Run button.

    The expander wrapping this is owned by the caller so each script
    can have its own parameter widgets above the Run button.
    """
    if warning:
        st.warning(warning, icon="⚠️")
    if help_text:
        st.caption(help_text)
    st.code(" ".join(cmd), language="bash")
    if st.button(f"▶ Run", key=f"run_{key}", type="primary", use_container_width=True):
        run_script(cmd, label, cwd=PROJECT_ROOT)
        st.cache_data.clear()


# ═════════════════════════════════════════════════════════════════════
# Section 1 — DATA REFRESH
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 📥 Data Refresh")
st.caption("Pull fresh prices + recompute features.  Run these first if data is stale.")

# ── refresh_etf_data ──
with st.expander("🔁 Refresh ETF data  ·  ~30s"):
    c1, c2 = st.columns(2)
    with c1:
        etf_refresh = st.checkbox("--refresh (download new bars)", value=True, key="etf_refresh")
    with c2:
        etf_force = st.checkbox("--force (re-download even if cache fresh)", value=True, key="etf_force")
    cmd = [PY, "refresh_etf_data.py"]
    if etf_refresh: cmd.append("--refresh")
    if etf_force:   cmd.append("--force")
    _run_block("Refresh ETF data", cmd, key="etf",
               help_text="SPY, QQQ, TQQQ, etc. — used as benchmarks and core positions.")


# ── research.py ──
with st.expander("📊 Refresh research / factor data  ·  2-5 min (incremental) / 20+ min (full)"):
    research_mode = st.radio(
        "Mode",
        ["incremental", "full"],
        index=0,
        horizontal=True,
        key="research_mode",
        help="incremental = only new days since last run.  full = recompute everything (slow).",
    )
    cmd = [PY, "research.py", f"--{research_mode}"]
    _run_block("Refresh research data", cmd, key="research",
               help_text=f"Mode: **{research_mode}**.  Updates per-ticker parquet files in `data/`.")


# ── feature_quality_diagnostic ──
with st.expander("🎯 Refresh feature quality report  ·  1-2 min"):
    top_n = st.slider("--top N (how many features to rank)", min_value=10, max_value=100,
                      value=48, step=2, key="fq_top")
    cmd = [PY, "feature_quality_diagnostic.py", "--top", str(top_n)]
    _run_block("Refresh feature quality", cmd, key="fq",
               help_text=f"Ranks top **{top_n}** features by predictive power, updates feature_quality_*.json.")


# ═════════════════════════════════════════════════════════════════════
# Section 2 — SIGNAL PIPELINE
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 Signal Pipeline")
st.caption("Full daily pipeline OR just regenerate today's signal.")

# ── daily_run.py ──
with st.expander("⚙️ Full daily pipeline (daily_run.py)  ·  3-5 min"):
    c1, c2 = st.columns(2)
    with c1:
        run_alpaca = st.checkbox("--alpaca", value=True, key="dr_alpaca",
                                  help="Run the Alpaca submit chain")
        run_moomoo = st.checkbox("--moomoo", value=False, key="dr_moomoo",
                                  help="Run the Moomoo submit chain (in addition to Alpaca)")
    with c2:
        skip_refresh = st.checkbox("--skip-refresh", value=True, key="dr_skip",
                                    help="Skip ETF + research refresh (use existing data)")
        dry_run = st.checkbox("--dry-run", value=False, key="dr_dry",
                               help="Show what would run without executing.  Safe.")
    c3, c4 = st.columns(2)
    with c3:
        force = st.checkbox("--force (ignore weekend/holiday guard)", value=False, key="dr_force")
        stress = st.checkbox("--stress (also run stress tests)", value=False, key="dr_stress",
                              help="Adds factor_decay, drawdown_throttle, execution_stress, survivorship steps")
    with c4:
        timeout = st.number_input("--timeout (sec per step)", min_value=60, max_value=1800,
                                   value=600, step=30, key="dr_timeout")

    cmd = [PY, "daily_run.py", "--timeout", str(int(timeout))]
    if run_alpaca:  cmd.append("--alpaca")
    if run_moomoo:  cmd.append("--moomoo")
    if skip_refresh: cmd.append("--skip-refresh")
    if dry_run:     cmd.append("--dry-run")
    if force:       cmd.append("--force")
    if stress:      cmd.append("--stress")

    _run_block("Full daily pipeline", cmd, key="daily_run",
               warning=(None if dry_run else "Will submit real paper orders if --alpaca and market is open."),
               help_text="Orchestrates ETF refresh → research → signal → submit → reconcile → health.")


# ── core_satellite_alpha (signal only) ──
with st.expander("🎯 Generate signal only (no orders)  ·  ~30s"):
    st.caption("Just runs core_satellite_alpha.py.  No CLI args needed — reads the approved live config.")
    cmd = [PY, "core_satellite_alpha.py"]
    _run_block("Generate signal", cmd, key="signal",
               help_text="Regenerates `signals/core_satellite_alpha_signal.csv`.")


# ═════════════════════════════════════════════════════════════════════
# Section 3 — TRADING ACTIONS
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 💸 Trading Actions")
st.caption("Send/manage orders on the Alpaca paper account.")

# ── alpaca submit ──
with st.expander("📤 Submit orders to Alpaca  ·  ~1 min"):
    c1, c2 = st.columns(2)
    with c1:
        submit_force = st.checkbox("--force (override duplicate-submission guard)",
                                    value=False, key="submit_force",
                                    help="Lets you submit again on the same day.")
    with c2:
        submit_dry = st.checkbox("--dry-run (show plan only)", value=False, key="submit_dry")
    cmd = [PY, "alpaca_paper_trading.py", "--submit"]
    if submit_force: cmd.append("--force")
    if submit_dry:   cmd.append("--dry-run")
    _run_block("Submit orders", cmd, key="submit",
               warning=("Will submit paper orders to Alpaca." if not submit_dry else None),
               help_text="Reads today's signal and submits the rebalance orders.")


# ── alpaca reconcile ──
with st.expander("✅ Reconcile fill statuses  ·  ~30s"):
    st.caption("Checks pending orders for fills/cancellations.  No new orders submitted.  Safe.")
    cmd = [PY, "alpaca_paper_trading.py", "--reconcile"]
    _run_block("Reconcile fills", cmd, key="reconcile",
               help_text="Updates the order log with broker-side status (filled/cancelled/expired).")


# ── alpaca status ──
with st.expander("👀 Show Alpaca account status  ·  ~10s"):
    st.caption("Read-only print of current positions, cash, equity.")
    cmd = [PY, "alpaca_paper_trading.py", "--status"]
    _run_block("Show account status", cmd, key="status",
               help_text="Pretty-prints account.  No state changes.")


# ── execution_guard ──
with st.expander("🛡 Execution guard (cancel stale orders, repair stops)  ·  ~15s"):
    st.caption(
        "Runs one cycle: repairs ETF protective stops, cancels orders older than threshold, "
        "checks intraday P&L guard."
    )
    cmd = [PY, "execution_guard.py", "--once"]
    _run_block("Execution guard (once)", cmd, key="exec_guard",
               help_text="Safe-ish: only cancels obviously-stale orders and re-places stops.")


# ═════════════════════════════════════════════════════════════════════
# Section 4 — DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 🩺 Diagnostics & Health")
st.caption("Read-only checks.  Safe to run anytime.")

# ── broker_health ──
with st.expander("📡 Broker health check  ·  ~10s"):
    c1, c2 = st.columns(2)
    with c1:
        bh_alpaca = st.checkbox("--alpaca", value=True, key="bh_alpaca")
    with c2:
        bh_moomoo = st.checkbox("--moomoo", value=False, key="bh_moomoo")
    cmd = [PY, "broker_health.py"]
    if bh_alpaca: cmd.append("--alpaca")
    if bh_moomoo: cmd.append("--moomoo")
    _run_block("Broker health", cmd, key="broker_health",
               help_text="Pings broker API(s), returns connectivity + equity + latency.")


# ── paper_health ──
with st.expander("📋 Paper health summary  ·  ~30s"):
    c1, c2 = st.columns(2)
    with c1:
        ph_broker = st.selectbox("--broker", ["alpaca", "moomoo"], index=0, key="ph_broker")
    with c2:
        ph_json = st.checkbox("--json (raw JSON output)", value=False, key="ph_json")
    cmd = [PY, "paper_health.py", "--broker", ph_broker]
    if ph_json: cmd.append("--json")
    _run_block(f"Paper health ({ph_broker})", cmd, key="paper_health",
               help_text="Builds drift detection, slippage analysis, concentration check, scorecard.")


# ── alpaca_paper_gauntlet ──
with st.expander("🛡 Alpaca paper gauntlet  ·  ~20s"):
    c1, c2 = st.columns(2)
    with c1:
        g_verbose = st.checkbox("--verbose", value=False, key="g_verbose")
    with c2:
        g_json = st.checkbox("--json", value=False, key="g_json")
    cmd = [PY, "alpaca_paper_gauntlet.py"]
    if g_verbose: cmd.append("--verbose")
    if g_json:    cmd.append("--json")
    _run_block("Alpaca paper gauntlet", cmd, key="gauntlet",
               help_text="Go-live readiness: trading days, fill rate, Sharpe, drawdown, signal age.")


# ── walkforward_analyzer ──
with st.expander("🔬 Walkforward analyzer  ·  ~5s"):
    custom_csv = st.text_input("--csv (path to walkforward CSV)",
                                value="signals/core_satellite_nested_walkforward.csv",
                                key="wfa_csv")
    save_json = st.checkbox("--json (also write analyzer JSON report)", value=False, key="wfa_json")
    cmd = [PY, "walkforward_analyzer.py", "--csv", custom_csv]
    if save_json: cmd.append("--json")
    _run_block("Walkforward analyzer", cmd, key="wf_analyzer",
               help_text="Runs 4 checks: score predictiveness, calibration, concentration vulnerability, config stability.")


# ── fill_monitor ──
with st.expander("📜 Fill monitor (recent orders)  ·  ~10s"):
    fm_days = st.number_input("--days (look back N trading days)",
                               min_value=1, max_value=30, value=2, step=1, key="fm_days")
    cmd = [PY, "fill_monitor.py", "--days", str(int(fm_days))]
    _run_block("Fill monitor", cmd, key="fill_monitor",
               help_text="Lists orders from the last N days with their fill status.")


# ═════════════════════════════════════════════════════════════════════
# Section 5 — HEAVY OPERATIONS
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 🏋️ Heavy Operations")
st.caption("Long-running or destructive.  Read the warnings.")

# ── nested walkforward ──
with st.expander("🔬 Nested walkforward (re-validate strategy)  ·  2-6 HOURS"):
    st.warning(
        "Takes hours.  Don't close the dashboard tab while it runs.  "
        "Consider running on a server / VPS instead.",
        icon="⚠️",
    )
    c1, c2 = st.columns(2)
    with c1:
        wf_strategy = st.selectbox("--strategy", ["core-alpha", "tqqq", "both"],
                                    index=0, key="wf_strategy")
        wf_full = st.checkbox("--full (skip successive-halving screen)",
                               value=True, key="wf_full",
                               help="Recommended.  Halving screen risks dropping the true best config.")
        wf_fast = st.checkbox("--fast (smoke-test grid only)",
                               value=False, key="wf_fast",
                               help="48 configs instead of 384.  For quick sanity checks.")
    with c2:
        wf_publish = st.selectbox(
            "Publish live config?",
            ["auto", "always (--publish-live-config)", "never (--no-publish-live-config)"],
            index=0, key="wf_publish",
            help="auto = publish only when running --full and no --fast/--stable-grid flags.",
        )
        wf_max_folds = st.number_input("--max-folds (limit fold count, 0=all)",
                                        min_value=0, max_value=20, value=0, step=1, key="wf_max_folds")

    confirm = st.checkbox("I understand this takes hours and may cost CPU/electricity",
                          key="wf_confirm")

    cmd = [PY, "core_satellite_nested_walkforward.py", "--strategy", wf_strategy]
    if wf_full:    cmd.append("--full")
    if wf_fast:    cmd.append("--fast")
    if "always" in wf_publish:
        cmd.append("--publish-live-config")
    elif "never" in wf_publish:
        cmd.append("--no-publish-live-config")
    if wf_max_folds > 0:
        cmd.extend(["--max-folds", str(int(wf_max_folds))])

    if confirm:
        _run_block("Nested walkforward", cmd, key="walkforward",
                   warning="HOURS — don't close this tab.")
    else:
        st.code(" ".join(cmd), language="bash")
        st.button("▶ Run (locked)", disabled=True, use_container_width=True,
                  help="Check the confirmation box above to enable.")


# ── publish_live_config_from_csv ──
with st.expander("📡 Publish live config from walkforward CSV  ·  ~5s"):
    c1, c2 = st.columns(2)
    with c1:
        publish_source = st.selectbox(
            "--source",
            ["most_common", "latest", "best_sharpe", "top_family"],
            index=0, key="pub_source",
            help=(
                "most_common = most-frequent exact config.\n"
                "latest = most recent fold's selection.\n"
                "best_sharpe = highest single-fold OOS Sharpe.\n"
                "top_family = most-frequent shape+weighting combo."
            ),
        )
    with c2:
        publish_force = st.checkbox("--force (publish even if approval fails)",
                                     value=False, key="pub_force")
        publish_dry = st.checkbox("--dry-run", value=False, key="pub_dry")
    cmd = [PY, "publish_live_config_from_csv.py", "--source", publish_source]
    if publish_force: cmd.append("--force")
    if publish_dry:   cmd.append("--dry-run")
    _run_block("Publish live config", cmd, key="publish",
               warning="Overwrites signals/core_satellite_live_configs.json.  Backup auto-saved as .bak.",
               help_text="Promotes a config from the walkforward CSV to live trading without rerunning the WF.")


# ═════════════════════════════════════════════════════════════════════
# Section 6 — RECENT LOGS
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 📜 Recent Dashboard Runs")

logs_dir = PROJECT_ROOT / "logs"
if logs_dir.exists():
    dashboard_logs = sorted(
        logs_dir.glob("dashboard_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:10]
    if dashboard_logs:
        import datetime as _dt
        for log_path in dashboard_logs:
            mtime = _dt.datetime.fromtimestamp(log_path.stat().st_mtime)
            size_kb = log_path.stat().st_size / 1024
            with st.expander(f"📄 {log_path.name}  ·  {mtime.strftime('%H:%M:%S')}  ·  {size_kb:.1f} KB"):
                try:
                    content = log_path.read_text(errors="replace")
                    lines = content.splitlines()[-100:]
                    st.code("\n".join(lines), language="text")
                except Exception as e:
                    st.error(f"Couldn't read log: {e}")
    else:
        st.caption("No dashboard runs yet.")
else:
    st.caption("`logs/` folder will be created on first run.")
