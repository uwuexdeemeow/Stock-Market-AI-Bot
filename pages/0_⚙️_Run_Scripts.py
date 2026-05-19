"""
Run Scripts — full control panel for backend operations

PLAIN ENGLISH: Click into a section, set parameters, click Run.  Every
CLI flag in every backend script has a widget here.  Output streams
live below.  Full logs saved to logs/dashboard_*.log.

The buttons are grouped by what they do:
  1. Data refresh    — ETF data, research panel, feature quality
  2. Signal          — full daily_run OR signal-only
  3. Trading         — submit / reconcile / status / exec guard
  4. Diagnostics     — broker health, paper health, gauntlet, analyzer, fill monitor
  5. Heavy ops       — nested walkforward (hours), publish live config
  6. Recent logs     — viewer for past dashboard runs
"""

import sys
from pathlib import Path

import streamlit as st

from dashboard.components import sidebar_refresh, run_script


st.set_page_config(page_title="Run Scripts", page_icon="⚙️", layout="wide")
sidebar_refresh()
st.title("⚙️ Run Backend Scripts")
st.caption(
    "Every backend CLI flag exposed as a widget.  Defaults match the "
    "GitHub Actions cron behavior — tweak as needed before clicking Run."
)

PY = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Helper — render command preview + Run button inside an expander
# ─────────────────────────────────────────────────────────────────────
def _run_block(label: str, cmd: list[str], *,
               key: str,
               warning: str | None = None,
               help_text: str | None = None) -> None:
    """Show resolved command + Run button.  Lives inside an expander."""
    if warning:
        st.warning(warning, icon="⚠️")
    if help_text:
        st.caption(help_text)
    st.code(" ".join(cmd), language="bash")
    if st.button("▶ Run", key=f"run_{key}", type="primary", use_container_width=True):
        run_script(cmd, label, cwd=PROJECT_ROOT)
        st.cache_data.clear()


# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA REFRESH
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 📥 Data Refresh")
st.caption("Pull fresh prices and recompute features.")

# ── refresh_etf_data.py ──────────────────────────────────────────────
with st.expander("🔁 Refresh ETF data  ·  ~30s"):
    c1, c2, c3 = st.columns(3)
    with c1:
        etf_refresh = st.checkbox("--refresh", value=True, key="etf_refresh",
                                   help="Download new bars from yfinance")
    with c2:
        etf_force = st.checkbox("--force", value=True, key="etf_force",
                                 help="Re-download even if cache is fresh")
    with c3:
        etf_json = st.checkbox("--json", value=False, key="etf_json",
                                help="Output result as JSON")
    etf_symbols = st.text_input(
        "--symbols (comma-separated, blank = default list)",
        value="", key="etf_symbols",
        help="e.g. SPY,QQQ,TQQQ.  Leave blank to use the standard list.",
    )
    cmd = [PY, "refresh_etf_data.py"]
    if etf_refresh: cmd.append("--refresh")
    if etf_force:   cmd.append("--force")
    if etf_json:    cmd.append("--json")
    if etf_symbols.strip():
        cmd.extend(["--symbols", etf_symbols.strip()])
    _run_block("Refresh ETF data", cmd, key="etf")


# ── research.py ──────────────────────────────────────────────────────
with st.expander("📊 Refresh research / factor data  ·  2-5 min (incremental) / 20+ min (full)"):
    research_mode = st.radio(
        "Mode",
        ["incremental (fast)", "all (full rebuild)", "xs-only (cross-sectional refresh only)", "test (smoke test)"],
        index=0, horizontal=True, key="research_mode",
        help="incremental: only new days (fast).  all: rebuild from scratch (slow).  xs-only: skip parquet rebuild, only re-run cross-sectional ranks.  test: smoke test on a few tickers.",
    )
    c1, c2 = st.columns(2)
    with c1:
        research_ticker = st.text_input("--ticker (single ticker)", value="", key="research_ticker",
                                         help="Build/refresh just one ticker.  Blank = use mode-default universe.")
    with c2:
        research_tickers = st.text_input("--tickers (space-separated list)", value="", key="research_tickers",
                                          help="e.g. AAPL MSFT NVDA")

    cmd = [PY, "research.py"]
    if "incremental" in research_mode:  cmd.append("--incremental")
    elif "all" in research_mode:        cmd.append("--all")
    elif "xs-only" in research_mode:    cmd.append("--xs-only")
    elif "test" in research_mode:       cmd.append("--test")
    if research_ticker.strip():
        cmd.extend(["--ticker", research_ticker.strip()])
    if research_tickers.strip():
        cmd.extend(["--tickers", *research_tickers.split()])
    _run_block("Refresh research data", cmd, key="research",
               help_text="Updates per-ticker parquets in `data/`.  Required before signal generation.")


# ── feature_quality_diagnostic.py ────────────────────────────────────
with st.expander("🎯 Refresh feature quality report  ·  1-2 min"):
    top_n = st.slider("--top N (features to rank)", min_value=10, max_value=100,
                      value=48, step=2, key="fq_top")
    cmd = [PY, "feature_quality_diagnostic.py", "--top", str(top_n)]
    _run_block("Refresh feature quality", cmd, key="fq",
               help_text=f"Ranks top **{top_n}** features by predictive power.")


# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — SIGNAL PIPELINE
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 Signal Pipeline")

# ── daily_run.py ─────────────────────────────────────────────────────
with st.expander("⚙️ Full daily pipeline (daily_run.py)  ·  3-5 min"):
    c1, c2 = st.columns(2)
    with c1:
        dr_alpaca = st.checkbox("--alpaca", value=True, key="dr_alpaca")
        dr_moomoo = st.checkbox("--moomoo", value=False, key="dr_moomoo")
        dr_skip_refresh = st.checkbox("--skip-refresh", value=True, key="dr_skip_refresh",
                                       help="Skip ETF + research refresh")
        dr_skip_factor = st.checkbox("--skip-factor-refresh", value=False, key="dr_skip_factor",
                                      help="Refresh ETFs but skip research/factor recompute")
    with c2:
        dr_dry = st.checkbox("--dry-run", value=False, key="dr_dry",
                              help="Show plan only — no execution")
        dr_force = st.checkbox("--force", value=False, key="dr_force",
                                help="Run even on weekends / holidays")
        dr_stress = st.checkbox("--stress", value=False, key="dr_stress",
                                 help="Also run factor_decay, drawdown_throttle, execution_stress, survivorship")
        dr_report = st.checkbox("--report", value=False, key="dr_report",
                                 help="Run side-by-side performance report at the end")
    dr_timeout = st.number_input("--timeout (sec per step)", min_value=60, max_value=1800,
                                  value=600, step=30, key="dr_timeout")

    cmd = [PY, "daily_run.py", "--timeout", str(int(dr_timeout))]
    if dr_alpaca:       cmd.append("--alpaca")
    if dr_moomoo:       cmd.append("--moomoo")
    if dr_skip_refresh: cmd.append("--skip-refresh")
    if dr_skip_factor:  cmd.append("--skip-factor-refresh")
    if dr_dry:          cmd.append("--dry-run")
    if dr_force:        cmd.append("--force")
    if dr_stress:       cmd.append("--stress")
    if dr_report:       cmd.append("--report")
    _run_block("Full daily pipeline", cmd, key="daily_run",
               warning=(None if dr_dry else "Will submit paper orders if market is open."),
               help_text="ETF refresh → research → signal → submit → reconcile → health checks.")


# ── core_satellite_alpha.py ──────────────────────────────────────────
with st.expander("🎯 Generate signal only (core_satellite_alpha.py)  ·  ~30s"):
    c1, c2 = st.columns(2)
    with c1:
        sig_ignore_stale = st.checkbox("--ignore-stale", value=False, key="sig_ignore_stale",
                                         help="Allow signal generation even if data is stale")
        sig_walkforward = st.checkbox("--walkforward (run nested WF inline)", value=False, key="sig_walkforward",
                                       help="Run nested walkforward as part of signal generation (SLOW)")
    with c2:
        sig_no_walkforward = st.checkbox("--no-walkforward", value=False, key="sig_no_walkforward",
                                          help="Skip nested walkforward")
        sig_min_train = st.number_input("--min-train-years", min_value=2, max_value=15,
                                          value=4, step=1, key="sig_min_train",
                                          help="Minimum training years required")

    cmd = [PY, "core_satellite_alpha.py"]
    if sig_ignore_stale:    cmd.append("--ignore-stale")
    if sig_walkforward:     cmd.append("--walkforward")
    if sig_no_walkforward:  cmd.append("--no-walkforward")
    if sig_min_train != 4:
        cmd.extend(["--min-train-years", str(int(sig_min_train))])
    _run_block("Generate signal", cmd, key="signal",
               help_text="Regenerates `signals/core_satellite_alpha_signal.csv`.")


# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — TRADING ACTIONS
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 💸 Trading Actions")

# ── alpaca submit ────────────────────────────────────────────────────
with st.expander("📤 Submit orders to Alpaca  ·  ~1 min"):
    c1, c2 = st.columns(2)
    with c1:
        ap_force = st.checkbox("--force", value=False, key="ap_force",
                                help="Skip drift thresholds AND duplicate-day check")
        ap_market = st.checkbox("--market-order (default on)", value=True, key="ap_market",
                                 help="Submit market orders.  Default behavior.")
    with c2:
        ap_stale_ok = st.checkbox("--allow-stale-signal", value=False, key="ap_stale",
                                   help="Allow submission even if signal freshness check fails")
        ap_signal_age = st.number_input("--max-signal-age-hours", min_value=0.5, max_value=72.0,
                                         value=24.0, step=0.5, key="ap_signal_age",
                                         help="Block submit if predicted_at is older than this")
    ap_factor_age = st.number_input("--max-factor-age-trading-days", min_value=1, max_value=14,
                                     value=3, step=1, key="ap_factor_age",
                                     help="Block submit if latest_factor_date is older than this many trading days")

    cmd = [PY, "alpaca_paper_trading.py", "--submit"]
    if ap_force:    cmd.append("--force")
    if not ap_market: pass  # --market-order is default-True; no flag to disable
    if ap_stale_ok: cmd.append("--allow-stale-signal")
    cmd.extend(["--max-signal-age-hours", str(float(ap_signal_age))])
    cmd.extend(["--max-factor-age-trading-days", str(int(ap_factor_age))])
    _run_block("Submit orders", cmd, key="submit",
               warning="Submits paper orders to Alpaca.",
               help_text="Reads today's signal and submits the rebalance orders.")


# ── alpaca reconcile ─────────────────────────────────────────────────
with st.expander("✅ Reconcile fill statuses  ·  ~30s"):
    st.caption("Checks pending orders for fills/cancellations.  Read-only.")
    cmd = [PY, "alpaca_paper_trading.py", "--reconcile"]
    _run_block("Reconcile fills", cmd, key="reconcile")


# ── alpaca status ────────────────────────────────────────────────────
with st.expander("👀 Show Alpaca account status  ·  ~10s"):
    st.caption("Print positions/cash/equity.  No state changes.")
    cmd = [PY, "alpaca_paper_trading.py", "--status"]
    _run_block("Show account status", cmd, key="status")


# ── execution_guard.py ───────────────────────────────────────────────
with st.expander("🛡 Execution guard  ·  ~15s"):
    c1, c2 = st.columns(2)
    with c1:
        eg_once = st.checkbox("--once (single cycle, then exit)", value=True, key="eg_once")
    with c2:
        eg_dry = st.checkbox("--dry-run (log only, no orders)", value=False, key="eg_dry")
    cmd = [PY, "execution_guard.py"]
    if eg_once: cmd.append("--once")
    if eg_dry:  cmd.append("--dry-run")
    _run_block("Execution guard", cmd, key="exec_guard",
               help_text="Repair stops, cancel stale orders, check intraday P&L.")


# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — DIAGNOSTICS / HEALTH
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 🩺 Diagnostics & Health")
st.caption("Read-only checks.")

# ── broker_health.py ─────────────────────────────────────────────────
with st.expander("📡 Broker health check  ·  ~10s"):
    c1, c2, c3 = st.columns(3)
    with c1:
        bh_alpaca = st.checkbox("--alpaca", value=True, key="bh_alpaca")
    with c2:
        bh_moomoo = st.checkbox("--moomoo", value=False, key="bh_moomoo")
    with c3:
        bh_json = st.checkbox("--json", value=False, key="bh_json")
    cmd = [PY, "broker_health.py"]
    if bh_alpaca: cmd.append("--alpaca")
    if bh_moomoo: cmd.append("--moomoo")
    if bh_json:   cmd.append("--json")
    _run_block("Broker health", cmd, key="broker_health")


# ── paper_health.py ──────────────────────────────────────────────────
with st.expander("📋 Paper health summary  ·  ~30s"):
    c1, c2 = st.columns(2)
    with c1:
        ph_broker = st.selectbox("--broker", ["alpaca", "moomoo"], index=0, key="ph_broker")
    with c2:
        ph_json = st.checkbox("--json", value=False, key="ph_json")
    cmd = [PY, "paper_health.py", "--broker", ph_broker]
    if ph_json: cmd.append("--json")
    _run_block(f"Paper health ({ph_broker})", cmd, key="paper_health",
               help_text="Drift detection, slippage, concentration, scorecard.")


# ── alpaca_paper_gauntlet.py ─────────────────────────────────────────
with st.expander("🛡 Alpaca paper gauntlet  ·  ~20s"):
    c1, c2, c3 = st.columns(3)
    with c1:
        g_verbose = st.checkbox("--verbose", value=False, key="g_verbose")
    with c2:
        g_json = st.checkbox("--json", value=False, key="g_json")
    with c3:
        g_snapshot = st.checkbox("--snapshot (just equity snap, skip gauntlet)",
                                  value=False, key="g_snapshot")
    cmd = [PY, "alpaca_paper_gauntlet.py"]
    if g_verbose:  cmd.append("--verbose")
    if g_json:     cmd.append("--json")
    if g_snapshot: cmd.append("--snapshot")
    _run_block("Alpaca paper gauntlet", cmd, key="gauntlet",
               help_text="Go-live readiness: trading days, fill rate, Sharpe, drawdown.")


# ── walkforward_analyzer.py ──────────────────────────────────────────
with st.expander("🔬 Walkforward analyzer  ·  ~5s"):
    wfa_csv = st.text_input("--csv", value="signals/core_satellite_nested_walkforward.csv",
                             key="wfa_csv")
    c1, c2 = st.columns(2)
    with c1:
        wfa_qqq = st.text_input("--qqq (path to QQQ parquet, blank = default)",
                                 value="", key="wfa_qqq")
    with c2:
        wfa_spy = st.text_input("--spy (path to SPY parquet, blank = default)",
                                 value="", key="wfa_spy")
    wfa_json = st.checkbox("--json (also write analyzer JSON)", value=False, key="wfa_json")
    cmd = [PY, "walkforward_analyzer.py", "--csv", wfa_csv]
    if wfa_qqq.strip(): cmd.extend(["--qqq", wfa_qqq.strip()])
    if wfa_spy.strip(): cmd.extend(["--spy", wfa_spy.strip()])
    if wfa_json:        cmd.append("--json")
    _run_block("Walkforward analyzer", cmd, key="wf_analyzer",
               help_text="4 checks: predictiveness, calibration, concentration, config stability.")


# ── fill_monitor.py ──────────────────────────────────────────────────
with st.expander("📜 Fill monitor (recent orders)  ·  ~10s"):
    c1, c2 = st.columns(2)
    with c1:
        fm_days = st.number_input("--days", min_value=1, max_value=30, value=2, step=1, key="fm_days")
    with c2:
        fm_quiet = st.checkbox("--quiet", value=False, key="fm_quiet",
                                help="Suppress detailed output")
    cmd = [PY, "fill_monitor.py", "--days", str(int(fm_days))]
    if fm_quiet: cmd.append("--quiet")
    _run_block("Fill monitor", cmd, key="fill_monitor")


# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — HEAVY OPERATIONS
# ═════════════════════════════════════════════════════════════════════
st.markdown("## 🏋️ Heavy Operations")
st.caption("Long-running or destructive.")

# ── nested walkforward ───────────────────────────────────────────────
with st.expander("🔬 Nested walkforward  ·  2-6 HOURS"):
    st.warning(
        "Takes hours.  Don't close the dashboard tab — Streamlit kills the subprocess if you do. "
        "Consider running on a server/VPS for long jobs.",
        icon="⚠️",
    )

    # Grid / strategy controls
    c1, c2 = st.columns(2)
    with c1:
        wf_strategy = st.selectbox("--strategy", ["core-alpha", "tqqq", "both"], index=0, key="wf_strategy")
        wf_full = st.checkbox("--full (exhaustive 768-config grid)", value=True, key="wf_full")
        wf_fast = st.checkbox("--fast (smoke ~48 configs, ~15 min)", value=False, key="wf_fast")
    with c2:
        wf_stable = st.checkbox("--stable-grid (pinned ~24 configs)", value=False, key="wf_stable")
        wf_recent = st.checkbox("--recent-alpha-grid (~48 focused configs)", value=False, key="wf_recent")
        wf_output = st.text_input("--output-prefix", value="core_satellite_nested_walkforward",
                                   key="wf_output")

    # Year + fold controls
    c3, c4 = st.columns(2)
    with c3:
        wf_start_year = st.number_input("--start-year (blank = auto)", min_value=0, max_value=2099,
                                         value=0, step=1, key="wf_start_year",
                                         help="0 means use script default")
        wf_end_year = st.number_input("--end-year (blank = auto)", min_value=0, max_value=2099,
                                       value=0, step=1, key="wf_end_year")
        wf_min_train = st.number_input("--min-train-years", min_value=1, max_value=15,
                                        value=4, step=1, key="wf_min_train")
        wf_min_inner = st.number_input("--min-inner-train-years (0 = auto)", min_value=0, max_value=15,
                                        value=0, step=1, key="wf_min_inner")
    with c4:
        wf_max_folds = st.number_input("--max-folds (0 = no limit)", min_value=0, max_value=20,
                                        value=0, step=1, key="wf_max_folds")
        wf_max_configs = st.number_input("--max-configs (0 = no limit)", min_value=0, max_value=1000,
                                          value=0, step=10, key="wf_max_configs")
        wf_max_specs = st.number_input("--max-specs (0 = use default)", min_value=0, max_value=1000,
                                        value=0, step=10, key="wf_max_specs")

    # Publish control
    wf_publish = st.selectbox(
        "Publish live config?",
        ["auto", "always (--publish-live-config)", "never (--no-publish-live-config)"],
        index=0, key="wf_publish",
    )

    # Build the command
    cmd = [PY, "core_satellite_nested_walkforward.py", "--strategy", wf_strategy]
    if wf_full:   cmd.append("--full")
    if wf_fast:   cmd.append("--fast")
    if wf_stable: cmd.append("--stable-grid")
    if wf_recent: cmd.append("--recent-alpha-grid")
    if "always" in wf_publish: cmd.append("--publish-live-config")
    elif "never" in wf_publish: cmd.append("--no-publish-live-config")
    if wf_start_year > 0:    cmd.extend(["--start-year", str(int(wf_start_year))])
    if wf_end_year > 0:      cmd.extend(["--end-year", str(int(wf_end_year))])
    if wf_min_train != 4:    cmd.extend(["--min-train-years", str(int(wf_min_train))])
    if wf_min_inner > 0:     cmd.extend(["--min-inner-train-years", str(int(wf_min_inner))])
    if wf_max_folds > 0:     cmd.extend(["--max-folds", str(int(wf_max_folds))])
    if wf_max_configs > 0:   cmd.extend(["--max-configs", str(int(wf_max_configs))])
    if wf_max_specs > 0:     cmd.extend(["--max-specs", str(int(wf_max_specs))])
    if wf_output != "core_satellite_nested_walkforward":
        cmd.extend(["--output-prefix", wf_output])

    confirm = st.checkbox("I understand this takes hours", key="wf_confirm")
    if confirm:
        _run_block("Nested walkforward", cmd, key="walkforward",
                   warning="HOURS — don't close this tab.")
    else:
        st.code(" ".join(cmd), language="bash")
        st.button("▶ Run (locked)", disabled=True, use_container_width=True,
                  help="Check the confirmation box above to enable.")


# ── publish_live_config_from_csv.py ──────────────────────────────────
with st.expander("📡 Publish live config from walkforward CSV  ·  ~5s"):
    c1, c2 = st.columns(2)
    with c1:
        pub_source = st.selectbox(
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
        pub_force = st.checkbox("--force (publish even if approval fails)",
                                 value=False, key="pub_force")
        pub_dry = st.checkbox("--dry-run", value=False, key="pub_dry")
    cmd = [PY, "publish_live_config_from_csv.py", "--source", pub_source]
    if pub_force: cmd.append("--force")
    if pub_dry:   cmd.append("--dry-run")
    _run_block("Publish live config", cmd, key="publish",
               warning="Overwrites signals/core_satellite_live_configs.json (.bak backup made).",
               help_text="Promotes a config from the walkforward CSV without rerunning WF.")


# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — RECENT LOGS
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
