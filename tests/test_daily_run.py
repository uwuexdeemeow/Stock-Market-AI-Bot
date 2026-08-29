from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import daily_run


def _names(steps):
    return [step.name for step in steps]


def test_default_steps_include_one_shared_signal_before_alpaca():
    steps = daily_run.build_steps(
        skip_refresh=False,
        run_alpaca=True,
    )
    names = _names(steps)
    assert names.count("core_satellite_signal") == 1
    assert names.index("factor_data_health") < names.index("core_satellite_signal")
    assert names.index("factor_data_health") < names.index("drift_monitor")
    assert names.index("drift_monitor") < names.index("core_satellite_signal")
    assert names.index("core_satellite_signal") < names.index("alpaca_submit")


def test_daily_refresh_forces_etf_download():
    etf_step = next(step for step in daily_run.DATA_REFRESH_STEPS if step.name == "refresh_etf_data")
    assert "--refresh" in etf_step.cmd
    assert "--force" in etf_step.cmd
    assert "--strict" in etf_step.cmd


def test_daily_workflow_pins_execution_safety_env():
    # PLAIN ENGLISH: the GitHub workflow writes its own .env file, so this test
    # catches accidental removal of the safety knobs the live submit script needs.
    workflow = Path(".github/workflows/daily_paper_trading.yml").read_text(encoding="utf-8")
    required_lines = [
        "ALPACA_MAX_GROSS_EXPOSURE=1.00",
        "ALPACA_ORDER_TYPE=limit",
        "ALPACA_ALLOW_MARKET_ORDER_OVERRIDE=0",
        "ALPACA_LIMIT_REFERENCE=quote",
        "ALPACA_LIMIT_OFFSET_BPS_ETF=5",
        "ALPACA_LIMIT_OFFSET_BPS_OVERLAY=12",
        "ALPACA_TWO_STAGE_EXECUTION=1",
        "ALPACA_EXECUTION_STAGE1_WAIT_SECONDS=15",
        "ALPACA_EXECUTION_STAGE2_OFFSET_BPS_ETF=1",
        "ALPACA_EXECUTION_STAGE2_OFFSET_BPS_OVERLAY=3",
        "ALPACA_EXECUTION_QUOTE_MAX_AGE_SECONDS=5",
        "ALPACA_EXECUTION_WINDOW_START=09:35",
        "ALPACA_EXECUTION_WINDOW_END=10:30",
        "ALPACA_EXECUTION_RISK_ENABLED=1",
        "ALPACA_EXECUTION_RISK_REPORT_MAX_AGE_HOURS=168",
        "ALPACA_EXECUTION_RISK_WARN_SCORE=40",
        "ALPACA_EXECUTION_RISK_HIGH_SCORE=60",
        "ALPACA_EXECUTION_RISK_WARN_BUY_SCALE=0.75",
        "ALPACA_EXECUTION_RISK_HIGH_BUY_SCALE=0.50",
        "ALPACA_EXECUTION_SCORECARD_THROTTLE=1",
        "ALPACA_EXECUTION_SCORECARD_MAX_AGE_HOURS=72",
        "ALPACA_EXECUTION_SCORECARD_FAIL_BUY_SCALE=0.75",
        "ALPACA_EXECUTION_SCORECARD_SEVERE_SCORE=50",
        "ALPACA_EXECUTION_SCORECARD_SEVERE_BUY_SCALE=0.50",
        "EXECUTION_SCORECARD_MAX_AVG_SLIPPAGE_BPS=10",
        "EXECUTION_SCORECARD_MAX_BAD_SLIPPAGE_RATE=0.60",
        "EXECUTION_SCORECARD_MIN_FILL_RATE=0.80",
        "EXECUTION_SCORECARD_MAX_SKIPPED_RATE=0.35",
        "EXECUTION_SCORECARD_MAX_ADVERSE_15M_RATE=0.60",
        "EXECUTION_SCORECARD_MAX_ADVERSE_60M_RATE=0.70",
        "EXECUTION_SCORECARD_LOOKBACK_DAYS=30",
        "EXECUTION_SCORECARD_WARN_AVG_SLIPPAGE_BPS=5",
        "EXECUTION_SCORECARD_WARN_BAD_SLIPPAGE_RATE=0.40",
        "EXECUTION_SCORECARD_MIN_REBALANCE_FILLS=10",
        "EXECUTION_SCORECARD_MIN_REBALANCE_SESSIONS=3",
        "BROKER_TRUTH_QTY_TOLERANCE=0.001",
        "BROKER_TRUTH_WEIGHT_TOLERANCE=0.02",
        "BROKER_TRUTH_REQUIRE_LIVE_ORDERS=0",
        "ALPACA_BROKER_TRUTH_GATE=1",
        "ALPACA_BROKER_TRUTH_BLOCK_BUYS_ON_FAIL=1",
        "ALPACA_REQUIRE_QUOTE_FOR_SUBMIT=1",
        "MAX_SPREAD_PCT_ETF=0.001",
        "MAX_SPREAD_PCT_OVERLAY=0.005",
        "ALPACA_ALLOW_CLOSED_MARKET_QUEUE=0",
        "ALPACA_SKIP_BUYS_UNTIL_SELLS_FILLED=1",
        "ALPACA_SKIP_BUYS_WHEN_CASH_BELOW=0",
        "ALPACA_BUY_CASH_BUFFER_PCT=0.005",
        "ALPACA_BUY_CASH_BUFFER_DOLLARS=0",
        "ALPACA_SELL_FILL_WAIT_SECONDS=20",
        "ALPACA_SELL_FILL_POLL_SECONDS=2",
        "ALPACA_BUY_FILL_WAIT_SECONDS=20",
        "ALPACA_BUY_FILL_POLL_SECONDS=2",
        "ALPACA_MARGIN_WARN_GROSS=1.02",
        "TQQQ_FAST_DD_FAIL_CLOSED=1",
        "SPREAD_GUARD_ALERT_TTL_HOURS=20",
        "GUARD_REPLACE_STALE_SELLS=1",
        "GUARD_REPLACE_STALE_SELL_OFFSET_BPS=20",
        "GUARD_BROKER_TRUTH_REFRESH=1",
    ]
    for line in required_lines:
        assert line in workflow


def test_daily_workflow_never_retries_the_trading_pipeline():
    """A partial order failure must stay red instead of being hidden by retry."""
    workflow = Path(".github/workflows/daily_paper_trading.yml").read_text(encoding="utf-8")

    # PLAIN ENGLISH: one failed run may already have changed the Alpaca account.
    # The workflow must stop and report that failure, not execute the whole trade
    # plan again and accidentally treat duplicate protection as success.
    assert workflow.count("python3 daily_run.py $FLAGS") == 1
    assert "retrying in 30s" not in workflow


def test_signal_latest_workflows_share_publish_lock():
    # PLAIN ENGLISH: both workflows force-push signals/latest. One shared
    # concurrency group makes GitHub queue them instead of letting them race.
    daily_workflow = Path(".github/workflows/daily_paper_trading.yml").read_text(encoding="utf-8")
    shadow_workflow = Path(".github/workflows/shadow_paper_journal.yml").read_text(encoding="utf-8")

    assert "group: signals-latest-publisher" in daily_workflow
    assert "group: signals-latest-publisher" in shadow_workflow
    assert "cancel-in-progress: false" in daily_workflow
    assert "cancel-in-progress: false" in shadow_workflow


def test_shadow_workflow_requires_safe_defaults_and_verified_evidence():
    """The shadow job stays strict and publishes only matching fresh evidence."""
    workflow = Path(".github/workflows/shadow_paper_journal.yml").read_text(encoding="utf-8")
    assert workflow.count('default: "false"') >= 2
    assert "validation_bundle_valid" in workflow
    assert "validation_fingerprint_match" in workflow
    assert "Paper-vs-shadow comparison is stale" in workflow
    # The journal observes a strategy; it never submits or mutates an order.
    forbidden = ["--submit", "--reconcile", "execution_guard.py", "cancel_order", "replace_order"]
    assert all(token not in workflow for token in forbidden)


def test_alpaca_only_still_generates_shared_signal():
    steps = daily_run.build_steps(
        skip_refresh=True,
        run_alpaca=True,
    )
    names = _names(steps)
    assert names.count("core_satellite_signal") == 1
    assert names.index("factor_data_health") < names.index("core_satellite_signal")
    assert names.index("core_satellite_signal") < names.index("alpaca_submit")
    assert names.index("alpaca_execution_guard") < names.index("broker_truth")
    assert names.index("broker_truth") < names.index("alpaca_paper_health")
    assert names.index("alpaca_paper_health") < names.index("alpaca_execution_scorecard")
    assert names.index("alpaca_execution_scorecard") < names.index("alpaca_gauntlet")


def test_skip_factor_refresh_keeps_etf_refresh_and_signal():
    steps = daily_run.build_steps(
        skip_refresh=False,
        skip_factor_refresh=True,
        run_alpaca=True,
    )
    names = _names(steps)
    assert "refresh_etf_data" in names
    assert "refresh_factor_data" not in names
    assert "refresh_feature_quality" not in names
    assert "factor_data_health" in names
    assert names.index("refresh_etf_data") < names.index("core_satellite_signal")
    assert names.index("factor_data_health") < names.index("core_satellite_signal")
    assert names.index("core_satellite_signal") < names.index("alpaca_submit")


def test_health_only_uses_synced_signal_without_order_submission():
    steps = daily_run.build_steps(
        skip_refresh=False,
        run_alpaca=True,
        health_only=True,
    )
    names = _names(steps)
    assert "refresh_etf_data" not in names
    assert "refresh_factor_data" not in names
    assert "core_satellite_signal" not in names
    assert "alpaca_submit" not in names
    assert "alpaca_execution_guard" not in names
    assert names == [
        "fill_monitor",
        "broker_health",
        "alpaca_status",
        "alpaca_reconcile",
        "broker_truth",
        "alpaca_paper_health",
        "alpaca_execution_scorecard",
        "execution_cost_calibration",
        "alpaca_gauntlet",
        "paper_validation_epoch",
        "regime_monitor",
        "monitor_heartbeat",
        "log_cleanup",
    ]


def test_factor_data_health_failure_blocks_signal_and_orders(monkeypatch):
    steps = daily_run.build_steps(
        skip_refresh=True,
        run_alpaca=True,
    )
    called: list[str] = []

    def fake_run_step(name, cmd, description, dry_run=False, timeout=300, env=None):
        called.append(name)
        if name == "factor_data_health":
            return {"name": name, "status": "failed", "elapsed": 0.1}
        return {"name": name, "status": "ok", "elapsed": 0.1}

    monkeypatch.setattr(daily_run, "run_step", fake_run_step)
    results = daily_run.run_steps(steps, dry_run=False, timeout=1)
    assert called == [
        "fill_monitor", "broker_health", "factor_data_health", "alpaca_reconcile",
        "alpaca_execution_guard", "broker_truth", "alpaca_paper_health",
        "alpaca_execution_scorecard", "execution_cost_calibration",
        "alpaca_gauntlet", "paper_validation_epoch", "monitor_heartbeat",
        "log_cleanup",
    ]
    assert next(r for r in results if r["name"] == "core_satellite_signal")["status"] == "blocked"
    assert next(r for r in results if r["name"] == "drift_monitor")["status"] == "blocked"
    assert next(r for r in results if r["name"] == "alpaca_submit")["blocked_by"] == "factor_data_health"
    assert next(r for r in results if r["name"] == "monitor_heartbeat")["upstream_blocked_by"] == "factor_data_health"


def test_core_signal_failure_blocks_broker_steps(monkeypatch):
    steps = [
        daily_run.CORE_SATELLITE_SIGNAL_STEP,
        *daily_run.ALPACA_STEPS[:1],
    ]
    called: list[str] = []

    def fake_run_step(name, cmd, description, dry_run=False, timeout=300, env=None):
        called.append(name)
        if name == "core_satellite_signal":
            return {"name": name, "status": "failed", "elapsed": 0.1}
        return {"name": name, "status": "ok", "elapsed": 0.1}

    monkeypatch.setattr(daily_run, "run_step", fake_run_step)
    results = daily_run.run_steps(steps, dry_run=False, timeout=1)
    assert called == ["core_satellite_signal"]
    assert [r["status"] for r in results] == ["failed", "blocked"]
    assert results[1]["blocked_by"] == "core_satellite_signal"


def test_refresh_failure_blocks_all_later_steps(monkeypatch):
    steps = daily_run.build_steps(
        skip_refresh=False,
        run_alpaca=True,
    )
    called: list[str] = []

    def fake_run_step(name, cmd, description, dry_run=False, timeout=300, env=None):
        called.append(name)
        if name == "refresh_factor_data":
            return {"name": name, "status": "error", "elapsed": 0.1}
        return {"name": name, "status": "ok", "elapsed": 0.1}

    monkeypatch.setattr(daily_run, "run_step", fake_run_step)
    results = daily_run.run_steps(steps, dry_run=False, timeout=1)
    assert called == [
        "refresh_etf_data", "refresh_factor_data", "fill_monitor",
        "alpaca_reconcile", "alpaca_execution_guard", "broker_truth",
        "alpaca_paper_health", "alpaca_execution_scorecard",
        "execution_cost_calibration", "alpaca_gauntlet",
        "paper_validation_epoch", "monitor_heartbeat", "log_cleanup",
    ]
    blocked = [r for r in results if r["status"] == "blocked"]
    assert blocked
    assert all(r["blocked_by"] == "refresh_factor_data" for r in blocked)
    assert next(r for r in results if r["name"] == "core_satellite_signal")["status"] == "blocked"
    assert next(r for r in results if r["name"] == "alpaca_submit")["status"] == "blocked"
    assert next(r for r in results if r["name"] == "fill_monitor")["upstream_blocked_by"] == "refresh_factor_data"


def test_refresh_factor_data_uses_longer_step_timeout(monkeypatch):
    captured: dict[str, int] = {}
    step = next(step for step in daily_run.DATA_REFRESH_STEPS if step.name == "refresh_factor_data")

    def fake_run_step(name, cmd, description, dry_run=False, timeout=300, env=None):
        captured[name] = timeout
        return {"name": name, "status": "ok", "elapsed": 0.1}

    monkeypatch.setattr(daily_run, "run_step", fake_run_step)
    daily_run.run_steps([step], dry_run=False, timeout=1)

    assert captured["refresh_factor_data"] == 1800


def test_run_step_timeout_stops_quiet_process():
    start = time.monotonic()
    result = daily_run.run_step(
        "quiet_timeout",
        [sys.executable, "-c", "import time; time.sleep(3)"],
        "Quiet process should still time out",
        timeout=1,
    )

    assert result["status"] == "timeout"
    assert time.monotonic() - start < 2.5


def test_dry_run_skipped_critical_steps_do_not_block(monkeypatch):
    steps = [
        daily_run.DATA_REFRESH_STEPS[0],
        daily_run.CORE_SATELLITE_SIGNAL_STEP,
        daily_run.ALPACA_STEPS[0],
    ]

    def fake_run_step(name, cmd, description, dry_run=False, timeout=300, env=None):
        return {"name": name, "status": "skipped", "elapsed": 0.0}

    monkeypatch.setattr(daily_run, "run_step", fake_run_step)
    results = daily_run.run_steps(steps, dry_run=True, timeout=1)
    assert [r["status"] for r in results] == ["skipped", "skipped", "skipped"]


def test_startup_stubs_use_signal_dir_and_refresh_existing_fill_file(tmp_path, monkeypatch, capsys):
    signal_dir = tmp_path / "signals_abs"
    log_dir = tmp_path / "logs_abs"
    signal_dir.mkdir()
    old_fill = signal_dir / "fill_monitor.json"
    old_fill.write_text(json.dumps({"status": "stale"}))

    monkeypatch.setattr(daily_run, "SIGNAL_DIR", str(signal_dir))
    monkeypatch.setattr(daily_run, "LOGS", log_dir)

    now = datetime(2026, 5, 19, 9, 35)
    daily_run._write_startup_stubs(now, [daily_run.FILL_MONITOR_STEP])

    fill_payload = json.loads(old_fill.read_text())
    assert fill_payload["status"] == "pending"
    assert fill_payload["stub_path"] == str(old_fill)
    assert fill_payload["stub_existed_before"] is True

    run_stub = log_dir / "daily_run_20260519.json"
    assert json.loads(run_stub.read_text())["status"] == "running"
    assert f"Wrote fill_monitor stub → {old_fill}" in capsys.readouterr().out


def test_sync_latest_github_signals_fetches_branch_and_copies_files(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_run, "PROJECT_ROOT", tmp_path)
    calls: list[list[str]] = []

    class FakeProc:
        def __init__(self, returncode=0, stdout=b"", stderr=b""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    remote_files = "\n".join([
        "signals/core_satellite_alpha_signal.csv",
        "signals/fill_monitor.json",
        "logs/etf_data_health.json",
        "logs/daily_run_20260515.json",
        "logs/daily_run_20260516.json",
        "logs/daily_run_20260517.json",
        "logs/daily_run_20260518.json",
        "logs/daily_run_20260519.json",
        "logs/daily_run_20260520.json",
    ]).encode()

    def fake_git(args):
        calls.append(args)
        if args[0] == "fetch":
            return FakeProc()
        if args[:3] == ["ls-tree", "-r", "--name-only"]:
            return FakeProc(stdout=remote_files)
        if args[0] == "show":
            rel_path = args[1].split(":", 1)[1]
            if rel_path in {
                "signals/core_satellite_alpha_signal.csv",
                "signals/fill_monitor.json",
                "logs/etf_data_health.json",
                "logs/daily_run_20260516.json",
                "logs/daily_run_20260517.json",
                "logs/daily_run_20260518.json",
                "logs/daily_run_20260519.json",
                "logs/daily_run_20260520.json",
            }:
                return FakeProc(stdout=f"payload:{rel_path}".encode())
            return FakeProc(returncode=128, stderr=b"missing")
        raise AssertionError(args)

    monkeypatch.setattr(daily_run, "_git_output", fake_git)

    result = daily_run.sync_latest_github_signals(fetch=True)

    assert result["status"] == "ok"
    assert (tmp_path / "signals/core_satellite_alpha_signal.csv").read_text() == (
        "payload:signals/core_satellite_alpha_signal.csv"
    )
    assert (tmp_path / "signals/fill_monitor.json").read_text() == "payload:signals/fill_monitor.json"
    assert (tmp_path / "logs/etf_data_health.json").read_text() == "payload:logs/etf_data_health.json"
    assert not (tmp_path / "logs/daily_run_20260515.json").exists()
    assert (tmp_path / "logs/daily_run_20260520.json").exists()
    assert any(call[0] == "fetch" for call in calls)
