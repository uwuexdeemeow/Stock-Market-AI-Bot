"""Tests for v_rule_signal.py and the rule V paper mode.

PLAIN ENGLISH: the paper account now trades rule V.  These tests check that
the daily signal (1) uses exactly the tested weight, (2) decides on the last
completed session of each month only, (3) is read the same way by every
safety check, and (4) switches the daily run and watchdog to the V steps.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import broker_truth
import daily_run
import v_rule_signal as vs
import workflow_watchdog
from signal_freshness import extract_signal_weights, validate_signal_sanity

spec = importlib.util.spec_from_file_location(
    "index_risk_check", Path("research_evidence/index_risk_20261016/index_risk_check.py"))
ir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir)


def _closes(end: str, daily_vol: float = 0.02, seed: int = 3) -> pd.Series:
    """Random-walk closes on NYSE sessions up to `end`."""
    days = vs._nyse_sessions(pd.Timestamp("2026-01-02"), pd.Timestamp(end))
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, daily_vol, len(days))), index=days)


def _utc(text: str) -> datetime:
    return pd.Timestamp(text, tz="America/New_York").tz_convert("UTC").to_pydatetime()


def test_month_end_schedule():
    sched = vs.month_end_schedule(pd.Timestamp("2026-10-20"))
    assert sched["last"] == pd.Timestamp("2026-09-30")
    assert sched["next"] == pd.Timestamp("2026-10-30")
    assert sched["scheduled_today"] is False
    on_day = vs.month_end_schedule(pd.Timestamp("2026-10-30"))
    assert on_day["last"] == pd.Timestamp("2026-10-30") and on_day["scheduled_today"] is True
    assert on_day["next"] == pd.Timestamp("2026-11-30")


def test_signal_day_weight_matches_tested_rule():
    qqq = _closes("2026-10-30")
    # 11/2 10:00 New York: 10/30 is the latest completed session = signal day.
    row = vs.build_signal_row(qqq, qqq * 0 + 91.0, now=_utc("2026-11-02 10:00"))
    assert row["paper_ready"] and row["scheduled_rebalance_today"]
    assert row["last_scheduled_rebalance_date"] == "2026-10-30"
    expected = ir.vol_weight(qqq, pd.Timestamp("2026-10-30"))
    assert row["target_qqq_weight"] == pytest.approx(expected, abs=1e-6)
    assert row["target_qqq_weight"] + row["target_bil_weight"] == pytest.approx(1.0)


def test_partial_bar_on_month_end_is_not_used():
    # On 10/30 at 10:00 the provider may already hold a partial 10/30 bar.
    qqq = _closes("2026-10-30")
    row = vs.build_signal_row(qqq, qqq * 0 + 91.0, now=_utc("2026-10-30 10:00"))
    assert row["latest_factor_date"] == "2026-10-29"
    assert row["last_scheduled_rebalance_date"] == "2026-09-30"
    assert row["scheduled_rebalance_today"] is False


def test_catch_up_day_keeps_signal_day_weight():
    qqq = _closes("2026-11-03", daily_vol=0.03)
    row = vs.build_signal_row(qqq, qqq * 0 + 91.0, now=_utc("2026-11-04 10:00"))
    assert row["paper_ready"] and not row["scheduled_rebalance_today"]
    assert row["last_scheduled_rebalance_date"] == "2026-10-30"
    expected = ir.vol_weight(qqq, pd.Timestamp("2026-10-30"))
    assert row["target_qqq_weight"] == pytest.approx(expected, abs=1e-6)


def test_stale_prices_block_the_signal():
    qqq = _closes("2026-10-28")
    row = vs.build_signal_row(qqq, qqq * 0 + 91.0, now=_utc("2026-11-02 10:00"))
    assert not row["paper_ready"] and "qqq_prices_not_current" in row["reason"]
    bil = _closes("2026-10-29")
    row = vs.build_signal_row(_closes("2026-10-30"), bil, now=_utc("2026-11-02 10:00"))
    assert not row["paper_ready"] and "bil_prices_not_current" in row["reason"]
    assert row["target_qqq_weight"] == 0.0 and row["target_bil_weight"] == 0.0


def test_every_check_reads_the_same_targets(tmp_path):
    qqq = _closes("2026-10-30", daily_vol=0.03)
    row = vs.build_signal_row(qqq, qqq * 0 + 91.0, now=_utc("2026-11-02 10:00"))
    assert 0.0 < row["target_qqq_weight"] < 1.0
    signal = pd.Series(row)
    weights = extract_signal_weights(signal)
    assert weights == pytest.approx({"QQQ": row["target_qqq_weight"], "BIL": row["target_bil_weight"]})
    ok, issues = validate_signal_sanity(signal)
    assert ok, issues
    path = vs.write_signal(row, tmp_path / "signal.csv")
    truth_weights, _ = broker_truth.load_target_weights(path)
    nonzero = {k: v for k, v in truth_weights.items() if abs(v) > 1e-9}
    assert nonzero == pytest.approx(weights)


def test_daily_run_uses_v_steps_only_in_v_mode():
    v_names = [s.name for s in daily_run.build_steps(skip_refresh=False, strategy="v_rule")]
    assert "v_rule_signal" in v_names and "core_satellite_signal" not in v_names
    for core_only in ("refresh_factor_data", "refresh_feature_quality", "factor_data_health", "drift_monitor"):
        assert core_only not in v_names
    assert "refresh_etf_data" in v_names and "alpaca_submit" in v_names
    core_names = [s.name for s in daily_run.build_steps(skip_refresh=False, strategy="core_satellite_alpha")]
    assert "core_satellite_signal" in core_names and "v_rule_signal" not in core_names


def test_watchdog_skips_paused_core_alpha_jobs():
    assert set(workflow_watchdog.active_workflows("v_rule")) == {"daily_paper", "execution_quality"}
    assert set(workflow_watchdog.active_workflows("core_satellite_alpha")) == set(workflow_watchdog.WORKFLOWS)


def test_workflow_and_watchdog_modes_agree():
    daily = Path(".github/workflows/daily_paper_trading.yml").read_text(encoding="utf-8")
    watchdog = Path(".github/workflows/independent_workflow_watchdog.yml").read_text(encoding="utf-8")
    assert "PAPER_MODE: v_rule" in daily
    assert "PAPER_MODE_STRATEGY: v_rule" in watchdog


# ── Forward Test Rule FV measures (forward_test_monitor.py --rule v) ─────────

import forward_test_monitor as ftm


def _ohlc(sessions: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.015, len(sessions))), index=sessions)
    return pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]), "Close": close})


def test_fv_rebalance_count_and_signal_days():
    sessions = vs._nyse_sessions(pd.Timestamp("2026-09-01"), pd.Timestamp("2027-03-31"))
    days = ftm.month_end_signal_days(sessions)
    assert days[0] == pd.Timestamp("2026-10-30")
    assert ftm.v_rebalances_done(days, sessions, "2026-11-01") == 0
    assert ftm.v_rebalances_done(days, sessions, "2026-11-02") == 1
    assert ftm.v_rebalances_done(days, sessions, "2027-01-04") == 3


def test_fv_account_tracking_qqq_matches_qqq_numbers():
    sessions = vs._nyse_sessions(pd.Timestamp("2026-01-02"), pd.Timestamp("2027-02-26"))
    qqq, bil = _ohlc(sessions, 1), _ohlc(sessions, 2) * 0 + 91.0
    live = qqq.loc["2026-10-30":, "Close"]
    equity = pd.DataFrame({"date": live.index.strftime("%Y-%m-%d"), "equity": 1000 * live.to_numpy()})
    out = ftm.v_measures(equity, qqq, bil, sessions, "2027-02-26")
    assert out["status"] == "ok" and out["base_day"] == "2026-10-30"
    assert out["account"]["max_drawdown_pct"] == pytest.approx(out["qqq"]["max_drawdown_pct"])
    assert out["account"]["sharpe"] == pytest.approx(out["qqq"]["sharpe"])
    assert [g["month"] for g in out["v4_monthly"]] == ["2026-11", "2026-12", "2027-01", "2027-02"]


def test_fv_weighted_cost():
    fills = pd.DataFrame({
        "submitted_at": ["2026-11-02T14:40:00+00:00", "2026-11-02T14:41:00+00:00"],
        "ticker": ["QQQ", "MU"], "side": ["buy", "sell"],
        "quantity": [100, 10], "filled_avg_price": [101.0, 99.0],
    })
    opens = {("QQQ", pd.Timestamp("2026-11-02")): 100.0, ("MU", pd.Timestamp("2026-11-02")): 100.0}
    out = ftm.weighted_fill_cost(fills, opens)
    # QQQ buy +100 bps on $10,100; MU sell 1% below Open = +100 bps on $990.
    assert out["fills"] == 2 and out["mean_bps"] == pytest.approx(100.0)
