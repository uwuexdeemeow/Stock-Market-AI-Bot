from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_performance_audit import (
    DailyAuditPath,
    audited_metrics,
    candidate_gate,
    daily_mark_to_market,
    shadow_candidate_configs,
    stitch_daily_paths,
)


def _prices(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    """Make a tiny raw-price history for deterministic audit tests."""
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    return pd.DataFrame({"Open": opens or closes, "Close": closes}, index=dates)


def _trade(exit_date: str = "2020-01-07", delay: int = 0) -> pd.DataFrame:
    """Make one saved target-weight row like the strategy emits."""
    return pd.DataFrame([{
        "date": "2020-01-01",
        "exit_date": exit_date,
        "entry_delay_days": delay,
        "core_gross": 0.0,
        "core_weights_json": "{}",
        "overlay_weights_json": json.dumps({"AAA": 1.0}),
    }])


def test_daily_drawdown_hidden_between_rebalances_is_visible():
    """A mid-hold crash must appear even when entry and exit both look fine."""
    aaa = _prices([100.0, 100.0, 50.0, 100.0, 100.0], opens=[100.0] * 5)
    qqq = _prices([100.0] * 5)
    result = daily_mark_to_market(_trade(), lambda _: aaa, qqq, cost_bps=0.0)
    metrics = audited_metrics(result, qqq, bootstrap_samples=10)
    assert metrics["strategy_total_return_pct"] == pytest.approx(0.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(-50.0)


def test_signal_day_prices_cannot_change_next_open_audit_return():
    """The audit enters after signal close, preventing signal-day lookahead."""
    original = _prices([10.0, 100.0, 110.0, 110.0, 110.0], opens=[10.0, 100.0, 110.0, 110.0, 110.0])
    changed = original.copy()
    changed.iloc[0] = [9_999.0, 1.0]
    qqq = _prices([100.0] * 5)
    first = daily_mark_to_market(_trade(), lambda _: original, qqq, cost_bps=0.0)
    second = daily_mark_to_market(_trade(), lambda _: changed, qqq, cost_bps=0.0)
    pd.testing.assert_series_equal(first.returns, second.returns)


def test_one_day_delay_waits_one_additional_market_session():
    aaa = _prices([10.0, 100.0, 200.0, 220.0, 220.0], opens=[10.0, 100.0, 200.0, 220.0, 220.0])
    qqq = _prices([100.0] * 5)
    normal = daily_mark_to_market(_trade(delay=0), lambda _: aaa, qqq, cost_bps=0.0)
    delayed = daily_mark_to_market(_trade(delay=1), lambda _: aaa, qqq, cost_bps=0.0)
    assert normal.interval_rows[0]["entry_date"] == "2020-01-02"
    assert delayed.interval_rows[0]["entry_date"] == "2020-01-03"


def test_full_turnover_and_cost_include_initial_core_and_overlay_targets():
    prices = _prices([100.0] * 5)
    trades = _trade()
    trades.loc[0, "core_gross"] = 0.5
    trades.loc[0, "core_weights_json"] = json.dumps({"QQQ": 1.0})
    result = daily_mark_to_market(trades, lambda _: prices, prices, cost_bps=10.0)
    assert result.turnover == pytest.approx(1.5)
    assert result.cost_paid_fraction == pytest.approx(0.0015)


def test_fold_boundary_carries_prior_target_for_turnover():
    """A new outer year does not pretend the portfolio first liquidated to cash."""
    dates = pd.bdate_range("2020-01-01", periods=12)
    prices = pd.DataFrame({"Open": [100.0] * 12, "Close": [100.0] * 12}, index=dates)
    trades = pd.DataFrame([
        {
            "date": "2020-01-01",
            "exit_date": "2020-01-07",
            "core_gross": 0.0,
            "core_weights_json": "{}",
            "overlay_weights_json": json.dumps({"AAA": 1.0}),
        },
        {
            "date": "2020-01-07",
            "exit_date": "2020-01-15",
            "core_gross": 0.0,
            "core_weights_json": "{}",
            "overlay_weights_json": json.dumps({"AAA": 1.0}),
        },
    ])

    result = daily_mark_to_market(trades, lambda _: prices, prices, cost_bps=10.0)

    assert result.turnover == pytest.approx(1.0)
    assert result.interval_rows[1]["full_turnover"] == pytest.approx(0.0)


def test_stitch_rejects_overlapping_outer_fold_days():
    index = pd.bdate_range("2020-01-01", periods=2)
    path = DailyAuditPath(pd.Series([0.0, 0.0], index=index), pd.Series([1.0, 1.0], index=index), 0.0, 1, 0.0, [])
    with pytest.raises(ValueError, match="overlap"):
        stitch_daily_paths([path, path])


def test_irregular_intervals_use_elapsed_calendar_time_for_cagr():
    index = pd.DatetimeIndex(["2020-01-02", "2022-01-02"])
    returns = pd.Series([0.0, 1.0], index=index)
    path = DailyAuditPath(returns, 100_000 * (1 + returns).cumprod(), 0.0, 1, 0.0, [])
    # Include one earlier benchmark close so QQQ has a return on the first
    # strategy day, just as the full raw history does in production.
    qqq_index = pd.DatetimeIndex(["2019-12-31", *index])
    qqq = pd.DataFrame({"Open": [100.0] * 3, "Close": [100.0] * 3}, index=qqq_index)
    metrics = audited_metrics(path, qqq, bootstrap_samples=10)
    assert metrics["strategy_cagr_pct"] == pytest.approx(41.42, abs=0.2)


def test_candidate_gate_requires_every_predeclared_guardrail():
    baseline = {
        "net": {"information_ratio_vs_qqq": 0.5, "max_drawdown_pct": -10.0, "annualized_turnover_pct": 100.0},
        "cost_25bps": {"information_ratio_vs_qqq": 0.3},
        "delay_1d": {"information_ratio_vs_qqq": 0.2},
    }
    candidate = {
        "net": {"information_ratio_vs_qqq": 0.61, "net_alpha_vs_qqq_pct": 1.0, "max_drawdown_pct": -11.0, "annualized_turnover_pct": 99.0},
        "recent_three_year": {"net_alpha_vs_qqq_pct": 0.5},
        "cost_25bps": {"information_ratio_vs_qqq": 0.31},
        "delay_1d": {"information_ratio_vs_qqq": 0.21},
    }
    assert candidate_gate(candidate, baseline, evidence_gates_pass=True)["passed"] is True
    assert candidate_gate(candidate, baseline, evidence_gates_pass=False)["passed"] is False


def test_shadow_variants_change_one_dimension_and_keep_tqqq_off():
    active = {
        "holding_days": 20,
        "overlay_gross": 0.5,
        "shape": "top3",
        "weighting": "sticky_score",
        "tqqq_weight": 0.0,
        "regime_preset": {
            "risk_on": {"overlay_gross": 0.5},
            "neutral": {"overlay_gross": 0.5},
            "risk_off": {"overlay_gross": 0.25},
        },
    }
    variants = shadow_candidate_configs(active)
    assert variants["top_five"]["shape"] == "top5"
    assert variants["overlay_40pct"]["regime_preset"]["risk_on"]["overlay_gross"] == 0.4
    assert variants["overlay_40pct"]["regime_preset"]["risk_off"]["overlay_gross"] == 0.25
    assert variants["sticky_blend_80pct"]["sticky_blend"] == 0.8
    assert all(float(config.get("tqqq_weight", 0.0)) == 0.0 for config in variants.values())
