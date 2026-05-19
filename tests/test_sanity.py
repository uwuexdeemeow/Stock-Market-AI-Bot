"""
test_sanity.py — Smoke tests runnable with `python -m pytest tests/`.

PLAIN ENGLISH:
These aren't exhaustive — they are a minimum bar. If any of these break,
don't trade. Every test should take < 1 second.
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from labels import forward_return, vol_normalized_return, triple_barrier
from risk_sizing import vol_target_size, fractional_kelly, position_size_with_stop
from execution_model import realistic_fill_price, commission, capacity_warning
from data_validation import validate_price_frame
from core_satellite_alpha import (
    _core_tickers_for_config,
    _notify_earnings_blackout_if_needed,
    _overlay_weights,
    _paper_signal_timestamp,
    _resolve_allocation,
    _scale_paper_targets_to_gross,
    _select_sticky_holdings,
    _top_count,
)
from robustness_scoring import (
    DEFAULT_TURNOVER_FREE_PCT,
    DEFAULT_TURNOVER_PENALTY_SPAN_PCT,
    add_cost_stress_approval_columns,
    robustness_score_components,
)
from paper_health import (
    _current_order_lifecycle,
    _drift_breakdown,
    _equity_sanity_checks,
    _execution_slippage,
    _go_live_scorecard,
    _open_position_attribution,
    _position_concentration,
    _stale_open_order_alerts,
)
import moomoo_paper_trading
from moomoo_paper_trading import (
    _find_broker_order_match,
    _order_status_bucket,
    build_submit_readiness_report,
    evaluate_broker_submit_preflight,
)
import paper_gauntlet
import daily_paper_check
import core_satellite_nested_walkforward as nested_wf
import publish_live_config_from_csv as manual_publish
from daily_paper_check import _after_fill_verdict, _prune_snapshots, _verdict
from refresh_etf_data import _validate_etf_frame
from config_health import _requirement_ok
from core_satellite_nested_walkforward import (
    BASE_REGIME,
    InnerFold,
    STRATEGIES,
    build_fold_splits,
    build_inner_folds,
    build_regime_preset_variant,
    iter_candidate_configs,
    select_config_from_inner_folds,
)
from core_satellite_alpha import MAX_GROSS_EXPOSURE, REGIME_PRESETS
from portfolio_manager import PortfolioRiskManager, ProposedTrade
from ranker_utils import compute_adaptive_factor_weights, load_adaptive_factor_weights
from feature_quality_diagnostic import (
    add_oriented_feature,
    add_sector_excess_return_columns,
    find_return_column,
    ic_decay_curve,
)


def _price_frame(n=100, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1e6, 5e6, n),
    }, index=idx)


def test_forward_return_is_leak_free():
    df = _price_frame()
    r = forward_return(df["Close"], horizon=5)
    # last 5 rows must be NaN — nothing to look forward to
    assert r.tail(5).isna().all()


def test_vol_normalized_scales_with_vol():
    df = _price_frame()
    r = vol_normalized_return(df["Close"], horizon=5, vol_window=20).dropna()
    assert np.isfinite(r).any()


def test_triple_barrier_outputs_in_set():
    df = _price_frame()
    lab = triple_barrier(df["Close"], df["High"], df["Low"], max_hold=5).dropna()
    assert set(lab.unique()).issubset({-1.0, 0.0, 1.0})


def test_risk_sizing_nonnegative():
    assert vol_target_size(100_000, 0.20) > 0
    assert vol_target_size(100_000, 0.0) == 0
    # fractional_kelly takes p_win (win probability), not an edge offset.
    # p_win <= 0.5 → no edge → returns 0.  p_win > 0.5 → returns positive fraction.
    assert fractional_kelly(0.40) == 0   # 40% win rate → below 50% → no bet
    assert fractional_kelly(0.50) == 0   # 50% win rate → coin flip → no bet
    assert 0 < fractional_kelly(0.60) < 1  # 60% win rate → positive edge → size up
    assert fractional_kelly(0.60) < fractional_kelly(0.70)  # more edge → larger bet
    assert position_size_with_stop(100_000, 100, 2.0) > 0


def test_execution_costs_sane():
    px = realistic_fill_price(100.0, 1_000, 1_000_000, "buy")
    assert px > 100  # buying costs more than mid
    assert commission(100) > 0
    assert capacity_warning(100_000, 1_000_000) is True


def _price_from_returns(values, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(100.0 * np.cumprod(1.0 + np.asarray(values, dtype=float)), index=idx)


def test_portfolio_correlation_blocks_recent_spike_not_diluted_by_60d():
    base = np.array([0.01, -0.006, 0.004, -0.003] * 30, dtype=float)
    left = base.copy()
    right = np.roll(base, 1)
    right[-20:] = left[-20:]
    manager = PortfolioRiskManager(max_pair_correlation=0.85)
    candidates = [
        ProposedTrade("AAPL", pd.Timestamp("2024-06-01"), "LONG", 0.9, 1.0, 0.10),
        ProposedTrade("MSFT", pd.Timestamp("2024-06-01"), "LONG", 0.8, 1.0, 0.10),
    ]

    approved = manager.approve_day(
        candidates,
        {"AAPL": _price_from_returns(left), "MSFT": _price_from_returns(right)},
        pd.Series([10_000.0], index=[pd.Timestamp("2024-06-01")]),
    )

    assert [trade.ticker for trade in approved] == ["AAPL"]


def test_portfolio_correlation_checks_existing_open_tickers():
    ret = np.array([0.005, -0.004, 0.006, -0.003] * 30, dtype=float)
    manager = PortfolioRiskManager(max_pair_correlation=0.85)
    candidates = [ProposedTrade("MSFT", pd.Timestamp("2024-06-01"), "LONG", 0.9, 1.0, 0.10)]

    approved = manager.approve_day(
        candidates,
        {"AAPL": _price_from_returns(ret), "MSFT": _price_from_returns(ret)},
        pd.Series([10_000.0], index=[pd.Timestamp("2024-06-01")]),
        current_gross=0.10,
        current_net=0.10,
        open_tickers={"AAPL"},
    )

    assert approved == []


def test_portfolio_correlation_insufficient_data_does_not_block():
    ret = np.array([0.005, -0.004] * 5, dtype=float)
    manager = PortfolioRiskManager(max_pair_correlation=0.85)
    candidates = [
        ProposedTrade("AAPL", pd.Timestamp("2024-01-15"), "LONG", 0.9, 1.0, 0.10),
        ProposedTrade("MSFT", pd.Timestamp("2024-01-15"), "LONG", 0.8, 1.0, 0.10),
    ]

    approved = manager.approve_day(
        candidates,
        {"AAPL": _price_from_returns(ret), "MSFT": _price_from_returns(ret)},
        pd.Series([10_000.0], index=[pd.Timestamp("2024-01-15")]),
    )

    assert [trade.ticker for trade in approved] == ["AAPL", "MSFT"]


def test_portfolio_correlation_vix_stress_floor_blocks_pairs():
    left = np.array([0.005, -0.004, 0.003, -0.002] * 25, dtype=float)
    right = np.array([-0.003, 0.002, -0.005, 0.004] * 25, dtype=float)
    manager = PortfolioRiskManager(max_pair_correlation=0.50)
    candidates = [
        ProposedTrade("AAPL", pd.Timestamp("2024-06-01"), "LONG", 0.9, 1.0, 0.10),
        ProposedTrade("MSFT", pd.Timestamp("2024-06-01"), "LONG", 0.8, 1.0, 0.10),
    ]

    approved = manager.approve_day(
        candidates,
        {"AAPL": _price_from_returns(left), "MSFT": _price_from_returns(right)},
        pd.Series([10_000.0], index=[pd.Timestamp("2024-06-01")]),
        vix_level=35.0,
    )

    assert [trade.ticker for trade in approved] == ["AAPL"]


def test_validation_rejects_bad():
    import pytest
    good = _price_frame()
    validate_price_frame(good, "TEST", max_lag_days=10_000)  # bypass freshness in tests

    bad = good.copy()
    bad.loc[bad.index[5], "Close"] = -1
    with pytest.raises(ValueError):
        validate_price_frame(bad, "TEST", max_lag_days=10_000)


def test_core_satellite_paper_scaling_caps_gross():
    overlay = pd.Series({"AMAT": 0.233333, "C": 0.138866, "MU": 0.327801})
    spy, qqq, tqqq, paper_overlay, raw_gross, scale, scaled = _scale_paper_targets_to_gross(
        target_spy=0.0,
        target_qqq=0.55,
        overlay=overlay,
        max_gross=1.0,
    )
    assert scaled is True
    assert round(raw_gross, 6) == 1.25
    assert round(scale, 6) == 0.8
    assert round(abs(spy) + abs(qqq) + abs(tqqq) + float(paper_overlay.abs().sum()), 6) == 1.0


def test_robustness_score_is_sharpe_minus_penalties():
    score = robustness_score_components({
        "sharpe": 1.2,
        "max_drawdown_pct": -35.0,
        "turnover_pct": 15_000.0,
        "subperiod_stability_pass": False,
        "year_alpha_concentration_pass": True,
    })
    assert score["drawdown_penalty"] == 0.4
    expected_turnover_penalty = round(
        (15_000.0 - DEFAULT_TURNOVER_FREE_PCT) / DEFAULT_TURNOVER_PENALTY_SPAN_PCT,
        4,
    )
    assert score["turnover_penalty"] == expected_turnover_penalty
    assert score["instability_penalty"] == 0.25
    assert score["robustness_score"] == round(1.2 - 0.4 - expected_turnover_penalty - 0.25, 4)


def test_adaptive_factor_weights_use_forward_return_and_floor(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=80)
    horizon = 5
    for i in range(12):
        target = np.array([0.001 * (i - 5.5) + 0.00001 * j for j in range(len(dates))])
        close = np.full(len(dates), 100.0)
        for j in range(len(dates) - horizon):
            close[j + horizon] = 100.0 * (1.0 + target[j])
        df = pd.DataFrame({
            "Open": np.full(len(dates), 100.0),
            "Close": close,
            "good_factor": target,
            "bad_factor": -target,
        }, index=dates)
        df.to_parquet(tmp_path / f"T{i:02d}.parquet")

    weights = compute_adaptive_factor_weights(
        str(tmp_path),
        ["good_factor", "bad_factor"],
        lookback_days=60,
        halflife=20,
        floor=0.05,
        target_horizon_days=horizon,
        output_path=str(tmp_path / "weights.json"),
    )

    assert round(sum(weights.values()), 6) == 1.0
    assert weights["good_factor"] > weights["bad_factor"]
    assert weights["bad_factor"] >= 0.05
    payload = json.loads((tmp_path / "weights.json").read_text())
    assert payload["target_horizon_days"] == horizon
    assert payload["target_source"] == "open_next_to_close_horizon"
    assert payload["weight_status"] == "adaptive"


def test_adaptive_factor_weight_loader_validates_schema_and_staleness(tmp_path):
    fallback = {"a": 0.7, "b": 0.3}
    path = tmp_path / "adaptive.json"
    path.write_text(json.dumps({
        "weights": {"a": 0.6, "b": 0.4},
        "computed_at": "2026-05-16T00:00:00",
    }))

    weights, meta = load_adaptive_factor_weights(
        str(path),
        fallback,
        ["a", "b"],
        max_age_days=7,
        now=pd.Timestamp("2026-05-17T00:00:00"),
        return_metadata=True,
    )
    assert weights == {"a": 0.6, "b": 0.4}
    assert meta["adaptive_weight_status"] == "adaptive"

    path.write_text(json.dumps({
        "weights": {"a": 1.0},
        "computed_at": "2026-05-16T00:00:00",
    }))
    weights, meta = load_adaptive_factor_weights(
        str(path),
        fallback,
        ["a", "b"],
        now=pd.Timestamp("2026-05-17T00:00:00"),
        return_metadata=True,
    )
    assert weights == fallback
    assert meta["adaptive_weight_status"] == "fallback"
    assert "missing_weights" in meta["adaptive_weight_reason"]

    path.write_text(json.dumps({
        "weights": {"a": 0.6, "b": 0.4},
        "computed_at": "2026-04-01T00:00:00",
    }))
    weights, meta = load_adaptive_factor_weights(
        str(path),
        fallback,
        ["a", "b"],
        max_age_days=7,
        now=pd.Timestamp("2026-05-17T00:00:00"),
        return_metadata=True,
    )
    assert weights == fallback
    assert "stale_weights" in meta["adaptive_weight_reason"]


def _cost_stress_grid(overrides=None):
    overrides = overrides or {}
    rows = []
    for cost in (2.0, 3.0, 5.0):
        row = {
            "config": "A",
            "cost_stress": cost,
            "row_gate_pass": True,
            "alpha_vs_spy_pct": 1.0,
            "alpha_vs_qqq_pct": 1.0,
            "alpha_vs_blend_pct": 1.0,
            "holdout_alpha_vs_qqq_pct": 1.0,
            "holdout_alpha_vs_blend_pct": 1.0,
        }
        row.update(overrides.get(cost, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def test_cost_stress_approval_passes_complete_positive_group():
    out = add_cost_stress_approval_columns(
        _cost_stress_grid(),
        key_cols=["config"],
        required_costs=(2.0, 3.0, 5.0),
        row_gate_col="row_gate_pass",
    )
    assert out["robust_cost_stress_pass"].all()
    assert out["stress_has_required_costs"].all()
    assert out["stress_all_gates_pass"].all()


def test_cost_stress_approval_fails_missing_level():
    df = _cost_stress_grid()
    df = df[df["cost_stress"] != 5.0]
    out = add_cost_stress_approval_columns(
        df,
        key_cols=["config"],
        required_costs=(2.0, 3.0, 5.0),
        row_gate_col="row_gate_pass",
    )
    assert not bool(out["robust_cost_stress_pass"].iloc[0])
    assert not bool(out["stress_has_required_costs"].iloc[0])


def test_cost_stress_approval_fails_negative_alpha():
    out = add_cost_stress_approval_columns(
        _cost_stress_grid({5.0: {"alpha_vs_blend_pct": -0.1}}),
        key_cols=["config"],
        required_costs=(2.0, 3.0, 5.0),
        row_gate_col="row_gate_pass",
    )
    assert not bool(out["robust_cost_stress_pass"].iloc[0])
    assert out["stress_min_alpha_vs_blend_pct"].iloc[0] == -0.1


def test_cost_stress_approval_fails_hard_gate():
    out = add_cost_stress_approval_columns(
        _cost_stress_grid({3.0: {"row_gate_pass": False}}),
        key_cols=["config"],
        required_costs=(2.0, 3.0, 5.0),
        row_gate_col="row_gate_pass",
    )
    assert not bool(out["robust_cost_stress_pass"].iloc[0])
    assert not bool(out["stress_all_gates_pass"].iloc[0])


def test_nested_live_signal_config_omits_cost_stress():
    config = {
        "strategy": "core-alpha",
        "cost_stress": 5.0,
        "nested_params": {"holding_days": 10},
        "score_source": "factor_walkforward",
        "shape": "top10",
        "weighting": "sticky_vol_score",
    }
    live = nested_wf.live_signal_config(config)
    assert "cost_stress" not in live
    assert "nested_params" not in live
    assert live["shape"] == "top10"
    assert live["weighting"] == "sticky_vol_score"

    tqqq_live = nested_wf.live_signal_config({
        "strategy": "tqqq",
        "cost_stress": 5.0,
        "nested_params": {
            "holding_days": 10,
            "shape": "top15",
            "weighting": "sticky_vol_score",
            "tqqq_weight": 0.2,
        },
    })
    assert "cost_stress" not in tqqq_live
    assert tqqq_live["shape"] == "top15"
    assert tqqq_live["weighting"] == "sticky_vol_score"


def test_top15_and_sticky_vol_score_weights_are_supported():
    assert _top_count(62, "top5") == 5
    assert _top_count(62, "top10") == 10
    assert _top_count(62, "top15") == 15

    selected = pd.DataFrame({
        "ticker": ["LOWVOL", "HIGHVOL", "MIDVOL"],
        "_rank_score": [0.9, 0.9, 0.9],
        "hvol_20d": [0.10, 0.40, 0.20],
    })
    weights = _overlay_weights(selected, 0.30, "vol_score", max_single_name_weight=0.25)
    assert round(float(weights.sum()), 6) == 0.30
    assert weights["LOWVOL"] > weights["MIDVOL"] > weights["HIGHVOL"]


def test_nested_write_outputs_only_publishes_live_config_when_requested(tmp_path, monkeypatch):
    live_path = tmp_path / "core_satellite_live_configs.json"
    live_path.write_text(json.dumps({
        "approvals": {
            "core-alpha": {"approved": True},
            "tqqq": {"approved": True},
            "both": {"approved": True},
        },
        "approved_live_configs": {
            "tqqq": {"config": {"stale": True}},
            "both": {"config": {"stale": True}},
        },
    }))
    monkeypatch.setattr(nested_wf, "SIGNAL_DIR", str(tmp_path))
    monkeypatch.setattr(nested_wf, "LIVE_CONFIG_PATH", live_path)

    result = {
        "strategy": "core-alpha",
        "method": "nested_walk_forward",
        "folds": [],
        "live_config_approval": {"approved": False, "reasons": ["smoke"]},
    }

    nested_wf.write_outputs(result, output_prefix="smoke", publish_live_config=False)
    assert '"approved":true' in live_path.read_text().replace(" ", "")

    nested_wf.write_outputs(result, output_prefix="publish", publish_live_config=True)
    live = json.loads(live_path.read_text())
    assert live["source_json"].endswith("publish.json")
    assert live["approvals"]["core-alpha"] == {"approved": False, "reasons": ["smoke"]}
    assert "tqqq" not in live["approvals"]
    assert "both" not in live["approvals"]
    assert "tqqq" not in live["approved_live_configs"]
    assert "both" not in live["approved_live_configs"]


def test_nested_publish_decision_defaults_to_publish_only_for_full_runs():
    full = Namespace(
        publish_live_config=None,
        fast=False,
        stable_grid=False,
        recent_alpha_grid=False,
        max_folds=None,
        max_configs=None,
        start_year=None,
        end_year=None,
        max_specs=nested_wf.DEFAULT_MAX_SPECS,
        output_prefix=nested_wf.DEFAULT_OUTPUT_PREFIX,
    )
    assert nested_wf.live_config_publish_decision(full) == (True, "auto_full_nested_default")

    smoke = Namespace(**{**vars(full), "fast": True, "max_folds": 2})
    publish, reason = nested_wf.live_config_publish_decision(smoke)
    assert publish is False
    assert "--fast" in reason
    assert "--max-folds" in reason

    stable_grid = Namespace(**{**vars(full), "stable_grid": True})
    publish, reason = nested_wf.live_config_publish_decision(stable_grid)
    assert publish is False
    assert "--stable-grid" in reason

    stable_forced = Namespace(**{**vars(stable_grid), "publish_live_config": True})
    assert nested_wf.live_config_publish_decision(stable_forced) == (True, "forced_by_--publish-live-config")

    recent_alpha = Namespace(**{**vars(full), "recent_alpha_grid": True})
    publish, reason = nested_wf.live_config_publish_decision(recent_alpha)
    assert publish is False
    assert "--recent-alpha-grid" in reason

    forced = Namespace(**{**vars(smoke), "publish_live_config": True})
    assert nested_wf.live_config_publish_decision(forced) == (True, "forced_by_--publish-live-config")

    dry_run = Namespace(**{**vars(full), "publish_live_config": False})
    assert nested_wf.live_config_publish_decision(dry_run) == (False, "disabled_by_--no-publish-live-config")


def test_feature_quality_accepts_alpha_factor_return_columns():
    panel = pd.DataFrame({
        "date": list(pd.bdate_range("2024-01-01", periods=8)) * 6,
        "ticker": [f"T{i}" for i in range(6) for _ in range(8)],
        "feature": np.tile(np.arange(8, dtype=float), 6),
        "forward_return": np.tile(np.linspace(-0.02, 0.02, 8), 6),
        "forward_return_10d": np.tile(np.linspace(-0.01, 0.03, 8), 6),
        "forward_return_20d": np.tile(np.linspace(0.0, 0.04, 8), 6),
    })
    panel["sector"] = np.repeat(["TECH", "TECH", "TECH", "FIN", "FIN", "FIN"], 8)
    panel = add_sector_excess_return_columns(panel)
    assert find_return_column(panel) == "fwd_sector_excess_5"
    decay = ic_decay_curve(panel, "feature", horizons=[5, 10, 20])
    assert decay["horizons"]["5"]["available"] is True
    assert decay["horizons"]["10"]["available"] is True
    assert decay["horizons"]["20"]["available"] is True
    oriented = add_oriented_feature(panel, "feature", "low_is_good")
    assert panel[oriented].iloc[0] == -panel["feature"].iloc[0]


def test_core_satellite_allocation_accepts_regime_override_with_tqqq():
    config = {
        "regime_mode": "qqq_trend_switch_overlay70_core55",
        "regime_preset": {
            "risk_on": {"core_weights": {"SPY": 0.0, "QQQ": 0.8, "TQQQ": 0.2}, "core_gross": 0.55, "overlay_gross": 0.70},
            "neutral": {"core_weights": {"SPY": 0.25, "QQQ": 0.75, "TQQQ": 0.0}, "core_gross": 0.50, "overlay_gross": 0.60},
            "risk_off": {"core_weights": {"SPY": 0.6, "QQQ": 0.4, "TQQQ": 0.0}, "core_gross": 0.55, "overlay_gross": 0.25},
        },
    }
    regime, weights, core_gross, overlay_gross = _resolve_allocation(
        pd.Timestamp("2026-01-01"),
        {**config, "current_regime": "risk_on"},
        None,
    )
    assert regime == "risk_on"
    assert weights["TQQQ"] == 0.2
    assert core_gross == 0.55
    assert overlay_gross == 0.70
    assert "TQQQ" in _core_tickers_for_config(config)


def test_earnings_blackout_records_skipped_overlay_candidates():
    day = pd.DataFrame({
        "ticker": ["NVDA", "MSFT", "CAT"],
        "sector": ["XLK", "XLK", "XLI"],
        "factor_score": [0.95, 0.80, 0.70],
        "days_to_next_earnings": [2.0, 10.0, 20.0],
    })
    diagnostics = {}

    selected = _select_sticky_holdings(
        day,
        set(),
        score_col="factor_score",
        return_col=None,
        shape="top5",
        exit_rank_floor=0.0,
        max_per_sector=2,
        earnings_blackout_days=5,
        diagnostics=diagnostics,
    )

    assert "NVDA" not in set(selected["ticker"])
    assert diagnostics["earnings_blackout_skips"][0]["ticker"] == "NVDA"
    assert diagnostics["earnings_blackout_skips"][0]["days_to_next_earnings"] == 2.0
    assert diagnostics["earnings_blackout_skips"][0]["rank_score"] == 1.0


def test_earnings_blackout_notification_helper_only_fires_when_needed():
    calls = []

    def fake_notify(message, *, title):
        calls.append((message, title))

    assert _notify_earnings_blackout_if_needed({}, notifier=fake_notify) is False
    assert calls == []
    assert _notify_earnings_blackout_if_needed(
        {
            "earnings_blackout_skipped_count": 2,
            "earnings_blackout_skipped_tickers": "NVDA,MSFT",
        },
        notifier=fake_notify,
    ) is True
    assert calls == [("Earnings blackout skipped 2 overlay candidate(s): NVDA,MSFT", "Earnings Blackout")]


def test_nested_regime_variant_tunes_overlay_without_mutating_base():
    base = REGIME_PRESETS[BASE_REGIME]
    original_risk_on = dict(base["risk_on"]["core_weights"])
    variant = build_regime_preset_variant(
        base_preset=base,
        risk_on_overlay_gross=0.60,
        ma_window=75,
        high_vol=0.25,
        high_vol_mode="fixed",
        tqqq_weight=0.20,
    )
    assert base["risk_on"]["core_weights"] == original_risk_on
    assert variant["ma_window"] == 75
    assert variant["high_vol"] == 0.25
    assert variant["risk_on"]["overlay_gross"] == 0.60
    assert round(variant["risk_on"]["core_gross"] + variant["risk_on"]["overlay_gross"], 6) == MAX_GROSS_EXPOSURE
    assert variant["risk_on"]["core_weights"]["QQQ"] == 0.80
    assert variant["risk_on"]["core_weights"]["TQQQ"] == 0.20
    assert variant["neutral"]["core_weights"]["TQQQ"] == 0.0
    assert variant["risk_off"]["core_weights"]["TQQQ"] == 0.0


def test_nested_fold_splits_keep_outer_year_unseen():
    rows = []
    for year in range(2020, 2025):
        for dt in pd.bdate_range(f"{year}-01-01", periods=25):
            rows.append({"date": dt, "ticker": "TEST"})
    panel = pd.DataFrame(rows)
    splits = build_fold_splits(panel, min_train_years=3)
    outer_2024 = [split for split in splits if split.outer_year == 2024][0]
    assert outer_2024.train_end == pd.Timestamp("2023-12-31")
    assert outer_2024.inner_validation_year == 2023
    assert outer_2024.inner_train_end == pd.Timestamp("2022-12-31")
    assert outer_2024.outer_start == pd.Timestamp("2024-01-01")
    inner = build_inner_folds([2020, 2021, 2022, 2023], min_inner_train_years=2)
    assert [fold.validation_year for fold in inner] == [2022, 2023]
    assert inner[0].train_end == pd.Timestamp("2021-12-31")
    assert inner[-1].validation_end == pd.Timestamp("2023-12-31")


def test_nested_candidate_grid_includes_requested_tuning_dimensions():
    configs = iter_candidate_configs(
        holding_days=(5, 10),
        overlay_gross=(0.50, 0.70),
        ma_windows=(75,),
        high_vol_values=(0.25,),
        high_vol_modes=("fixed",),
        score_sources=("factor_walkforward", "regime_adaptive"),
        shapes=("top5", "top10", "top15"),
        weightings=("sticky_score", "sticky_vol_score"),
        tqqq_weights=(0.0, 0.20),
    )
    params = [config["nested_params"] for config in configs]
    assert {p["holding_days"] for p in params} == {5, 10}
    assert {p["risk_on_overlay_gross"] for p in params} == {0.50, 0.70}
    assert {p["score_source"] for p in params} == {"factor_walkforward", "regime_adaptive"}
    assert {p["shape"] for p in params} == {"top5", "top10", "top15"}
    assert {p["weighting"] for p in params} == {"sticky_score", "sticky_vol_score"}
    assert {p["tqqq_weight"] for p in params} == {0.0, 0.20}
    for config in configs:
        risk_on = config["regime_preset"]["risk_on"]
        assert risk_on["core_gross"] + risk_on["overlay_gross"] <= MAX_GROSS_EXPOSURE + 1e-9
    assert set(STRATEGIES) == {"core-alpha"}
    alpha_default = iter_candidate_configs(strategy="core-alpha", high_vol_values=(0.30,), max_configs=8)
    tqqq_alias_default = iter_candidate_configs(strategy="tqqq", high_vol_values=(0.30,), max_configs=8)
    assert alpha_default == tqqq_alias_default
    assert any(c["nested_params"]["tqqq_weight"] > 0.0 for c in alpha_default)
    assert {c["shape"] for c in alpha_default}.issubset({"top3", "top5", "top10", "top15"})
    assert {c["weighting"] for c in alpha_default}.issubset({"sticky_score", "risk_parity", "sticky_vol_score"})


def test_nested_stable_grid_pins_consensus_dimensions():
    configs = nested_wf.stable_grid_candidate_configs()

    assert len(configs) == 24
    params = [config["nested_params"] for config in configs]
    assert {p["holding_days"] for p in params} == {20}
    assert {p["overlay_gross"] for p in params} == {0.50}
    assert {p["risk_on_overlay_gross"] for p in params} == {0.50}
    assert {p["ma_window"] for p in params} == {100}
    assert {p["high_vol"] for p in params} == {0.30}
    assert {p["score_source"] for p in params} == {"regime_adaptive"}
    assert {p["weighting"] for p in params} == {"sticky_score"}
    assert {p["risk_control_mode"] for p in params} == {"defensive"}
    assert {p["shape"] for p in params} == {"top5", "top10", "top15"}
    assert {p["tqqq_weight"] for p in params} == {0.0, 0.10, 0.20, 0.30}
    assert {p["high_vol_mode"] for p in params} == {"fixed", "percentile"}


def test_nested_recent_alpha_grid_focuses_new_regime_dimensions():
    configs = nested_wf.recent_alpha_grid_candidate_configs()

    assert len(configs) == 48
    params = [config["nested_params"] for config in configs]
    assert {p["holding_days"] for p in params} == {20}
    assert {p["overlay_gross"] for p in params} == {0.50, 0.70}
    assert {p["ma_window"] for p in params} == {100}
    assert {p["high_vol"] for p in params} == {0.30}
    assert {p["score_source"] for p in params} == {"regime_adaptive"}
    assert {p["risk_control_mode"] for p in params} == {"off"}
    assert {p["shape"] for p in params} == {"top3", "top5", "top15"}
    assert {p["weighting"] for p in params} == {"sticky_score", "risk_parity"}
    assert {p["tqqq_weight"] for p in params} == {0.0, 0.10}
    assert {p["high_vol_mode"] for p in params} == {"fixed", "percentile"}


def test_nested_live_approval_uses_specific_config_frequency_and_turnover():
    result = {
        "valid": True,
        "strategy": "core-alpha",
        "fold_count": 14,
        "best_config_frequency": 0.30,
        "approved_config_frequency": 0.07,
        "mean_oos_sharpe": 1.4,
        "oos_positive_alpha_hit_rate": 0.70,
        "cost_stress_approval_pass": True,
        "mean_oos_max_drawdown_pct": -12.0,
        "worst_oos_max_drawdown_pct": -28.0,
        "worst_oos_turnover_pct": 991.0,
        "selection_bias_gap_sharpe": 0.4,
        "most_common_config": "h=20,ov=0.5,ma=100,vol=fixed:0.3,score=regime_adaptive,shape=top5,weighting=sticky_score,tqqq=0.0,risk=off",
    }
    approval = nested_wf.approval_status(result)

    assert approval["approved"] is False
    assert any(reason.startswith("config_frequency<") for reason in approval["reasons"])
    assert any(reason.startswith("worst_oos_turnover=") for reason in approval["reasons"])


def _wf_sig(
    *,
    shape: str = "top5",
    weighting: str = "sticky_score",
    tqqq: float = 0.0,
    risk: str = "off",
    h: int = 20,
    ov: float = 0.5,
    vol_mode: str = "fixed",
) -> str:
    return (
        f"h={h},ov={ov},ma=100,vol={vol_mode}:0.3,"
        f"score=regime_adaptive,shape={shape},weighting={weighting},"
        f"tqqq={tqqq},risk={risk}"
    )


def test_manual_publish_preserves_risk_mode_controls():
    off_config = manual_publish.build_full_config(
        manual_publish.parse_config_signature(_wf_sig(risk="off")),
        {},
    )
    defensive_config = manual_publish.build_full_config(
        manual_publish.parse_config_signature(_wf_sig(risk="defensive")),
        {},
    )

    assert off_config["risk_control_mode"] == "off"
    assert off_config["drawdown_circuit_breaker"] == 0.0
    assert off_config["vol_target"] == 0.0
    assert defensive_config["risk_control_mode"] == "defensive"
    assert defensive_config["drawdown_circuit_breaker"] == 0.15
    assert defensive_config["vol_target"] == 0.15


def test_stable_family_signature_keeps_tqqq_separate():
    no_tqqq = nested_wf.stable_family_signature_from_config_signature(_wf_sig(tqqq=0.0))
    with_tqqq = nested_wf.stable_family_signature_from_config_signature(_wf_sig(tqqq=0.1))

    assert no_tqqq != with_tqqq
    assert no_tqqq.endswith("tqqq=0.0")
    assert with_tqqq.endswith("tqqq=0.1")


def test_stable_family_tiebreak_prefers_no_tqqq_then_lower_drawdown():
    rows = [
        {"selected_config": _wf_sig(shape="top3", tqqq=0.1), "outer_year": 2025, "oos_cagr_pct": 20, "oos_sharpe": 3.0, "oos_max_drawdown_pct": -4.0, "oos_turnover_pct": 100, "oos_alpha_vs_spy_pct": 5, "oos_alpha_vs_qqq_pct": 4},
        {"selected_config": _wf_sig(shape="top3", tqqq=0.1, vol_mode="percentile"), "outer_year": 2026, "oos_cagr_pct": 21, "oos_sharpe": 3.1, "oos_max_drawdown_pct": -4.5, "oos_turnover_pct": 110, "oos_alpha_vs_spy_pct": 5, "oos_alpha_vs_qqq_pct": 4},
        {"selected_config": _wf_sig(shape="top5", tqqq=0.0), "outer_year": 2023, "oos_cagr_pct": 12, "oos_sharpe": 1.1, "oos_max_drawdown_pct": -5.0, "oos_turnover_pct": 90, "oos_alpha_vs_spy_pct": 4, "oos_alpha_vs_qqq_pct": 3},
        {"selected_config": _wf_sig(shape="top5", tqqq=0.0, vol_mode="percentile"), "outer_year": 2024, "oos_cagr_pct": 13, "oos_sharpe": 1.0, "oos_max_drawdown_pct": -6.0, "oos_turnover_pct": 95, "oos_alpha_vs_spy_pct": 4, "oos_alpha_vs_qqq_pct": 3},
        {"selected_config": _wf_sig(shape="top15", tqqq=0.0), "outer_year": 2021, "oos_cagr_pct": 15, "oos_sharpe": 2.0, "oos_max_drawdown_pct": -10.0, "oos_turnover_pct": 80, "oos_alpha_vs_spy_pct": 4, "oos_alpha_vs_qqq_pct": 3},
        {"selected_config": _wf_sig(shape="top15", tqqq=0.0, vol_mode="percentile"), "outer_year": 2022, "oos_cagr_pct": 16, "oos_sharpe": 2.2, "oos_max_drawdown_pct": -11.0, "oos_turnover_pct": 85, "oos_alpha_vs_spy_pct": 4, "oos_alpha_vs_qqq_pct": 3},
    ]

    families = nested_wf.stable_family_frequency(rows)

    assert families[0]["stable_family_signature"] == "score=regime_adaptive,shape=top5,weighting=sticky_score,risk=off,tqqq=0.0"
    assert families[0]["uses_tqqq"] is False


def test_manual_publish_approval_rejects_one_off_selected_config():
    df = pd.DataFrame({
        "fold_year": [2020, 2021, 2022, 2023, 2024, 2025],
        "selected_config": ["A", "A", "B", "C", "D", "E"],
        "oos_return_pct": [10.0, 12.0, 8.0, 9.0, 11.0, 13.0],
        "oos_sharpe": [1.0, 1.1, 0.9, 1.2, 1.3, 1.4],
        "oos_max_drawdown_pct": [-5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
        "oos_turnover_pct": [100.0, 120.0, 130.0, 140.0, 150.0, 160.0],
        "oos_alpha_vs_spy_pct": [4.0, 5.0, 3.0, 4.0, 5.0, 6.0],
        "oos_alpha_vs_qqq_pct": [2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
    })

    metrics = manual_publish.compute_aggregate_metrics(df)
    selected = manual_publish.compute_selected_config_metrics(df, "A")
    approval = manual_publish.build_approval(metrics, "A", force=False, selected_config_metrics=selected)

    assert metrics["best_config_frequency"] == round(2 / 6, 4)
    assert selected["selected_config_frequency"] == round(2 / 6, 4)
    assert approval["approved"] is True

    selected_too_rare = manual_publish.compute_selected_config_metrics(df, "E")
    approval_too_rare = manual_publish.build_approval(
        metrics,
        "E",
        force=False,
        selected_config_metrics=selected_too_rare,
    )
    assert approval_too_rare["approved"] is False
    assert any("selected_config_freq" in reason for reason in approval_too_rare["reasons"])


def test_manual_publish_stable_family_skips_latest_one_off_config():
    df = pd.DataFrame({
        "fold_year": [2020, 2021, 2022, 2023, 2024, 2025],
        "selected_config": [
            _wf_sig(shape="top5", h=10),
            _wf_sig(shape="top5", h=20),
            _wf_sig(shape="top5", h=10, vol_mode="percentile"),
            _wf_sig(shape="top3", tqqq=0.1),
            _wf_sig(shape="top15", weighting="risk_parity"),
            _wf_sig(shape="top3", weighting="risk_parity", tqqq=0.1),
        ],
        "oos_return_pct": [10.0, 12.0, 8.0, 9.0, 11.0, 13.0],
        "oos_sharpe": [1.0, 1.1, 0.9, 1.2, 1.3, 1.4],
        "oos_max_drawdown_pct": [-5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
        "oos_turnover_pct": [100.0, 120.0, 130.0, 140.0, 150.0, 160.0],
        "oos_alpha_vs_spy_pct": [4.0, 5.0, 3.0, 4.0, 5.0, 6.0],
        "oos_alpha_vs_qqq_pct": [2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
    })

    pick = manual_publish.select_config(df, "stable_family")
    selected_metrics = manual_publish.compute_selected_config_metrics(df, str(pick["row"].selected_config))
    family_metrics = pick["family_metrics"]
    approval = manual_publish.build_approval(
        manual_publish.compute_aggregate_metrics(df),
        pick["family_signature"],
        force=False,
        selected_config_metrics=selected_metrics,
        stable_family_metrics=family_metrics,
    )

    assert int(pick["row"].fold_year) == 2022
    assert family_metrics["approved_family_fold_count"] == 3
    assert approval["approved"] is True
    assert approval["approved_config_frequency"] < approval["approved_family_frequency"]
    assert approval["warnings"]


def test_inner_selection_scores_validation_folds_only(monkeypatch):
    configs = [
        {"name": "train_fit", "nested_params": {"holding_days": 10}},
        {"name": "validation_winner", "nested_params": {"holding_days": 10}},
    ]
    folds = [
        InnerFold(
            validation_year=2022,
            train_end=pd.Timestamp("2021-12-31"),
            validation_start=pd.Timestamp("2022-01-01"),
            validation_end=pd.Timestamp("2022-12-31"),
        ),
        InnerFold(
            validation_year=2023,
            train_end=pd.Timestamp("2022-12-31"),
            validation_start=pd.Timestamp("2023-01-01"),
            validation_end=pd.Timestamp("2023-12-31"),
        ),
    ]
    calls = []

    def fake_evaluate_window(panel, config, start, end):
        calls.append((config["name"], pd.Timestamp(start), pd.Timestamp(end)))
        assert pd.Timestamp(start).year in {2022, 2023}
        assert pd.Timestamp(end).year in {2022, 2023}
        sharpe = 0.1 if config["name"] == "train_fit" else 1.0
        return {
            "sharpe": sharpe,
            "total_return_pct": sharpe,
            "max_drawdown_pct": -5.0,
            "turnover_pct": 100.0,
            "alpha_vs_spy_pct": 1.0,
            "alpha_vs_qqq_pct": 1.0,
            "alpha_vs_blend_pct": 1.0,
        }

    monkeypatch.setattr(nested_wf, "evaluate_window", fake_evaluate_window)
    selected = select_config_from_inner_folds(pd.DataFrame(), configs, folds)

    assert selected["config"]["name"] == "validation_winner"
    assert len(calls) == len(configs) * len(folds) * 3


def test_inner_selection_rejects_configs_above_turnover_cap(monkeypatch):
    configs = [
        {"name": "high_sharpe_churn", "nested_params": {"holding_days": 10}},
        {"name": "lower_sharpe_clean", "nested_params": {"holding_days": 10}},
    ]
    folds = [
        InnerFold(
            validation_year=2022,
            train_end=pd.Timestamp("2021-12-31"),
            validation_start=pd.Timestamp("2022-01-01"),
            validation_end=pd.Timestamp("2022-12-31"),
        ),
        InnerFold(
            validation_year=2023,
            train_end=pd.Timestamp("2022-12-31"),
            validation_start=pd.Timestamp("2023-01-01"),
            validation_end=pd.Timestamp("2023-12-31"),
        ),
    ]

    def fake_evaluate_window(panel, config, start, end):
        churn = config["name"] == "high_sharpe_churn"
        return {
            "sharpe": 5.0 if churn else 1.0,
            "total_return_pct": 25.0 if churn else 5.0,
            "max_drawdown_pct": -5.0,
            "turnover_pct": nested_wf.MAX_INNER_MEAN_TURNOVER_PCT + 50.0 if churn else 100.0,
            "alpha_vs_spy_pct": 1.0,
            "alpha_vs_qqq_pct": 1.0,
            "alpha_vs_blend_pct": 1.0,
        }

    monkeypatch.setattr(nested_wf, "evaluate_window", fake_evaluate_window)
    selected = select_config_from_inner_folds(pd.DataFrame(), configs, folds)

    assert selected["config"]["name"] == "lower_sharpe_clean"
    assert selected["metrics"]["inner_mean_turnover_pct"] == 100.0


def test_inner_selection_can_treat_stress_gate_as_diagnostic(monkeypatch):
    configs = [{"name": "research_candidate", "nested_params": {"holding_days": 20}}]
    folds = [
        InnerFold(
            validation_year=2022,
            train_end=pd.Timestamp("2021-12-31"),
            validation_start=pd.Timestamp("2022-01-01"),
            validation_end=pd.Timestamp("2022-12-31"),
        ),
        InnerFold(
            validation_year=2023,
            train_end=pd.Timestamp("2022-12-31"),
            validation_start=pd.Timestamp("2023-01-01"),
            validation_end=pd.Timestamp("2023-12-31"),
        ),
    ]

    def fake_evaluate_window(panel, config, start, end):
        return {
            "sharpe": 1.0,
            "total_return_pct": 10.0,
            "max_drawdown_pct": -5.0,
            "turnover_pct": 100.0,
            "alpha_vs_spy_pct": 2.0,
            "alpha_vs_qqq_pct": 2.0,
            "alpha_vs_blend_pct": 2.0,
        }

    def fake_stress_approval(*args, **kwargs):
        return {
            "cost_stress_approval_pass": False,
            "cost_stress_summary": {"all_gates_pass": False},
        }

    monkeypatch.setattr(nested_wf, "evaluate_window", fake_evaluate_window)
    monkeypatch.setattr(nested_wf, "nested_cost_stress_approval", fake_stress_approval)

    strict = select_config_from_inner_folds(pd.DataFrame(), configs, folds)
    relaxed = select_config_from_inner_folds(
        pd.DataFrame(),
        configs,
        folds,
        skip_stress_gate=True,
    )

    assert strict["config"] is None
    assert relaxed["config"]["name"] == "research_candidate"
    assert relaxed["metrics"]["inner_cost_stress_approval_pass"] is False


def test_signal_timestamp_is_timezone_aware():
    ts = pd.Timestamp(_paper_signal_timestamp())
    assert ts.tzinfo is not None


def test_gauntlet_counts_local_signal_fills(monkeypatch):
    monkeypatch.setattr(paper_gauntlet, "DEFAULT_SIGNAL_TIMEZONE", "Asia/Singapore")
    trades = pd.DataFrame({
        "submitted": [True],
        "submitted_at": ["2026-05-04T14:57:29+00:00"],
        "fill_status": ["filled"],
    })
    status = {"signal_predicted_at": "2026-05-04 22:45"}
    current = paper_gauntlet._filter_current_signal_trades(trades, status)
    assert len(current) == 1
    assert paper_gauntlet._order_fill_stats(current)["fill_rate"] == 1.0


def test_execution_slippage_is_adverse_by_side():
    buy_bps, buy_dollars = _execution_slippage({
        "action": "BUY",
        "price": 100.0,
        "broker_dealt_avg_price": 100.10,
        "broker_dealt_qty": 10,
    })
    sell_bps, sell_dollars = _execution_slippage({
        "action": "SELL",
        "price": 100.0,
        "broker_dealt_avg_price": 99.90,
        "broker_dealt_qty": 10,
    })
    assert round(buy_bps, 6) == 10.0
    assert round(sell_bps, 6) == 10.0
    assert round(buy_dollars, 6) == 1.0
    assert round(sell_dollars, 6) == 1.0


def test_health_concentration_separates_core_etfs_from_overlay_stocks():
    concentration = _position_concentration({
        "account_equity": 100_000,
        "position_values": {
            "QQQ": 44_000,
            "MU": 20_000,
            "CAT": 12_000,
        },
    })
    assert concentration["max_ticker"] == "QQQ"
    assert concentration["max_core_ticker"] == "QQQ"
    assert concentration["max_overlay_ticker"] == "MU"
    assert concentration["overlay_ticker_concentration_warning"] is False
    assert concentration["core_ticker_concentration_warning"] is False


def test_health_concentration_empty_positions_is_neutral():
    concentration = _position_concentration({
        "account_equity": 100_000,
        "position_values": {},
    })
    assert concentration["ticker_weights"] == {}
    assert concentration["max_ticker"] == ""
    assert concentration["max_ticker_weight"] == 0.0
    assert concentration["ticker_concentration_warning"] is False


def test_health_concentration_unusable_position_values_is_neutral():
    concentration = _position_concentration({
        "account_equity": 100_000,
        "position_values": {
            "AAPL": 0.0,
            "MSFT": np.nan,
            "CAT": "not-a-number",
        },
    })
    assert concentration["ticker_weights"] == {}
    assert concentration["max_ticker"] == ""
    assert concentration["max_sector"] == ""
    assert concentration["overlay_sector_concentration_warning"] is False


def test_health_concentration_core_only_keeps_overlay_fields_neutral():
    concentration = _position_concentration({
        "account_equity": 100_000,
        "position_values": {
            "SPY": 30_000,
            "QQQ": 40_000,
        },
    })
    assert concentration["max_ticker"] == "QQQ"
    assert concentration["max_core_ticker"] == "QQQ"
    assert concentration["max_overlay_ticker"] == ""
    assert concentration["max_overlay_ticker_weight"] == 0.0
    assert concentration["max_overlay_sector"] == ""
    assert concentration["overlay_ticker_concentration_warning"] is False


def test_moomoo_status_bucket_accepts_common_filled_variants():
    assert _order_status_bucket("FILLED_ALL", 0, 0) == "filled"
    assert _order_status_bucket("ALL_TRADED", 0, 0) == "filled"
    assert _order_status_bucket("SUBMITTED", 0, 10) == "open"
    assert _order_status_bucket("CANCELLED_ALL", 0, 10) == "cancelled"


def test_reconcile_fallback_matches_by_ticker_side_and_qty():
    trade = pd.Series({
        "ticker": "MU",
        "action": "BUY",
        "broker_qty": 25,
        "submitted_at": "2026-05-08T13:30:00+00:00",
    })
    orders = pd.DataFrame([{
        "order_id": "123456789",
        "code": "US.MU",
        "trd_side": "BUY",
        "qty": 25,
        "updated_time": "2026-05-08 13:31:00",
    }])
    match, match_type = _find_broker_order_match(trade, orders)
    assert match is not None
    assert match_type == "ticker_side_qty"
    assert str(match["order_id"]) == "123456789"


def test_open_position_attribution_skips_zero_priced_positions():
    attribution = _open_position_attribution(
        {
            "positions": {"QQQ": 10, "MU": 5},
            "position_values": {"QQQ": 0.0, "MU": 550.0},
        },
        pd.DataFrame([
            {"ticker": "QQQ", "action": "BUY", "fill_status": "filled", "broker_dealt_avg_price": 100.0},
            {"ticker": "MU", "action": "BUY", "fill_status": "filled", "broker_dealt_avg_price": 100.0},
        ]),
    )
    assert "QQQ" in attribution["skipped_unpriced_tickers"]
    assert "QQQ" not in attribution["by_ticker"]
    assert attribution["by_ticker"]["MU"]["open_pnl"] == 50.0


def test_daily_check_verdict_waits_for_open_orders():
    verdict, reasons = _verdict(
        {
            "current_signal_open_orders": 2,
            "approved_for_real_capital": False,
            "readiness_flags": {
                "strategy_ready": True,
                "signal_fresh": True,
                "broker_synced": True,
                "account_aligned": False,
                "slippage_ok": True,
                "drawdown_ok": True,
                "concentration_ok": True,
            },
        },
        [{"name": "health", "status": "ok"}],
    )
    assert verdict == "WAIT_FOR_OPEN_ORDERS"
    assert "current_signal_open_orders=2" in reasons


def test_after_fill_verdict_distinguishes_open_and_aligned():
    open_verdict, _ = _after_fill_verdict(
        {"current_signal_open_orders": 1, "readiness_flags": {"broker_synced": True, "account_aligned": False}},
        [{"name": "sync", "status": "ok"}],
    )
    aligned_verdict, _ = _after_fill_verdict(
        {"current_signal_open_orders": 0, "readiness_flags": {"broker_synced": True, "account_aligned": True}},
        [{"name": "sync", "status": "ok"}],
    )
    assert open_verdict == "STILL_OPEN"
    assert aligned_verdict == "ALIGNED"


def test_prune_snapshots_removes_only_old_timestamped_dirs(tmp_path, monkeypatch):
    root = tmp_path / "paper_snapshots"
    old_dir = root / "20250101T000000Z"
    new_dir = root / "20260501T000000Z"
    odd_dir = root / "keep_me"
    old_dir.mkdir(parents=True)
    new_dir.mkdir()
    odd_dir.mkdir()
    monkeypatch.setattr(daily_paper_check, "LOGS", tmp_path)
    result = _prune_snapshots(keep_days=30, now=datetime(2026, 5, 9, tzinfo=timezone.utc))
    assert not old_dir.exists()
    assert new_dir.exists()
    assert odd_dir.exists()
    assert str(old_dir) in result["removed"]


def test_submit_readiness_blocks_over_gross_and_closed_market(monkeypatch):
    monkeypatch.setattr(moomoo_paper_trading, "us_regular_market_is_open", lambda: (False, "closed for test"))
    report = build_submit_readiness_report(
        signal=pd.Series({"paper_ready": True, "gates_all_pass": True}),
        orders=pd.DataFrame([{
            "ticker": "QQQ",
            "action": "BUY",
            "target_weight": 1.25,
            "order_value": 1000.0,
            "reason": "rebalance_to_target",
        }]),
        freshness_ok=True,
        freshness_issues=[],
        submit_allowed=True,
        max_submit_gross_exposure=1.0,
    )
    assert report["can_submit_during_regular_market"] is False
    assert any("target gross" in item for item in report["blockers"])
    assert "closed for test" in report["blockers"]


def test_submit_preflight_blocks_current_signal_open_orders(monkeypatch):
    monkeypatch.setattr(moomoo_paper_trading, "us_regular_market_is_open", lambda: (True, "open for test"))
    monkeypatch.setattr(
        moomoo_paper_trading,
        "current_signal_open_orders",
        lambda signal: pd.DataFrame([{"ticker": "CAT", "fill_status": "open"}]),
    )
    ok, blockers = evaluate_broker_submit_preflight(
        signal=pd.Series({"paper_ready": True, "gates_all_pass": True}),
        orders=pd.DataFrame([{
            "ticker": "CAT",
            "action": "BUY",
            "target_weight": 1.0,
            "order_value": 1000.0,
            "reason": "rebalance_to_target",
        }]),
        freshness_ok=True,
        freshness_issues=[],
        submit_allowed=True,
        allow_stale_signal=False,
        allow_closed_market_submit=False,
        allow_leverage_submit=False,
        allow_submit_with_open_orders=False,
        max_submit_gross_exposure=1.0,
    )
    assert ok is False
    assert any("open paper orders" in blocker for blocker in blockers)


def test_current_order_lifecycle_summarizes_open_quantity(monkeypatch):
    monkeypatch.setattr(paper_gauntlet, "DEFAULT_SIGNAL_TIMEZONE", "Asia/Singapore")
    trades = pd.DataFrame([{
        "ticker": "CAT",
        "action": "BUY",
        "broker_order_id": "123",
        "fill_status": "open",
        "broker_requested_qty": 139,
        "broker_dealt_qty": 0,
        "submitted_at": "2026-05-08T13:35:35+00:00",
    }])
    lifecycle = _current_order_lifecycle(trades, {"signal_predicted_at": "2026-05-08T21:17+08:00"})
    assert lifecycle[0]["ticker"] == "CAT"
    assert lifecycle[0]["fill_status"] == "open"
    assert lifecycle[0]["unfilled_qty"] == 139


def test_drift_breakdown_ranks_largest_drift_first():
    breakdown = _drift_breakdown(pd.DataFrame([
        {"ticker": "QQQ", "action": "SKIP", "current_weight": 0.42, "target_weight": 0.44, "drift_abs": 0.02, "order_value": 10},
        {"ticker": "INTC", "action": "BUY", "current_weight": 0.0, "target_weight": 0.18, "drift_abs": 0.18, "order_value": 100},
    ]))
    assert breakdown[0]["ticker"] == "INTC"
    assert breakdown[0]["drift_abs"] == 0.18


def test_stale_open_order_alerts_only_active_old_orders():
    alerts = _stale_open_order_alerts(
        [
            {
                "ticker": "CAT",
                "action": "BUY",
                "fill_status": "open",
                "submitted_at": "2026-05-08T13:00:00+00:00",
                "unfilled_qty": 139,
                "broker_order_id": "1",
            },
            {
                "ticker": "MU",
                "action": "SELL",
                "fill_status": "filled",
                "submitted_at": "2026-05-08T13:00:00+00:00",
                "unfilled_qty": 0,
                "broker_order_id": "2",
            },
        ],
        now=pd.Timestamp("2026-05-08T15:00:00+00:00"),
    )
    assert len(alerts) == 1
    assert alerts[0]["ticker"] == "CAT"
    assert alerts[0]["age_minutes"] == 120.0


def test_go_live_scorecard_tracks_progress_not_just_fail():
    scorecard = _go_live_scorecard({
        "paper_equity_days": 3,
        "current_signal_open_orders": 2,
        "fill_rate": 0.6,
        "max_drift_abs": 0.18,
        "approved_for_real_capital": False,
        "slippage": {"avg_slippage_bps": 1.0},
        "readiness_flags": {
            "strategy_ready": True,
            "signal_fresh": True,
            "broker_synced": True,
            "drawdown_ok": True,
            "concentration_ok": True,
        },
    })
    assert scorecard["passed"] < scorecard["total"]
    names = {item["name"]: item for item in scorecard["criteria"]}
    assert names["paper_equity_days"]["passed"] is False
    assert names["strategy_ready"]["passed"] is True


def test_equity_sanity_flags_large_moves_and_duplicates():
    sanity = _equity_sanity_checks(pd.DataFrame({
        "date": ["2026-05-01", "2026-05-01", "2026-05-02"],
        "equity": [100.0, 130.0, 90.0],
    }))
    assert sanity["ok"] is False
    assert "duplicate_equity_dates" in sanity["warnings"]
    assert any(item.startswith("large_equity_move") for item in sanity["warnings"])


def test_etf_validator_rejects_flat_recent_close():
    idx = pd.bdate_range(start="2025-01-01", periods=260)
    frame = pd.DataFrame({"Close": [100.0] * 260}, index=idx)
    report = _validate_etf_frame(frame, symbol="SPY", max_age_business_days=10_000)
    assert report["ok"] is False
    assert "flat_recent_close" in report["issues"]


def test_config_health_requirement_detects_installed_package():
    check = _requirement_ok("yfinance", ">=0")
    assert check["ok"] is True
    assert check["installed"]
