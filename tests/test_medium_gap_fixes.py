from __future__ import annotations

import json
import gc
import weakref

import pandas as pd

import core_satellite_alpha as csa
import core_satellite_nested_walkforward as nwf
import factor_decay_monitor


def test_panel_day_cache_rebuilds_when_dataframe_id_entry_is_stale():
    """A recycled DataFrame id must not return a day map from an old panel."""
    csa.clear_panel_day_cache()
    date = pd.Timestamp("2024-01-01")

    old_panel = pd.DataFrame({"date": [date], "sentinel": [-1]})
    old_ref = weakref.ref(old_panel)
    old_map = {date: old_panel}
    del old_panel
    gc.collect()

    panel = pd.DataFrame({"date": [date], "sentinel": [42]})
    csa._PANEL_DAY_CACHE[id(panel)] = (old_ref, old_map)

    day_map = csa._panel_day_map(panel)

    assert int(day_map[date]["sentinel"].iloc[0]) == 42
    assert len(csa._PANEL_DAY_CACHE) <= csa._MAX_PANEL_DAY_ENTRIES
    csa.clear_panel_day_cache()


def test_factor_decay_classifies_negative_rank_ic_positive_edge_as_advisory():
    row = {
        "daily_ic_mean": -0.05,
        "top_bucket_excess_return_pct": 0.2,
        "overlay_alpha_sum_pct": 1.0,
    }

    assert factor_decay_monitor.edge_health_status(row) == "advisory"


def test_factor_decay_classifies_nonpositive_top_bucket_as_warning():
    row = {
        "daily_ic_mean": 0.05,
        "top_bucket_excess_return_pct": 0.0,
        "overlay_alpha_sum_pct": 1.0,
    }

    assert factor_decay_monitor.edge_health_status(row) == "warning"


def test_factor_decay_classifies_material_negative_overlay_alpha_as_block():
    """Real block fires only when the sample is wide enough AND the
    cumulative drawdown is materially negative.  This is the canonical
    120-day window shape: 4+ overlay trades and a meaningful loss."""
    row = {
        "daily_ic_mean": 0.05,
        "top_bucket_excess_return_pct": 0.5,
        "overlay_alpha_sum_pct": -2.5,
        "overlay_periods": 5,
    }

    assert factor_decay_monitor.edge_health_status(row) == "block"


def test_factor_decay_thin_sample_negative_overlay_does_not_block():
    """A 60-day window typically has only 1–2 overlay trades at
    holding_days=20.  A single bad trade pushes cumulative overlay
    alpha well below zero but is too thin to call "block" — it should
    downgrade to advisory/warning instead of halting live trading."""
    row = {
        "daily_ic_mean": -0.05,
        "top_bucket_excess_return_pct": 0.5,
        "overlay_alpha_sum_pct": -2.5,
        "overlay_periods": 2,
    }

    assert factor_decay_monitor.edge_health_status(row) != "block"


def test_factor_decay_small_negative_overlay_with_sample_does_not_block():
    """Magnitude gate: even with 4+ periods, a -0.1% cumulative is noise
    and must not trigger a real-capital block."""
    row = {
        "daily_ic_mean": 0.05,
        "top_bucket_excess_return_pct": 0.5,
        "overlay_alpha_sum_pct": -0.1,
        "overlay_periods": 6,
    }

    assert factor_decay_monitor.edge_health_status(row) != "block"


def _passing_survivorship():
    return {
        "survivorship_adjusted_score": 0.9,
        "rows": [
            {"scenario": "watchlist_plus_failed_audit_tickers", "paper_ready": True, "audit_rebalance_selections": 0},
            {"scenario": "delta_stressed_minus_base", "total_return_pct": -100.0, "max_drawdown_pct": -1.0},
        ],
    }


def _passing_execution():
    return {
        "rows": [
            {
                "scenario": "delay_1d",
                "paper_ready": True,
                "alpha_vs_qqq_pct": 1.0,
                "alpha_vs_blend_pct": 1.0,
                "max_drawdown_pct": -20.0,
            },
        ],
    }


def test_medium_risk_review_blocks_material_survivorship_failure():
    bad_survivorship = _passing_survivorship()
    bad_survivorship["survivorship_adjusted_score"] = 0.5

    review = nwf.medium_risk_review_from_reports(
        survivorship=bad_survivorship,
        execution=_passing_execution(),
        factor_decay={"edge_health_status": "pass"},
    )

    assert review["pass"] is False
    assert "survivorship_review_failed" in review["reasons"]


def test_medium_risk_review_blocks_execution_drawdown_failure():
    bad_execution = _passing_execution()
    bad_execution["rows"][0]["max_drawdown_pct"] = -36.0

    review = nwf.medium_risk_review_from_reports(
        survivorship=_passing_survivorship(),
        execution=bad_execution,
        factor_decay={"edge_health_status": "pass"},
    )

    assert review["pass"] is False
    assert "execution_stress_review_failed" in review["reasons"]


def test_apply_medium_risk_review_removes_approved_live_config_when_failed():
    summary = {
        "live_config_approval": {"approved": True, "reasons": []},
        "approved_live_config": {"config": {"risk_control_mode": "off"}, "source_metrics": {}},
    }
    review = {"pass": False, "reasons": ["execution_stress_review_failed"]}

    out = nwf.apply_medium_risk_review(summary, review)

    assert out["live_config_approval"]["approved"] is False
    assert "approved_live_config" not in out
    assert "medium_risk_review_failed:execution_stress_review_failed" in out["live_config_approval"]["reasons"]


def test_live_gate_requires_medium_risk_review_pass():
    metrics = {
        "core_satellite_gate_results": {"all_pass": True},
        "feature_health_gate_pass": True,
    }
    live = {
        "approval": {"approved": True, "reasons": []},
        "source_metrics": {"cost_stress_approval_pass": True},
        "medium_risk_review": {"pass": False, "reasons": ["factor_decay_review_warning"]},
    }

    out = csa._apply_nested_live_approval_gates(metrics, live, {"fresh": True})

    assert out["paper_ready"] is False
    assert out["core_satellite_gate_results"]["all_pass"] is False
    assert out["medium_risk_review_pass"] is False
    assert any("medium_risk_review_failed" in reason for reason in out["live_gate_reasons"])


