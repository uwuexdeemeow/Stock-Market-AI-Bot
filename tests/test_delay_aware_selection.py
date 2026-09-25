"""Delay-aware inner selection for the nested walk-forward.

PLAIN ENGLISH: with WALKFORWARD_SELECTION_ENTRY_DELAY_DAYS=1 the selector
replays every inner fold with fills one trading day late and keeps the worse
score. A candidate that only works with perfect timing must lose to one that
survives a late fill. The default (0) must behave exactly as before.
"""

from __future__ import annotations

import pandas as pd
import pytest

import core_satellite_nested_walkforward as nested_wf
from core_satellite_nested_walkforward import InnerFold, select_config_from_inner_folds


FOLDS = [
    InnerFold(
        validation_year=year,
        train_end=pd.Timestamp(f"{year - 1}-12-31"),
        validation_start=pd.Timestamp(f"{year}-01-01"),
        validation_end=pd.Timestamp(f"{year}-12-31"),
    )
    for year in (2021, 2022)
]

CONFIGS = [
    {"name": "timing_fragile", "nested_params": {"holding_days": 10}},
    {"name": "timing_robust", "nested_params": {"holding_days": 10}},
]


def _metrics(strength: float) -> dict:
    return {
        "sharpe": strength,
        "total_return_pct": strength * 10.0,
        "max_drawdown_pct": -5.0,
        "turnover_pct": 100.0,
        "alpha_vs_spy_pct": strength * 5.0,
        "alpha_vs_qqq_pct": strength * 5.0,
        "alpha_vs_blend_pct": strength * 5.0,
    }


def _fake_evaluate_window(calls: list):
    def fake(panel, config, start, end):
        delay = int(config.get("entry_delay_days", 0))
        calls.append((config["name"], delay))
        if config["name"] == "timing_fragile":
            # Best on time, collapses when fills are one day late.
            return _metrics(2.0 if delay == 0 else -1.0)
        return _metrics(1.0)
    return fake


def test_default_selection_ignores_delay_and_picks_on_time_winner(monkeypatch):
    monkeypatch.delenv("WALKFORWARD_SELECTION_ENTRY_DELAY_DAYS", raising=False)
    calls: list = []
    monkeypatch.setattr(nested_wf, "evaluate_window", _fake_evaluate_window(calls))

    selected = select_config_from_inner_folds(pd.DataFrame(), CONFIGS, FOLDS)

    assert selected["config"]["name"] == "timing_fragile"
    assert all(delay == 0 for _, delay in calls)
    assert selected["metrics"]["selection_entry_delay_days"] == 0


def test_delay_aware_selection_prefers_timing_robust_candidate(monkeypatch):
    monkeypatch.setenv("WALKFORWARD_SELECTION_ENTRY_DELAY_DAYS", "1")
    calls: list = []
    monkeypatch.setattr(nested_wf, "evaluate_window", _fake_evaluate_window(calls))

    selected = select_config_from_inner_folds(pd.DataFrame(), CONFIGS, FOLDS)

    assert selected["config"]["name"] == "timing_robust"
    assert any(delay == 1 for _, delay in calls)
    assert selected["metrics"]["selection_entry_delay_days"] == 1
    fold = selected["fold_metrics"][0]
    assert fold["selection_entry_delay_days"] == 1
    assert fold["score"] == min(fold["on_time_score"], fold["delayed_score"])


def test_delay_aware_selection_rejects_tqqq_candidates(monkeypatch):
    monkeypatch.setenv("WALKFORWARD_SELECTION_ENTRY_DELAY_DAYS", "1")
    monkeypatch.setattr(nested_wf, "evaluate_window", _fake_evaluate_window([]))
    tqqq = {"name": "timing_robust", "nested_params": {"holding_days": 10, "tqqq_weight": 0.1}}

    result = nested_wf._evaluate_one_config(tqqq, pd.DataFrame(), FOLDS, low_memory=True)

    assert result["config"] is None
    assert result["rejection_reason"] == "delay_selection_unsupported_tqqq"


def test_worst_case_timing_metrics_takes_lower_alpha_and_higher_turnover():
    on_time = {**_metrics(2.0), "turnover_pct": 100.0}
    delayed = {**_metrics(-1.0), "turnover_pct": 120.0}

    merged = nested_wf.worst_case_timing_metrics(on_time, delayed)

    assert merged["alpha_vs_qqq_pct"] == -5.0
    assert merged["turnover_pct"] == 120.0
    assert merged["on_time_alpha_vs_qqq_pct"] == 10.0
    assert merged["delayed_alpha_vs_qqq_pct"] == -5.0


@pytest.mark.parametrize("raw", ["2", "-1", "one"])
def test_invalid_selection_delay_setting_fails_fast(monkeypatch, raw):
    monkeypatch.setenv("WALKFORWARD_SELECTION_ENTRY_DELAY_DAYS", raw)
    with pytest.raises(ValueError):
        nested_wf.selection_entry_delay_days_from_env()


def test_selection_delay_changes_checkpoint_key_only_when_enabled(monkeypatch):
    monkeypatch.delenv("WALKFORWARD_SELECTION_ENTRY_DELAY_DAYS", raising=False)
    default_key = nested_wf._ckpt_key("core-alpha", 5, CONFIGS, None, None)
    monkeypatch.setenv("WALKFORWARD_SELECTION_ENTRY_DELAY_DAYS", "0")
    assert nested_wf._ckpt_key("core-alpha", 5, CONFIGS, None, None) == default_key
    monkeypatch.setenv("WALKFORWARD_SELECTION_ENTRY_DELAY_DAYS", "1")
    assert nested_wf._ckpt_key("core-alpha", 5, CONFIGS, None, None) != default_key
