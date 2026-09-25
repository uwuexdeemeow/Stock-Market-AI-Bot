"""Research-candidate mode for the execution-stress and survivorship scripts.

PLAIN ENGLISH: a research winner (for example from Colab) must be testable
without publishing it.  These tests use fake panels and fake evaluations so
they never touch real market data, and they check that candidate runs write
only to their own ``logs/research_candidate_*`` files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import core_satellite_execution_stress as execution_stress
import core_satellite_survivorship_audit as survivorship_stress


def _fake_metrics() -> dict:
    """Smallest metrics dict both scripts can turn into report rows."""
    comps = {k: {"alpha_pct": 1.0} for k in ("SPY", "QQQ", "BLEND")}
    return {
        "benchmark_comparisons": comps,
        "core_satellite_gate_results": {"all_pass": True, "alpha_pass": True},
        "holdout_2023_2026": {"alpha_vs_qqq_pct": 0.5, "alpha_vs_blend_pct": 0.5},
        "total_return_pct": 10.0,
        "cagr_pct": 2.0,
        "sharpe": 1.0,
        "max_drawdown_pct": -5.0,
        "turnover_pct": 100.0,
        "estimated_cost_pct": 0.1,
    }


def _patch_common(monkeypatch, tmp_path, module, seen_configs):
    logs = tmp_path / "logs"
    signals = tmp_path / "signals"
    monkeypatch.setattr(module, "LOG_DIR", str(logs))
    monkeypatch.setattr(module, "SIGNAL_DIR", str(signals))
    # The official output paths are redirected too, so a regression that
    # wrote to them would show up in tmp_path rather than the real repo.
    monkeypatch.setattr(module, "OUT_JSON", logs / f"official_{module.__name__}.json")
    monkeypatch.setattr(module, "OUT_CSV", signals / f"official_{module.__name__}.csv")
    monkeypatch.setattr(module, "add_validation_context", lambda payload, config: dict(payload))

    def _fake_evaluate(_panel, config):
        seen_configs.append(dict(config))
        trades = pd.DataFrame({"overlay_tickers": ["AAA,BBB"]})
        return _fake_metrics(), pd.DataFrame(), trades

    monkeypatch.setattr(module.core, "evaluate", _fake_evaluate)

    def _official_config_must_not_load():
        raise AssertionError("candidate mode must not read the approved live config")

    if module is execution_stress:
        monkeypatch.setattr(module, "_selected_config", _official_config_must_not_load)
        monkeypatch.setattr(module, "load_feature_specs", lambda: [])
        monkeypatch.setattr(module, "load_factor_panel", lambda *_a, **_k: pd.DataFrame())
        monkeypatch.setattr(module, "load_prediction_scores", lambda: None)
        monkeypatch.setattr(module, "attach_scores", lambda panel, *_a: panel)
        monkeypatch.setattr(module.core, "_ensure_robust_score_columns", lambda panel: panel)
    else:
        monkeypatch.setattr(module, "_load_selected_config", _official_config_must_not_load)
        monkeypatch.setattr(module, "_build_panel", lambda tickers: pd.DataFrame())
        monkeypatch.setattr(module, "existing_audit_profiles", lambda: {})
        monkeypatch.setattr(module, "available_audit_tickers", lambda _profiles: ["BBB"])
        monkeypatch.setattr(module, "membership_status", lambda: {"complete": False})
        monkeypatch.setattr(module, "WATCHLIST", ["AAA"])
    return logs, signals


def _walkforward_result(tmp_path: Path) -> Path:
    path = tmp_path / "wf_delay_robust_lowturnover_20260926.json"
    path.write_text(json.dumps({
        "live_config_approval": {"approved": True},
        "approved_live_config": {"config": {"top_n": 7, "regime_preset": "x"}},
    }), encoding="utf-8")
    return path


def test_load_candidate_config_reads_walkforward_and_plain_json(tmp_path):
    wf = _walkforward_result(tmp_path)
    assert execution_stress.load_candidate_config(wf) == {"top_n": 7, "regime_preset": "x"}

    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"top_n": 3}), encoding="utf-8")
    assert execution_stress.load_candidate_config(plain) == {"top_n": 3}


def test_load_candidate_config_rejects_walkforward_without_selection(tmp_path):
    path = tmp_path / "rejected.json"
    path.write_text(json.dumps({"live_config_approval": {"approved": False}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        execution_stress.load_candidate_config(path)


def test_candidate_name_is_file_safe(tmp_path):
    assert execution_stress.candidate_name(tmp_path / "wf run.json") == "wf_run"
    assert execution_stress.candidate_name("x.json", "../evil name") == "evil_name"


@pytest.mark.parametrize("module,prefix", [
    (execution_stress, "research_candidate_execution_stress"),
    (survivorship_stress, "research_candidate_survivorship_audit"),
])
def test_candidate_mode_writes_only_separate_research_outputs(tmp_path, monkeypatch, module, prefix):
    seen: list[dict] = []
    logs, signals = _patch_common(monkeypatch, tmp_path, module, seen)
    candidate = _walkforward_result(tmp_path)

    module.main(["--candidate-json", str(candidate)])

    name = "wf_delay_robust_lowturnover_20260926"
    out_json = logs / f"{prefix}_{name}.json"
    out_csv = logs / f"{prefix}_{name}.csv"
    assert out_json.exists() and out_csv.exists()
    # The official reports read by the paper gate were not created.
    assert not module.OUT_JSON.exists()
    assert not module.OUT_CSV.exists()
    assert not signals.exists() or not any(signals.iterdir())

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["research_candidate"] is True
    assert payload["approves_trading"] is False
    assert payload["selected_config"] == {"top_n": 7, "regime_preset": "x"}
    # Every evaluation used the candidate's settings.
    assert seen and all(cfg["top_n"] == 7 for cfg in seen)


@pytest.mark.parametrize("module", [execution_stress, survivorship_stress])
def test_default_mode_still_writes_official_outputs_without_candidate_stamp(tmp_path, monkeypatch, module):
    seen: list[dict] = []
    _patch_common(monkeypatch, tmp_path, module, seen)
    official = {"top_n": 5}
    attr = "_selected_config" if module is execution_stress else "_load_selected_config"
    monkeypatch.setattr(module, attr, lambda: dict(official))

    module.main([])

    payload = json.loads(module.OUT_JSON.read_text(encoding="utf-8"))
    assert module.OUT_CSV.exists()
    assert "research_candidate" not in payload
    assert "approves_trading" not in payload
    assert payload["selected_config"] == official
