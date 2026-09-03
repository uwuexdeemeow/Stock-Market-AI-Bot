from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import alpaca_paper_trading as alpaca
import core_satellite_nested_walkforward as walkforward
import core_satellite_execution_stress as execution_stress
import core_satellite_survivorship_audit as survivorship_stress
import data_manifest
import factor_decay_monitor
import validation_bundle


def test_validation_file_hash_ignores_cross_platform_line_endings(tmp_path):
    """The same text must keep one evidence checksum on Windows and Linux."""
    windows_copy = tmp_path / "windows.json"
    linux_copy = tmp_path / "linux.json"
    # PLAIN ENGLISH: both files say the same thing; only their line-break bytes
    # differ, as they can after Git moves a file between Windows and Linux.
    windows_copy.write_bytes(b'{\r\n  "approved": true\r\n}\r\n')
    linux_copy.write_bytes(b'{\n  "approved": true\n}\n')

    assert validation_bundle.file_sha256(windows_copy) == validation_bundle.file_sha256(linux_copy)


def _passing_walkforward_result() -> dict:
    """Return the smallest result that clears every reliability approval gate."""
    return {
        "valid": True,
        "strategy": "core-alpha",
        "fold_count": 5,
        "approved_family_frequency": 0.6,
        "approved_config_frequency": 0.6,
        "mean_oos_sharpe": 1.0,
        "oos_positive_alpha_hit_rate": 0.8,
        "cost_stress_approval_pass": True,
        "mean_oos_max_drawdown_pct": -10.0,
        "worst_oos_max_drawdown_pct": -15.0,
        "worst_oos_turnover_pct": 100.0,
        "selection_bias_gap_sharpe": 0.2,
        "inner_score_vs_oos_qqq_alpha_correlation": 0.3,
        "overconfidence_gap_pct": 5.0,
        "fallback_rate": 0.0,
        "recent_fallback_years": [],
        "frozen_baseline_available": True,
        "selector_sharpe_uplift_vs_baseline": 0.0,
        "selector_alpha_hit_uplift_vs_baseline": 0.0,
    }


def test_walkforward_rejects_negative_selector_correlation_and_hidden_fallbacks():
    result = _passing_walkforward_result()
    assert walkforward.approval_status(result)["approved"] is True

    result["inner_score_vs_oos_qqq_alpha_correlation"] = -0.1
    result["fallback_rate"] = 0.4
    result["recent_fallback_years"] = [2026]
    result["overconfidence_gap_pct"] = 25.0
    approval = walkforward.approval_status(result)

    assert approval["approved"] is False
    assert any(reason.startswith("selector_alpha_correlation=") for reason in approval["reasons"])
    assert any(reason.startswith("fallback_rate=") for reason in approval["reasons"])
    assert "recent_relaxed_fallbacks:2026" in approval["reasons"]
    assert any(reason.startswith("overconfidence_gap=") for reason in approval["reasons"])


def test_fixed_frozen_baseline_skips_only_selector_specific_gates():
    """A one-config incumbent run has no selector, but all other gates remain active."""
    result = _passing_walkforward_result()
    result.update({
        "selection_mode": "fixed_frozen_baseline",
        "inner_score_vs_oos_qqq_alpha_correlation": -0.9,
        "overconfidence_gap_pct": 99.0,
        "selector_sharpe_uplift_vs_baseline": -5.0,
    })

    assert walkforward.approval_status(result)["approved"] is True

    result["worst_oos_max_drawdown_pct"] = -40.0
    approval = walkforward.approval_status(result)
    assert approval["approved"] is False
    assert any(reason.startswith("worst_oos_drawdown=") for reason in approval["reasons"])


def test_validation_bundle_detects_stale_report_fingerprint(tmp_path, monkeypatch):
    source = tmp_path / "walkforward.json"
    source.write_text("{}", encoding="utf-8")
    config = {"score_source": "regime_adaptive", "holding_days": 20}
    config_fingerprint = validation_bundle.strategy_config_fingerprint(config)
    dataset_fingerprint = "dataset-v1"
    report_paths = {}
    for name in ("survivorship", "execution_stress", "factor_decay"):
        path = tmp_path / f"{name}.json"
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "validation_context": {
                "config_fingerprint": config_fingerprint,
                "dataset_fingerprint": dataset_fingerprint,
            },
        }
        if name == "survivorship":
            payload.update({
                "survivorship_adjusted_score": 0.8,
                "rows": [
                    {"scenario": "watchlist_plus_failed_audit_tickers", "paper_ready": True, "audit_rebalance_selections": 0},
                    {"scenario": "delta_stressed_minus_base", "total_return_pct": 0.0, "max_drawdown_pct": 0.0},
                ],
            })
        elif name == "execution_stress":
            payload["rows"] = [{
                "scenario": "base",
                "paper_ready": True,
                "alpha_vs_qqq_pct": 1.0,
                "alpha_vs_blend_pct": 1.0,
                "max_drawdown_pct": -10.0,
            }]
        else:
            payload["edge_health_status"] = "pass"
        path.write_text(json.dumps(payload), encoding="utf-8")
        report_paths[name] = path
    monkeypatch.setattr(validation_bundle, "membership_status", lambda: {"complete": False})
    result = {
        "strategy": "core-alpha",
        "folds": [{"outer_year": 2025, "valid": True}],
        "live_config_approval": {"approved": True},
        "approved_live_config": {"config": config},
    }
    dataset = {"dataset_fingerprint": dataset_fingerprint, "manifest_path": "test"}

    bundle = validation_bundle.build_validation_bundle(
        result,
        source_json=str(source),
        report_paths=report_paths,
        dataset_context=dataset,
    )
    assert bundle["deployment"]["paper_approved"] is True
    assert bundle["deployment"]["real_capital_approved"] is False
    assert validation_bundle.validate_validation_bundle(bundle) == (True, [])

    stale = json.loads(report_paths["factor_decay"].read_text(encoding="utf-8"))
    stale["validation_context"]["dataset_fingerprint"] = "older-dataset"
    report_paths["factor_decay"].write_text(json.dumps(stale), encoding="utf-8")
    stale_bundle = validation_bundle.build_validation_bundle(
        result,
        source_json=str(source),
        report_paths=report_paths,
        dataset_context=dataset,
    )
    assert stale_bundle["robustness_reports"]["factor_decay"]["match"] is False
    assert stale_bundle["deployment"]["integrity_status"] == "provisional"
    assert stale_bundle["deployment"]["paper_approved"] is False


def test_validation_bundle_rebuild_restores_only_matching_approved_folds(tmp_path, monkeypatch):
    """Repair keeps matching evidence and never promotes paper state to real money."""
    source = tmp_path / "source.json"
    canonical = tmp_path / "signals" / "core_satellite_nested_walkforward.json"
    live_path = tmp_path / "signals" / "core_satellite_live_configs.json"
    bundle_path = tmp_path / "signals" / "core_satellite_validation_bundle.json"
    config = {"score_source": "regime_adaptive", "shape": "top3", "holding_days": 20}
    source.write_text(json.dumps({
        "strategy": "core-alpha",
        "fold_count": 1,
        "folds": [{"outer_year": 2025, "valid": True}],
        "live_config_approval": {"approved": True},
        "approved_live_config": {"config": config},
    }), encoding="utf-8")
    live_path.parent.mkdir(parents=True)
    live_path.write_text(json.dumps({
        "approved_live_configs": {"core-alpha": {"config": config}},
        "real_capital_approved": True,
    }), encoding="utf-8")
    monkeypatch.setattr(validation_bundle, "membership_status", lambda: {"complete": False})

    validation_bundle.rebuild_from_walkforward(
        source,
        live_config_path=live_path,
        bundle_path=bundle_path,
        canonical_source_path=canonical,
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    live = json.loads(live_path.read_text(encoding="utf-8"))
    assert len(bundle["folds"]) == 1
    assert "walkforward_folds_missing" not in bundle["deployment"]["reasons"]
    assert "walkforward_source_missing" not in bundle["deployment"]["reasons"]
    assert live["source_json"] == str(canonical)
    assert live["validation_bundle_hash"] == bundle["validation_bundle_hash"]
    assert live["real_capital_approved"] is False


def test_validation_bundle_rebuild_refuses_different_live_config(tmp_path):
    """Evidence for top-five holdings cannot be attached to a top-three bot."""
    source = tmp_path / "source.json"
    live_path = tmp_path / "live.json"
    source.write_text(json.dumps({
        "folds": [{"outer_year": 2025}],
        "live_config_approval": {"approved": True},
        "approved_live_config": {"config": {"shape": "top5"}},
    }), encoding="utf-8")
    live_path.write_text(json.dumps({
        "approved_live_configs": {"core-alpha": {"config": {"shape": "top3"}}},
    }), encoding="utf-8")

    try:
        validation_bundle.rebuild_from_walkforward(
            source,
            live_config_path=live_path,
            bundle_path=tmp_path / "bundle.json",
            canonical_source_path=tmp_path / "canonical.json",
        )
    except ValueError as exc:
        assert str(exc) == "walkforward_live_config_mismatch"
    else:
        raise AssertionError("mismatched strategy evidence was accepted")


def test_robustness_reports_preserve_full_live_config_identity(tmp_path, monkeypatch):
    """Every report must fingerprint the same behavior-changing live fields."""
    metrics_path = tmp_path / "core_satellite_alpha_metrics.json"
    expected = {
        "score_source": "regime_adaptive",
        "shape": "top3",
        "weighting": "sticky_score",
        "holding_days": 20,
        "overlay_gross": 0.5,
        "regime_ma_window": 100,
        "regime_high_vol": 0.3,
        "high_vol_mode": "percentile",
        "risk_control_mode": "off",
    }
    metrics_path.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(execution_stress, "SIGNAL_DIR", str(tmp_path))
    monkeypatch.setattr(survivorship_stress, "SIGNAL_DIR", str(tmp_path))
    monkeypatch.setattr(factor_decay_monitor, "METRICS_PATH", metrics_path)

    configs = (
        execution_stress._selected_config(),
        survivorship_stress._load_selected_config(),
        factor_decay_monitor._selected_config(),
    )
    expected_fingerprint = validation_bundle.strategy_config_fingerprint(expected)
    assert all(
        validation_bundle.strategy_config_fingerprint(config) == expected_fingerprint
        for config in configs
    )


def test_provider_overlap_accepts_adjusted_match_and_rejects_price_scale_change():
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    existing = pd.DataFrame({"Close": range(100, 110)}, index=dates)
    matching = pd.DataFrame({"Close": [value * 1.001 for value in range(100, 110)]}, index=dates)
    split_scaled = pd.DataFrame({"Close": [value / 2 for value in range(100, 110)]}, index=dates)

    assert data_manifest.compare_provider_overlap(existing, matching)["ok"] is True
    mismatch = data_manifest.compare_provider_overlap(existing, split_scaled)
    assert mismatch["ok"] is False
    assert mismatch["reason"] == "provider_overlap_price_mismatch"


def test_provider_overlap_allows_small_dividend_adjustment_gap():
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    existing = pd.DataFrame({"Close": range(100, 110)}, index=dates)
    dividend_adjusted = pd.DataFrame(
        {"Close": [value * 0.9925 for value in range(100, 110)]},
        index=dates,
    )

    result = data_manifest.compare_provider_overlap(existing, dividend_adjusted)

    assert result["ok"] is True
    assert result["median_limit_pct"] == 1.0


def test_submit_outcome_is_fail_closed_and_journaled_once(tmp_path, monkeypatch):
    outcome_path = tmp_path / "outcome.json"
    journal_path = tmp_path / "outcomes.csv"
    monkeypatch.setattr(alpaca, "SUBMIT_OUTCOME_FILE", outcome_path)
    monkeypatch.setattr(alpaca, "SUBMIT_OUTCOME_JOURNAL_FILE", journal_path)

    alpaca._begin_submit_outcome()
    initial = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert initial["status"] == "failed"
    assert initial["reason_code"] == "process_exited_before_final_outcome"

    alpaca._set_submit_outcome("no_action", "portfolio_already_aligned", planned_orders=0)
    alpaca._set_submit_outcome("no_action", "portfolio_already_aligned", planned_orders=0)
    journal = pd.read_csv(journal_path)
    assert len(journal) == 1
    assert journal.iloc[0]["status"] == "no_action"


def test_every_planned_order_receives_a_final_execution_state(tmp_path, monkeypatch):
    monkeypatch.setattr(alpaca, "SUBMIT_OUTCOME_FILE", tmp_path / "outcome.json")
    monkeypatch.setattr(alpaca, "SUBMIT_OUTCOME_JOURNAL_FILE", tmp_path / "outcomes.csv")
    monkeypatch.setattr(
        alpaca,
        "_write_broker_truth_gate_report",
        lambda: {"status": "pass", "summary": {"fail_count": 0, "warning_count": 0}},
    )
    statuses = {"accepted-filled": "filled", "accepted-open": "new"}
    monkeypatch.setattr(alpaca, "_order_status", lambda broker, order_id: statuses[order_id])

    alpaca._begin_submit_outcome()
    outcome = alpaca._finalize_submit_outcome(
        object(),
        planned_count=4,
        order_ids=["accepted-filled", "accepted-open", "SKIPPED:cash", "ERROR:rejected"],
    )

    accounted = (
        outcome["filled_orders"] + outcome["open_orders"]
        + outcome["skipped_orders"] + outcome["failed_orders"]
        + outcome["rejected_orders"]
    )
    assert accounted == outcome["planned_orders"]
    assert outcome["status"] == "failed"
