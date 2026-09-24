"""Protect the local research runbook from an unrealistically short timeout."""

import sys

import refresh_local_research_data as refresh


def test_feature_quality_gets_time_for_full_panel(monkeypatch):
    # PLAIN ENGLISH: record the planned steps without downloading prices or
    # running the expensive feature grader during this quick test.
    planned = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["refresh_local_research_data.py", "--skip-research", "--skip-feature-research"],
    )
    monkeypatch.setattr(
        refresh,
        "_run_step",
        lambda step, dry_run: (planned.append(step) or True, 0.0),
    )
    monkeypatch.setattr(refresh, "_verify_outputs", lambda verbose=True: True)

    assert refresh.main() == 0
    feature_quality = next(step for step in planned if step.name == "feature_quality")
    assert feature_quality.timeout_seconds >= 3600
    names = [step.name for step in planned]
    assert names.index("feature_quality") < names.index("feature_health")
    assert names.index("feature_health") < names.index("factor_data_health")
