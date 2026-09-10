import json

import workflow_failure_annotation as annotation


def test_failed_step_ids_reports_only_failed_or_cancelled_steps():
    raw = '{"refresh":{"outcome":"success"},"verify":{"outcome":"failure"},"wait":{"outcome":"cancelled"}}'

    assert annotation.failed_step_ids(raw) == ["verify", "wait"]


def test_failed_step_ids_tolerates_bad_json():
    assert annotation.failed_step_ids("not-json") == []


def test_failure_details_reports_child_pipeline_root_cause(tmp_path):
    older = tmp_path / "daily_run_20260909.json"
    older.write_text(json.dumps({"results": [{"name": "old", "status": "failed"}]}))
    newest = tmp_path / "daily_run_20260910.json"
    newest.write_text(json.dumps({"results": [
        {"name": "factor_data_health", "status": "ok"},
        {
            "name": "core_satellite_signal",
            "status": "failed",
            "stderr_tail": "context\ncurrent_robustness_failed:factor_decay:report_stale",
        },
        {"name": "alpaca_submit", "status": "blocked", "blocked_by": "core_satellite_signal"},
    ]}))

    assert annotation.failure_details(str(tmp_path / "daily_run_*.json")) == [
        "core_satellite_signal: failed (current_robustness_failed:factor_decay:report_stale)"
    ]


def test_failure_details_tolerates_missing_or_bad_logs(tmp_path):
    assert annotation.failure_details(str(tmp_path / "missing*.json")) == []
    bad = tmp_path / "daily_run_bad.json"
    bad.write_text("not-json")
    assert annotation.failure_details(str(tmp_path / "daily_run_*.json")) == []
