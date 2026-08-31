import workflow_failure_annotation as annotation


def test_failed_step_ids_reports_only_failed_or_cancelled_steps():
    raw = '{"refresh":{"outcome":"success"},"verify":{"outcome":"failure"},"wait":{"outcome":"cancelled"}}'

    assert annotation.failed_step_ids(raw) == ["verify", "wait"]


def test_failed_step_ids_tolerates_bad_json():
    assert annotation.failed_step_ids("not-json") == []
