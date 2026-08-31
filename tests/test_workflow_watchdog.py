from datetime import datetime, timezone

import workflow_watchdog


def _run(created_at, conclusion="success"):
    return {"event": "schedule", "created_at": created_at, "conclusion": conclusion, "html_url": "https://example.test/run"}


def _manual_run(created_at, conclusion="success", title="Daily Paper Trading (paper-session)"):
    row = _run(created_at, conclusion)
    row["event"] = "workflow_dispatch"
    row["display_title"] = title
    return row


def test_watchdog_detects_missing_due_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(workflow_watchdog, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(workflow_watchdog, "REPORT_FILE", tmp_path / "report.json")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(workflow_watchdog, "_is_nyse_session", lambda _clock: True)
    monkeypatch.setattr(workflow_watchdog, "_github_runs", lambda *_args: [])
    monkeypatch.setattr(workflow_watchdog, "_dispatch_workflow", lambda *_args, **_kwargs: False)
    sent = []
    monkeypatch.setattr(workflow_watchdog, "_send_telegram", lambda message: sent.append(message) or True)
    report = workflow_watchdog.check_workflows(now=datetime(2026, 8, 31, 23, tzinfo=timezone.utc))
    assert report["status"] == "fail"
    assert any(item.startswith("daily_paper:") for item in report["problems"])
    assert len(sent) == 1


def test_watchdog_ignores_offseason_duplicate_and_accepts_expected_run(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_watchdog, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(workflow_watchdog, "REPORT_FILE", tmp_path / "report.json")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(workflow_watchdog, "_is_nyse_session", lambda _clock: True)
    rows = [
        _run("2026-08-31T14:35:00Z", "success"),
        _run("2026-08-31T13:35:00Z", "success"),
        _run("2026-08-31T13:55:00Z", "success"),
        _run("2026-08-31T11:30:00Z", "success"),
        _run("2026-08-31T21:15:00Z", "success"),
    ]
    monkeypatch.setattr(workflow_watchdog, "_github_runs", lambda *_args: rows)
    monkeypatch.setattr(workflow_watchdog, "_send_telegram", lambda _message: True)
    report = workflow_watchdog.check_workflows(now=datetime(2026, 8, 31, 23, tzinfo=timezone.utc))
    assert report["status"] == "pass"


def test_watchdog_dispatches_missing_daily_run_only_inside_safe_window(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_watchdog, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(workflow_watchdog, "REPORT_FILE", tmp_path / "report.json")
    monkeypatch.setattr(
        workflow_watchdog,
        "WORKFLOWS",
        {"daily_paper": ("daily.yml", workflow_watchdog.time(9, 35), workflow_watchdog.time(9, 45), workflow_watchdog.time(10, 25), workflow_watchdog.time(11, 0))},
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(workflow_watchdog, "_is_nyse_session", lambda _clock: True)
    monkeypatch.setattr(workflow_watchdog, "_github_runs", lambda *_args: [])
    dispatched = []
    monkeypatch.setattr(
        workflow_watchdog,
        "_dispatch_workflow",
        lambda _repo, workflow, _token, *, inputs: dispatched.append((workflow, inputs)) or True,
    )

    report = workflow_watchdog.check_workflows(now=datetime(2026, 8, 31, 13, 50, tzinfo=timezone.utc))

    assert report["recovery_dispatches"] == ["daily_paper"]
    assert report["checks"]["daily_paper"]["reason"] == "fallback_dispatched"
    assert dispatched == [("daily.yml", {"force": "false", "dry_run": "false"})]


def test_watchdog_never_dispatches_daily_run_after_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_watchdog, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(workflow_watchdog, "REPORT_FILE", tmp_path / "report.json")
    monkeypatch.setattr(
        workflow_watchdog,
        "WORKFLOWS",
        {"daily_paper": ("daily.yml", workflow_watchdog.time(9, 35), workflow_watchdog.time(9, 45), workflow_watchdog.time(10, 25), workflow_watchdog.time(11, 0))},
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(workflow_watchdog, "_is_nyse_session", lambda _clock: True)
    monkeypatch.setattr(workflow_watchdog, "_github_runs", lambda *_args: [])
    monkeypatch.setattr(
        workflow_watchdog,
        "_dispatch_workflow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("late dispatch")),
    )

    report = workflow_watchdog.check_workflows(now=datetime(2026, 8, 31, 15, 30, tzinfo=timezone.utc))

    assert report["recovery_dispatches"] == []
    assert report["checks"]["daily_paper"]["recovery_dispatched"] is False


def test_watchdog_retries_failed_shadow_only_once(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    monkeypatch.setattr(workflow_watchdog, "STATE_FILE", state)
    monkeypatch.setattr(workflow_watchdog, "REPORT_FILE", tmp_path / "report.json")
    monkeypatch.setattr(workflow_watchdog, "WORKFLOWS", {
        "shadow_paper": ("shadow.yml", workflow_watchdog.time(9, 55), workflow_watchdog.time(10, 5), workflow_watchdog.time(16), workflow_watchdog.time(11, 30)),
    })
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(workflow_watchdog, "_is_nyse_session", lambda _clock: True)
    monkeypatch.setattr(workflow_watchdog, "_github_runs", lambda *_args: [_run("2026-08-31T13:55:00Z", "failure")])
    dispatched = []
    monkeypatch.setattr(workflow_watchdog, "_dispatch_workflow", lambda *_args, **_kwargs: dispatched.append(1) or True)

    workflow_watchdog.check_workflows(now=datetime(2026, 8, 31, 16, tzinfo=timezone.utc))
    workflow_watchdog.check_workflows(now=datetime(2026, 8, 31, 16, 10, tzinfo=timezone.utc))

    assert len(dispatched) == 1


def test_manual_daily_dry_run_does_not_count_as_real_session(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_watchdog, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(workflow_watchdog, "REPORT_FILE", tmp_path / "report.json")
    monkeypatch.setattr(workflow_watchdog, "WORKFLOWS", {
        "daily_paper": ("daily.yml", workflow_watchdog.time(9, 35), workflow_watchdog.time(9, 45), workflow_watchdog.time(10, 25), workflow_watchdog.time(11)),
    })
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(workflow_watchdog, "_is_nyse_session", lambda _clock: True)
    monkeypatch.setattr(workflow_watchdog, "_github_runs", lambda *_args: [_manual_run("2026-08-31T13:50:00Z", title="Daily Paper Trading (dry-run)")])
    monkeypatch.setattr(workflow_watchdog, "_dispatch_workflow", lambda *_args, **_kwargs: False)

    report = workflow_watchdog.check_workflows(now=datetime(2026, 8, 31, 15, 30, tzinfo=timezone.utc))

    assert report["status"] == "fail"
    assert report["checks"]["daily_paper"]["ran_today"] is False


def test_incident_key_stays_stable_when_reason_changes(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text('{"problems":["shadow_paper:run_pending:in_progress"],"problem_keys":["workflow:shadow_paper"]}')
    monkeypatch.setattr(workflow_watchdog, "STATE_FILE", state)
    monkeypatch.setattr(workflow_watchdog, "REPORT_FILE", tmp_path / "report.json")
    monkeypatch.setattr(workflow_watchdog, "WORKFLOWS", {
        "shadow_paper": ("shadow.yml", workflow_watchdog.time(9, 55), workflow_watchdog.time(10, 5), workflow_watchdog.time(16), workflow_watchdog.time(11, 30)),
    })
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(workflow_watchdog, "_is_nyse_session", lambda _clock: True)
    monkeypatch.setattr(workflow_watchdog, "_github_runs", lambda *_args: [_run("2026-08-31T13:55:00Z", "failure")])
    monkeypatch.setattr(workflow_watchdog, "_dispatch_workflow", lambda *_args, **_kwargs: False)
    sent = []
    monkeypatch.setattr(workflow_watchdog, "_send_telegram", lambda message: sent.append(message) or True)

    report = workflow_watchdog.check_workflows(now=datetime(2026, 8, 31, 17, tzinfo=timezone.utc))

    assert report["new_problems"] == []
    assert report["recovered_problems"] == []
    assert sent == []
