from datetime import datetime, timezone

import workflow_watchdog


def _run(created_at, conclusion="success"):
    return {"event": "schedule", "created_at": created_at, "conclusion": conclusion, "html_url": "https://example.test/run"}


def test_watchdog_detects_missing_due_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(workflow_watchdog, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(workflow_watchdog, "REPORT_FILE", tmp_path / "report.json")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(workflow_watchdog, "_is_nyse_session", lambda _clock: True)
    monkeypatch.setattr(workflow_watchdog, "_github_runs", lambda *_args: [])
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
