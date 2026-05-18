from __future__ import annotations

import daily_run


def _names(steps):
    return [step.name for step in steps]


def test_default_steps_include_one_shared_signal_before_brokers():
    steps = daily_run.build_steps(
        skip_refresh=False,
        run_moomoo=True,
        run_alpaca=True,
    )
    names = _names(steps)
    assert names.count("core_satellite_signal") == 1
    assert names.index("core_satellite_signal") < names.index("moomoo_submit")
    assert names.index("core_satellite_signal") < names.index("alpaca_submit")
    assert "moomoo_signal" not in names


def test_daily_refresh_forces_etf_download():
    etf_step = next(step for step in daily_run.DATA_REFRESH_STEPS if step.name == "refresh_etf_data")
    assert "--refresh" in etf_step.cmd
    assert "--force" in etf_step.cmd


def test_alpaca_only_still_generates_shared_signal():
    steps = daily_run.build_steps(
        skip_refresh=True,
        run_moomoo=False,
        run_alpaca=True,
    )
    names = _names(steps)
    assert names.count("core_satellite_signal") == 1
    assert names.index("core_satellite_signal") < names.index("alpaca_submit")
    assert "moomoo_submit" not in names


def test_skip_factor_refresh_keeps_etf_refresh_and_signal():
    steps = daily_run.build_steps(
        skip_refresh=False,
        skip_factor_refresh=True,
        run_moomoo=False,
        run_alpaca=True,
    )
    names = _names(steps)
    assert "refresh_etf_data" in names
    assert "refresh_factor_data" not in names
    assert "refresh_feature_quality" not in names
    assert names.index("refresh_etf_data") < names.index("core_satellite_signal")
    assert names.index("core_satellite_signal") < names.index("alpaca_submit")


def test_core_signal_failure_blocks_broker_steps(monkeypatch):
    steps = [
        daily_run.CORE_SATELLITE_SIGNAL_STEP,
        *daily_run.MOOMOO_STEPS[:1],
        *daily_run.ALPACA_STEPS[:1],
    ]
    called: list[str] = []

    def fake_run_step(name, cmd, description, dry_run=False, timeout=300):
        called.append(name)
        if name == "core_satellite_signal":
            return {"name": name, "status": "failed", "elapsed": 0.1}
        return {"name": name, "status": "ok", "elapsed": 0.1}

    monkeypatch.setattr(daily_run, "run_step", fake_run_step)
    results = daily_run.run_steps(steps, dry_run=False, timeout=1)
    assert called == ["core_satellite_signal"]
    assert [r["status"] for r in results] == ["failed", "blocked", "blocked"]
    assert results[1]["blocked_by"] == "core_satellite_signal"
    assert results[2]["blocked_by"] == "core_satellite_signal"


def test_refresh_failure_blocks_all_later_steps(monkeypatch):
    steps = daily_run.build_steps(
        skip_refresh=False,
        run_moomoo=True,
        run_alpaca=True,
    )
    called: list[str] = []

    def fake_run_step(name, cmd, description, dry_run=False, timeout=300):
        called.append(name)
        if name == "refresh_factor_data":
            return {"name": name, "status": "error", "elapsed": 0.1}
        return {"name": name, "status": "ok", "elapsed": 0.1}

    monkeypatch.setattr(daily_run, "run_step", fake_run_step)
    results = daily_run.run_steps(steps, dry_run=False, timeout=1)
    assert called == ["refresh_etf_data", "refresh_factor_data"]
    blocked = [r for r in results if r["status"] == "blocked"]
    assert blocked
    assert all(r["blocked_by"] == "refresh_factor_data" for r in blocked)
    assert next(r for r in results if r["name"] == "core_satellite_signal")["status"] == "blocked"
    assert next(r for r in results if r["name"] == "alpaca_submit")["status"] == "blocked"


def test_refresh_factor_data_uses_longer_step_timeout(monkeypatch):
    captured: dict[str, int] = {}
    step = next(step for step in daily_run.DATA_REFRESH_STEPS if step.name == "refresh_factor_data")

    def fake_run_step(name, cmd, description, dry_run=False, timeout=300):
        captured[name] = timeout
        return {"name": name, "status": "ok", "elapsed": 0.1}

    monkeypatch.setattr(daily_run, "run_step", fake_run_step)
    daily_run.run_steps([step], dry_run=False, timeout=1)

    assert captured["refresh_factor_data"] == 1800


def test_dry_run_skipped_critical_steps_do_not_block(monkeypatch):
    steps = [
        daily_run.DATA_REFRESH_STEPS[0],
        daily_run.CORE_SATELLITE_SIGNAL_STEP,
        daily_run.ALPACA_STEPS[0],
    ]

    def fake_run_step(name, cmd, description, dry_run=False, timeout=300):
        return {"name": name, "status": "skipped", "elapsed": 0.0}

    monkeypatch.setattr(daily_run, "run_step", fake_run_step)
    results = daily_run.run_steps(steps, dry_run=True, timeout=1)
    assert [r["status"] for r in results] == ["skipped", "skipped", "skipped"]
