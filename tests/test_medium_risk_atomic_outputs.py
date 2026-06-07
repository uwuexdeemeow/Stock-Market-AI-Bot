from __future__ import annotations

from pathlib import Path


def test_medium_risk_scripts_use_atomic_output_writers():
    scripts = [
        Path("core_satellite_survivorship_audit.py"),
        Path("core_satellite_execution_stress.py"),
        Path("core_satellite_drawdown_throttle.py"),
        Path("factor_decay_monitor.py"),
    ]

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        assert "atomic_write_csv" in source
        assert "atomic_write_json" in source
        assert ".to_csv(OUT_CSV" not in source
        assert "OUT_JSON.write_text" not in source
