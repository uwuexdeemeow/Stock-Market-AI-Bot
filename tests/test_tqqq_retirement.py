from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import core_satellite_tqqq as tqqq


def test_write_tqqq_signal_is_retired_and_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(tqqq, "SIGNAL_DIR", str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        tqqq.write_tqqq_signal(pd.DataFrame())

    assert "core_satellite_alpha.py" in str(exc.value)
    assert not (tmp_path / "core_satellite_tqqq_signal.csv").exists()


def test_tqqq_research_backtest_function_remains_importable():
    assert callable(tqqq.run_tqqq_backtest)


def test_tqqq_default_cli_exits_before_live_generation():
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "core_satellite_tqqq.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "live signal generation is retired" in combined
    assert "core_satellite_alpha.py" in combined
