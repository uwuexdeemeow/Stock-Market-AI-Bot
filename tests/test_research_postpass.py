from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_research_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "research.py"
    spec = importlib.util.spec_from_file_location("stock_research_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_xs_rank_summary_rejects_write_errors():
    research = _load_research_module()

    with pytest.raises(RuntimeError) as exc:
        research._validate_xs_rank_summary({
            "updated": 1,
            "new_cols": ["xs_rank_market_mom_20d"],
            "write_errors": ["AAPL: permission denied"],
        })

    assert "write_errors" in str(exc.value)


def test_validate_xs_rank_summary_rejects_zero_updates():
    research = _load_research_module()

    with pytest.raises(RuntimeError) as exc:
        research._validate_xs_rank_summary({
            "updated": 0,
            "new_cols": ["xs_rank_market_mom_20d"],
            "write_errors": [],
        })

    assert "updated 0" in str(exc.value)


def test_validate_xs_rank_summary_rejects_empty_new_columns():
    research = _load_research_module()

    with pytest.raises(RuntimeError) as exc:
        research._validate_xs_rank_summary({
            "updated": 3,
            "new_cols": [],
            "write_errors": [],
        })

    assert "produced no xs-rank columns" in str(exc.value)


def test_research_xs_only_exits_nonzero_on_bad_summary(monkeypatch):
    research = _load_research_module()
    monkeypatch.setattr(research, "WATCHLIST", ["AAPL"])
    monkeypatch.setattr(research, "SURVIVORSHIP_TRAINING_TICKERS", [])
    monkeypatch.setattr(
        research,
        "apply_cross_sectional_rank_features",
        lambda tickers, data_dir: {"updated": 0, "new_cols": [], "skipped": [], "write_errors": []},
    )
    monkeypatch.setattr(sys, "argv", ["research.py", "--xs-only"])

    with pytest.raises(SystemExit) as exc:
        research.main()

    assert exc.value.code == 1
