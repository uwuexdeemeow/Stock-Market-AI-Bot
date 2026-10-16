"""Tests for v_rule_helper.py: it must give the same split as the H-index-risk test.

PLAIN ENGLISH: the helper is the hand-use version of rule V.  If its
arithmetic drifted from the tested rule, you'd be following an untested
rule without knowing it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import v_rule_helper as helper

spec = importlib.util.spec_from_file_location(
    "index_risk_check", Path("research_evidence/index_risk_20261016/index_risk_check.py"))
ir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir)


@pytest.mark.parametrize("daily_vol", [0.005, 0.02, 0.04])
def test_matches_the_tested_rule(daily_vol):
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2024-01-01", periods=120)
    closes = pd.Series(100 * np.cumprod(1 + rng.normal(0, daily_vol, len(days))), index=days)
    assert helper.qqq_weight(closes)["qqq_weight"] == pytest.approx(ir.vol_weight(closes, days[-1]))


def test_needs_enough_history():
    closes = pd.Series([100.0] * 10, index=pd.bdate_range("2024-01-01", periods=10))
    with pytest.raises(ValueError):
        helper.qqq_weight(closes)
