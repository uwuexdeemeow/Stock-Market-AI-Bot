"""Regression tests for failed-company symbol identity checks."""

import pandas as pd

import survivorship_audit as audit


def _prices(start: str, periods: int = 600) -> pd.DataFrame:
    """Build a small business-day price history for identity tests."""
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({"Close": range(1, periods + 1)}, index=dates)


def test_reused_shld_symbol_is_not_sears_history():
    """Modern SHLD data begins years after Sears failed and must be rejected."""
    profile = audit._profile_frame("SHLD", _prices("2023-09-14"))

    assert profile["status"] == "symbol_reuse"
    assert profile["symbol_history_valid"] is False
    assert audit.available_audit_tickers([profile]) == []


def test_history_reaching_before_failure_is_accepted():
    """A long FRC history beginning before its failure remains useful evidence."""
    profile = audit._profile_frame("FRC", _prices("2018-01-02", periods=1_500))
    profile["status"] = "available"

    assert profile["symbol_history_valid"] is True
    assert audit.available_audit_tickers([profile]) == ["FRC"]
