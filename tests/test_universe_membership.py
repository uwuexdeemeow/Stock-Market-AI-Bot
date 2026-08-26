from __future__ import annotations

import json
import sys

import pandas as pd

import universe_membership


def test_status_cli_reports_missing_table(tmp_path, capsys):
    original_argv = sys.argv
    try:
        sys.argv = [
            "universe_membership.py",
            "--status",
            "--path",
            str(tmp_path / "missing.csv"),
        ]
        assert universe_membership.main() == 0
    finally:
        sys.argv = original_argv

    payload = json.loads(capsys.readouterr().out)
    assert payload["complete"] is False
    assert "membership_table_missing" in payload["reasons"]


def _panel() -> pd.DataFrame:
    """Small two-stock panel spanning dates before and after one membership."""
    return pd.DataFrame([
        {"ticker": "AAA", "date": "2020-01-02", "score": 1.0},
        {"ticker": "AAA", "date": "2020-06-02", "score": 2.0},
        {"ticker": "BBB", "date": "2020-06-02", "score": 3.0},
        {"ticker": "BBB", "date": "2021-02-01", "score": 4.0},
    ])


def test_complete_membership_filters_ineligible_ticker_dates(tmp_path):
    """Rows survive only inside each ticker's effective membership interval."""
    path = tmp_path / "membership.csv"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame([
        {
            "ticker": "AAA",
            "effective_from": "2020-05-01",
            "effective_to": "",
            "status": "active",
            "source": "test-fixture",
        },
        {
            "ticker": "BBB",
            "effective_from": "2019-01-01",
            "effective_to": "2020-12-31",
            "status": "delisted",
            "source": "test-fixture",
        },
    ]).to_csv(path, index=False)
    # The strict production gate requires price files for the historical
    # universe. Tiny fixtures lower the population thresholds explicitly.
    pd.DataFrame({"Close": [1.0]}).to_parquet(data_dir / "AAA.parquet")
    pd.DataFrame({"Close": [1.0]}).to_parquet(data_dir / "BBB.parquet")

    filtered, status = universe_membership.apply_membership_if_complete(
        _panel(),
        path=path,
        required_tickers=["AAA"],
        data_dir=data_dir,
        coverage_start="2020-01-01",
        coverage_end="2021-12-31",
        min_active_members=1,
        min_inactive_members=1,
        min_price_coverage=1.0,
    )

    assert status["applied"] is True
    assert status["removed_rows"] == 2
    assert list(zip(filtered["ticker"], filtered["date"].dt.strftime("%Y-%m-%d"))) == [
        ("AAA", "2020-06-02"),
        ("BBB", "2020-06-02"),
    ]


def test_incomplete_membership_never_partially_filters_panel(tmp_path):
    """A partial table leaves all research rows intact and reports the blocker."""
    path = tmp_path / "membership.csv"
    pd.DataFrame([
        {
            "ticker": "AAA",
            "effective_from": "2020-05-01",
            "effective_to": "",
            "status": "active",
            "source": "test-fixture",
        },
    ]).to_csv(path, index=False)

    filtered, status = universe_membership.apply_membership_if_complete(
        _panel(),
        path=path,
        required_tickers=["AAA", "BBB"],
        data_dir=tmp_path,
        coverage_start="2020-01-01",
        coverage_end="2021-12-31",
        min_active_members=1,
        min_inactive_members=1,
        min_price_coverage=1.0,
    )

    assert status["applied"] is False
    assert "required_ticker_membership_missing" in status["reasons"]
    pd.testing.assert_frame_equal(filtered, _panel())


def test_current_only_membership_cannot_fake_historical_completeness(tmp_path):
    path = tmp_path / "membership.csv"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {
            "ticker": f"NOW{i}",
            "effective_from": "2024-01-01",
            "effective_to": "",
            "status": "active",
            "source": "test-fixture",
        }
        for i in range(10)
    ]
    rows.append({
        "ticker": "OLD",
        "effective_from": "2015-01-01",
        "effective_to": "2018-01-01",
        "status": "removed",
        "source": "test-fixture",
    })
    pd.DataFrame(rows).to_csv(path, index=False)

    status = universe_membership.membership_status(
        path,
        required_tickers=[],
        data_dir=data_dir,
        coverage_start="2015-01-01",
        coverage_end="2026-01-01",
        min_active_members=400,
        min_inactive_members=17,
        min_price_coverage=0.95,
    )

    assert status["complete"] is False
    assert "historical_universe_too_small" in status["reasons"]
    assert "inactive_membership_coverage_too_small" in status["reasons"]
    assert "historical_price_coverage_incomplete" in status["reasons"]
