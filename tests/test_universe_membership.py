from __future__ import annotations

import pandas as pd

import universe_membership


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

    filtered, status = universe_membership.apply_membership_if_complete(
        _panel(),
        path=path,
        required_tickers=["AAA"],
        data_dir=data_dir,
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
    )

    assert status["applied"] is False
    assert "required_ticker_membership_missing" in status["reasons"]
    pd.testing.assert_frame_equal(filtered, _panel())
