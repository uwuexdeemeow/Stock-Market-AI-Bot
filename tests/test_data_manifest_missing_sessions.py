"""A price file with even one missing trading day must be flagged.

PLAIN ENGLISH: on 2026-09-24 a backup price source skipped 2026-09-22 for
44 stocks, and the old check allowed up to 2 missing days, so the hole got
saved.  Now a single missing NYSE session is a quality issue.
"""
from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd

from data_manifest import frame_quality_issues


def _frame(sessions):
    n = len(sessions)
    return pd.DataFrame({"Open": [10.0] * n, "High": [11.0] * n, "Low": [9.0] * n,
                         "Close": [10.0] * n, "Volume": [1000.0] * n}, index=sessions)


def _sessions():
    days = xcals.get_calendar("XNYS").sessions_in_range("2026-08-03", "2026-09-28")
    return pd.DatetimeIndex(days).tz_localize(None)


def test_complete_frame_has_no_session_issue():
    assert not any(i.startswith("missing_recent_sessions") for i in frame_quality_issues(_frame(_sessions())))


def test_one_missing_session_is_flagged():
    sessions = _sessions()
    gap = sessions[sessions != pd.Timestamp("2026-09-22")]
    assert "missing_recent_sessions:1" in frame_quality_issues(_frame(gap))


def test_market_holidays_are_not_missing_sessions():
    # Labor Day (2026-09-07) is not an NYSE session, so skipping it is fine.
    sessions = _sessions()
    assert pd.Timestamp("2026-09-07") not in sessions
    assert not any(i.startswith("missing_recent_sessions") for i in frame_quality_issues(_frame(sessions)))
