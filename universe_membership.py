"""Point-in-time universe membership helpers for survivorship-safe research.

PLAIN ENGLISH: Today's stock list must not be pretended to exist unchanged in
2012. This module reads effective start/end dates and filters each historical
day to names that were genuinely eligible then. Until the membership table is
complete, validation reports the limitation and real capital stays blocked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from settings import DATA_DIR, WATCHLIST


DEFAULT_MEMBERSHIP_PATH = Path("data/universe_membership.csv")
REQUIRED_COLUMNS = {"ticker", "effective_from", "effective_to", "status", "source"}


def load_membership(path: Path = DEFAULT_MEMBERSHIP_PATH) -> pd.DataFrame:
    """Load and normalize the date-effective membership table."""
    if not path.exists():
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("membership table missing columns: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["effective_from"] = pd.to_datetime(frame["effective_from"], errors="coerce")
    frame["effective_to"] = pd.to_datetime(frame["effective_to"], errors="coerce")
    return frame


def membership_status(
    path: Path = DEFAULT_MEMBERSHIP_PATH,
    *,
    required_tickers: Iterable[str] = WATCHLIST,
    data_dir: Path = Path(DATA_DIR),
) -> dict:
    """Report whether the universe is complete enough for real-capital claims."""
    try:
        frame = load_membership(path)
    except ValueError as exc:
        return {"complete": False, "path": str(path), "reasons": [str(exc)]}
    required = {str(ticker).upper() for ticker in required_tickers}
    covered = set(frame["ticker"]) if not frame.empty else set()
    missing = sorted(required - covered)
    invalid_dates = frame[frame["effective_from"].isna()] if not frame.empty else frame
    delisted = frame[frame["status"].astype(str).str.lower() == "delisted"] if not frame.empty else frame
    delisted_with_data = [
        ticker for ticker in delisted["ticker"].astype(str).tolist()
        if (data_dir / f"{ticker}.parquet").exists()
    ]
    reasons: list[str] = []
    if not path.exists():
        reasons.append("membership_table_missing")
    if missing:
        reasons.append("required_ticker_membership_missing")
    if len(invalid_dates):
        reasons.append("effective_from_missing")
    if len(delisted) == 0:
        reasons.append("delisted_membership_missing")
    return {
        "complete": not reasons,
        "path": str(path),
        "rows": int(len(frame)),
        "required_tickers": len(required),
        "missing_required_tickers": missing,
        "delisted_tickers": int(len(delisted)),
        "delisted_tickers_with_data": sorted(delisted_with_data),
        "reasons": reasons,
    }


def apply_membership_if_complete(
    panel: pd.DataFrame,
    *,
    path: Path = DEFAULT_MEMBERSHIP_PATH,
    required_tickers: Iterable[str] = WATCHLIST,
    data_dir: Path = Path(DATA_DIR),
) -> tuple[pd.DataFrame, dict]:
    """Apply point-in-time membership only when the table passes validation.

    PLAIN ENGLISH: silently filtering with a half-built membership file can be
    worse than not filtering because it removes companies unevenly.  This
    helper therefore has two honest outcomes: use a complete table, or leave
    the panel unchanged and return the reasons validation is still blocked.
    """
    status = membership_status(
        path,
        required_tickers=required_tickers,
        data_dir=data_dir,
    )
    if not status.get("complete", False):
        return panel.copy(), {**status, "applied": False}
    membership = load_membership(path)
    filtered = filter_panel_point_in_time(panel, membership)
    return filtered, {
        **status,
        "applied": True,
        "input_rows": int(len(panel)),
        "output_rows": int(len(filtered)),
        "removed_rows": int(len(panel) - len(filtered)),
    }


def filter_panel_point_in_time(panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Keep a ticker/date row only while that ticker's membership is active."""
    if panel.empty:
        return panel.copy()
    required_panel = {"ticker", "date"}
    if not required_panel.issubset(panel.columns):
        raise ValueError("panel must contain ticker and date columns")
    rows = panel.copy()
    rows["ticker"] = rows["ticker"].astype(str).str.upper()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    joined = rows.merge(membership, on="ticker", how="inner")
    active = joined["date"] >= joined["effective_from"]
    active &= joined["effective_to"].isna() | (joined["date"] <= joined["effective_to"])
    return joined.loc[active, panel.columns].copy()
