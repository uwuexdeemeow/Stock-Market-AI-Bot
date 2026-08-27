"""Point-in-time universe membership helpers for survivorship-safe research.

PLAIN ENGLISH: Today's stock list must not be pretended to exist unchanged in
2012. This module reads effective start/end dates and filters each historical
day to names that were genuinely eligible then. Until the membership table is
complete, validation reports the limitation and real capital stays blocked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from settings import DATA_DIR, SURVIVORSHIP_AUDIT_TICKERS, TRAIN_START, WATCHLIST


DEFAULT_MEMBERSHIP_PATH = Path("data/universe_membership.csv")
REQUIRED_COLUMNS = {"ticker", "effective_from", "effective_to", "status", "source"}
# Production-quality membership needs enough provenance for another person to
# retrieve and legally inspect the same free source. A ticker/date table with
# only a vague source label remains useful research input but cannot clear the
# survivorship gate.
PROVENANCE_COLUMNS = {"source_url", "retrieved_at", "license", "access_cost"}
DEFAULT_MIN_ACTIVE_MEMBERS = 400
DEFAULT_MIN_INACTIVE_MEMBERS = len(SURVIVORSHIP_AUDIT_TICKERS)


def _canonical_ticker(value: object) -> str:
    """Match common data-file symbols such as BRK-B to source symbol BRK.B."""
    return str(value).upper().strip().replace(".", "-")


def load_membership(path: Path = DEFAULT_MEMBERSHIP_PATH) -> pd.DataFrame:
    """Load and normalize the date-effective membership table."""
    if not path.exists():
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("membership table missing columns: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].map(_canonical_ticker)
    frame["effective_from"] = pd.to_datetime(frame["effective_from"], errors="coerce")
    frame["effective_to"] = pd.to_datetime(frame["effective_to"], errors="coerce")
    return frame


def membership_status(
    path: Path = DEFAULT_MEMBERSHIP_PATH,
    *,
    required_tickers: Iterable[str] = WATCHLIST,
    data_dir: Path = Path(DATA_DIR),
    coverage_start: object = TRAIN_START,
    coverage_end: object | None = None,
    min_active_members: int = DEFAULT_MIN_ACTIVE_MEMBERS,
    min_inactive_members: int = DEFAULT_MIN_INACTIVE_MEMBERS,
    min_price_coverage: float = 0.95,
) -> dict:
    """Report whether the universe is complete enough for real-capital claims."""
    try:
        frame = load_membership(path)
    except ValueError as exc:
        return {"complete": False, "path": str(path), "reasons": [str(exc)]}
    required = {_canonical_ticker(ticker) for ticker in required_tickers}
    covered = set(frame["ticker"]) if not frame.empty else set()
    missing = sorted(required - covered)
    invalid_dates = frame[frame["effective_from"].isna()] if not frame.empty else frame
    invalid_sources = frame[frame["source"].fillna("").astype(str).str.strip() == ""] if not frame.empty else frame
    missing_provenance_columns = sorted(PROVENANCE_COLUMNS - set(frame.columns))
    if missing_provenance_columns:
        invalid_source_urls = frame
        invalid_retrieved_at = frame
        invalid_licenses = frame
        non_free_sources = frame
    else:
        source_urls = frame["source_url"].fillna("").astype(str).str.strip()
        invalid_source_urls = frame[~source_urls.str.match(r"^https?://", case=False)]
        retrieved_at = pd.to_datetime(frame["retrieved_at"], errors="coerce", utc=True)
        invalid_retrieved_at = frame[retrieved_at.isna()]
        invalid_licenses = frame[frame["license"].fillna("").astype(str).str.strip() == ""]
        access_cost = frame["access_cost"].fillna("").astype(str).str.strip().str.lower()
        non_free_sources = frame[access_cost != "free"]
    start = pd.Timestamp(coverage_start).tz_localize(None).normalize()
    end = (
        pd.Timestamp(coverage_end).tz_localize(None).normalize()
        if coverage_end is not None
        else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    )
    # Historical scope includes every constituent whose membership overlaps
    # the backtest era. A current-only list of 62 winners cannot pass by adding
    # one token delisted name; a genuine S&P source should have about 500 names
    # active at the start and many removals over the full period.
    overlaps = (
        (frame["effective_from"] <= end)
        & (frame["effective_to"].isna() | (frame["effective_to"] >= start))
    ) if not frame.empty else pd.Series(dtype=bool)
    historical = frame.loc[overlaps].copy() if not frame.empty else frame
    active_at_start = historical[
        (historical["effective_from"] <= start)
        & (historical["effective_to"].isna() | (historical["effective_to"] >= start))
    ] if not historical.empty else historical
    inactive = historical[historical["effective_to"].notna()] if not historical.empty else historical
    historical_tickers = sorted(set(historical["ticker"].astype(str)))
    historical_with_data = [ticker for ticker in historical_tickers if (data_dir / f"{ticker}.parquet").exists()]
    historical_without_data = sorted(set(historical_tickers) - set(historical_with_data))
    price_coverage = len(historical_with_data) / max(len(historical_tickers), 1)
    reasons: list[str] = []
    if not path.exists():
        reasons.append("membership_table_missing")
    if missing:
        reasons.append("required_ticker_membership_missing")
    if len(invalid_dates):
        reasons.append("effective_from_missing")
    if len(invalid_sources):
        reasons.append("membership_source_missing")
    if missing_provenance_columns:
        reasons.append("membership_provenance_columns_missing")
    if len(invalid_source_urls):
        reasons.append("membership_source_url_missing_or_invalid")
    if len(invalid_retrieved_at):
        reasons.append("membership_retrieved_at_missing_or_invalid")
    if len(invalid_licenses):
        reasons.append("membership_license_missing")
    if len(non_free_sources):
        reasons.append("membership_source_not_verified_free")
    if int(active_at_start["ticker"].nunique()) < int(min_active_members):
        reasons.append("historical_universe_too_small")
    if int(inactive["ticker"].nunique()) < int(min_inactive_members):
        reasons.append("inactive_membership_coverage_too_small")
    if price_coverage < float(min_price_coverage):
        reasons.append("historical_price_coverage_incomplete")
    return {
        "complete": not reasons,
        "path": str(path),
        "rows": int(len(frame)),
        "source_count": int(frame["source"].nunique()) if not frame.empty else 0,
        "required_provenance_columns": sorted(PROVENANCE_COLUMNS),
        "missing_provenance_columns": missing_provenance_columns,
        "provenance_valid_rows": int(
            len(frame)
            - len(set(invalid_source_urls.index) | set(invalid_retrieved_at.index) | set(invalid_licenses.index) | set(non_free_sources.index))
        ) if not missing_provenance_columns else 0,
        "required_tickers": len(required),
        "missing_required_tickers": missing,
        "coverage_start": start.strftime("%Y-%m-%d"),
        "coverage_end": end.strftime("%Y-%m-%d"),
        "active_members_at_coverage_start": int(active_at_start["ticker"].nunique()),
        "minimum_active_members": int(min_active_members),
        "inactive_members": int(inactive["ticker"].nunique()),
        "minimum_inactive_members": int(min_inactive_members),
        "historical_tickers": len(historical_tickers),
        "historical_tickers_with_data": len(historical_with_data),
        "historical_price_coverage": round(float(price_coverage), 6),
        "minimum_price_coverage": float(min_price_coverage),
        "missing_historical_price_tickers": historical_without_data,
        "reasons": reasons,
    }


def apply_membership_if_complete(
    panel: pd.DataFrame,
    *,
    path: Path = DEFAULT_MEMBERSHIP_PATH,
    required_tickers: Iterable[str] = WATCHLIST,
    data_dir: Path = Path(DATA_DIR),
    coverage_start: object = TRAIN_START,
    coverage_end: object | None = None,
    min_active_members: int = DEFAULT_MIN_ACTIVE_MEMBERS,
    min_inactive_members: int = DEFAULT_MIN_INACTIVE_MEMBERS,
    min_price_coverage: float = 0.95,
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
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        min_active_members=min_active_members,
        min_inactive_members=min_inactive_members,
        min_price_coverage=min_price_coverage,
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


def main() -> int:
    """Print an honest membership-coverage report for people and automation."""
    parser = argparse.ArgumentParser(
        description="Check whether point-in-time universe membership evidence is complete."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the coverage report (also the default action).",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_MEMBERSHIP_PATH,
        help="Membership CSV to validate.",
    )
    args = parser.parse_args()
    status = membership_status(args.path)
    print(json.dumps(status, indent=2, default=str))
    # Incomplete coverage is a valid diagnostic result. Deployment code reads
    # complete=false and keeps real capital blocked; the status command works.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
