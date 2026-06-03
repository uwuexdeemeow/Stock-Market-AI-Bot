"""paper_shadow_compare.py — Compare Alpaca paper equity with shadow paper equity.

PLAIN ENGLISH: Alpaca paper equity is the broker account that actually receives
paper orders.  Shadow paper equity is a "what if" account that follows a
candidate config without submitting orders.  This script puts both curves on a
common return scale so you can quickly see which one is ahead.

Usage:
    python3 paper_shadow_compare.py
    python3 paper_shadow_compare.py --json-out signals/paper_shadow_compare.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from safe_io import atomic_write_csv, atomic_write_json
from settings import SIGNAL_DIR


# PLAIN ENGLISH: These are the default files written by the real paper broker,
# the shadow journal, and this comparison script.
ALPACA_EQUITY_FILE = Path(SIGNAL_DIR) / "alpaca_paper_equity.csv"
SHADOW_EQUITY_FILE = Path(SIGNAL_DIR) / "shadow_paper_equity.csv"
COMPARE_CSV_FILE = Path(SIGNAL_DIR) / "paper_shadow_compare.csv"
COMPARE_JSON_FILE = Path(SIGNAL_DIR) / "paper_shadow_compare.json"


def _read_equity_csv(path: Path, *, equity_column: str = "equity") -> pd.DataFrame:
    """Read an equity CSV and normalize its date/equity columns.

    PLAIN ENGLISH: CSV files store everything as text.  This turns the date
    column into real dates and the equity column into numbers so we can compare
    returns safely.
    """
    if not path.exists():
        return pd.DataFrame(columns=["date", "equity"])
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["date", "equity"])
    if "date" not in df.columns or equity_column not in df.columns:
        return pd.DataFrame(columns=["date", "equity"])

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["equity"] = pd.to_numeric(out[equity_column], errors="coerce")
    out = out.dropna(subset=["date", "equity"]).sort_values("date")
    return out[["date", "equity"]].drop_duplicates(subset=["date"], keep="last")


def _first_row_on_or_after(df: pd.DataFrame, start_date: object) -> pd.Series | None:
    """Return the first row whose date is on/after start_date."""
    if df.empty:
        return None
    subset = df[df["date"] >= start_date]
    if subset.empty:
        return None
    return subset.iloc[0]


def _return_pct(latest: float | None, start: float | None) -> float | None:
    """Calculate percentage return, or None if the inputs are unusable."""
    if latest is None or start is None or start <= 0:
        return None
    return (float(latest) / float(start) - 1.0) * 100.0


def _rounded(value: object, digits: int = 4) -> float | None:
    """Round a number for JSON output while preserving None for missing values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def build_comparison_payload(
    *,
    alpaca_path: Path = ALPACA_EQUITY_FILE,
    shadow_path: Path = SHADOW_EQUITY_FILE,
) -> tuple[dict, pd.DataFrame]:
    """Build a JSON summary and row-by-row CSV comparison.

    PLAIN ENGLISH: The two curves do not always update on the same date.  We
    choose the later first date as the shared start, then compare each latest
    value from that point forward.  The summary clearly flags when the latest
    dates do not match.
    """
    alpaca = _read_equity_csv(alpaca_path)
    shadow = _read_equity_csv(shadow_path)

    empty_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "missing_data",
        "reason": "alpaca_or_shadow_equity_missing",
    }
    if alpaca.empty or shadow.empty:
        return empty_summary, pd.DataFrame()

    common_start_date = max(alpaca["date"].min(), shadow["date"].min())
    alpaca_start = _first_row_on_or_after(alpaca, common_start_date)
    shadow_start = _first_row_on_or_after(shadow, common_start_date)
    if alpaca_start is None or shadow_start is None:
        empty_summary["reason"] = "no_common_start_date"
        return empty_summary, pd.DataFrame()

    alpaca_latest = alpaca.iloc[-1]
    shadow_latest = shadow.iloc[-1]
    alpaca_start_equity = float(alpaca_start["equity"])
    shadow_start_equity = float(shadow_start["equity"])
    alpaca_return = _return_pct(float(alpaca_latest["equity"]), alpaca_start_equity)
    shadow_return = _return_pct(float(shadow_latest["equity"]), shadow_start_equity)
    return_spread = (
        None if alpaca_return is None or shadow_return is None
        else alpaca_return - shadow_return
    )

    # PLAIN ENGLISH: This merged table is useful for the dashboard chart.  It
    # shows whatever each curve knows on each date, without inventing prices.
    merged = pd.merge(
        alpaca.rename(columns={"equity": "alpaca_equity"}),
        shadow.rename(columns={"equity": "shadow_equity"}),
        on="date",
        how="outer",
    ).sort_values("date")
    merged = merged[merged["date"] >= common_start_date].copy()
    merged["alpaca_return_pct"] = (
        pd.to_numeric(merged["alpaca_equity"], errors="coerce") / alpaca_start_equity - 1.0
    ) * 100.0
    merged["shadow_return_pct"] = (
        pd.to_numeric(merged["shadow_equity"], errors="coerce") / shadow_start_equity - 1.0
    ) * 100.0
    merged["return_spread_pct"] = merged["alpaca_return_pct"] - merged["shadow_return_pct"]
    for col in ("alpaca_return_pct", "shadow_return_pct", "return_spread_pct"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").round(4)
    merged["date"] = merged["date"].astype(str)

    latest_dates_match = str(alpaca_latest["date"]) == str(shadow_latest["date"])
    aligned_rows = merged.dropna(subset=["alpaca_equity", "shadow_equity"])
    winner = "tie"
    if return_spread is not None:
        if return_spread > 0:
            winner = "alpaca"
        elif return_spread < 0:
            winner = "shadow"

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
        "common_start_date": str(common_start_date),
        "aligned_days": int(len(aligned_rows)),
        "latest_dates_match": bool(latest_dates_match),
        "alpaca": {
            "latest_date": str(alpaca_latest["date"]),
            "latest_equity": _rounded(alpaca_latest["equity"], 2),
            "start_equity": _rounded(alpaca_start_equity, 2),
            "return_pct_since_common_start": _rounded(alpaca_return, 4),
        },
        "shadow": {
            "latest_date": str(shadow_latest["date"]),
            "latest_equity": _rounded(shadow_latest["equity"], 2),
            "start_equity": _rounded(shadow_start_equity, 2),
            "return_pct_since_common_start": _rounded(shadow_return, 4),
        },
        "spread": {
            "alpaca_minus_shadow_return_pct": _rounded(return_spread, 4),
            "leader": winner,
        },
        "sources": {
            "alpaca_equity_csv": str(alpaca_path),
            "shadow_equity_csv": str(shadow_path),
        },
    }
    return summary, merged


def write_comparison(
    *,
    alpaca_path: Path = ALPACA_EQUITY_FILE,
    shadow_path: Path = SHADOW_EQUITY_FILE,
    csv_out: Path = COMPARE_CSV_FILE,
    json_out: Path = COMPARE_JSON_FILE,
) -> dict:
    """Build the comparison and write both CSV and JSON outputs."""
    summary, table = build_comparison_payload(alpaca_path=alpaca_path, shadow_path=shadow_path)
    atomic_write_json(summary, json_out)
    if not table.empty:
        atomic_write_csv(table, csv_out)
    else:
        atomic_write_csv(pd.DataFrame(), csv_out)
    return summary


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Alpaca paper equity against shadow paper equity.",
    )
    parser.add_argument("--alpaca-equity", type=Path, default=ALPACA_EQUITY_FILE,
                        help="Path to alpaca_paper_equity.csv")
    parser.add_argument("--shadow-equity", type=Path, default=SHADOW_EQUITY_FILE,
                        help="Path to shadow_paper_equity.csv")
    parser.add_argument("--csv-out", type=Path, default=COMPARE_CSV_FILE,
                        help="CSV output path for row-by-row comparison")
    parser.add_argument("--json-out", type=Path, default=COMPARE_JSON_FILE,
                        help="JSON output path for compact dashboard summary")
    args = parser.parse_args()

    summary = write_comparison(
        alpaca_path=args.alpaca_equity,
        shadow_path=args.shadow_equity,
        csv_out=args.csv_out,
        json_out=args.json_out,
    )
    print(
        "Paper vs shadow:",
        summary.get("status"),
        "leader=" + str((summary.get("spread") or {}).get("leader", "?")),
        "spread=" + str((summary.get("spread") or {}).get("alpaca_minus_shadow_return_pct", "?")),
    )
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
