"""
watchlist_membership_check.py — how "survivor-picked" is the stock list?

PLAIN ENGLISH: the strategy picks stocks from a fixed list (settings.WATCHLIST)
that was chosen recently.  A backtest that starts in 2010 then only "knows"
companies that did well enough to still be large today.  This script
measures that with a free, community-built history of S&P 500 membership:

  * how many S&P 500 members in January 2010 are no longer members today
    (the companies a 2010 investor could have picked, but our list can't);
  * which names on our list were NOT in the S&P 500 when the backtest starts
    (we only know to include them because of what happened later).

It is a DIAGNOSTIC.  The source is a community reconstruction (see
Documentation/doc_audit_evidence_recovery.md): useful for sizing the
problem, not verified enough to pass the point-in-time membership gate.
Nothing here changes a gate, a config or the trading universe.

Run from the project root (needs internet access to raw.githubusercontent.com,
or pass a saved copy of the CSV):
    python research_evidence/survivorship_check_20260926/watchlist_membership_check.py [--csv PATH]
Output: research_evidence/survivorship_check_20260926/watchlist_membership_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))

SOURCE_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
              "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
SOURCE_LICENSE = "MIT (https://github.com/fja05680/sp500/blob/master/LICENSE)"
OUT = Path(__file__).resolve().parent / "watchlist_membership_check.json"
CHECK_DATES = ("2010-01-04", "2013-01-02")   # backtest start, decision-window start
# Same company, older ticker (checked by hand).  A rename is not hindsight:
# the company was in the index under its old name.
PREDECESSORS = {"META": ("FB",), "RTX": ("UTX",), "LIN": ("PX",)}
# Caveat: the "no longer in index" count also includes plain ticker changes
# (like UTX -> RTX), so the true share of companies that left is lower.


def load_snapshots(text: str) -> pd.DataFrame:
    """Rows of (date, set of tickers).  'BRK.B' style becomes 'BRK-B' like settings.py."""
    frame = pd.read_csv(StringIO(text))
    frame["date"] = pd.to_datetime(frame["date"])
    frame["members"] = frame["tickers"].map(lambda s: {t.strip().replace(".", "-") for t in str(s).split(",") if t.strip()})
    return frame.sort_values("date")[["date", "members"]].reset_index(drop=True)


def members_on(snapshots: pd.DataFrame, day) -> set[str]:
    """Index members on `day` (the latest snapshot on or before it)."""
    earlier = snapshots[snapshots["date"] <= pd.Timestamp(day)]
    if earlier.empty:
        raise ValueError(f"no snapshot on or before {day}")
    return set(earlier.iloc[-1]["members"])


def first_joined(snapshots: pd.DataFrame, ticker: str, after) -> str | None:
    """First snapshot date on or after `after` that lists the ticker (or its old ticker)."""
    names = {ticker, *PREDECESSORS.get(ticker, ())}
    later = snapshots[snapshots["date"] >= pd.Timestamp(after)]
    for date, members in zip(later["date"], later["members"]):
        if names & members:
            return str(date.date())
    return None


def analyse(snapshots: pd.DataFrame, watchlist: list[str]) -> dict:
    cutoff = snapshots["date"].max()
    now = members_on(snapshots, cutoff)
    names = [t for t in watchlist if t != "BTC-USD"]   # not a stock
    result = {"source_cutoff": str(cutoff.date()), "watchlist_size": len(names), "by_date": {}}
    for day in CHECK_DATES:
        then = members_on(snapshots, day)
        missing, renamed = [], []
        for ticker in names:
            if ticker in then:
                continue
            if any(old in then for old in PREDECESSORS.get(ticker, ())):
                renamed.append(ticker)
            else:
                missing.append({"ticker": ticker, "first_in_index": first_joined(snapshots, ticker, day)})
        gone = sorted(then - now)
        result["by_date"][day] = {
            "index_members": len(then),
            "members_no_longer_in_index_at_cutoff": len(gone),
            "share_no_longer_in_index_pct": round(100.0 * len(gone) / max(len(then), 1), 1),
            "watchlist_names_that_later_left_index": sorted(set(names) & set(gone)),
            "watchlist_not_in_index_then": missing,
            "watchlist_in_index_under_older_ticker": renamed,
        }
    return result


def held_before_joining(trades: pd.DataFrame, join_dates: dict[str, str]) -> dict:
    """Count overlay holdings of a name before it joined the index.

    `trades` is signals/core_satellite_alpha_trades.csv (one row per
    rebalance; `overlay_weights_json` lists that period's stock picks).
    """
    held = []
    for _, row in trades.iterrows():
        weights = json.loads(row["overlay_weights_json"]) if isinstance(row.get("overlay_weights_json"), str) else {}
        held += [(pd.Timestamp(row["date"]), ticker) for ticker in weights]
    early = [(day, t) for day, t in held if t in join_dates and join_dates[t] and day < pd.Timestamp(join_dates[t])]
    hindsight = [t for _, t in held if t in join_dates]
    return {
        "overlay_holdings": len(held),
        "holdings_of_names_not_in_index_at_start": len(hindsight),
        "holdings_before_name_joined_index": len(early),
        "before_joining_by_ticker": pd.Series([t for _, t in early], dtype=str).value_counts().to_dict(),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Survivorship diagnostic for the stock list (research only).")
    parser.add_argument("--csv", help="saved copy of the source CSV (default: download it)")
    args = parser.parse_args(argv)
    if args.csv:
        raw = Path(args.csv).read_bytes()
    else:
        import requests
        response = requests.get(SOURCE_URL, timeout=60)
        response.raise_for_status()
        raw = response.content
    from settings import WATCHLIST
    analysis = analyse(load_snapshots(raw.decode("utf-8")), list(WATCHLIST))
    report = {
        "research_only": True, "approves_trading": False, "verified_membership": False,
        "source_url": SOURCE_URL, "source_license": SOURCE_LICENSE, "access_cost": "free",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        **analysis,
    }
    trades_path = Path("signals/core_satellite_alpha_trades.csv")
    if trades_path.exists():
        start = analysis["by_date"][CHECK_DATES[0]]["watchlist_not_in_index_then"]
        report["incumbent_holdings"] = held_before_joining(
            pd.read_csv(trades_path), {row["ticker"]: row["first_in_index"] for row in start})
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("by_date", "incumbent_holdings") if k in report}, indent=1))
    return report


if __name__ == "__main__":
    main()
