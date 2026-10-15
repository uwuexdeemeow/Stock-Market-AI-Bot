"""
wide_universe_check.py — pre-registered "H-wide-universe" test, step 0 (data).

PLAIN ENGLISH: the strategy has only ever been tested on 62 big companies
picked in 2026.  H-wide-universe asks two questions: does the edge survive
when companies that later shrank or failed are included (universe U), and
does it hold when the strategy can pick from ALL S&P 500 members (universe
W)?  Before any backtest, the price data has to be good enough.

This file does the data part:

  1. "wiki" step: downloads the free Quandl WIKI price archive (frozen in
     March 2018, but it still has many companies that no longer trade) for
     every company that was in the S&P 500 at some point in 2012-10-01 to
     2022-12-31.
  2. "coverage" step: joins ("stitches") WIKI with the Tiingo files already
     downloaded for H-full-universe, then checks gate C1: do enough of
     those companies have prices?  If not, the test stops here.

The rules were written down BEFORE any WIKI download: see "Hypothesis
H-wide-universe" and its implementation notes in
Documentation/DELAY_STRESS_PAPER_ADVISORY.md.  Research only: nothing here
changes a gate, a config, the live stock list or the data/ folder.

Run from the project root (needs internet and a free Nasdaq Data Link key
in .env as NASDAQ_DATA_LINK_API_KEY):
    python research_evidence/wide_universe_20261016/wide_universe_check.py --step wiki
    python research_evidence/wide_universe_20261016/wide_universe_check.py --step coverage
Output: research_evidence/wide_universe_20261016/wide_universe_coverage.json
Downloaded data: research_evidence/wide_universe_20261016/data/ (not in Git)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Let Python find the project modules when run from the project root.
sys.path.insert(0, str(Path.cwd()))

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = OUT_DIR / "data"                 # everything downloaded lives here (not in Git)
WIKI_DIR = DATA_ROOT / "wiki"                # one WIKI price file per ticker
STITCHED_DIR = DATA_ROOT / "stitched"        # one joined WIKI+Tiingo file per company
WIKI_LOG = DATA_ROOT / "wiki_log.json"       # which tickers were asked for and what came back
COVERAGE_JSON = OUT_DIR / "wide_universe_coverage.json"

HYPOTHESIS = "H-wide-universe"
FULL_UNIVERSE_SCRIPT = OUT_DIR.parent / "full_universe_20260929" / "full_universe_check.py"

# WIKI stopped updating in March 2018.  Its last full day:
WIKI_END = "2018-03-27"
WIKI_URL = "https://data.nasdaq.com/api/v3/datatables/WIKI/PRICES.json"
WIKI_COLUMNS = ("ticker,date,open,high,low,close,volume,"
                "adj_open,adj_high,adj_low,adj_close,adj_volume")
WIKI_BATCH = 20                  # tickers per request (the API pages at 10,000 rows)

# Stitch check (fixed in the implementation notes before any download).
MIN_OVERLAP_DAYS = 20
MIN_RETURN_CORRELATION = 0.95


def _load_full_universe():
    """The H-full-universe script: pool, membership, C1 and Tiingo loader.

    PLAIN ENGLISH: we reuse its tested code instead of copying it, so C1 is
    computed exactly the same way, with the same thresholds.
    """
    spec = importlib.util.spec_from_file_location("full_universe_check", FULL_UNIVERSE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fu = _load_full_universe()


# ── Pure helpers (tested) ──────────────────────────────────────────────────

def wiki_candidates(aliases) -> list[str]:
    """Every WIKI ticker a company might be filed under.

    PLAIN ENGLISH: the membership list writes "BRK-B"; WIKI may write
    "BRK_B" or "BRK.B".  We try each spelling of each name the company had.
    """
    out = []
    for name in sorted(aliases):
        for spelling in (name, name.replace("-", "_"), name.replace("-", ".")):
            if spelling not in out:
                out.append(spelling)
    return out


def wiki_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """WIKI rows -> the same OHLCV frame shape as the Tiingo loader.

    Open/High/Low/Close/Volume are adjusted for splits and dividends;
    RawClose/RawVolume are the real traded numbers.  Bars with a missing or
    non-positive price are dropped, and the file is cut at its first long
    gap (a gap means a different security reused the ticker).
    """
    if raw.empty:
        return pd.DataFrame()
    index = pd.DatetimeIndex(pd.to_datetime(raw["date"])).normalize()
    frame = pd.DataFrame({
        "Open": raw["adj_open"].to_numpy(float), "High": raw["adj_high"].to_numpy(float),
        "Low": raw["adj_low"].to_numpy(float), "Close": raw["adj_close"].to_numpy(float),
        "Volume": raw["adj_volume"].to_numpy(float),
        "RawClose": raw["close"].to_numpy(float), "RawVolume": raw["volume"].to_numpy(float),
    }, index=index)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.loc[frame.index <= pd.Timestamp(WIKI_END)]
    good = (frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    return fu.first_segment(frame.loc[good])


def pick_wiki_name(frames: dict[str, pd.DataFrame], member_days: pd.DatetimeIndex) -> str | None:
    """The WIKI ticker with the most bars on the company's member days."""
    best, best_count = None, 0
    for name, frame in frames.items():
        count = int(member_days.isin(frame.index).sum()) if len(frame) else 0
        if count > best_count:
            best, best_count = name, count
    return best


def stitch(tiingo: pd.DataFrame, wiki: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join one company's WIKI and Tiingo prices.

    PLAIN ENGLISH (rule fixed before any download):
      * Tiingo is used wherever it has a bar.
      * WIKI only fills the days before Tiingo's first bar (or every day,
        if Tiingo has nothing).
      * WIKI's adjusted prices are multiplied by one number (the median
        Tiingo/WIKI Close ratio on shared days), so price levels line up
        and daily returns are unchanged.
      * If the two overlap on fewer than 20 days, or their daily returns
        on shared days correlate below 0.95, they may not be the same
        security, so the WIKI part is dropped.
    Returns (joined frame, info about what happened).
    """
    if wiki.empty:
        return tiingo, {"source": "tiingo" if len(tiingo) else "none"}
    if tiingo.empty:
        return wiki, {"source": "wiki"}
    shared = tiingo.index.intersection(wiki.index)
    info = {"overlap_days": int(len(shared))}
    if len(shared) < MIN_OVERLAP_DAYS:
        return tiingo, {**info, "source": "tiingo", "wiki_dropped": "overlap_too_short"}
    t_ret = tiingo.loc[shared, "Close"].pct_change().dropna()
    w_ret = wiki.loc[shared, "Close"].pct_change().dropna()
    corr = float(t_ret.corr(w_ret))
    info["return_correlation"] = round(corr, 4)
    if not np.isfinite(corr) or corr < MIN_RETURN_CORRELATION:
        return tiingo, {**info, "source": "tiingo", "wiki_dropped": "returns_disagree"}
    scale = float((tiingo.loc[shared, "Close"] / wiki.loc[shared, "Close"]).median())
    info["scale"] = round(scale, 6)
    early = wiki.loc[wiki.index < tiingo.index[0]].copy()
    if early.empty:
        return tiingo, {**info, "source": "tiingo"}
    for column in ("Open", "High", "Low", "Close"):
        early[column] = early[column] * scale
    # Adjusted volume moves the opposite way to adjusted price.
    early["Volume"] = early["Volume"] / scale
    joined = pd.concat([early, tiingo]).sort_index()
    return joined, {**info, "source": "wiki+tiingo", "wiki_days_added": int(len(early))}


# ── WIKI download ──────────────────────────────────────────────────────────

def _wiki_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
    except Exception:
        pass
    key = os.environ.get("NASDAQ_DATA_LINK_API_KEY") or os.environ.get("QUANDL_API_KEY")
    if not key:
        raise SystemExit("No Nasdaq Data Link key: add NASDAQ_DATA_LINK_API_KEY=... to .env "
                         "(free account at data.nasdaq.com)")
    return key


def _wiki_fetch(session, tickers: list[str], key: str) -> pd.DataFrame:
    """All WIKI rows for a batch of tickers, following the page cursor."""
    params = {"ticker": ",".join(tickers), "date.gte": fu.PRICE_START,
              "qopts.columns": WIKI_COLUMNS, "api_key": key}
    frames, cursor = [], None
    while True:
        if cursor:
            params["qopts.cursor_id"] = cursor
        for attempt in range(5):
            response = session.get(WIKI_URL, params=params, timeout=120)
            if response.status_code == 429:          # too many requests: wait and retry
                time.sleep(15 * (attempt + 1))
                continue
            break
        response.raise_for_status()
        payload = response.json()["datatable"]
        names = [c["name"] for c in payload["columns"]]
        frames.append(pd.DataFrame(payload["data"], columns=names))
        cursor = response.json().get("meta", {}).get("next_cursor_id")
        if not cursor:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pool_and_names(args):
    """Same pool and names as H-full-universe (same membership source)."""
    membership, snapshots, membership_sha = fu._snapshots(args)
    pool = fu.build_pool(snapshots)
    names = fu.membership_names(pool, membership.PREDECESSORS)
    latest = set(snapshots["members"].iloc[-1])
    current = {c for c, aliases in names.items() if aliases & latest}
    return membership, snapshots, membership_sha, pool, names, current


def step_wiki(args) -> dict:
    import requests

    key = _wiki_key()
    _, _, _, _, names, _ = _pool_and_names(args)
    wanted = sorted({t for aliases in names.values() for t in wiki_candidates(aliases)})
    log = fu._read_json(WIKI_LOG, {})
    todo = [t for t in wanted if t not in log]
    print(f"WIKI tickers to ask for: {len(todo)} (of {len(wanted)})", flush=True)
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for start in range(0, len(todo), WIKI_BATCH):
        batch = todo[start:start + WIKI_BATCH]
        rows = _wiki_fetch(session, batch, key)
        for ticker in batch:
            part = rows[rows["ticker"] == ticker] if len(rows) else rows
            if len(part):
                part.to_parquet(WIKI_DIR / f"{ticker}.parquet", index=False)
            log[ticker] = {"rows": int(len(part))}
        fu._write_json(WIKI_LOG, log)
        print(f"  {min(start + WIKI_BATCH, len(todo))}/{len(todo)}", flush=True)
    found = sum(1 for v in log.values() if v["rows"])
    print(f"WIKI tickers with data: {found} of {len(log)}", flush=True)
    return {"asked": len(log), "with_data": found}


def load_wiki(ticker: str) -> pd.DataFrame:
    path = WIKI_DIR / f"{ticker}.parquet"
    return wiki_frame(pd.read_parquet(path)) if path.exists() else pd.DataFrame()


# ── Coverage (gate C1) ─────────────────────────────────────────────────────

def step_coverage(args) -> dict:
    import core_satellite_alpha as core

    if not WIKI_LOG.exists():
        raise SystemExit("Run --step wiki first.")
    membership, snapshots, membership_sha, pool, names, current = _pool_and_names(args)
    tiingo_log = fu._read_json(fu.DOWNLOAD_LOG, {})
    sessions = core._nyse_sessions(*fu.COVERAGE_WINDOW)
    members = fu.member_matrix(snapshots, sessions, names)

    STITCHED_DIR.mkdir(parents=True, exist_ok=True)
    price_dates, sources = {}, {}
    for company in members.columns:
        tiingo = fu.load_raw(company) if tiingo_log.get(company, {}).get("status") == "ok" else pd.DataFrame()
        member_days = members.index[members[company].to_numpy()]
        frames = {t: load_wiki(t) for t in wiki_candidates(names[company])}
        wiki_name = pick_wiki_name(frames, member_days)
        wiki = frames.get(wiki_name, pd.DataFrame()) if wiki_name else pd.DataFrame()
        joined, info = stitch(tiingo, wiki)
        sources[company] = {**info, "wiki_ticker": wiki_name}
        if len(joined):
            joined.to_parquet(STITCHED_DIR / f"{company}.parquet")
            price_dates[company] = joined.index

    table = fu.coverage_table(members, price_dates, current)
    table["source"] = table["company"].map(lambda c: sources[c]["source"])
    result = fu.judge_coverage(table)
    payload = {
        "hypothesis": HYPOTHESIS, "step": "coverage", "research_only": True, "approves_trading": False,
        "price_source": f"Quandl WIKI (to {WIKI_END}) stitched with Tiingo (H-full-universe files)",
        "membership_source_url": membership.SOURCE_URL, "membership_source_sha256": membership_sha,
        "pool_companies": len(pool),
        "thresholds": {"min_session_share": fu.C1_MIN_SESSION_SHARE,
                       "min_removed_share": fu.C1_MIN_REMOVED_SHARE,
                       "min_current_share": fu.C1_MIN_CURRENT_SHARE},
        "result": result,
        "source_counts": pd.Series([s["source"] for s in sources.values()]).value_counts().to_dict(),
        "wiki_dropped": {c: s["wiki_dropped"] for c, s in sources.items() if s.get("wiki_dropped")},
        "companies": [{**row, **sources[row["company"]]} for row in table.to_dict("records")],
    }
    fu._write_json(COVERAGE_JSON, payload)
    print(json.dumps({"C1": result["C1_pass"], "removed": result["removed_covered_share"],
                      "current": result["current_covered_share"],
                      "sources": payload["source_counts"]}, indent=1), flush=True)
    return payload


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Pre-registered H-wide-universe data step (research only).")
    parser.add_argument("--step", choices=("wiki", "coverage"), required=True)
    parser.add_argument("--membership-csv", default=str(fu.DATA_ROOT / "sp500_membership.csv"),
                        help="membership CSV (default: the copy saved by H-full-universe)")
    args = parser.parse_args(argv)
    return step_wiki(args) if args.step == "wiki" else step_coverage(args)


if __name__ == "__main__":
    main()
