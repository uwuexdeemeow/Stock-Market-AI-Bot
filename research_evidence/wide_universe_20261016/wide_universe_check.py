"""
wide_universe_check.py — pre-registered "H-wide-universe" test, step 0 (data).

PLAIN ENGLISH: the strategy has only ever been tested on 62 big companies
picked in 2026.  H-wide-universe asks two questions: does the edge survive
when companies that later shrank or failed are included (universe U), and
does it hold when the strategy can pick from ALL S&P 500 members (universe
W)?  Before any backtest, the price data has to be good enough.

This file does the data part:

  1. "wiki" step: reads a public Kaggle copy of the free Quandl WIKI price
     archive (frozen in March 2018, but it still has many companies that no
     longer trade) and keeps the rows for every company that was in the S&P 500 at some point in 2012-10-01 to
     2022-12-31.
  2. "coverage" step: joins ("stitches") WIKI with the Tiingo files already
     downloaded for H-full-universe, then checks gate C1: do enough of
     those companies have prices?  If not, the test stops here.

The rules were written down BEFORE any WIKI download: see "Hypothesis
H-wide-universe" and its implementation notes in
Documentation/DELAY_STRESS_PAPER_ADVISORY.md.  Research only: nothing here
changes a gate, a config, the live stock list or the data/ folder.

Run from the project root.  First save the public Kaggle copy of WIKI
(marketneutral/quandl-wiki-prices-us-equites, no login needed) as
research_evidence/wide_universe_20261016/data/kaggle_wiki/wiki.zip:
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
from pathlib import Path

import numpy as np
import pandas as pd

# Let Python find the project modules when run from the project root.
sys.path.insert(0, str(Path.cwd()))

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = OUT_DIR / "data"                 # everything downloaded lives here (not in Git)
WIKI_DIR = DATA_ROOT / "wiki"                # one WIKI price file per ticker
STITCHED_DIR = DATA_ROOT / "stitched"        # one joined WIKI+Tiingo file per company
WIKI_LOG = DATA_ROOT / "wiki_log.json"       # file hash, and rows found per ticker
COVERAGE_JSON = OUT_DIR / "wide_universe_coverage.json"

HYPOTHESIS = "H-wide-universe"
FULL_UNIVERSE_SCRIPT = OUT_DIR.parent / "full_universe_20260929" / "full_universe_check.py"

# WIKI stopped updating in March 2018.  Its last full day:
WIKI_END = "2018-03-27"

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


# ── WIKI file (Kaggle copy) ────────────────────────────────────────────────

# The official WIKI API now needs a paid/locked key, so we use a public
# Kaggle copy of the same table (saved by hand, see the implementation notes).
KAGGLE_WIKI_REF = "marketneutral/quandl-wiki-prices-us-equites"
KAGGLE_WIKI_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_WIKI_REF}"
WIKI_ZIP = DATA_ROOT / "kaggle_wiki" / "wiki.zip"


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool_and_names(args):
    """Same pool and names as H-full-universe (same membership source)."""
    membership, snapshots, membership_sha = fu._snapshots(args)
    pool = fu.build_pool(snapshots)
    names = fu.membership_names(pool, membership.PREDECESSORS)
    latest = set(snapshots["members"].iloc[-1])
    current = {c for c, aliases in names.items() if aliases & latest}
    return membership, snapshots, membership_sha, pool, names, current


def step_wiki(args) -> dict:
    """Split the big WIKI CSV into one small file per ticker we need.

    PLAIN ENGLISH: the Kaggle file holds ~15 million rows for ~3,200
    tickers.  We read it in pieces, keep only the tickers that could belong
    to a pool company and only dates from 2010 on, and save each ticker on
    its own so the coverage step can load it fast.
    """
    import zipfile

    zip_path = Path(args.wiki_zip)
    if not zip_path.exists():
        raise SystemExit(f"WIKI file not found: {zip_path}. Download it from {KAGGLE_WIKI_URL}")
    _, _, _, _, names, _ = _pool_and_names(args)
    wanted = set(t for aliases in names.values() for t in wiki_candidates(aliases))
    parts: dict[str, list[pd.DataFrame]] = {}
    total_rows, last_date = 0, ""
    with zipfile.ZipFile(zip_path) as archive, archive.open("WIKI_PRICES.csv") as handle:
        for chunk in pd.read_csv(handle, chunksize=1_000_000):
            total_rows += len(chunk)
            last_date = max(last_date, str(chunk["date"].max()))
            keep = chunk[chunk["ticker"].isin(wanted) & (chunk["date"] >= fu.PRICE_START)]
            for ticker, rows in keep.groupby("ticker"):
                parts.setdefault(ticker, []).append(rows)
            print(f"  read {total_rows:,} rows", flush=True)
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    log = {}
    for ticker in sorted(wanted):
        rows = pd.concat(parts[ticker], ignore_index=True) if ticker in parts else pd.DataFrame()
        if len(rows):
            rows.to_parquet(WIKI_DIR / f"{ticker}.parquet", index=False)
        log[ticker] = {"rows": int(len(rows))}
    summary = {"source": KAGGLE_WIKI_REF, "zip_sha256": _sha256(zip_path), "csv_rows": total_rows,
               "csv_last_date": last_date, "asked": len(log),
               "with_data": sum(1 for v in log.values() if v["rows"])}
    fu._write_json(WIKI_LOG, {"summary": summary, "tickers": log})
    print(json.dumps(summary, indent=1), flush=True)
    return summary


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




# ═══════════════════════════════════════════════════════════════════════════
# Step "sectors": one sector label per company (SEC SIC code, fixed table)
# ═══════════════════════════════════════════════════════════════════════════

SEC_DIR = DATA_ROOT / "sec"
WIKIPEDIA_CHANGES_URL = ("https://en.wikipedia.org/w/index.php?"
                         "title=Historical_components_of_the_S%26P_500&action=raw")


def _wiki_text(cell: str) -> str:
    """'[[Bed Bath & Beyond Inc.|Bed Bath & Beyond]]' -> 'Bed Bath & Beyond'."""
    import re
    cell = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", cell)
    cell = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", cell)
    cell = re.sub(r"\{\{[^}]*\}\}", "", cell)
    return cell.replace("'''", "").replace("''", "").strip()


def wikipedia_names(text: str) -> dict[str, str]:
    """Ticker -> company name from Wikipedia's S&P 500 changes table.

    PLAIN ENGLISH: companies that left the index often have no current SEC
    ticker, so we need their NAME to find them at the SEC.  Wikipedia's
    "Historical components of the S&P 500" table lists each change as
    (date, added ticker, added name, removed ticker, removed name, ...).
    The table runs newest first; for a ticker that was reused by a later
    company, the OLDEST mention wins, which is the one inside our window
    more often.  Cells may start with "|" or "||".
    """
    start = text.find('id="changes"')
    if start < 0:
        return {}
    body = text[start:text.find("\n|}", start)]
    names: dict[str, str] = {}
    for block in body.split("\n|-")[1:]:
        cells = [_wiki_text(line.lstrip("|")) for line in block.strip().split("\n")
                 if line.startswith("|") and not line.startswith("|-")]
        if len(cells) < 5:
            continue
        for ticker, name in ((cells[1], cells[2]), (cells[3], cells[4])):
            ticker = ticker.strip().upper().replace(".", "-")
            if ticker and name:
                names[ticker] = name      # later rows are older, so they overwrite
    return names


def pq_schema_names(path) -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_schema(path).names


def stitched_companies() -> list[str]:
    return sorted(p.stem for p in STITCHED_DIR.glob("*.parquet"))


def load_stitched(company: str) -> pd.DataFrame:
    """One company's joined WIKI+Tiingo prices (same shape as the Tiingo loader)."""
    path = STITCHED_DIR / f"{company}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def step_sectors(args) -> dict:
    """Find each company's SIC code with the H-full-universe lookup.

    Companies still listed are found by ticker in the SEC list.  Others are
    found by name on SEC EDGAR; the name comes from Wikipedia's changes
    table, and only if Wikipedia has none, from Tiingo (slow free limit).
    Resumable: run again until it says "done".
    """
    import shutil
    import requests

    SEC_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("company_tickers.json", "sectors.json"):
        if not (SEC_DIR / name).exists() and (fu.SEC_DIR / name).exists():
            shutil.copy(fu.SEC_DIR / name, SEC_DIR / name)
    wiki_path = SEC_DIR / "wikipedia_sp500_changes.txt"
    if not wiki_path.exists():
        response = requests.get(WIKIPEDIA_CHANGES_URL, timeout=60,
                                headers={"User-Agent": fu.DEFAULT_SEC_USER_AGENT})
        response.raise_for_status()
        wiki_path.write_text(response.text, encoding="utf-8")
    wiki_names = wikipedia_names(wiki_path.read_text(encoding="utf-8"))

    companies = stitched_companies()
    _, _, _, _, names, _ = _pool_and_names(args)
    tiingo_log = fu._read_json(fu.DOWNLOAD_LOG, {})
    # Companies priced from WIKI alone have no Tiingo entry; give them one so
    # the lookup treats them as "stopped trading" and searches by name.
    log = {}
    for company in companies:
        entry = dict(tiingo_log.get(company, {}))
        if entry.get("status") != "ok":
            entry = {"status": "ok", "last": str(load_stitched(company).index.max().date())}
        log[company] = entry
    cache_path = SEC_DIR / "sectors.json"
    cache = fu._read_json(cache_path, {})
    for company in companies:
        found = next((wiki_names[t] for t in sorted(names.get(company, {company})) if t in wiki_names), None)
        cached = cache.get(company, {})
        if found and cached.get("tiingo_name") is None and not cached.get("searched_v3"):
            cache[company] = {**cached, "tiingo_name": found, "name_source": "wikipedia"}
    fu._write_json(cache_path, cache)

    old_dir = fu.SEC_DIR
    fu.SEC_DIR = SEC_DIR
    try:
        sectors = fu.lookup_sectors(companies, log, per_hour=args.per_hour)
    finally:
        fu.SEC_DIR = old_dir
    done = all(c in sectors for c in companies)
    summary = {"companies": len(companies), "done": done,
               "with_sic": sum(1 for c in companies if sectors.get(c, {}).get("sic")),
               "sector_counts": pd.Series([sectors.get(c, {}).get("sector", "OTHER")
                                           for c in companies]).value_counts().to_dict()}
    print(json.dumps(summary, indent=1), flush=True)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Step "runs": features, universes, engine runs and the pre-registered gates
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_DIR = DATA_ROOT / "features"
R0N_FEATURE_DIR = DATA_ROOT / "features_r0n"   # R0n: built exactly like the project's data/
RESULT_JSON = OUT_DIR / "wide_universe_check.json"
RUN_END = "2022-12-30"            # every run ends here; 2023-2026 is kept for the exam
FEATURE_END = "2023-03-01"        # enough later prices for the last 20-day labels
PANEL_START = "2012-06-01"        # rows before this are never eligible (saves memory)
C2_TOLERANCE = 0.25
MIN_SHARE = 0.50                  # W1, W3 (and U1) need at least 50%
P95 = 95
TOP_K = 3


def rerank_cross_sectional(source: pd.DataFrame, eligible: pd.Series, sector_map: dict,
                           source_cols=None, min_group: int | None = None) -> pd.DataFrame:
    """Recompute the xs_rank_* columns using only eligible rows.

    PLAIN ENGLISH: "rank within sector" only means something relative to
    the stocks you can actually pick that day.  The pre-registered rule says
    ranks are taken within each day's eligible list, so the live rule from
    cross_sectional_features.py is applied to those rows only: percentile
    rank per date (market) and per date+sector (sector), sector falls back
    to market when a sector has fewer than 6 names that day, a lone value
    gets 0.5, and missing ranks get 0.5.
    """
    import cross_sectional_features as xs

    source_cols = source_cols or xs.SOURCE_COLS
    min_group = xs.MIN_GROUP_SIZE if min_group is None else min_group
    panel = source.loc[eligible.to_numpy()].copy()
    panel["sector"] = panel["ticker"].map(lambda t: sector_map.get(t, "OTHER"))
    out = panel[["date", "ticker"]].copy()
    by_date = panel.groupby("date")
    by_sector = panel.groupby(["date", "sector"])
    sector_size = by_sector["ticker"].transform("count")

    def pct(groups, values):
        ranks = groups[values].rank(method="average", pct=True)
        counts = groups[values].transform("count")
        return ranks.where(counts != 1, 0.5)

    for col in source_cols:
        if col not in panel.columns:
            continue
        sector_col, market_col = xs._rank_column_names(col)
        market = pct(by_date, col)
        sector = pct(by_sector, col)
        out[market_col] = market.fillna(0.5).clip(0.0, 1.0)
        out[sector_col] = np.where(sector_size < min_group, market, sector)
        out[sector_col] = out[sector_col].astype(float).fillna(0.5).clip(0.0, 1.0)
    return out


def eligible_rows(source: pd.DataFrame, lists: dict | None, tickers=None) -> pd.Series:
    """Rows whose ticker is on that month's list (or in `tickers` on every day)."""
    if lists is None:
        return source["ticker"].isin(set(tickers))
    return fu.universe_mask(source, lists)


def judge(rows: list[dict]) -> dict:
    """Apply the pre-registered H-wide-universe rules (C2, U1, U2, W1, W2, W3)."""
    med = lambda test, delay: fu._median(fu._alphas(rows, test, delay))  # noqa: E731
    r0, r0n = med("R0", 1), med("R0n", 1)
    c2 = (r0 is not None and r0n is not None and r0 != 0
          and abs(r0n - r0) <= C2_TOLERANCE * abs(r0))
    out = {"numbers": {"R0_median_late": r0, "R0n_median_late": r0n,
                       "R0n_vs_R0": (r0n / r0) if (r0 and r0n is not None) else None},
           "gates": {"C2_source_agreement": c2}}
    if not c2:
        out["verdict"] = "source disagrees with project data"
        return out

    def p95(test):
        values = fu._alphas(rows, test, 0)
        return float(np.percentile(values, P95)) if values else None

    u_late, u_on, mu95 = med("U", 1), med("U", 0), p95("MU")
    w_late, w_on, mw95 = med("W", 1), med("W", 0), p95("MW")
    wn3 = med("WN3", 1)
    u1 = u_late is not None and u_late > 0 and u_late >= MIN_SHARE * r0n
    u2 = u_on is not None and mu95 is not None and u_on > mu95
    w1 = w_late is not None and w_late > 0 and w_late >= MIN_SHARE * r0n
    w2 = w_on is not None and mw95 is not None and w_on > mw95
    w3 = wn3 is not None and w_late is not None and wn3 > 0 and wn3 >= MIN_SHARE * w_late
    if u1 and u2:
        u_verdict = "edge survives the full survivorship test"
    elif u2:
        u_verdict = "edge real but mostly list hindsight"
    else:
        u_verdict = "edge not shown"
    if w1 and w2 and w3:
        w_verdict = "edge holds on a wide universe"
    elif w2:
        w_verdict = "real but too small or too narrow"
    else:
        w_verdict = "no stock-picking edge on an unbiased universe"
    out["numbers"].update({
        "U_median_late": u_late, "U_median_on_time": u_on, "MU_p95": mu95, "MU_median": fu._median(fu._alphas(rows, "MU", 0)),
        "W_median_late": w_late, "W_median_on_time": w_on, "MW_p95": mw95, "MW_median": fu._median(fu._alphas(rows, "MW", 0)),
        "WN3_median_late": wn3,
        "U_share_of_R0n": (u_late / r0n) if (u_late is not None and r0n) else None,
        "W_share_of_R0n": (w_late / r0n) if (w_late is not None and r0n) else None,
        "WN3_share_of_W": (wn3 / w_late) if (wn3 is not None and w_late) else None,
    })
    out["gates"].update({"U1_survivorship": u1, "U2_beats_random_picks": u2,
                         "W1_width": w1, "W2_skill": w2, "W3_breadth": w3})
    out["U_verdict"], out["W_verdict"] = u_verdict, w_verdict

    def spread(test):
        values = fu._alphas(rows, test, 0)
        return (max(values) - min(values)) if values else None

    def delay_cost(test):
        late = {r["offset"]: r["decision_alpha_vs_qqq_pct"] for r in rows
                if r["test"] == test and r["delay"] == 1 and r.get("decision_alpha_vs_qqq_pct") is not None}
        costs = [r["decision_alpha_vs_qqq_pct"] - late[r["offset"]] for r in rows
                 if r["test"] == test and r["delay"] == 0 and r["offset"] in late
                 and r.get("decision_alpha_vs_qqq_pct") is not None]
        return float(np.mean(costs)) if costs else None

    out["diagnostics"] = {
        "on_time_spread": {t: spread(t) for t in ("R0", "R0n", "U", "W")},
        "mean_delay_cost": {t: delay_cost(t) for t in ("R0", "R0n", "U", "W")},
        "UD_median_on_time": med("UD", 0), "WD_median_on_time": med("WD", 0),
    }
    return out


def _done_key(row: dict) -> tuple:
    return (row["test"], row["delay"], row["offset"], row.get("seed"))


def step_runs(args) -> dict:
    import contextlib
    import time
    import alpha_factor_backtest as afb
    import core_satellite_alpha as core
    from settings import WATCHLIST
    from validation_bundle import load_approved_research_config

    coverage = fu._read_json(COVERAGE_JSON, {})
    if not coverage.get("result", {}).get("C1_pass"):
        raise SystemExit("C1 has not passed. By the pre-registered rule, no runs.")
    sectors = fu._read_json(SEC_DIR / "sectors.json", {})
    companies = stitched_companies()
    if not all(c in sectors for c in companies):
        raise SystemExit("Sector lookup unfinished: run --step sectors until it says done.")
    sector_map = {c: sectors[c].get("sector", "OTHER") for c in companies}
    edge = fu._edge_module()
    top_names = fu._load_module("top_names_check", OUT_DIR.parent / "top_names_20260929" / "top_names_check.py")
    membership, snapshots, _sha, pool, names, current = _pool_and_names(args)

    # ── Features from the stitched prices (offline, cached) ──
    # PLAIN ENGLISH: re-use H-full-universe's feature builder, but point its
    # price loader at the stitched files and its output at our own folder.
    old = {"load_raw": fu.load_raw, "FEATURE_DIR": fu.FEATURE_DIR}
    fu.load_raw, fu.FEATURE_DIR = load_stitched, FEATURE_DIR
    try:
        built = fu.build_features(companies, sector_map, rebuild=args.rebuild_features, end=FEATURE_END)
        sessions_all = core._nyse_sessions(fu.PRICE_START, FEATURE_END)
        dollar_volume = fu.dollar_volume_table(built, sessions_all)
    finally:
        fu.load_raw, fu.FEATURE_DIR = old["load_raw"], old["FEATURE_DIR"]
    print(f"[features] built for {len(built)} companies", flush=True)

    # R0n is the project's own recipe on the new prices: the project's
    # SECTOR_MAP, cross-sectional ranks across WATCHLIST + the survivorship
    # training names (as research.py does), and the live rank function.
    from settings import SECTOR_MAP as PROJECT_SECTOR_MAP, SURVIVORSHIP_TRAINING_TICKERS
    r0n_names = sorted(set(WATCHLIST) | {t for t in SURVIVORSHIP_TRAINING_TICKERS
                                         if (STITCHED_DIR / f"{t}.parquet").exists()})
    fu.load_raw, fu.FEATURE_DIR = load_stitched, R0N_FEATURE_DIR
    try:
        fu.build_features(r0n_names, PROJECT_SECTOR_MAP, rebuild=args.rebuild_features, end=FEATURE_END)
    finally:
        fu.load_raw, fu.FEATURE_DIR = old["load_raw"], old["FEATURE_DIR"]

    last_real = {c: load_stitched(c).index.max() for c in built}
    data_end = max(last_real.values())
    last_real = {c: d for c, d in last_real.items() if d < data_end - pd.Timedelta(days=10)}
    members = fu.member_matrix(snapshots, sessions_all, {c: names[c] for c in built})
    u_lists = fu.monthly_universe(dollar_volume, members)
    w_lists = fu.monthly_universe(dollar_volume, members, size=10**6)

    # ── Source columns for the cross-sectional re-ranking ──
    import cross_sectional_features as xs
    frames = []
    for company in built:
        path = FEATURE_DIR / f"{company}.parquet"
        frame = pd.read_parquet(path, columns=[c for c in pq_schema_names(path) if c in xs.SOURCE_COLS])
        frame.index = pd.DatetimeIndex(frame.index)
        frame = frame.loc[frame.index >= pd.Timestamp(PANEL_START)]
        cols = [c for c in xs.SOURCE_COLS if c in frame.columns]
        sub = frame[cols].copy()
        sub["ticker"], sub["date"] = company, sub.index
        frames.append(sub.reset_index(drop=True))
    source = pd.concat(frames, ignore_index=True)
    del frames
    xs_tables = {
        "U": rerank_cross_sectional(source, eligible_rows(source, u_lists), sector_map),
        "W": rerank_cross_sectional(source, eligible_rows(source, w_lists), sector_map),
    }
    del source

    specs = afb.load_feature_specs(write_health_outputs=False)
    import pyarrow.parquet as pq
    # Columns load_factor_panel can use: the score features, its fixed extras
    # and the prices for the labels.  Everything else is never read.
    panel_columns = ({sp["feature"] for sp in specs} | {"Open", "High", "Low", "Close", "Volume"}
                     | {"days_to_next_earnings", "days_since_earnings", "factor_beta_252_spy",
                        "factor_idio_vol_252_spy", "hvol_20d"}
                     | {c for c in xs_tables["U"].columns if c.startswith("xs_rank_")})
    feature_dir = str(FEATURE_DIR)
    original_read = pd.read_parquet

    def build_panel(universe: str, tickers, lists, haircut: float) -> pd.DataFrame:
        table = xs_tables[universe].set_index(["ticker", "date"])
        xs_cols = list(table.columns)
        by_ticker = {t: g.droplevel(0) for t, g in table.groupby(level=0)}

        def read(path, *a, **k):
            company = Path(str(path)).stem
            if str(Path(str(path)).parent) != feature_dir:
                return original_read(path, *a, **k)
            # Read only the columns the panel uses (saves most of the memory).
            available = pq.read_schema(path).names
            frame = original_read(path, columns=[c for c in available if c in panel_columns or c == "__index_level_0__"])
            index = pd.DatetimeIndex(pd.to_datetime(frame.index))
            frame.index = index.tz_localize(None) if index.tz is not None else index
            frame = frame.sort_index()
            frame = frame.loc[frame.index >= pd.Timestamp(PANEL_START)].copy()
            ranks = by_ticker.get(company)
            for col in xs_cols:
                frame[col] = (ranks[col].reindex(frame.index) if ranks is not None
                              else pd.Series(np.nan, index=frame.index)).fillna(0.5).to_numpy()
            if company in last_real:
                frame = fu.pad_after_delisting(frame, sessions_all, haircut)
            return frame

        with fu.patched(afb, WATCHLIST=list(tickers), DATA_DIR=feature_dir, SECTOR_MAP=sector_map), \
                fu.patched(pd, read_parquet=read):
            panel = afb.load_factor_panel(specs, require_forward_returns=False)
        status = panel.attrs.get("point_in_time_membership", {})
        notes.setdefault("project_membership_filter", {})[universe] = bool(status.get("applied"))
        last = panel["ticker"].map(last_real)
        panel = panel[last.isna() | (panel["date"] <= last)]
        if lists is not None:
            panel = panel.loc[fu.universe_mask(panel, lists)]
        panel = panel.reset_index(drop=True)
        empty_ml = pd.DataFrame(columns=["date", "ticker", "ml_score"])
        return core._ensure_robust_score_columns(afb.attach_scores(panel, specs, empty_ml))

    def build_r0n_panel() -> pd.DataFrame:
        with fu.patched(afb, WATCHLIST=list(WATCHLIST), DATA_DIR=str(R0N_FEATURE_DIR)):
            panel = afb.load_factor_panel(specs, require_forward_returns=False)
        notes.setdefault("project_membership_filter", {})["R0n"] = bool(
            panel.attrs.get("point_in_time_membership", {}).get("applied"))
        empty_ml = pd.DataFrame(columns=["date", "ticker", "ml_score"])
        return core._ensure_robust_score_columns(afb.attach_scores(panel, specs, empty_ml))

    notes: dict = {
        "price_source": coverage.get("price_source"), "companies_with_features": len(built),
        "stopped_trading_companies": len(last_real),
        "u_months": len(u_lists), "w_months": len(w_lists),
        "w_median_list_size": float(np.median([len(v) for v in w_lists.values()])) if w_lists else None,
        "u_companies_ever_listed": len({t for v in u_lists.values() for t in v}),
        "run_end": RUN_END,
    }

    # ── Runs ──
    config = load_approved_research_config(Path("signals") / "core_satellite_alpha_metrics.json")
    r0_panel = core._ensure_robust_score_columns(afb.attach_scores(
        afb.load_factor_panel(specs, require_forward_returns=False), specs, afb.load_prediction_scores()))
    notes.setdefault("project_membership_filter", {})["R0"] = bool(
        r0_panel.attrs.get("point_in_time_membership", {}).get("applied"))
    sessions = core._nyse_sessions(r0_panel["date"].min(), r0_panel["date"].max())
    end = pd.Timestamp(RUN_END)
    holding_days = int(config.get("holding_days", afb.HORIZON_DAYS))
    label = top_names.label_column(holding_days, afb.HORIZON_DAYS)
    later_removed = {c for c in built if not (names[c] & set(snapshots["members"].iloc[-1]))}

    out_path = Path(args.out)
    previous = fu._read_json(out_path, {})
    valid = args.offsets == fu.N_OFFSETS and args.monkeys == fu.N_MONKEYS
    same_setup = previous.get("offsets") == args.offsets and previous.get("monkeys") == args.monkeys
    rows: list[dict] = list(previous.get("rows", [])) if same_setup else []
    done = {_done_key(r) for r in rows}

    def save(result=None):
        payload = {"hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
                   "valid_full_test": valid, "end_date": RUN_END, "offsets": args.offsets,
                   "monkeys": args.monkeys, "notes": notes, "rows": rows}
        if result is not None:
            payload["result"] = result
        fu._write_json(out_path, payload)

    def holdings(trades: pd.DataFrame) -> dict:
        picks = removed = delist = 0
        if "overlay_tickers" not in trades:
            return {}
        for tickers, exit_day in zip(trades["overlay_tickers"].fillna(""), trades["exit_date"]):
            for t in [x for x in str(tickers).split(",") if x]:
                picks += 1
                removed += t in later_removed
                delist += bool(t in last_real and pd.Timestamp(exit_day) > last_real[t])
        return {"stock_positions": picks, "later_removed_positions": removed, "delisting_exits": delist}

    start_days = {"R0": sessions, "R0n": sessions}

    def run(test, panel, delay, offset, *, seed=None, labels=None, randomise=False):
        key = (test, delay, offset, seed)
        if key in done:
            return
        t0 = time.time()
        frame = edge.randomized_scores(panel, seed) if randomise else panel
        equity, trades, extra = core.run_core_satellite(
            frame, {**config, "entry_delay_days": delay},
            evaluation_start=start_days[test][offset], evaluation_end=end)
        if randomise:
            unexpected = {c for c in trades.get("score_col", pd.Series(dtype=str)).astype(str).unique()
                          if c not in edge.SCORE_COLS and not c.startswith("_blended_regime_score_")}
            if unexpected:
                raise SystemExit(f"Random-pick runs used non-randomised score columns: {sorted(unexpected)}")
        row = edge._summarise(core, test, equity, extra, delay=delay, offset=offset, seed=seed, t0=t0)
        row.update(holdings(trades))
        if labels is not None:
            row["contributions"] = top_names.ticker_contributions(trades, labels)
        rows.append(row)
        done.add(key)
        print(json.dumps({k: v for k, v in row.items() if k != "contributions"}), flush=True)
        save()

    # R0 and R0n first: gate C2 decides whether anything else runs.
    r0n_panel = build_r0n_panel()
    notes["r0n_xs_names"] = r0n_names
    notes["r0n_panel_rows"] = int(len(r0n_panel))
    for test, panel in (("R0", r0_panel), ("R0n", r0n_panel)):
        for delay in fu.DELAYS:
            for offset in range(args.offsets):
                run(test, panel, delay, offset)
    del r0_panel, r0n_panel
    c2 = judge(rows)
    # A smoke test (not a valid run) carries on past C2 so every code path is exercised.
    if valid and not c2["gates"]["C2_source_agreement"]:
        save(c2)
        print(json.dumps(c2, indent=1), flush=True)
        return c2

    # PLAIN ENGLISH: one universe at a time, so only one ~700-stock panel
    # is in memory at once (two at once ran the computer out of memory).
    # U and W have no rows before their first monthly list, so their 20
    # start days are the first 20 sessions of their own panels.  (The
    # project panel's 2010 start days would make all 20 calendars identical,
    # because the engine can't start before the data does.)
    def own_start_days(panel, family):
        own = core._nyse_sessions(panel["date"].min(), panel["date"].max())
        start_days.update({t: own for t in family})
        notes.setdefault("start_days", {}).update({t: str(pd.Timestamp(own[0]).date()) for t in family})

    # ── U family: U, MU, UD ──
    panel = build_panel("U", built, u_lists, 0.0)
    notes["u_panel_rows"] = int(len(panel))
    own_start_days(panel, ("U", "MU", "UD"))
    for delay in fu.DELAYS:
        for offset in range(args.offsets):
            run("U", panel, delay, offset)
    for seed in range(args.monkeys):
        run("MU", panel, 0, seed % fu.N_OFFSETS, seed=seed, randomise=True)
    del panel
    panel = build_panel("U", built, u_lists, fu.DELIST_HAIRCUT)
    for offset in range(args.offsets):
        run("UD", panel, 0, offset)
    del panel

    # ── W family: W, WN3 (top 3 chosen from W's on-time runs first), MW, WD ──
    panel = build_panel("W", built, w_lists, 0.0)
    notes["w_panel_rows"] = int(len(panel))
    own_start_days(panel, ("W", "MW", "WN3", "WD"))
    w_labels = top_names.label_lookup(panel, label)
    for delay in fu.DELAYS:
        for offset in range(args.offsets):
            run("W", panel, delay, offset, labels=w_labels if delay == 0 else None)
    del w_labels
    contributions = [r["contributions"] for r in rows if r["test"] == "W" and r["delay"] == 0]
    top3 = top_names.top_contributors(contributions, TOP_K)
    total = sum(sum(c.values()) for c in contributions) / max(len(contributions), 1)
    top_avg = {t: sum(c.get(t, 0.0) for c in contributions) / max(len(contributions), 1) for t in top3}
    notes.update({"W_top3_tickers": top3, "W_top3_avg_contribution": top_avg,
                  "W_top3_share_of_overlay_profit": (sum(top_avg.values()) / total) if total else None})
    save()
    print(f"[W] top 3 contributors, chosen before WN3: {top3}", flush=True)
    wn3_panel = panel[~panel["ticker"].astype(str).str.upper().isin(top3)].reset_index(drop=True)
    for offset in range(args.offsets):
        run("WN3", wn3_panel, 1, offset)
    del wn3_panel
    for seed in range(args.monkeys):
        run("MW", panel, 0, seed % fu.N_OFFSETS, seed=seed, randomise=True)
    del panel
    panel = build_panel("W", built, w_lists, fu.DELIST_HAIRCUT)
    for offset in range(args.offsets):
        run("WD", panel, 0, offset)
    del panel

    result = judge(rows)
    save(result)
    print(json.dumps({k: result.get(k) for k in ("gates", "U_verdict", "W_verdict", "verdict")}, indent=1),
          flush=True)
    return result


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Pre-registered H-wide-universe check (research only).")
    parser.add_argument("--step", choices=("wiki", "coverage", "sectors", "runs"), required=True)
    parser.add_argument("--wiki-zip", default=str(WIKI_ZIP), help="Kaggle WIKI zip (wiki step)")
    parser.add_argument("--membership-csv", default=str(fu.DATA_ROOT / "sp500_membership.csv"),
                        help="membership CSV (default: the copy saved by H-full-universe)")
    parser.add_argument("--per-hour", type=int, default=45, help="Tiingo requests per hour (sectors step)")
    parser.add_argument("--offsets", type=int, default=fu.N_OFFSETS)
    parser.add_argument("--monkeys", type=int, default=fu.N_MONKEYS)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--out", default=str(RESULT_JSON))
    args = parser.parse_args(argv)
    steps = {"wiki": step_wiki, "coverage": step_coverage, "sectors": step_sectors, "runs": step_runs}
    return steps[args.step](args)


if __name__ == "__main__":
    main()
