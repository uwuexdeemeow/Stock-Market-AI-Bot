"""
full_universe_check.py — pre-registered "H-full-universe" survivorship test.

PLAIN ENGLISH: the incumbent strategy picks stocks from a fixed list of 62
big companies that were chosen in 2026.  Companies that shrank, failed or
were bought between 2013 and 2022 can never be picked, because we had no
price files for them.  That makes the backtest look better than reality
("survivorship bias").

This script rebuilds the test on a fairer stock list:

  1. It downloads prices (from Tiingo) for EVERY company that was in the
     S&P 500 at some point from 2012-10-01 to 2022-12-31, including ones
     that later left the index.
  2. Step "coverage": before any backtest runs, it checks that enough of
     those companies have prices (gate C1).  If not, it stops.
  3. Step "runs": each month it takes the 62 biggest-trading S&P 500
     members of that month (the "U" list), builds the same features the
     project always uses, and runs the unchanged incumbent strategy on it.
     It compares that against the current list (R0) and against 100
     random-pick runs (MU), and applies the pass/fail rules.

The rules were written down BEFORE any data was downloaded: see
"Hypothesis H-full-universe" in Documentation/DELAY_STRESS_PAPER_ADVISORY.md.
Nothing here changes a gate, a config, the live stock list or the data/
folder.  Research only.

Run from the project root (needs data/, signals/, internet, and a Tiingo
key in .env as TIINGO_KEY):
    python research_evidence/full_universe_20260929/full_universe_check.py --step coverage
    # only if C1 passes:
    python research_evidence/full_universe_20260929/full_universe_check.py --step runs
Quick smoke test (NOT a valid result):
    ... --step runs --offsets 2 --monkeys 4
Output: research_evidence/full_universe_20260929/full_universe_check.json
Downloaded data: research_evidence/full_universe_20260929/data/ (not in Git)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Let Python find the project modules when run from the project root.
sys.path.insert(0, str(Path.cwd()))

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = OUT_DIR / "data"            # everything downloaded lives here
RAW_DIR = DATA_ROOT / "raw"             # one Tiingo price file per company
FEATURE_DIR = DATA_ROOT / "features"    # feature files built from RAW_DIR
SEC_DIR = DATA_ROOT / "sec"             # cached SEC answers (sector codes)
DOWNLOAD_LOG = DATA_ROOT / "download_log.json"
COVERAGE_JSON = OUT_DIR / "full_universe_coverage.json"
RESULT_JSON = OUT_DIR / "full_universe_check.json"

# ── Fixed settings (pre-registered; do not change after results) ───────────
HYPOTHESIS = "H-full-universe"
POOL_WINDOW = ("2012-10-01", "2022-12-31")   # who counts as "was a member"
COVERAGE_WINDOW = ("2013-01-01", "2022-12-31")
C1_MIN_SESSION_SHARE = 0.90       # a company is "covered" at 90% of its member sessions
C1_MIN_REMOVED_SHARE = 0.85       # ... and 85% of the removed companies must be covered
C1_MIN_CURRENT_SHARE = 0.95       # ... and 95% of current members
UNIVERSE_SIZE = 62                # same size as settings.WATCHLIST
DOLLAR_VOLUME_SESSIONS = 63       # trailing window for the monthly ranking
STILL_TRADING_SESSIONS = 5        # a stock must have traded in the last 5 sessions
DELIST_HAIRCUT = 0.30             # diagnostic UD row only
N_OFFSETS = 20
N_MONKEYS = 100
DELAYS = (0, 1)
U1_MIN_SHARE_OF_R0 = 0.5
U2_PERCENTILE = 95.0
PRICE_START = "2010-01-01"        # features need about a year of history first

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
# The SEC asks every script to say who it is.  Set SEC_USER_AGENT in .env to
# "Your Name your@email" if the SEC ever refuses the default.
DEFAULT_SEC_USER_AGENT = "StockBot personal research script"

# ── Ticker changes (same company, new ticker) ──────────────────────────────
# PLAIN ENGLISH: Tiingo stores a company's whole history under its LATEST
# ticker.  Bank of New York Mellon traded as BK until 2024 and as BNY after,
# so its prices sit under "BNY".  This map says "the S&P list's old ticker ->
# the ticker Tiingo keeps the history under".  Each pair was checked by hand
# (a rename, not an acquisition by a different company).  The coverage step
# still checks the dates: a wrong pair shows up as "not covered".
RENAMES = {
    "ABC": "COR",      # AmerisourceBergen -> Cencora
    "ADS": "BFH",      # Alliance Data Systems -> Bread Financial
    "ANTM": "ELV",     # Anthem -> Elevance Health
    "ARNC": "HWM",     # Arconic Inc -> Howmet Aerospace
    "BHGE": "BKR",     # Baker Hughes, a GE company -> Baker Hughes
    "BK": "BNY",       # Bank of New York Mellon
    "BLL": "BALL",     # Ball Corp
    "CBS": "PARA",     # CBS -> ViacomCBS -> Paramount Global
    "CDAY": "DAY",     # Ceridian -> Dayforce
    "COG": "CTRA",     # Cabot Oil & Gas -> Coterra
    "CTL": "LUMN",     # CenturyLink -> Lumen
    "DISCA": "WBD",    # Discovery -> Warner Bros. Discovery
    "DWDP": "DD",      # DowDuPont -> DuPont
    "FB": "META",      # Facebook -> Meta
    "FBHS": "FBIN",    # Fortune Brands Home & Security -> Fortune Brands Innovations
    "FII": "FHI",      # Federated Investors -> Federated Hermes
    "FLT": "CPAY",     # FleetCor -> Corpay
    "GPS": "GAP",      # Gap Inc
    "HCP": "DOC",      # HCP -> Healthpeak (PEAK) -> DOC
    "PEAK": "DOC",
    "HFC": "DINO",     # HollyFrontier -> HF Sinclair
    "HRS": "LHX",      # Harris -> L3Harris
    "JEC": "J",        # Jacobs Engineering
    "KORS": "CPRI",    # Michael Kors -> Capri Holdings
    "LB": "BBWI",      # L Brands -> Bath & Body Works
    "MMC": "MRSH",     # Marsh & McLennan
    "NLOK": "GEN",     # NortonLifeLock -> Gen Digital
    "SYMC": "GEN",     # Symantec -> NortonLifeLock -> Gen Digital
    "PKI": "RVTY",     # PerkinElmer -> Revvity
    "RE": "EG",        # Everest Re -> Everest Group
    "TMK": "GL",       # Torchmark -> Globe Life
    "UTX": "RTX",      # United Technologies -> Raytheon Technologies -> RTX
    "VIAC": "PARA",    # ViacomCBS -> Paramount Global
    "WLTW": "WTW",     # Willis Towers Watson
    "WYND": "TNL",     # Wyndham Destinations -> Travel + Leisure
}

# ── Sector from SEC SIC code (fixed before any download) ───────────────────
# PLAIN ENGLISH: the project labels each stock with one of 11 sector ETFs
# (XLK = technology, XLF = financials, ...).  For old companies we don't have
# that label, so every company - old and new - gets its sector from the
# industry code (SIC) the SEC keeps on file.  Rows are (first code, last
# code, sector ETF); the FIRST matching row wins, so narrow ranges come
# before the wide ones they sit inside.
SIC_SECTOR_TABLE = (
    (100, 999, "XLP"),       # agriculture
    (1000, 1099, "XLB"),     # metal mining
    (1200, 1399, "XLE"),     # coal, oil and gas
    (1400, 1499, "XLB"),     # other mining
    (1520, 1531, "XLY"),     # homebuilders
    (1500, 1799, "XLI"),     # construction
    (2000, 2199, "XLP"),     # food, drinks, tobacco
    (2200, 2399, "XLY"),     # textiles, apparel
    (2400, 2499, "XLB"),     # lumber
    (2500, 2599, "XLY"),     # furniture
    (2600, 2699, "XLB"),     # paper, packaging
    (2700, 2799, "XLC"),     # publishing
    (2830, 2836, "XLV"),     # drugs
    (2840, 2844, "XLP"),     # soap, cosmetics
    (2800, 2899, "XLB"),     # chemicals
    (2900, 2999, "XLE"),     # oil refining
    (3000, 3099, "XLB"),     # rubber, plastics
    (3100, 3199, "XLY"),     # leather
    (3200, 3399, "XLB"),     # glass, cement, metals
    (3400, 3499, "XLI"),     # fabricated metal
    (3570, 3579, "XLK"),     # computers
    (3500, 3599, "XLI"),     # machinery
    (3630, 3651, "XLY"),     # household appliances, audio/video
    (3660, 3679, "XLK"),     # telecom equipment, semiconductors
    (3600, 3699, "XLI"),     # other electrical equipment
    (3710, 3716, "XLY"),     # cars and car parts
    (3751, 3751, "XLY"),     # motorcycles
    (3700, 3799, "XLI"),     # aircraft, ships, rail
    (3812, 3812, "XLI"),     # defence electronics
    (3820, 3829, "XLK"),     # measuring instruments
    (3840, 3851, "XLV"),     # medical instruments
    (3800, 3899, "XLK"),     # other instruments, photo equipment
    (3900, 3999, "XLY"),     # toys, jewellery, other
    (4000, 4799, "XLI"),     # transport
    (4800, 4899, "XLC"),     # telephone, broadcasting, cable
    (4950, 4959, "XLI"),     # waste management
    (4900, 4999, "XLU"),     # utilities
    (5122, 5122, "XLV"),     # drug wholesale
    (5140, 5149, "XLP"),     # grocery wholesale
    (5170, 5172, "XLE"),     # fuel wholesale
    (5000, 5199, "XLI"),     # other wholesale
    (5310, 5331, "XLP"),     # department / variety / discount stores
    (5400, 5499, "XLP"),     # grocery stores
    (5912, 5912, "XLP"),     # drug stores
    (5200, 5999, "XLY"),     # other retail, restaurants
    (6320, 6324, "XLV"),     # health insurers
    (6500, 6553, "XLRE"),    # real estate
    (6798, 6798, "XLRE"),    # REITs
    (6000, 6799, "XLF"),     # banks, insurers, brokers
    (7000, 7099, "XLY"),     # hotels
    (7200, 7299, "XLY"),     # personal services
    (7370, 7379, "XLK"),     # software, IT services
    (7800, 7829, "XLC"),     # film, TV production
    (7900, 7999, "XLY"),     # entertainment, casinos
    (7300, 7399, "XLI"),     # other business services
    (8000, 8099, "XLV"),     # health services
    (8200, 8299, "XLY"),     # education
    (8700, 8799, "XLI"),     # engineering, consulting
)
SECTOR_ETFS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")


def sic_to_sector(sic) -> str:
    """Map one SEC SIC code to a sector ETF symbol, or "OTHER"."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "OTHER"
    for low, high, sector in SIC_SECTOR_TABLE:
        if low <= code <= high:
            return sector
    return "OTHER"


# ── Helpers shared with earlier checks ─────────────────────────────────────

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _membership_module():
    return _load_module("watchlist_membership_check",
                        OUT_DIR.parent / "survivorship_check_20260926" / "watchlist_membership_check.py")


def _edge_module():
    return _load_module("edge_check", OUT_DIR.parent / "edge_check_20260926" / "edge_check.py")


# ── Pool and membership (pure functions, tested) ───────────────────────────

def price_ticker(ticker: str) -> str:
    """The ticker whose price file holds this S&P ticker's history."""
    return RENAMES.get(ticker, ticker)


def build_pool(snapshots: pd.DataFrame, window=POOL_WINDOW) -> dict[str, set[str]]:
    """Companies that were members at any time in `window`.

    PLAIN ENGLISH: returns {price ticker: every S&P ticker it was listed
    under}.  The snapshot in force on the window's first day counts too,
    because it was the list on that day.
    """
    dates = pd.DatetimeIndex(snapshots["date"])
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    first = max(int(dates.searchsorted(start, side="right")) - 1, 0)
    keep = [i for i in range(len(dates)) if (i >= first and dates[i] <= end)]
    pool: dict[str, set[str]] = {}
    for i in keep:
        for ticker in snapshots["members"].iloc[i]:
            pool.setdefault(price_ticker(ticker), set()).add(ticker)
    return pool


def membership_names(pool: dict[str, set[str]], predecessors: dict) -> dict[str, set[str]]:
    """All names a company may appear under in the snapshots."""
    out = {}
    for company, names in pool.items():
        out[company] = set(names) | {company} | set(predecessors.get(company, ()))
    return out


def member_matrix(snapshots: pd.DataFrame, sessions: pd.DatetimeIndex,
                  names: dict[str, set[str]]) -> pd.DataFrame:
    """True where a company was an index member on a session.

    Uses the latest snapshot on or before each session.  Sessions before the
    first snapshot are False.
    """
    snap_dates = pd.DatetimeIndex(snapshots["date"])
    members = list(snapshots["members"])
    positions = snap_dates.searchsorted(sessions, side="right") - 1
    out = {}
    for company, aliases in names.items():
        flags = np.zeros(len(sessions), dtype=bool)
        cache: dict[int, bool] = {}
        for i, pos in enumerate(positions):
            if pos < 0:
                continue
            if pos not in cache:
                cache[pos] = bool(aliases & members[pos])
            flags[i] = cache[pos]
        out[company] = flags
    return pd.DataFrame(out, index=sessions)


def coverage_table(members: pd.DataFrame, price_dates: dict[str, pd.DatetimeIndex],
                   current: set[str]) -> pd.DataFrame:
    """One row per company: member sessions, sessions with a price, share."""
    rows = []
    for company in members.columns:
        member_days = members.index[members[company].to_numpy()]
        have = price_dates.get(company, pd.DatetimeIndex([]))
        covered = int(member_days.isin(have).sum()) if len(member_days) else 0
        rows.append({
            "company": company,
            "current_member": company in current,
            "member_sessions": int(len(member_days)),
            "priced_sessions": covered,
            "share": (covered / len(member_days)) if len(member_days) else None,
        })
    return pd.DataFrame(rows)


def judge_coverage(table: pd.DataFrame) -> dict:
    """Apply gate C1 to the coverage table."""
    usable = table[table["member_sessions"] > 0]
    ok = usable["share"].fillna(0.0) >= C1_MIN_SESSION_SHARE
    removed = usable[~usable["current_member"]]
    current = usable[usable["current_member"]]
    removed_share = float(ok[removed.index].mean()) if len(removed) else 0.0
    current_share = float(ok[current.index].mean()) if len(current) else 0.0
    return {
        "removed_companies": int(len(removed)), "removed_covered_share": round(removed_share, 4),
        "current_companies": int(len(current)), "current_covered_share": round(current_share, 4),
        "C1_pass": bool(removed_share >= C1_MIN_REMOVED_SHARE and current_share >= C1_MIN_CURRENT_SHARE),
        "not_covered": sorted(usable.loc[~ok, "company"].tolist()),
    }


# ── Monthly universe (pure function, tested) ───────────────────────────────

def monthly_universe(dollar_volume: pd.DataFrame, members: pd.DataFrame,
                     size: int = UNIVERSE_SIZE, lookback: int = DOLLAR_VOLUME_SESSIONS,
                     still_trading: int = STILL_TRADING_SESSIONS,
                     start=POOL_WINDOW[0]) -> dict[pd.Timestamp, list[str]]:
    """The U list for every month: {first session of month: tickers}.

    PLAIN ENGLISH: `dollar_volume` has one row per NYSE session and one
    column per company (NaN = no trade that day).  On the first session of
    each month we look only at data up to the DAY BEFORE, keep companies
    that are index members that day, have at least `lookback` sessions of
    history and traded in the last `still_trading` sessions, and take the
    `size` with the highest median dollar volume over the last `lookback`
    sessions.
    """
    dv = dollar_volume.sort_index()
    sessions = dv.index
    later = sessions[sessions >= pd.Timestamp(start)]
    month_starts = later.to_series().groupby([later.year, later.month]).min()
    history = dv.notna().cumsum()
    out = {}
    for day in month_starts:
        pos = sessions.get_loc(day)
        if pos < lookback:
            continue
        window = dv.iloc[pos - lookback:pos]            # ends the day before
        enough = history.iloc[pos - 1] >= lookback
        recent = dv.iloc[pos - still_trading:pos].notna().any()
        member = members.loc[day] if day in members.index else pd.Series(False, index=dv.columns)
        eligible = enough & recent & member.reindex(dv.columns, fill_value=False)
        median = window.median(skipna=True)[eligible]
        ranked = median.dropna().sort_values(ascending=False, kind="mergesort")
        out[pd.Timestamp(day)] = ranked.index[:size].tolist()
    return out


def universe_mask(panel: pd.DataFrame, lists: dict[pd.Timestamp, list[str]]) -> pd.Series:
    """True for panel rows whose ticker is on that month's U list."""
    if not lists:
        return pd.Series(False, index=panel.index)
    starts = pd.DatetimeIndex(sorted(lists))
    pos = starts.searchsorted(pd.DatetimeIndex(panel["date"]), side="right") - 1
    sets = [set(lists[d]) for d in starts]
    tickers = panel["ticker"].astype(str).str.upper().to_numpy()
    allowed = np.array([p >= 0 and t in sets[p] for p, t in zip(pos, tickers)], dtype=bool)
    return pd.Series(allowed, index=panel.index)


# ── Delisting (pure function, tested) ──────────────────────────────────────

def pad_after_delisting(frame: pd.DataFrame, sessions: pd.DatetimeIndex, haircut: float = 0.0) -> pd.DataFrame:
    """Extend a stopped stock's prices with flat bars at its last Close.

    PLAIN ENGLISH: if a held stock stops trading (bought out, bankrupt), the
    backtest needs a price to sell it at.  We add one flat bar per later
    session at the last Close (times 1 - haircut for the diagnostic row).
    The strategy never BUYS on these rows: they are removed from the panel
    after the sell prices are worked out.
    """
    if frame.empty:
        return frame
    last_day = frame.index.max()
    later = sessions[sessions > last_day]
    if len(later) == 0:
        return frame
    price = float(frame["Close"].iloc[-1]) * (1.0 - haircut)
    pad = pd.DataFrame(index=later, columns=frame.columns, dtype=float)
    for col in ("Open", "High", "Low", "Close"):
        pad[col] = price
    pad["Volume"] = 0.0
    return pd.concat([frame, pad])


# ── Judging the runs (pure function, tested) ───────────────────────────────

def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def _alphas(rows, test, delay, key="decision_alpha_vs_qqq_pct"):
    return [r[key] for r in rows if r["test"] == test and r["delay"] == delay and r.get(key) is not None]


def judge(rows: list[dict]) -> dict:
    """Apply the pre-registered H-full-universe rules."""
    r0_late = _median(_alphas(rows, "R0", 1))
    u_late = _median(_alphas(rows, "U", 1))
    u_on_time_list = _alphas(rows, "U", 0)
    u_on_time = _median(u_on_time_list)
    monkeys = _alphas(rows, "MU", 0)
    monkey_p95 = float(np.percentile(monkeys, U2_PERCENTILE)) if monkeys else None

    u1 = (u_late is not None and r0_late is not None and u_late > 0
          and u_late >= U1_MIN_SHARE_OF_R0 * r0_late)
    u2 = u_on_time is not None and monkey_p95 is not None and u_on_time > monkey_p95
    if u1 and u2:
        verdict = "edge survives the full survivorship test"
    elif u2:
        verdict = "edge real but mostly list hindsight"
    else:
        verdict = "edge not shown"

    late_by_offset = {r["offset"]: r["decision_alpha_vs_qqq_pct"] for r in rows
                      if r["test"] == "U" and r["delay"] == 1 and r.get("decision_alpha_vs_qqq_pct") is not None}
    delay_costs = [r["decision_alpha_vs_qqq_pct"] - late_by_offset[r["offset"]] for r in rows
                   if r["test"] == "U" and r["delay"] == 0 and r["offset"] in late_by_offset
                   and r.get("decision_alpha_vs_qqq_pct") is not None]
    return {
        "numbers": {
            "R0_median_late_alpha": r0_late, "U_median_late_alpha": u_late,
            "U_median_on_time_alpha": u_on_time, "MU_p95_alpha": monkey_p95,
            "MU_median_alpha": _median(monkeys),
            "U_share_of_R0_late": (u_late / r0_late) if (u_late is not None and r0_late) else None,
        },
        "diagnostics": {
            "U_on_time_spread": (max(u_on_time_list) - min(u_on_time_list)) if u_on_time_list else None,
            "U_mean_delay_cost": float(np.mean(delay_costs)) if delay_costs else None,
            "UD_median_on_time_alpha": _median(_alphas(rows, "UD", 0)),
            "R0_2023_2026_median": _median(_alphas(rows, "R0", 0, "diagnostic_alpha_vs_qqq_pct")),
            "U_2023_2026_median": _median(_alphas(rows, "U", 0, "diagnostic_alpha_vs_qqq_pct")),
            "MU_2023_2026_median": _median(_alphas(rows, "MU", 0, "diagnostic_alpha_vs_qqq_pct")),
        },
        "gates": {"U1_survivorship": u1, "U2_beats_random_picks": u2},
        "verdict": verdict,
    }


# ── Downloads ──────────────────────────────────────────────────────────────

class QuotaReached(RuntimeError):
    """Tiingo said the hourly/daily/monthly allowance is used up."""


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def _tiingo_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
    except Exception:
        pass
    key = os.environ.get("TIINGO_KEY") or os.environ.get("TIINGO_API_KEY")
    if not key:
        raise SystemExit("No Tiingo key: add TIINGO_KEY=... to .env")
    return key


class Throttle:
    """Wait between requests so we stay under Tiingo's hourly limit."""

    def __init__(self, per_hour: int):
        self.gap = 3600.0 / max(per_hour, 1)
        self.last = 0.0

    def wait(self) -> None:
        pause = self.last + self.gap - time.time()
        if pause > 0:
            time.sleep(pause)
        self.last = time.time()


def _tiingo_get(session, url: str, key: str, params: dict | None = None):
    response = session.get(url, params=params or {}, headers={"Authorization": f"Token {key}"}, timeout=60)
    text = response.text[:300].lower()
    if response.status_code == 429 or "allocation" in text or "rate limit" in text:
        raise QuotaReached(response.text[:300])
    return response


def download_prices(companies: list[str], *, per_hour: int, end: str) -> dict:
    """Download each company's full daily history once (resumable)."""
    import requests

    key = _tiingo_key()
    log = _read_json(DOWNLOAD_LOG, {})
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    throttle = Throttle(per_hour)
    session = requests.Session()
    todo = [c for c in companies if c not in log]
    print(f"[download] {len(companies) - len(todo)} done before, {len(todo)} to go "
          f"(about {len(todo) * 3600 / max(per_hour, 1) / 3600:.1f} hours at {per_hour}/hour)", flush=True)
    for n, company in enumerate(todo, 1):
        throttle.wait()
        try:
            response = _tiingo_get(session, TIINGO_URL.format(ticker=company) + "/prices", key,
                                   {"startDate": PRICE_START, "endDate": end})
        except QuotaReached as exc:
            print(f"[download] Tiingo allowance used up: {exc}\n"
                  f"[download] progress saved; run the same command again later.", flush=True)
            _write_json(DOWNLOAD_LOG, log)
            return {"complete": False, "reason": "quota", "done": len(log), "total": len(companies)}
        except Exception as exc:   # network trouble: skip for now, retry next run
            print(f"[download] {company}: {exc}", flush=True)
            continue
        if response.status_code == 404:
            log[company] = {"status": "not_found"}
        elif response.ok:
            rows = response.json()
            if not rows:
                log[company] = {"status": "empty"}
            else:
                frame = pd.DataFrame(rows)
                path = RAW_DIR / f"{company}.parquet"
                frame.to_parquet(path, index=False)
                log[company] = {
                    "status": "ok", "rows": len(frame),
                    "first": str(frame["date"].iloc[0])[:10], "last": str(frame["date"].iloc[-1])[:10],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "downloaded": str(date.today()),
                }
        else:
            print(f"[download] {company}: HTTP {response.status_code} {response.text[:120]}", flush=True)
            continue
        if n % 10 == 0 or n == len(todo):
            _write_json(DOWNLOAD_LOG, log)
            print(f"[download] {len(log)}/{len(companies)}", flush=True)
    _write_json(DOWNLOAD_LOG, log)
    missing = [c for c in companies if c not in log]
    return {"complete": not missing, "reason": "network errors" if missing else None,
            "done": len(log), "total": len(companies)}


def load_raw(company: str) -> pd.DataFrame:
    """One company's Tiingo prices as the project's OHLCV frame.

    Open/High/Low/Close/Volume are split- and dividend-adjusted (like the
    project's other price files).  RawClose/RawVolume are the real traded
    numbers, used for the dollar-volume ranking.
    """
    path = RAW_DIR / f"{company}.parquet"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_parquet(path)
    index = pd.DatetimeIndex(pd.to_datetime(raw["date"], utc=True)).tz_localize(None).normalize()
    frame = pd.DataFrame({
        "Open": raw["adjOpen"].to_numpy(float), "High": raw["adjHigh"].to_numpy(float),
        "Low": raw["adjLow"].to_numpy(float), "Close": raw["adjClose"].to_numpy(float),
        "Volume": raw["adjVolume"].to_numpy(float),
        "RawClose": raw["close"].to_numpy(float), "RawVolume": raw["volume"].to_numpy(float),
    }, index=index)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    # A bar needs positive prices to be usable.
    good = (frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    return frame.loc[good]


# ── Sectors from the SEC ───────────────────────────────────────────────────

def _sec_get(session, url: str, params: dict | None = None):
    agent = os.environ.get("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT)
    time.sleep(0.2)   # the SEC allows 10 requests a second; stay well under
    response = session.get(url, params=params or {}, headers={"User-Agent": agent}, timeout=60)
    response.raise_for_status()
    return response


def lookup_sectors(companies: list[str], download_log: dict, *, per_hour: int) -> dict:
    """Find each company's SIC code and sector (cached, resumable).

    PLAIN ENGLISH: companies that still trade are found in the SEC's
    ticker list.  For companies that stopped trading, their ticker may now
    belong to someone else, so we ask Tiingo for the company NAME and look
    that name up in the SEC's company search instead.
    """
    import requests

    cache_path = SEC_DIR / "sectors.json"
    cache = _read_json(cache_path, {})
    session = requests.Session()
    tickers_path = SEC_DIR / "company_tickers.json"
    if not tickers_path.exists():
        SEC_DIR.mkdir(parents=True, exist_ok=True)
        tickers_path.write_bytes(_sec_get(session, SEC_TICKERS_URL).content)
    sec_by_ticker = {str(v["ticker"]).upper().replace(".", "-"): int(v["cik_str"])
                     for v in json.loads(tickers_path.read_text()).values()}
    key = None
    throttle = Throttle(per_hour)
    recent = pd.Timestamp(date.today()) - pd.Timedelta(days=30)
    for company in companies:
        if company in cache:
            continue
        info = download_log.get(company, {})
        if info.get("status") != "ok":
            continue
        entry = {"cik": None, "sic": None, "method": None}
        still_listed = pd.Timestamp(info.get("last", "1900-01-01")) >= recent
        try:
            if still_listed and company in sec_by_ticker:
                entry.update(cik=sec_by_ticker[company], method="sec_ticker")
            else:
                key = key or _tiingo_key()
                throttle.wait()
                meta = _tiingo_get(session, TIINGO_URL.format(ticker=company), key).json()
                name = str(meta.get("name") or "").strip()
                entry["tiingo_name"] = name
                if name:
                    hits = _sec_get(session, SEC_SEARCH_URL, {"keysTyped": name}).json()["hits"]["hits"]
                    if hits:
                        best = max(hits, key=lambda h: (h.get("_score", 0), h["_source"].get("rank", 0)))
                        entry.update(cik=int(best["_id"]), method="sec_name_search",
                                     sec_name=best["_source"].get("entity"))
            if entry["cik"]:
                sub = _sec_get(session, SEC_SUBMISSIONS_URL.format(cik=entry["cik"])).json()
                entry.update(sic=sub.get("sic") or None, sic_description=sub.get("sicDescription"),
                             sec_name=sub.get("name"))
        except QuotaReached as exc:
            print(f"[sectors] Tiingo allowance used up: {exc}; run again later.", flush=True)
            _write_json(cache_path, cache)
            return cache
        except Exception as exc:
            entry["error"] = str(exc)[:200]
        entry["sector"] = sic_to_sector(entry.get("sic"))
        cache[company] = entry
        _write_json(cache_path, cache)
    return cache


# ── Step 1: coverage ───────────────────────────────────────────────────────

def _snapshots(args):
    membership = _membership_module()
    if args.membership_csv:
        raw = Path(args.membership_csv).read_bytes()
    else:
        import requests
        response = requests.get(membership.SOURCE_URL, timeout=60)
        response.raise_for_status()
        raw = response.content
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / "sp500_membership.csv").write_bytes(raw)
    return membership, membership.load_snapshots(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _download_order(pool: dict, current: set[str]) -> list[str]:
    """Removed companies first: they decide C1 and are the point of the test."""
    removed = sorted(c for c in pool if c not in current)
    return removed + sorted(c for c in pool if c in current)


def step_coverage(args) -> dict:
    import core_satellite_alpha as core

    membership, snapshots, membership_sha = _snapshots(args)
    pool = build_pool(snapshots)
    names = membership_names(pool, membership.PREDECESSORS)
    latest = set(snapshots["members"].iloc[-1])
    current = {c for c, aliases in names.items() if aliases & latest}
    companies = _download_order(pool, current)
    status = download_prices(companies, per_hour=args.per_hour, end=args.end)
    if not status["complete"]:
        return {"step": "coverage", "status": "downloads unfinished", **status}

    log = _read_json(DOWNLOAD_LOG, {})
    sectors = lookup_sectors(companies, log, per_hour=args.per_hour)
    sessions = core._nyse_sessions(*COVERAGE_WINDOW)
    members = member_matrix(snapshots, sessions, names)
    price_dates = {c: load_raw(c).index for c in companies if log.get(c, {}).get("status") == "ok"}
    table = coverage_table(members, price_dates, current)
    result = judge_coverage(table)
    payload = {
        "hypothesis": HYPOTHESIS, "step": "coverage", "research_only": True, "approves_trading": False,
        "price_source": "Tiingo end-of-day API (adjusted OHLCV)",
        "membership_source_url": membership.SOURCE_URL, "membership_source_sha256": membership_sha,
        "pool_companies": len(pool), "renames_used": {k: v for k, v in RENAMES.items() if v in pool},
        "result": result,
        "sector_counts": pd.Series([sectors.get(c, {}).get("sector", "OTHER") for c in companies]).value_counts().to_dict(),
        "sector_lookup_errors": sorted(c for c, e in sectors.items() if e.get("error")),
        "download_status_counts": pd.Series([v.get("status") for v in log.values()]).value_counts().to_dict(),
        "companies": table.to_dict("records"),
    }
    _write_json(COVERAGE_JSON, payload)
    print(json.dumps({"C1": result["C1_pass"], "removed": result["removed_covered_share"],
                      "current": result["current_covered_share"]}, indent=1))
    return payload


# ── Step 2: features and runs ──────────────────────────────────────────────

LOCAL_SYMBOLS = ("SPY", "QQQ", "GLD", "IEF") + SECTOR_ETFS


@contextlib.contextmanager
def patched(module, **values):
    """Temporarily replace module attributes, then put them back."""
    old = {k: getattr(module, k) for k in values if hasattr(module, k)}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k in values:
            if k in old:
                setattr(module, k, old[k])
            else:
                delattr(module, k)


def _local_etf(symbol: str, start=None, end=None, **_kwargs) -> pd.DataFrame:
    """Serve SPY/QQQ/sector ETFs from data/, so no feature build needs the internet."""
    from settings import DATA_DIR
    if symbol.upper() not in LOCAL_SYMBOLS:
        raise RuntimeError(f"offline feature build: {symbol} not available")
    frame = pd.read_parquet(Path(DATA_DIR) / f"{symbol.upper()}.parquet",
                            columns=["Open", "High", "Low", "Close", "Volume"])
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.index = index.tz_localize(None) if index.tz is not None else index
    if start:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end:
        frame = frame[frame.index < pd.Timestamp(end)]
    return frame.sort_index()


def build_features(companies: list[str], sector_map: dict[str, str], *, rebuild: bool, end: str) -> list[str]:
    """Build the project's feature files for every company (offline)."""
    import cross_sectional_features as xs
    import fundamental_features as ff
    import pipeline_shared as ps

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    built = []

    def offline_prices(ticker, start, stop):
        frame = load_raw(ticker)[["Open", "High", "Low", "Close", "Volume"]]
        frame = frame[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(stop))]
        return frame.copy()

    def no_multi_download(*_a, **_k):
        raise RuntimeError("offline feature build")

    with contextlib.ExitStack() as stack:
        stack.enter_context(patched(ps, fetch_price_data=offline_prices, _dp_download_single=_local_etf,
                                    SECTOR_MAP=sector_map, USE_NEWS_SENTIMENT=False,
                                    SOCIAL_SENTIMENT_ALPHA_ENABLED=False))
        stack.enter_context(patched(ff, _dp_download_prices=no_multi_download, USE_FUNDAMENTAL_ZSCORES=False,
                                    SECTOR_MAP=sector_map))
        for n, company in enumerate(companies, 1):
            out = FEATURE_DIR / f"{company}.parquet"
            if out.exists() and not rebuild:
                built.append(company)
                continue
            try:
                frame = ps.build_research_feature_frame(company, PRICE_START, end)
            except Exception as exc:
                print(f"[features] {company}: {exc}", flush=True)
                continue
            if frame.empty:
                continue
            frame.to_parquet(out)
            built.append(company)
            if n % 25 == 0:
                print(f"[features] {n}/{len(companies)}", flush=True)
        # Cross-sectional ranks, recomputed across the whole pool.
        summary = xs.apply_cross_sectional_rank_features(built, str(FEATURE_DIR), sector_map=sector_map)
        print(f"[features] xs_rank: updated={summary.get('updated')} errors={len(summary.get('write_errors', []))}",
              flush=True)
    return built


def dollar_volume_table(companies: list[str], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    cols = {}
    for company in companies:
        raw = load_raw(company)
        if raw.empty:
            continue
        cols[company] = (raw["RawClose"] * raw["RawVolume"]).reindex(sessions)
    return pd.DataFrame(cols, index=sessions)


def build_u_panel(companies, sector_map, lists, last_real, sessions, *, haircut: float):
    """The incumbent's panel on the U list (scores ranked within the list)."""
    import alpha_factor_backtest as afb
    import core_satellite_alpha as core

    original_read = pd.read_parquet
    feature_dir = str(FEATURE_DIR)

    def read_with_padding(path, *a, **k):
        frame = original_read(path, *a, **k)
        company = Path(str(path)).stem
        if str(Path(str(path)).parent) == feature_dir and company in last_real:
            index = pd.DatetimeIndex(pd.to_datetime(frame.index))
            frame.index = index.tz_localize(None) if index.tz is not None else index
            frame = pad_after_delisting(frame.sort_index(), sessions, haircut)
        return frame

    specs = afb.load_feature_specs(write_health_outputs=False)
    with patched(afb, WATCHLIST=list(companies), DATA_DIR=feature_dir, SECTOR_MAP=sector_map), \
            patched(pd, read_parquet=read_with_padding):
        panel = afb.load_factor_panel(specs, require_forward_returns=False)
    membership_status = panel.attrs.get("point_in_time_membership", {})
    if membership_status.get("applied"):
        raise SystemExit("The project's own membership filter was applied; the U test expects it off.")
    # Drop the padded rows (after the stop date): nobody can buy those.
    last = panel["ticker"].map(last_real)
    panel = panel[last.isna() | (panel["date"] <= last)]
    panel = panel.loc[universe_mask(panel, lists)].reset_index(drop=True)
    empty_ml = pd.DataFrame(columns=["date", "ticker", "ml_score"])
    return core._ensure_robust_score_columns(afb.attach_scores(panel, specs, empty_ml))


def step_runs(args) -> dict:
    from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
    import core_satellite_alpha as core
    from validation_bundle import load_approved_research_config

    coverage = _read_json(COVERAGE_JSON, {})
    if not coverage.get("result", {}).get("C1_pass"):
        raise SystemExit("C1 has not passed (run --step coverage first). By the pre-registered rule, no runs.")
    edge = _edge_module()
    membership, snapshots, _sha = _snapshots(args)
    pool = build_pool(snapshots)
    names = membership_names(pool, membership.PREDECESSORS)
    latest = set(snapshots["members"].iloc[-1])
    log = _read_json(DOWNLOAD_LOG, {})
    companies = sorted(c for c in pool if log.get(c, {}).get("status") == "ok")
    sectors = _read_json(SEC_DIR / "sectors.json", {})
    sector_map = {c: sectors.get(c, {}).get("sector", "OTHER") for c in companies}

    config = load_approved_research_config(Path("signals") / "core_satellite_alpha_metrics.json")
    # R0 first, from the untouched project data (same as H-edge's R0).
    specs = load_feature_specs(write_health_outputs=False)
    r0_panel = core._ensure_robust_score_columns(attach_scores(
        load_factor_panel(specs, require_forward_returns=False), specs, load_prediction_scores()))
    sessions = core._nyse_sessions(r0_panel["date"].min(), r0_panel["date"].max())
    end = sessions[-30]

    all_sessions = core._nyse_sessions(PRICE_START, args.end)
    built = build_features(companies, sector_map, rebuild=args.rebuild_features, end=args.end)
    last_real = {c: load_raw(c).index.max() for c in built}
    data_end = max(last_real.values())
    last_real = {c: d for c, d in last_real.items() if d < data_end - pd.Timedelta(days=10)}
    members = member_matrix(snapshots, all_sessions, {c: names[c] for c in built})
    lists = monthly_universe(dollar_volume_table(built, all_sessions), members)
    u_panel = build_u_panel(built, sector_map, lists, last_real, all_sessions, haircut=0.0)
    ud_panel = build_u_panel(built, sector_map, lists, last_real, all_sessions, haircut=DELIST_HAIRCUT)

    later_removed = {c for c in built if not (names[c] & latest)}
    notes = {
        "u_companies_ever_listed": len({t for v in lists.values() for t in v}),
        "u_months": len(lists), "u_panel_rows": int(len(u_panel)),
        "stopped_trading_companies": len(last_real),
        "later_removed_companies_in_pool": len(later_removed),
        "universe_lists": {str(k.date()): v for k, v in lists.items()},
    }
    rows: list[dict] = []
    valid = args.offsets == N_OFFSETS and args.monkeys == N_MONKEYS

    def save(result=None):
        payload = {"hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
                   "valid_full_test": valid, "end_date": str(pd.Timestamp(end).date()),
                   "offsets": args.offsets, "monkeys": args.monkeys, "notes": notes, "rows": rows}
        if result is not None:
            payload["result"] = result
        _write_json(Path(args.out), payload)

    def holdings_stats(trades: pd.DataFrame) -> dict:
        """How many picks were later-removed names, and how many stopped trading while held."""
        picks = removed = delist = 0
        if "overlay_tickers" not in trades:
            return {}
        for tickers, exit_day in zip(trades["overlay_tickers"].fillna(""), trades["exit_date"]):
            for t in [x for x in str(tickers).split(",") if x]:
                picks += 1
                removed += t in later_removed
                delist += bool(t in last_real and pd.Timestamp(exit_day) > last_real[t])
        return {"stock_positions": picks, "later_removed_positions": removed, "delisting_exits": delist}

    def record(test, equity, trades, extra, *, delay, offset, seed=None, t0=None):
        row = edge._summarise(core, test, equity, extra, delay=delay, offset=offset, seed=seed, t0=t0)
        row.update(holdings_stats(trades))
        used = set(str(c) for c in trades.get("score_col", pd.Series(dtype=str)).unique())
        row["score_cols"] = sorted(used)
        rows.append(row)
        print(json.dumps(row), flush=True)
        save()

    for test, panel in (("R0", r0_panel), ("U", u_panel)):
        for delay in DELAYS:
            for offset in range(args.offsets):
                t0 = time.time()
                equity, trades, extra = core.run_core_satellite(
                    panel, {**config, "entry_delay_days": delay}, evaluation_start=sessions[offset], evaluation_end=end)
                record(test, equity, trades, extra, delay=delay, offset=offset, t0=t0)
    for offset in range(args.offsets):
        t0 = time.time()
        equity, trades, extra = core.run_core_satellite(
            ud_panel, {**config, "entry_delay_days": 0}, evaluation_start=sessions[offset], evaluation_end=end)
        record("UD", equity, trades, extra, delay=0, offset=offset, t0=t0)
    for seed in range(args.monkeys):
        t0 = time.time()
        offset = seed % N_OFFSETS
        equity, trades, extra = core.run_core_satellite(
            edge.randomized_scores(u_panel, seed), {**config, "entry_delay_days": 0},
            evaluation_start=sessions[offset], evaluation_end=end)
        unexpected = {c for c in trades.get("score_col", pd.Series(dtype=str)).astype(str).unique()
                      if c not in edge.SCORE_COLS and not c.startswith("_blended_regime_score_")}
        if unexpected:
            raise SystemExit(f"Random-pick runs used non-randomised score columns: {sorted(unexpected)}")
        record("MU", equity, trades, extra, delay=0, offset=offset, seed=seed, t0=t0)

    result = judge(rows)
    save(result)
    print(json.dumps({k: result[k] for k in ("gates", "verdict")}, indent=1))
    return result


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Pre-registered H-full-universe check (research only).")
    parser.add_argument("--step", choices=("coverage", "runs"), required=True)
    parser.add_argument("--per-hour", type=int, default=45, help="Tiingo requests per hour (free plan: 50)")
    parser.add_argument("--end", default=str(date.today()), help="last price date to download")
    parser.add_argument("--membership-csv", help="saved copy of the membership CSV (default: download it)")
    parser.add_argument("--offsets", type=int, default=N_OFFSETS)
    parser.add_argument("--monkeys", type=int, default=N_MONKEYS)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--out", default=str(RESULT_JSON))
    args = parser.parse_args(argv)
    return step_coverage(args) if args.step == "coverage" else step_runs(args)


if __name__ == "__main__":
    main()
