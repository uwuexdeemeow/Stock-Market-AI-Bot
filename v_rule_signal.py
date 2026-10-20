"""
v_rule_signal.py — the daily paper-trading signal for rule V (QQQ + BIL).

PLAIN ENGLISH: from 2026-10-30 the paper account follows rule V, the
volatility-target rule that passed the pre-registered H-index-risk test
(see Documentation/V_RULE_FORWARD_TEST.md).  The rule:

    On the last trading day of each month, measure how jumpy QQQ has been
    over its last 63 daily returns (its "volatility", as a yearly number).
    QQQ share = 20% / that volatility, never more than 100%.
    The rest goes in BIL (a T-bill fund, basically cash).
    Trade at the next session, then hold until the next month end.

This script runs every trading day inside daily_run.py, but it only does the
arithmetic.  It writes the same signal file the broker script already reads
(signals/core_satellite_alpha_signal.csv), so the broker's tested
"rebalance on the signal day, hold otherwise" calendar does the rest:

  * last_scheduled_rebalance_date = the latest month-end signal day;
  * scheduled_rebalance_today     = True when today's data IS that day;
  * on every other day the broker holds.

It never contacts the broker and never sends an order.

How to run (project root; data/QQQ.parquet and data/BIL.parquet must be
fresh, which refresh_etf_data.py takes care of):
    python v_rule_signal.py            # write the signal file
    python v_rule_signal.py --print    # also print the row
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from run_evidence import current_run_id
from safe_io import atomic_write_csv
from settings import DATA_DIR, SIGNAL_DIR
from signal_freshness import latest_completed_us_trading_day
from v_rule_helper import VOL_SESSIONS, VOL_TARGET, qqq_weight

SIGNAL_TYPE = "v_rule"
REBALANCE_POLICY = "v_rule_month_end"
SIGNAL_PATH = Path(SIGNAL_DIR) / "core_satellite_alpha_signal.csv"
# Same timestamp zone as core_satellite_alpha.py, so freshness checks agree.
PAPER_SIGNAL_TIMEZONE = os.environ.get("PAPER_SIGNAL_TIMEZONE", os.environ.get("TZ", "Asia/Singapore"))


def _load_closes(symbol: str, data_dir: str | Path = DATA_DIR) -> pd.Series:
    """Daily adjusted closes for one ETF, indexed by plain dates."""
    frame = pd.read_parquet(Path(data_dir) / f"{symbol}.parquet", columns=["Close"])
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
    closes = pd.to_numeric(frame["Close"], errors="coerce")
    closes = closes[~closes.index.duplicated(keep="last")].sort_index()
    return closes.dropna()


def _nyse_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """NYSE trading days between two dates (holidays excluded)."""
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    return pd.DatetimeIndex(calendar.sessions_in_range(start, end)).tz_localize(None).normalize()


def month_end_schedule(latest: pd.Timestamp) -> dict:
    """Find the signal days around the newest data date.

    PLAIN ENGLISH: a signal day is the last NYSE session of a month.  The
    "current" one is the newest signal day on or before the latest data;
    the "next" one is the first after it.
    """
    latest = pd.Timestamp(latest).normalize()
    sessions = _nyse_sessions(latest - pd.Timedelta(days=70), latest + pd.Timedelta(days=70))
    month_ends = pd.DatetimeIndex(
        sessions.to_series().groupby([sessions.year, sessions.month]).max().to_numpy()
    )
    done = month_ends[month_ends <= latest]
    upcoming = month_ends[month_ends > latest]
    if done.empty or upcoming.empty:
        raise ValueError(f"Could not place {latest.date()} in the month-end calendar")
    return {
        "last": pd.Timestamp(done[-1]),
        "next": pd.Timestamp(upcoming[0]),
        "scheduled_today": bool(done[-1] == latest),
    }


def build_signal_row(
    qqq_closes: pd.Series,
    bil_closes: pd.Series,
    *,
    now: datetime | None = None,
) -> dict:
    """Turn the price history into one broker-ready signal row.

    PLAIN ENGLISH: the weights always come from closes up to the signal day,
    even on a catch-up day after a failed run, because that is when the
    tested rule decides.  Any data problem makes the row "not paper-ready"
    and the broker refuses to trade it.
    """
    problems: list[str] = []
    # A provider may already include today's unfinished bar while the market
    # is open. The tested rule decides on completed closes only, so cut both
    # histories at the latest fully completed NYSE session.
    completed = pd.Timestamp(latest_completed_us_trading_day(now=now)).normalize()
    qqq_closes = qqq_closes.loc[:completed]
    bil_closes = bil_closes.loc[:completed]
    latest = pd.Timestamp(qqq_closes.index.max()).normalize()
    if latest != completed:
        problems.append(f"qqq_prices_not_current:{latest.date()}_vs_{completed.date()}")
    schedule = month_end_schedule(latest)
    signal_day = schedule["last"]

    weight = 0.0
    volatility = float("nan")
    try:
        result = qqq_weight(qqq_closes.loc[:signal_day], VOL_SESSIONS, VOL_TARGET)
        if pd.Timestamp(result["as_of"]) != signal_day:
            problems.append(f"qqq_close_missing_on_signal_day_{signal_day.date()}")
        weight = float(result["qqq_weight"])
        volatility = float(result["volatility"])
    except ValueError as exc:
        problems.append(f"qqq_history_too_short:{exc}")
    if not 0.0 <= weight <= 1.0:
        problems.append(f"qqq_weight_out_of_range_{weight}")
    bil_latest = pd.Timestamp(bil_closes.index.max()).normalize() if len(bil_closes) else pd.NaT
    if pd.isna(bil_latest) or bil_latest != latest:
        problems.append(f"bil_prices_not_current:{bil_latest}")

    ready = not problems
    tz = ZoneInfo(str(PAPER_SIGNAL_TIMEZONE))
    stamp = (now.astimezone(tz) if now is not None else datetime.now(tz)).isoformat(timespec="minutes")
    qqq_share = round(weight, 6) if ready else 0.0
    bil_share = round(1.0 - weight, 6) if ready else 0.0
    return {
        "paper_signal_type": SIGNAL_TYPE,
        "strategy": SIGNAL_TYPE,
        "paper_ready": ready,
        "gates_all_pass": ready,
        # Rule V has no walk-forward approval or medium-risk review: it passed
        # its own pre-registered test.  This flag only means "no data problem".
        "medium_risk_review_pass": ready,
        "reason": "rule V signal ok" if ready else "; ".join(problems),
        "current_regime": SIGNAL_TYPE,
        "target_spy_weight": 0.0,
        "target_qqq_weight": qqq_share,
        "target_tqqq_weight": 0.0,
        "target_bil_weight": bil_share,
        "target_cash_weight": 0.0,
        "overlay_weights_json": "{}",
        "overlay_tickers": "",
        "gross_exposure": round(qqq_share + bil_share, 6),
        "max_gross_exposure": 1.0,
        "paper_max_gross_exposure": 1.0,
        "paper_weight_scale": 1.0,
        "live_config_hash": "",
        "rebalance_policy": REBALANCE_POLICY,
        "last_scheduled_rebalance_date": str(signal_day.date()),
        "next_scheduled_rebalance_date": str(schedule["next"].date()),
        "scheduled_rebalance_today": bool(schedule["scheduled_today"]),
        "v_signal_day": str(signal_day.date()),
        "v_qqq_volatility": round(volatility, 6) if volatility == volatility else "",
        "v_vol_sessions": VOL_SESSIONS,
        "v_vol_target": VOL_TARGET,
        "latest_factor_date": str(latest.date()),
        "live_regime_refresh_failed": False,
        "predicted_at": stamp,
        "run_id": current_run_id(),
    }


def write_signal(row: dict, path: Path = SIGNAL_PATH) -> Path:
    """Atomically replace the broker's signal file with this one row."""
    atomic_write_csv(pd.DataFrame([row]), path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the rule V paper signal (no trading).")
    parser.add_argument("--print", action="store_true", help="also print the signal row")
    args = parser.parse_args(argv)

    row = build_signal_row(_load_closes("QQQ"), _load_closes("BIL"))
    path = write_signal(row)
    print(f"Rule V signal -> {path}")
    print(f"  data up to {row['latest_factor_date']}, signal day {row['v_signal_day']}"
          f" (trade today: {row['scheduled_rebalance_today']})")
    print(f"  QQQ {row['target_qqq_weight']:.1%}, BIL {row['target_bil_weight']:.1%}"
          f", next signal day {row['next_scheduled_rebalance_date']}")
    if args.print:
        print(json.dumps(row, indent=2, default=str))
    if not row["paper_ready"]:
        print(f"  NOT paper-ready: {row['reason']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
