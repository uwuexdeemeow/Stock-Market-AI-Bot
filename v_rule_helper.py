"""
v_rule_helper.py — this month's QQQ / cash split for the "V" rule.

PLAIN ENGLISH: H-index-risk tested a simple risk rule for holding QQQ and
it passed (see Documentation/DELAY_STRESS_PAPER_ADVISORY.md):

    Once a month, look at how jumpy QQQ has been over the last 63 trading
    days (its "volatility", turned into a yearly number).
    QQQ share = 20% / that volatility, never more than 100%.
    The rest goes in cash (BIL, a T-bill fund).

When the market is calm (volatility 20% or less) you hold 100% QQQ.  When
it gets jumpy you hold less, which is what made the worst falls smaller in
the test.  This script only does the arithmetic with the latest prices.
It never trades.  Using it with real money is your own decision.

How to run (project root, needs internet):
    python v_rule_helper.py                  # print this month's split
    python v_rule_helper.py --amount 10000   # also show it in dollars
Use it on the last trading day of the month (after the close) or the next
morning, then trade at that day's open, as in the test.
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

VOL_SESSIONS = 63       # about 3 months of trading days (same as the test)
VOL_TARGET = 0.20       # 20% a year (same as the test)
TRADING_DAYS = 252


def qqq_weight(closes: pd.Series, sessions: int = VOL_SESSIONS, target: float = VOL_TARGET) -> dict:
    """The V rule on a series of daily closes: QQQ share and the volatility behind it."""
    returns = closes.pct_change().dropna().tail(sessions)
    if len(returns) < sessions:
        raise ValueError(f"Need at least {sessions + 1} daily closes, got {len(closes)}.")
    vol = float(returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    weight = float(min(1.0, target / vol)) if vol > 0 else 1.0
    return {"as_of": str(pd.Timestamp(closes.index[-1]).date()), "volatility": vol, "qqq_weight": weight}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="This month's QQQ/cash split for the V rule (no trading).")
    parser.add_argument("--amount", type=float, help="money to split, e.g. 10000")
    args = parser.parse_args(argv)

    from data_provider import download_single
    start = pd.Timestamp(date.today()) - pd.Timedelta(days=150)
    prices = download_single("QQQ", start=str(start.date()))
    closes = prices["Close"].dropna()
    result = qqq_weight(closes)

    print(f"QQQ prices up to:        {result['as_of']}")
    print(f"QQQ volatility (63 days): {100 * result['volatility']:.1f}% a year")
    print(f"QQQ share:               {100 * result['qqq_weight']:.0f}%")
    print(f"Cash (BIL) share:        {100 * (1 - result['qqq_weight']):.0f}%")
    if args.amount:
        print(f"\nOf ${args.amount:,.0f}: QQQ ${args.amount * result['qqq_weight']:,.0f}, "
              f"cash ${args.amount * (1 - result['qqq_weight']):,.0f}")
    return result


if __name__ == "__main__":
    main()
