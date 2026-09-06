"""Recover free source candidates and paper records using read-only GET requests.

Raw responses stay in ignored data/audit_recovery. Candidates never become
verified production inputs automatically, and this script cannot submit orders.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

from portfolio_ledger import replay_events
from safe_io import atomic_write_json, atomic_write_csv


def activity_pages(get_page, page_size=100, maximum_pages=1000):
    """Follow activity IDs, rejecting repeated pages instead of losing fills."""
    rows, seen, token = [], set(), None
    for _ in range(maximum_pages):
        batch = get_page(token, page_size)
        if not isinstance(batch, list):
            raise ValueError("Activity response is not a list")
        for row in batch:
            key = row.get("id")
            if not key or key in seen:
                raise ValueError("Activity pagination repeated or omitted an ID")
            seen.add(key)
            rows.append(row)
        if len(batch) < page_size:
            return rows
        token = batch[-1]["id"]
    raise ValueError("Activity pagination limit reached")


def activity_events(rows):
    """Keep separately billed fees separate; refuse unsupported cash/share events."""
    output = []
    for row in rows:
        kind = row["activity_type"]
        if kind == "FILL":
            if row["side"] not in {"buy", "sell"}:
                raise ValueError("Unsupported recorded fill side")
            output.append({"kind": "fill", "event_id": row["id"], "order_id": row["order_id"],
                           "timestamp": row["transaction_time"], "ticker": row["symbol"],
                           "quantity": float(row["qty"]) * (1 if row["side"] == "buy" else -1),
                           "price": float(row["price"]), "fee": 0., "fee_source": "separate_complete_cash_activity_stream"})
        elif kind == "FEE":
            output.append({"kind": "cash_adjustment", "event_id": row["id"],
                           "timestamp": row.get("created_at") or row["date"] + "T23:59:59Z",
                           "amount": float(row["net_amount"]), "source": "alpaca_fee_activity"})
        else:
            raise ValueError(f"Unmapped activity requires review: {kind}")
    return pd.DataFrame(output)


def candidate_intervals(snapshots, source_url, retrieved_at):
    """Convert observed constituent sets without extending past source coverage."""
    frame = snapshots.copy()
    frame["date"] = pd.to_datetime(frame.date)
    frame = frame.sort_values("date")
    if frame.empty or frame.date.duplicated().any():
        duplicates = frame.loc[frame.date.duplicated(False), "date"].dt.strftime("%Y-%m-%d").unique().tolist()
        raise ValueError(f"Empty or duplicate constituent snapshots: {duplicates}")
    active, rows = {}, []
    final = frame.date.iloc[-1]
    for row in frame.itertuples():
        names = {name.strip().upper().replace(".", "-") for name in row.tickers.split(",") if name.strip()}
        for ticker in sorted(set(active) - names):
            rows.append({"ticker": ticker, "effective_from": active.pop(ticker), "effective_to": row.date - pd.Timedelta(days=1), "status": "removed"})
        for ticker in names - set(active):
            active[ticker] = row.date
    rows.extend({"ticker": ticker, "effective_from": start, "effective_to": final, "status": "active_at_source_cutoff"} for ticker, start in active.items())
    result = pd.DataFrame(rows).sort_values(["ticker", "effective_from"])
    result["source"] = "community historical reconstruction; pending independent verification"
    result["source_url"], result["retrieved_at"] = source_url, retrieved_at
    result["license"] = "repository MIT; underlying baseline provenance requires review"
    result["access_cost"] = "free"
    return result


def read_json(url, *, headers=None, params=None):
    response = requests.get(url, headers=headers, params=params, timeout=45)
    response.raise_for_status()
    return response.json()


def recover_paper(folder, headers):
    """Reconcile recovered arithmetic, while distinguishing inferred opening cash."""
    base = "https://paper-api.alpaca.markets"
    before = read_json(base + "/v2/account", headers=headers)
    rows = activity_pages(lambda token, size: read_json(base + "/v2/account/activities", headers=headers,
                          params={"direction": "desc", "page_size": size, **({"page_token": token} if token else {})}))
    positions = read_json(base + "/v2/positions", headers=headers)
    after = read_json(base + "/v2/account", headers=headers)
    # An account changing while fetched cannot provide one coherent snapshot.
    if before["cash"] != after["cash"] or before["equity"] != after["equity"]:
        raise ValueError("Account changed during retrieval; retry after activity settles")
    events = activity_events(rows)
    first = pd.to_datetime(events.timestamp, utc=True).min()
    history = read_json(base + "/v2/account/portfolio/history", headers=headers,
                        params={"date_start": before["created_at"][:10], "date_end": str(first.date()), "timeframe": "1D"})
    baseline = [(stamp, value) for stamp, value in zip(history["timestamp"], history["equity"])
                if pd.Timestamp(stamp, unit="s", tz="UTC") < first and value is not None and value > 0]
    if not baseline:
        raise ValueError("No pre-trade equity evidence; opening cash remains unknown")
    for name, value in [("account", after), ("positions", positions), ("activities", rows), ("initial_equity", history)]:
        atomic_write_json(value, folder / (name + ".json"))
    atomic_write_json({"cash": float(after["cash"]), "holdings": {r["symbol"]: float(r["qty"]) for r in positions},
                       "verified": True, "source": "paper-api.alpaca.markets/v2/account and /v2/positions",
                       "observed_at": datetime.now(timezone.utc).isoformat(),
                       "purpose": "opening_snapshot_for_future_interval_only"}, folder / "current_balance_snapshot.json")
    atomic_write_csv(events, folder / "events.csv")
    # Equity before the first recorded trade supports a flat-start hypothesis;
    # it is not mislabeled as an independent historical cash/position statement.
    result = replay_events(events, opening_cash=float(baseline[-1][1]), opening_holdings={},
                           expected_cash=float(after["cash"]), expected_holdings={r["symbol"]: float(r["qty"]) for r in positions})
    summary = {**result.metrics, "history_complete": True, "activity_counts": dict(Counter(row["activity_type"] for row in rows)),
               "opening_balance_status": "inferred_from_pretrade_equity_with_flat_start", "opening_balances_verified": False,
               "certified_for_freeze": False, "gaps": result.data_quality + [{"reason": "independent_opening_cash_and_positions_statement_required"}],
               "first_activity": str(first), "last_activity": str(pd.to_datetime(events.timestamp, utc=True).max())}
    atomic_write_json(summary, folder / "reconciliation.json")
    atomic_write_csv(result.events, folder / "ledger_events.csv")
    return summary


def recover_sources(folder):
    """Save original files and licenses; compare candidates before any promotion."""
    report = []
    for repo, filename in [("fja05680/sp500", "S&P 500 Historical Components & Changes (Updated).csv"),
                            ("hanshof/sp500_constituents", "sp_500_historical_components.csv")]:
        listing = read_json("https://api.github.com/repos/" + repo + "/contents")
        source = next(row for row in listing if row["name"] == filename)
        owner = repo.split("/")[0]
        response = requests.get(source["download_url"], timeout=45)
        response.raise_for_status()
        path = folder / (owner + "_membership.csv")
        path.write_bytes(response.content)
        for name in ("LICENSE", "README.md"):
            url = next(row["download_url"] for row in listing if row["name"] == name)
            text = requests.get(url, timeout=45)
            text.raise_for_status()
            (folder / (owner + "_" + name)).write_bytes(text.content)
        snapshots = pd.read_csv(path)
        failure = None
        try:
            intervals = candidate_intervals(snapshots, source["download_url"], datetime.now(timezone.utc).isoformat())
            atomic_write_csv(intervals, folder / (owner + "_candidate_intervals.csv"))
            count = int(intervals.loc[intervals.effective_to >= "2012-01-01", "ticker"].nunique())
        except ValueError as exc:
            failure, count = str(exc), None
        report.append({"source": repo, "source_url": source["download_url"], "sha256": hashlib.sha256(response.content).hexdigest(),
                       "source_cutoff": str(pd.to_datetime(snapshots.date).max().date()), "historical_tickers_since_2012": count,
                       "verified": False, "reason": failure or "community reconstruction requires corroboration; no extension beyond source cutoff"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--sources", action="store_true")
    parser.add_argument("--price-probes", nargs="*", default=[])
    args = parser.parse_args(argv)
    clock = datetime.now(timezone.utc)
    folder = Path("data/audit_recovery") / clock.strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    cfg = dotenv_values(".env")
    headers = {"APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY") or cfg.get("ALPACA_API_KEY"),
               "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY") or cfg.get("ALPACA_SECRET_KEY")}
    report = {"generated_at": clock.isoformat(), "private_evidence_directory": str(folder), "read_only": True, "freeze_started": False}
    for enabled, key, work in [(args.paper, "paper", lambda: recover_paper(folder, headers)),
                                (args.sources, "membership_sources", lambda: recover_sources(folder))]:
        if enabled:
            try:
                report[key] = work()
            except (requests.RequestException, ValueError, KeyError) as exc:
                report[key] = {"complete": False, "error": type(exc).__name__, "detail": str(exc)}
            atomic_write_json(report, folder / "recovery_report.json")
    report["price_probes"] = []
    for ticker in args.price_probes:
        params = {"start": "2012-01-01T00:00:00Z", "end": clock.date().isoformat() + "T00:00:00Z",
                  "timeframe": "1Day", "adjustment": "raw", "feed": "sip", "limit": 10000}
        try:
            value = read_json(f"https://data.alpaca.markets/v2/stocks/{ticker}/bars", headers=headers, params=params)
            atomic_write_json(value, folder / (ticker + "_raw_bars.json"))
            bars = value.get("bars") or []
            report["price_probes"].append({"ticker": ticker, "rows": len(bars), "first": bars[0]["t"] if bars else None,
                                           "last": bars[-1]["t"] if bars else None, "pagination_remaining": bool(value.get("next_page_token")),
                                           "verified_full_coverage": False})
        except requests.RequestException as exc:
            report["price_probes"].append({"ticker": ticker, "error": type(exc).__name__, "verified_full_coverage": False})
    atomic_write_json(report, folder / "recovery_report.json")
    # The summary stays local too: account balances are never auto-published.
    print(json.dumps({"report": str(folder / "recovery_report.json"), "paper_matches": report.get("paper", {}).get("reconciled"), "freeze_started": False}))


if __name__ == "__main__":
    main()
