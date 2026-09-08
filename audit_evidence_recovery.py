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
from decimal import Decimal
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
        elif kind in {"FEE", "CSD", "CSW", "DIV", "CGD", "DIVCGL", "DIVCGS", "DIVFEE",
                      "DIVFT", "DIVNRA", "DIVROC", "DIVTW", "DIVTXEX", "INT", "INTNRA", "INTTW", "ACATC", "JNLC"}:
            # These documented activity types move cash only. Stock transfers,
            # mergers and reorganizations require separately verified share legs.
            output.append({"kind": "cash_adjustment", "event_id": row["id"],
                           "timestamp": row.get("created_at") or row["date"] + "T23:59:59Z",
                           "amount": float(row["net_amount"]), "source": "alpaca_" + kind.lower() + "_activity"})
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


def saved_broker_page(folder, name, url, headers, params):
    """Archive each received broker page before later parsing can fail."""
    page = read_json(url, headers=headers, params=params)
    suffix = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    atomic_write_json(page, folder / f'{name}_page_{suffix}.json')
    return page


def market_pages(get_page, field, *, maximum_pages=1000):
    """Keep following market-data tokens, including a short page with a token."""
    rows, tokens, identities = [], set(), set()
    token = None
    for _ in range(maximum_pages):
        page = get_page(token)
        batch = page.get(field)
        # Corporate actions are grouped by type; retain that type on every row.
        if field == "corporate_actions" and isinstance(batch, dict):
            batch = [{**row, "source_action_type": kind} for kind, values in batch.items() for row in values]
        if batch is None and field == "bars":
            batch = []
        if not isinstance(batch, list):
            raise ValueError("Malformed market-data page")
        for row in batch:
            identity = row.get("id") if field == "corporate_actions" else row.get("t")
            if not identity or identity in identities:
                raise ValueError("Duplicate or missing market-data identity")
            identities.add(identity)
            rows.append(row)
        token = page.get("next_page_token")
        if not token:
            return rows
        if token in tokens:
            raise ValueError("Repeated market-data pagination token")
        tokens.add(token)
    raise ValueError("Market-data pagination limit reached")


def saved_market_page(folder, name, url, headers, params, token):
    """Preserve each received page even if a later page fails validation."""
    page = read_json(url, headers=headers,
                     params={**params, **({"page_token": token} if token else {})})
    # A token hash gives each private page a stable filename without exposing
    # request headers or relying on provider token characters as a file path.
    suffix = hashlib.sha256(str(token).encode()).hexdigest()[:16]
    atomic_write_json(page, folder / f"{name}_page_{suffix}.json")
    return page


def recover_actions(folder, headers, symbols, end):
    """Recover original action records without inventing missing payment dates."""
    url = "https://data.alpaca.markets/v1/corporate-actions"
    params = {"symbols": ",".join(symbols), "start": "2012-01-01", "end": end,
              "limit": 1000, "sort": "asc"}
    rows = market_pages(lambda token: saved_market_page(folder, "actions", url, headers, params, token), "corporate_actions")
    path = folder / "corporate_actions.json"
    atomic_write_json(rows, path)
    facts_path = Path(__file__).with_name('research_evidence') / 'dividend_payment_facts.json'
    date_review = reconcile_action_dates(rows, json.loads(facts_path.read_text()).get('facts', []))
    atomic_write_json(date_review, folder / 'dividend_payment_review.json')
    # API success proves retrieval, not historical completeness. Payment dates
    # and unsupported event types remain explicit gaps for the ledger importer.
    dividends = [r for r in rows if r["source_action_type"] == "cash_dividends"]
    return {"source_url": url, "parameters": params, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
            "activity_counts": dict(Counter(r["source_action_type"] for r in rows)),
            "missing_dividend_payment_dates": sum(not r.get("payable_date") for r in dividends),
            "independently_supported_payment_dates": date_review['supported_missing_dates'],
            "remaining_missing_payment_dates": date_review['remaining_missing_dates'],
            "earliest_ex_date": min((r["ex_date"] for r in dividends if r.get("ex_date")), default=None),
            "pagination_complete": True, "verified_full_coverage": False,
            "next_action": "Corroborate per-symbol coverage, missing payment dates and event semantics with issuer evidence."}


def reconcile_action_dates(rows, facts):
    """Match issuer dates by security, ex-date and exact amount, preserving raw rows."""
    lookup = {}
    for fact in facts:
        key = (fact.get('symbol'), fact.get('cusip'), fact.get('ex_date'))
        payable = pd.to_datetime(fact.get('payable_date'), errors='coerce', utc=True)
        ex_date = pd.to_datetime(fact.get('ex_date'), errors='coerce', utc=True)
        if (key in lookup or fact.get('reviewed') is not True or not all(key)
                or not str(fact.get('source_url', '')).startswith('https://')
                or len(fact.get('source_sha256', '')) != 64
                or pd.isna(payable) or pd.isna(ex_date) or payable < ex_date):
            raise ValueError('Invalid dividend primary evidence')
        lookup[key] = fact
    results = []
    for row in rows:
        if row.get('source_action_type') != 'cash_dividends' or row.get('payable_date'):
            continue
        fact = lookup.get((row.get('symbol'), row.get('cusip'), row.get('ex_date')))
        matched = fact is not None and Decimal(str(row['rate'])) == Decimal(str(fact['rate']))
        results.append({'ticker': row['symbol'], 'ex_date': row['ex_date'],
                        'status': 'payment_date_supported' if matched else 'payment_date_unverified',
                        'payable_date': fact['payable_date'] if matched else None,
                        'source_url': fact['source_url'] if matched else None,
                        'source_sha256': fact['source_sha256'] if matched else None})
    supported = sum(row['status'] == 'payment_date_supported' for row in results)
    return {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'status': 'partial_primary_evidence', 'complete': False,
            'supported_missing_dates': supported, 'remaining_missing_dates': len(results) - supported,
            'raw_source_modified': False, 'full_action_coverage_verified': False, 'results': results}


def recover_paper(folder, headers):
    """Reconcile recovered arithmetic, while distinguishing inferred opening cash."""
    base = "https://paper-api.alpaca.markets"
    before = read_json(base + "/v2/account", headers=headers)
    rows = activity_pages(lambda token, size: saved_broker_page(folder, 'activities', base + "/v2/account/activities", headers,
                          {"direction": "desc", "page_size": size, **({"page_token": token} if token else {})}))
    positions = read_json(base + "/v2/positions", headers=headers)
    after = read_json(base + "/v2/account", headers=headers)
    # An account changing while fetched cannot provide one coherent snapshot.
    if before["cash"] != after["cash"] or before["equity"] != after["equity"]:
        raise ValueError("Account changed during retrieval; retry after activity settles")
    for name, value in (("account", after), ("positions", positions), ("activities", rows)):
        atomic_write_json(value, folder / (name + '.json'))
    events = activity_events(rows)
    if events.empty:
        raise ValueError('No historical activities; independent opening statement required')
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


def recover_verified_interval(folder, headers, opening_path, closing_path=None):
    """Replay independently documented balances over their exact time interval.

    A current API balance can close an interval, but cannot certify an inferred
    historical opening. Raw account data stays in the private recovery folder.
    """
    from broker_history import collect_order_history
    from corrected_audit import replay_certified
    base = "https://paper-api.alpaca.markets"
    opening = json.loads(Path(opening_path).read_text(encoding="utf-8"))
    if opening.get("verified") is not True or not opening.get("source") or not opening.get("observed_at"):
        raise ValueError("Independent attributed opening balance and timestamp required")
    start = pd.to_datetime(opening["observed_at"], utc=True, errors="raise")
    before = read_json(base + "/v2/account", headers=headers)
    positions_before = read_json(base + "/v2/positions", headers=headers)
    rows = activity_pages(lambda token, size: saved_broker_page(folder, 'activities', base + "/v2/account/activities", headers,
        {"direction": "asc", "page_size": size, **({"page_token": token} if token else {})}))
    positions_after = read_json(base + "/v2/positions", headers=headers)
    after = read_json(base + "/v2/account", headers=headers)
    holdings = lambda values: {r["symbol"]: float(r["qty"]) for r in values}
    if before["cash"] != after["cash"] or holdings(positions_before) != holdings(positions_after):
        raise ValueError("Account changed during interval retrieval")
    observed = datetime.now(timezone.utc).isoformat()
    current = {"cash": float(after["cash"]), "holdings": holdings(positions_after), "verified": True,
               "source": base + "/v2/account and /v2/positions", "observed_at": observed}
    closing = json.loads(Path(closing_path).read_text(encoding="utf-8")) if closing_path else current
    if closing.get("verified") is not True or not closing.get("source") or not closing.get("observed_at"):
        raise ValueError("Independent attributed closing balance and timestamp required")
    end = pd.to_datetime(closing["observed_at"], utc=True, errors="raise")
    if not start < end <= pd.Timestamp(observed):
        raise ValueError("Balance interval must be ordered and end no later than retrieval")
    # Save original evidence before interpretation; an unsupported cash action
    # must remain visible rather than being dropped to obtain a reconciliation.
    for name, value in (("activities", rows), ("opening_balances", opening), ("closing_balances", closing), ("current_balance_snapshot", current)):
        atomic_write_json(value, folder / (name + ".json"))
    # Date-only cash entries cannot be placed accurately inside an intraday
    # boundary. Require a timestamp instead of guessing a convenient ordering.
    selected = []
    for row in rows:
        stamp = row.get("transaction_time") or row.get("created_at")
        if not stamp:
            day = pd.Timestamp(row["date"], tz="UTC")
            if day <= end and day + pd.Timedelta(days=1) > start:
                raise ValueError("Activity timestamp missing inside balance interval")
            continue
        if start < pd.to_datetime(stamp, utc=True) <= end:
            selected.append(row)
    events = activity_events(selected)
    if events.empty:
        events = pd.DataFrame(columns=["kind", "event_id", "timestamp"])
    class ReadOnlyOrders:
        def list_orders(self, **params):
            return saved_broker_page(folder, 'orders', base + "/v2/orders", headers, params)
    # Include older parent orders: a fill inside this interval may belong to
    # an order submitted before the opening snapshot.
    history_start = pd.to_datetime(before.get("created_at", start), utc=True)
    order_history = collect_order_history(ReadOnlyOrders(), after=history_start, until=end)
    known_orders = {str(row.get("id", "")) for row in order_history.orders}
    fill_orders = {str(row["order_id"]) for row in selected if row.get("activity_type") == "FILL"}
    if not fill_orders.issubset(known_orders):
        order_history.complete = False
        order_history.errors.append("fill_parent_order_missing")
    atomic_write_json(order_history.orders, folder / "orders.json")
    result = replay_events(events, opening_cash=opening.get("cash"), opening_holdings=opening.get("holdings"),
                           expected_cash=closing.get("cash"), expected_holdings=closing.get("holdings"))
    summary = {**result.metrics, "source_history_complete": order_history.complete,
               "source_opening_balances_verified": True, "source_closing_balances_verified": True,
               "gaps": result.data_quality + [{"reason": e.split(":")[0]} for e in order_history.errors],
               "interval_start": start.isoformat(), "interval_end": end.isoformat(),
               "activity_stream_complete": True, "arrival_quotes_available": False}
    summary["activity_counts"] = {kind: sum(row.get("activity_type") == kind for row in selected) for kind in ("FILL", "FEE")}
    summary["evidence_scope"] = "recorded_activity_interval" if selected else "balance_continuity_only_no_activity"
    summary["certified_for_freeze"] = replay_certified(summary)
    atomic_write_json(summary, folder / "replay_reconciliation.json")
    atomic_write_json({"history_complete": order_history.complete, "after": start.isoformat(), "until": end.isoformat()}, folder / "broker_history_report.json")
    atomic_write_csv(events, folder / "events.csv")
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
    parser.add_argument("--opening-balances", type=Path, help="Verified opening cash/positions with observed_at and source")
    parser.add_argument("--closing-balances", type=Path, help="Optional verified closing snapshot; otherwise use current paper API")
    parser.add_argument("--price-probes", nargs="*", default=[])
    parser.add_argument("--action-probes", nargs="*", default=[], help="Read-only paginated corporate-action candidates")
    args = parser.parse_args(argv)
    if (args.opening_balances or args.closing_balances) and not args.paper:
        parser.error("Balance inputs require --paper")
    if args.closing_balances and not args.opening_balances:
        parser.error("Closing balances require independently verified opening balances")
    clock = datetime.now(timezone.utc)
    folder = Path("data/audit_recovery") / clock.strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    cfg = dotenv_values(".env")
    headers = {"APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY") or cfg.get("ALPACA_API_KEY"),
               "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY") or cfg.get("ALPACA_SECRET_KEY")}
    report = {"generated_at": clock.isoformat(), "private_evidence_directory": str(folder), "read_only": True, "freeze_started": False}
    for enabled, key, work in [(args.paper, "paper", lambda: recover_verified_interval(folder, headers, args.opening_balances, args.closing_balances) if args.opening_balances else recover_paper(folder, headers)),
                                (args.sources, "membership_sources", lambda: recover_sources(folder)),
                                (args.action_probes, "corporate_actions", lambda: recover_actions(folder, headers, args.action_probes, clock.date().isoformat()))]:
        if enabled:
            try:
                report[key] = work()
            except (requests.RequestException, ValueError, KeyError) as exc:
                report[key] = {"complete": False, "error": type(exc).__name__, "detail": str(exc)}
            atomic_write_json(report, folder / "recovery_report.json")
    report["price_probes"] = []
    for ticker in args.price_probes:
        params = {"start": "2012-01-01T00:00:00Z", "end": clock.date().isoformat() + "T00:00:00Z",
                  "timeframe": "1Day", "adjustment": "raw", "feed": "sip", "asof": "-", "limit": 10000}
        try:
            url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
            bars = market_pages(lambda token: saved_market_page(folder, ticker, url, headers, params, token), "bars")
            value = {"bars": bars, "next_page_token": None, "symbol": ticker}
            atomic_write_json(value, folder / (ticker + "_raw_bars.json"))
            report["price_probes"].append({"ticker": ticker, "rows": len(bars), "first": bars[0]["t"] if bars else None,
                                           "last": bars[-1]["t"] if bars else None, "pagination_remaining": bool(value.get("next_page_token")),
                                           "source_url": url, "parameters": params, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                           "sha256": hashlib.sha256((folder / (ticker + "_raw_bars.json")).read_bytes()).hexdigest(),
                                           "verified_full_coverage": False})
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            report["price_probes"].append({"ticker": ticker, "error": type(exc).__name__, "verified_full_coverage": False})
    atomic_write_json(report, folder / "recovery_report.json")
    # The summary stays local too: account balances are never auto-published.
    print(json.dumps({"report": str(folder / "recovery_report.json"), "paper_matches": report.get("paper", {}).get("reconciled"), "freeze_started": False}))


if __name__ == "__main__":
    main()
