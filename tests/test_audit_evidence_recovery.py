"""Offline recovery regressions: real formats, fake HTTP, no broker orders."""
import pandas as pd
import pytest

from audit_evidence_recovery import activity_pages, activity_events, candidate_intervals
from corrected_audit import replay_certified
from portfolio_ledger import Account, replay_events, LEDGER_VERSION


def test_activity_paging_recovers_244_rows():
    rows = [{"id": str(i)} for i in range(244)]
    def fetch(token, size):
        start = int(token) + 1 if token is not None else 0
        return rows[start:start + size]
    assert activity_pages(fetch) == rows


def test_activity_duplicate_or_interrupted_page_cannot_pass():
    with pytest.raises(ValueError, match="repeated"):
        activity_pages(lambda *_: [{"id": "one"}], page_size=1)
    def failed(*_):
        raise RuntimeError("retrieval interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        activity_pages(failed)


def test_separate_broker_fee_charged_once_and_margin_replayed():
    rows = [{"activity_type": "FILL", "id": "one", "order_id": "a", "transaction_time": "2026-05-08T14:00:00Z",
             "symbol": "SPY", "side": "buy", "qty": "2", "price": "100"},
            {"activity_type": "FILL", "id": "two", "order_id": "b", "transaction_time": "2026-05-08T15:00:00Z",
             "symbol": "SPY", "side": "sell", "qty": "2", "price": "110"},
            {"activity_type": "FEE", "id": "fee", "created_at": "2026-05-09T00:00:00Z", "net_amount": "-0.03"}]
    events = activity_events(rows)
    result = replay_events(events, opening_cash=100, opening_holdings={}, expected_cash=119.97, expected_holdings={})
    assert result.metrics["reconciled"]
    assert result.metrics["historical_margin_observed"]
    assert result.metrics["minimum_recorded_cash"] == -100
    with pytest.raises(ValueError, match="Duplicate"):
        replay_events(pd.concat([events, events.iloc[[-1]]]), opening_cash=100, opening_holdings={})
    with pytest.raises(ValueError, match="conservation"):
        Account(100).fill("2026-05-08", "SPY", 2, 100, 0, event_id="one", source="simulation")


def test_unknown_account_activity_is_not_silently_ignored():
    with pytest.raises(ValueError, match="Unmapped"):
        activity_events([{"activity_type": "SSP"}])


def test_membership_intervals_do_not_extend_source_or_erase_reentry():
    source = pd.DataFrame({"date": ["2024-01-02", "2024-02-01", "2024-03-01"], "tickers": ["A,B", "B,C", "A,C"]})
    result = candidate_intervals(source, "https://example.test/source", "2024-03-02T00:00:00Z")
    assert len(result.query("ticker == 'A'")) == 2
    assert result.effective_to.max() == pd.Timestamp("2024-03-01")
    assert result.query("ticker == 'B'").effective_to.iloc[0] == pd.Timestamp("2024-02-29")


def test_conflicting_membership_dates_block_import():
    source = pd.DataFrame({"date": ["2024-01-02"] * 2, "tickers": ["A,B", "A,C"]})
    with pytest.raises(ValueError, match="2024-01-02"):
        candidate_intervals(source, "https://example.test/source", "2024-03-02T00:00:00Z")


def test_inferred_opening_balances_cannot_certify_freeze():
    report = {"ledger_version": LEDGER_VERSION, "reconciled": True, "source_history_complete": True,
              "source_closing_balances_verified": True, "source_opening_balances_verified": False}
    assert not replay_certified(report)
    report["source_opening_balances_verified"] = True
    assert replay_certified(report)
