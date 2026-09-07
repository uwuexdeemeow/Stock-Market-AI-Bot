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


def test_verified_interval_partial_fills_fees_and_independent_balances(tmp_path, monkeypatch):
    import json
    import audit_evidence_recovery as recovery
    opening = tmp_path / 'opening.json'
    opening.write_text(json.dumps({'cash': 1000., 'holdings': {}, 'verified': True,
                                  'source': 'independent statement', 'observed_at': '2024-02-01T00:00:00Z'}))
    closing = tmp_path / 'closing.json'
    closing.write_text(json.dumps({'cash': 698.97, 'holdings': {'SPY': 3.}, 'verified': True,
                                  'source': 'independent statement', 'observed_at': '2024-02-03T00:00:00Z'}))
    rows = [{'activity_type': 'FILL', 'id': 'a', 'order_id': 'one', 'transaction_time': '2024-02-02T14:35:00Z',
             'symbol': 'SPY', 'side': 'buy', 'qty': '2', 'price': '100'},
            {'activity_type': 'FILL', 'id': 'b', 'order_id': 'one', 'transaction_time': '2024-02-02T14:36:00Z',
             'symbol': 'SPY', 'side': 'buy', 'qty': '1', 'price': '101'},
            {'activity_type': 'FEE', 'id': 'fee', 'created_at': '2024-02-02T21:00:00Z', 'net_amount': '-0.03'}]
    def get(url, **kwargs):
        if url.endswith('/activities'):
            return rows
        if url.endswith('/positions'):
            return [{'symbol': 'SPY', 'qty': '3'}]
        if url.endswith('/orders'):
            return [{'id': 'one', 'submitted_at': '2024-02-02T14:34:00Z'}]
        return {'cash': '698.97'}
    monkeypatch.setattr(recovery, 'read_json', get)
    report = recovery.recover_verified_interval(tmp_path, {}, opening, closing)
    assert report['certified_for_freeze']
    assert report['cash'] == pytest.approx(698.97)
    assert not report['arrival_quotes_available']
    # The very same arithmetic is not enough if the source order query fails.
    def incomplete(url, **kwargs):
        if url.endswith('/orders'):
            raise RuntimeError('interrupted')
        return get(url, **kwargs)
    monkeypatch.setattr(recovery, 'read_json', incomplete)
    assert not recovery.recover_verified_interval(tmp_path, {}, opening, closing)['certified_for_freeze']


def test_inferred_balance_input_is_rejected_before_network(tmp_path, monkeypatch):
    import audit_evidence_recovery as recovery
    opening = tmp_path / 'opening.json'
    opening.write_text('{"cash":1000,"holdings":{},"verified":false}')
    monkeypatch.setattr(recovery, 'read_json', lambda *a, **k: pytest.fail('should not fetch'))
    with pytest.raises(ValueError, match='Independent'):
        recovery.recover_verified_interval(tmp_path, {}, opening)
