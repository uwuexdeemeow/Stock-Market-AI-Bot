"""Boundary evidence must never promote incomplete membership or price history."""
import json

import pandas as pd
import pytest

from membership_reconciliation import reconcile, reviewed_events


def inputs(tmp_path):
    candidate = tmp_path / 'candidate.csv'
    comparison = tmp_path / 'comparison.csv'
    pd.DataFrame({'date': ['2024-01-02', '2024-02-01', '2024-03-01'],
                  'tickers': ['A,B', 'B,C', 'A,C']}).to_csv(candidate, index=False)
    pd.DataFrame({'date': ['2024-01-02', '2024-02-01', '2024-02-01', '2024-03-01'],
                  'tickers': ['A,B', 'B,C', 'A,C', 'A,C']}).to_csv(comparison, index=False)
    fact = {'event_id': 'add-c', 'ticker': 'C', 'company': 'Company C', 'index': 'SP500',
            'action': 'addition', 'announcement_date': '2024-01-20', 'effective_session': '2024-02-01',
            'reviewed': True, 'publisher': 'Fixture', 'source_url': 'https://example.test'}
    facts = tmp_path / 'facts.json'
    facts.write_text(json.dumps({'candidate_source_url': 'https://example.test/candidate', 'membership_events': [fact]}))
    return candidate, comparison, facts


def test_boundary_match_does_not_verify_whole_interval(tmp_path):
    queue, report = reconcile(*inputs(tmp_path), start='2024-01-01', end='2024-04-01')
    assert report['primary_boundary_checks'][0]['candidate_boundary_matches']
    assert queue.loc[queue.ticker == 'C', 'addition_boundary_supported'].all()
    assert not queue.interval_verified.any() and not queue.security_history_verified.any()
    assert not report['complete'] and report['verified_full_intervals'] == 0
    assert report['comparison_duplicate_dates'] == ['2024-02-01']
    assert report['common_dates'] == 2
    assert len(queue.loc[queue.ticker == 'A']) == 2
    assert queue.effective_to.max() == pd.Timestamp('2024-03-01')


def test_future_snapshot_cannot_supply_earlier_boundary(tmp_path):
    candidate, comparison, facts = inputs(tmp_path)
    body = json.loads(facts.read_text())
    body['membership_events'][0]['effective_session'] = '2024-01-25'
    facts.write_text(json.dumps(body))
    _, report = reconcile(candidate, comparison, facts, start='2024-01-01', end='2024-04-01')
    assert not report['primary_boundary_checks'][0]['candidate_boundary_matches']


def test_conflicting_or_unreviewed_facts_rejected(tmp_path):
    _, _, path = inputs(tmp_path)
    facts = json.loads(path.read_text())
    facts['membership_events'][0]['reviewed'] = False
    with pytest.raises(ValueError, match='Unreviewed'):
        reviewed_events(facts)
    event = facts['membership_events'][0]
    event['reviewed'] = True
    facts['membership_events'].append({**event, 'event_id': 'delete-c', 'action': 'deletion'})
    with pytest.raises(ValueError, match='Conflicting'):
        reviewed_events(facts)


def test_curated_primary_facts_have_unique_reviewed_claims():
    from pathlib import Path
    facts = json.loads(Path('research_evidence/membership_primary_facts.json').read_text())
    assert len(reviewed_events(facts)) == 26
    assert all(e['scope'] == 'membership_boundary_only' for e in facts['membership_events'])
