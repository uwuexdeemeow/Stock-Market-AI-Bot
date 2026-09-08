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


def transition_queue():
    """These intervals deliberately reproduce continuous ticker reuse."""
    return pd.DataFrame([
        {'ticker': ticker, 'effective_from': begin, 'effective_to': '2026-08-18'}
        for ticker, begin in [('FOXA', '2004-12-20'), ('FOX', '2015-09-21'),
                              ('IR', '2010-11-17'), ('TT', '2020-03-03')]])


def test_transition_review_separates_issuer_and_membership_dates():
    from membership_reconciliation import review_transitions
    report = review_transitions(transition_queue(), 'research_evidence/security_transition_facts.json',
                                start='2012-01-01', end='2026-09-08')
    fox, ir = report['transitions']
    assert all(c['result'] == 'unseparated_issuer_history' for c in fox['checks'][:2])
    assert all(c['result'] == 'candidate_disagrees' for c in fox['checks'][2:])
    checks = {(c['ticker'], c['date']): c for c in ir['checks']}
    assert checks['TT', '2020-03-02']['result'] == 'candidate_disagrees'
    assert checks['IR', '2020-03-03']['result'] == 'ticker_presence_matches_only'
    assert not checks['IR', '2020-03-03']['security_identity_verified']
    assert not report['complete'] and not report['production_inputs_changed']
    assert all(not e['executable'] and not e['full_history_verified'] for e in report['transitions'])


def test_transition_review_does_not_infer_events_outside_window():
    from membership_reconciliation import review_transitions
    report = review_transitions(transition_queue(), 'research_evidence/security_transition_facts.json',
                                start='2021-01-01', end='2026-09-08')
    assert not report['transitions']
    assert not report['complete']


@pytest.mark.parametrize('damage', ['review', 'source', 'duplicate', 'date', 'kind'])
def test_invalid_transition_evidence_rejected(tmp_path, damage):
    from pathlib import Path
    from membership_reconciliation import review_transitions
    facts = json.loads(Path('research_evidence/security_transition_facts.json').read_text())
    event = facts['transitions'][0]
    if damage == 'review':
        event['reviewed'] = False
    elif damage == 'source':
        event['source_ids'] = ['missing']
    elif damage == 'duplicate':
        facts['transitions'].append(dict(event))
    elif damage == 'date':
        event['legal_date'] = '2020-01-01'
    else:
        event['checks'][0]['kind'] = 'approve'
    path = tmp_path / 'facts.json'
    path.write_text(json.dumps(facts))
    with pytest.raises(ValueError):
        review_transitions(transition_queue(), path, start='2021-01-01', end='2026-09-08')


def test_absence_outside_candidate_coverage_is_not_a_match():
    from membership_reconciliation import review_transitions
    queue = transition_queue()
    queue['effective_to'] = '2018-01-01'
    report = review_transitions(queue, 'research_evidence/security_transition_facts.json',
                                start='2012-01-01', end='2026-09-08')
    assert all(c['result'] == 'candidate_coverage_unavailable'
               for e in report['transitions'] for c in e['checks'])
