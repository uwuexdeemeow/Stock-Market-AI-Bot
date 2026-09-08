"""Build an offline membership review dataset; never approve or replace inputs.

Each official addition/deletion verifies one boundary. It does not prove the
starting universe, uninterrupted membership, or a security's full price history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from audit_evidence_recovery import candidate_intervals
from safe_io import atomic_write_csv, atomic_write_json, atomic_write_text
from audit_gap_register import code_identity


def digest(path):
    """A fingerprint lets another researcher check which file was inspected."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_sets(path):
    """Keep conflicting same-day snapshots out of comparison, not out of history."""
    frame = pd.read_csv(path)
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if frame.empty or frame.date.isna().any() or frame.tickers.isna().any():
        raise ValueError('Empty or invalid source snapshots')
    frame['members'] = frame.tickers.map(lambda value: frozenset(
        name.strip().upper().replace('.', '-') for name in value.split(',') if name.strip()))
    if frame.members.map(len).eq(0).any():
        raise ValueError('Empty constituent set')
    ambiguous = frame.loc[frame.date.duplicated(False), 'date'].dt.strftime('%Y-%m-%d').unique().tolist()
    return frame.sort_values('date'), sorted(ambiguous)


def reviewed_events(facts):
    """Require explicit review and source attribution; reject conflicting claims."""
    events = facts.get('membership_events', [])
    ids, claims = set(), {}
    for event in events:
        if (event.get('reviewed') is not True or event.get('index') != 'SP500'
                or event.get('action') not in {'addition', 'deletion'}
                or not str(event.get('source_url', '')).startswith('https://')
                or not event.get('publisher') or not event.get('company')
                or not event.get('ticker') or not event.get('event_id')):
            raise ValueError('Unreviewed or unattributed membership fact')
        effective = pd.Timestamp(event['effective_session'])
        announced = pd.Timestamp(event['announcement_date'])
        if pd.isna(effective) or pd.isna(announced) or announced > effective:
            raise ValueError('Invalid announcement/effective date')
        if event['event_id'] in ids:
            raise ValueError('Duplicate event identity')
        ids.add(event['event_id'])
        # Opposite claims for the same boundary must be resolved, not ordered
        # conveniently according to whichever source happens to appear last.
        key = (event['ticker'], event['effective_session'])
        if key in claims and claims[key] != event['action']:
            raise ValueError('Conflicting membership claims')
        claims[key] = event['action']
    return events


def reconcile(candidate_path, comparison_path, facts_path, *, start, end):
    """Inventory every candidate interval and the exact evidence still missing."""
    first, first_duplicates = snapshot_sets(candidate_path)
    second, second_duplicates = snapshot_sets(comparison_path)
    if first_duplicates:
        raise ValueError('Candidate has ambiguous snapshot dates: ' + ', '.join(first_duplicates))
    facts = json.loads(Path(facts_path).read_text())
    events = reviewed_events(facts)
    # A hash of the fact collection tracks manual reviews independently from
    # downloaded snapshots. The source's cutoff is never extended to today.
    intervals = candidate_intervals(first[['date', 'tickers']], facts['candidate_source_url'],
                                    datetime.now(timezone.utc).isoformat())
    intervals = intervals.loc[(intervals.effective_from <= pd.Timestamp(end)) &
                              (intervals.effective_to >= pd.Timestamp(start))].copy()
    review = []
    for row in intervals.to_dict('records'):
        addition = [e['event_id'] for e in events if e['ticker'] == row['ticker'] and e['action'] == 'addition'
                    and pd.Timestamp(e['effective_session']) == row['effective_from']]
        # Source end dates are inclusive; a deletion before next day's open
        # corresponds to an interval ending on the preceding calendar day.
        deletion = [e['event_id'] for e in events if e['ticker'] == row['ticker'] and e['action'] == 'deletion'
                    and pd.Timestamp(e['effective_session']) - pd.Timedelta(days=1) == row['effective_to']]
        row.update({'addition_evidence_ids': ','.join(addition), 'deletion_evidence_ids': ','.join(deletion),
                    'identity_evidence_urls': ' | '.join(e['source_url'] for e in facts.get('identity_observations', [])
                                                       if e['ticker'] == row['ticker']),
                    'addition_boundary_supported': bool(addition), 'deletion_boundary_supported': bool(deletion),
                    'interval_verified': False, 'security_history_verified': False,
                    'next_action': 'Verify baseline/entry, all intervening changes, exit/cutoff and security identity with primary evidence.'})
        review.append(row)
    # Compare common observation dates only. Later snapshots cannot retroactively
    # establish the membership set of an earlier trading day.
    common = first.merge(second, on='date', suffixes=('_a', '_b'))
    common = common.loc[(common.date >= pd.Timestamp(start)) & (common.date <= pd.Timestamp(end)) &
                        ~common.date.dt.strftime('%Y-%m-%d').isin(second_duplicates)]
    disagreements, counts = [], Counter()
    for row in common.itertuples():
        left, right = sorted(row.members_a - row.members_b), sorted(row.members_b - row.members_a)
        if left or right:
            disagreements.append({'date': row.date.date().isoformat(), 'candidate_only': left, 'comparison_only': right})
            counts.update(left + right)
    checks = []
    for event in events:
        date = pd.Timestamp(event['effective_session'])
        before = first.loc[first.date < date]
        after = first.loc[first.date <= date]
        covered = not before.empty and not after.empty and date <= first.date.max()
        before_member = event['ticker'] in before.iloc[-1].members if covered else None
        after_member = event['ticker'] in after.iloc[-1].members if covered else None
        expected_after = event['action'] == 'addition'
        checks.append({**event, 'candidate_boundary_matches': covered and before_member != expected_after and after_member == expected_after,
                       'candidate_before': before_member, 'candidate_after': after_member,
                       'scope': 'single_boundary_only'})
    report = {'schema_version': 2, **code_identity(), 'generated_at': datetime.now(timezone.utc).isoformat(),
              'status': 'blocked_partial_primary_evidence', 'complete': False,
              'candidate_sha256': digest(candidate_path), 'comparison_sha256': digest(comparison_path),
              'facts_sha256': digest(facts_path), 'start': start, 'end': end,
              'candidate_cutoff': first.date.max().date().isoformat(),
              'candidate_symbols': int(intervals.ticker.nunique()), 'candidate_intervals': len(review),
              'comparison_duplicate_dates': second_duplicates, 'common_dates': len(common),
              'disagreement_dates': len(disagreements), 'disagreement_symbols': len(counts),
              'disagreements': disagreements, 'priority_symbols': counts.most_common(),
              'primary_boundary_checks': checks, 'identity_observations': facts.get('identity_observations', []),
              'verified_full_intervals': 0, 'production_inputs_changed': False,
              'blockers': ['independent_starting_membership_baseline', 'unresolved_source_disagreements',
                           'unverified_intervening_membership_events', 'source_cutoff_before_audit_end',
                           'full_security_identity_and_share_class_history'],
              'source_limitations': facts.get('source_limitations', [])}
    return pd.DataFrame(review), report


def review_transitions(queue, facts_path, *, start, end):
    """Compare local event facts with intervals; never certify an entire history."""
    facts = json.loads(Path(facts_path).read_text())
    if facts.get('scope') != 'review_only_not_executable_corporate_actions':
        raise ValueError('Transition facts must be review only')
    sources = facts.get('sources', [])
    source_ids = {source.get('id') for source in sources}
    if (not sources or None in source_ids or len(source_ids) != len(sources)
            or any(not source.get('url', '').startswith('https://')
                   or len(source.get('sha256', '')) != 64
                   or any(char not in '0123456789abcdef' for char in source.get('sha256', ''))
                   for source in sources)):
        raise ValueError('Invalid transition source identities')
    events, seen = [], set()
    lower, upper = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(lower) or pd.isna(upper) or lower > upper:
        raise ValueError('Invalid transition review dates')
    for original in facts.get('transitions', []):
        # Validate even out-of-window facts so changing the date range cannot
        # conceal a broken source reference or an unreviewed claim.
        event = dict(original)
        if (not event.get('id') or event['id'] in seen or event.get('reviewed') is not True
                or not event.get('source_ids') or not set(event['source_ids']) <= source_ids
                or any(not event.get(key) for key in ('identity', 'entitlements', 'membership', 'blockers', 'checks'))):
            raise ValueError('Invalid or unreviewed transition')
        seen.add(event['id'])
        legal, trading = pd.Timestamp(event['legal_date']), pd.Timestamp(event['first_trading_session'])
        if pd.isna(legal) or pd.isna(trading) or legal > trading:
            raise ValueError('Invalid transition dates')
        checks = []
        for claim in event['checks']:
            date = pd.Timestamp(claim['date'])
            if (pd.isna(date) or not claim.get('ticker')
                    or claim.get('kind') not in {'identity_break', 'expected_member', 'expected_absent'}):
                raise ValueError('Invalid transition check')
            if not lower <= date <= upper:
                continue
            rows = queue.loc[(queue.ticker == claim['ticker']) &
                             (pd.to_datetime(queue.effective_from) <= date) &
                             (pd.to_datetime(queue.effective_to) >= date)]
            covered = (not queue.empty and pd.to_datetime(queue.effective_from).min() <= date
                       <= pd.to_datetime(queue.effective_to).max())
            if not covered:
                result = 'candidate_coverage_unavailable'
            elif claim['kind'] == 'identity_break':
                crossing = rows.loc[pd.to_datetime(rows.effective_from) < date]
                result = 'unseparated_issuer_history' if not crossing.empty else 'full_identity_history_still_unverified'
            else:
                expected = claim['kind'] == 'expected_member'
                result = 'ticker_presence_matches_only' if bool(len(rows)) == expected else 'candidate_disagrees'
            # An exact ticker-presence match cannot prove that these shares
            # belong to the right issuer; keep both statements visible.
            checks.append({**claim, 'result': result, 'security_identity_verified': False,
                           'candidate_intervals': [
                               {'from': str(row.effective_from)[:10], 'to': str(row.effective_to)[:10]}
                               for row in rows.itertuples()]})
        if checks:
            events.append({**event, 'checks': checks, 'executable': False, 'full_history_verified': False})
    return {'schema_version': 2, **code_identity(), 'generated_at': datetime.now(timezone.utc).isoformat(),
            'status': 'blocked_partial_event_evidence', 'complete': False,
            'facts_sha256': digest(facts_path), 'sources': sources, 'transitions': events,
            'production_inputs_changed': False,
            'scope': 'Retrospective event review; source hashes identify archived documents, not complete history certification.'}


def main(argv=None):
    """Write a review queue and evidence report without touching production data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--facts', type=Path, default=Path('research_evidence/membership_primary_facts.json'))
    parser.add_argument('--transitions', type=Path, default=Path('research_evidence/security_transition_facts.json'))
    parser.add_argument('--output', type=Path, default=Path('signals/corrected_audit/membership_review'))
    parser.add_argument('--start', default='2012-01-01')
    parser.add_argument('--end', default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args(argv)
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        parser.error('Start must precede end')
    queue, report = reconcile(args.candidate, args.comparison, args.facts, start=args.start, end=args.end)
    # Event reviews describe supported corrections without rewriting the input
    # universe or inventing the cash actually paid to an account.
    transitions = review_transitions(queue, args.transitions, start=args.start, end=args.end)
    report['security_transitions'] = transitions
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(queue, args.output / 'membership_review_queue.csv')
    atomic_write_json(report, args.output / 'membership_identity_report.json')
    atomic_write_json(report['primary_boundary_checks'], args.output / 'primary_membership_events.json')
    atomic_write_json(transitions, args.output / 'security_transition_report.json')
    lines = ['# Historical membership and identity review', '',
             '**Partial primary evidence; full dataset remains blocked.**', '',
             f"Candidate: {report['candidate_symbols']} symbols, {report['candidate_intervals']} intervals; cutoff {report['candidate_cutoff']}.",
             f"Disagreement: {report['disagreement_dates']} of {report['common_dates']} comparable dates across {report['disagreement_symbols']} symbols.",
             'A confirmed boundary does not verify an entire interval or a raw price history.', '',
             '| Symbol | Action | Effective session | Candidate matches | Primary source |', '| --- | --- | --- | --- | --- |']
    for row in report['primary_boundary_checks']:
        lines.append(f"| {row['ticker']} | {row['action']} | {row['effective_session']} | {row['candidate_boundary_matches']} | [Announcement]({row['source_url']}) |")
    lines += ['', '## Identity observations', '']
    for row in report['identity_observations']:
        lines.append(f"- {row['ticker']}: {row['finding']} [Source]({row['source_url']}).")
    lines += ['', '## Remaining evidence', ''] + ['- ' + item.replace('_', ' ') for item in report['blockers']]
    lines += ['', '## Security transitions', '']
    for event in transitions['transitions']:
        lines += [f"### {event['id']}", '', event['identity'], '', event['entitlements'], '', event['membership'], '']
        for check in event['checks']:
            lines.append(f"- {check['ticker']} {check['date']} ({check['kind']}): {check['result']}.")
        lines += ['- Remaining: ' + item for item in event['blockers']]
        lines += ['- [Primary source](' + source['url'] + ')' for source in transitions['sources'] if source['id'] in event['source_ids']]
    lines += ['', 'No production membership, price manifest, strategy settings, orders or freeze were changed.']
    atomic_write_text(args.output / 'membership_identity_report.md', '\n'.join(lines) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'candidate_symbols', 'candidate_intervals', 'disagreement_dates', 'verified_full_intervals')}))


if __name__ == '__main__':
    main()
