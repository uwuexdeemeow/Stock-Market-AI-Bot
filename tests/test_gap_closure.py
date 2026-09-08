"""Prevent diagnostic improvements from manufacturing evidence approval."""
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from audit_gap_register import update_register, lock_changes
from evidence_audit import api_pages, snapshot_summary, REQUIRED
from order_accounting import stage_attempt_counts
from paper_health import _open_position_attribution


def test_original_findings_survive_disappearance():
    baseline = {'generated_at': 'old', 'findings': [{'reason': 'missing_balance'}]}
    report = {'generated_at': 'new', 'audit_code_commit': 'abc', 'blockers': []}
    first = update_register(report, baseline)
    second = update_register(report, baseline, first)
    assert first == second
    row = second['findings'][0]
    assert row['id'] == 'original-001' and row['status'] == 'blocked'
    assert not row['observed_in_current_audit'] and not row['closure_evidence']


def test_paginate_and_reject_repeated_ids(monkeypatch):
    import evidence_audit
    def page(root, *args):
        return json.dumps({'items': [{'id': n} for n in range(100)] if args[-1].endswith('&page=1')
                           else [{'id': 100}]}).encode()
    monkeypatch.setattr(evidence_audit, 'command', page)
    assert len(api_pages('.', '/test', 'items')) == 101
    monkeypatch.setattr(evidence_audit, 'command', lambda *a: json.dumps({'items': [{'id': 1}] * 100}).encode())
    with pytest.raises(ValueError, match='repeated'):
        api_pages('.', '/test', 'items')


def test_execution_manifest_does_not_borrow_daily_identity():
    files = {name: json.dumps({'run_id': 'observation'}).encode() for name in REQUIRED}
    manifest = {'profile': 'execution_observation_only', 'run_id': 'observation', 'status': 'complete',
                'generated_at': '2026-09-08T00:00:00Z',
                'files': {name: {'required': True, 'sha256': hashlib.sha256(value).hexdigest()} for name, value in files.items()}}
    files['execution_run_manifest.json'] = json.dumps(manifest).encode()
    now = datetime(2026, 9, 8, 1, tzinfo=timezone.utc)
    assert snapshot_summary(files, source='test', now=now)['complete']
    files[REQUIRED[0]] = b'{"run_id":"daily"}'
    assert not snapshot_summary(files, source='test', now=now)['complete']


def test_malformed_manifest_remains_deliverable():
    result = snapshot_summary({'paper_run_manifest.json': b'{"files":{"x":null}}'}, source='test')
    assert 'manifest_records_malformed' in result['issues']


def test_remaining_basis_uses_fractional_current_holdings_not_last_buy():
    status = {'broker': 'alpaca', 'generated_at': '2026-09-08T00:00:00Z', 'positions': {'A': 1.5},
              'position_details': [{'ticker': 'A', 'quantity': 1.5, 'avg_price': 100, 'market_value': 180}]}
    # The last fill is deliberately unrelated to remaining average basis.
    trades = pd.DataFrame([{'action': 'BUY', 'broker_dealt_avg_price': 200, 'fill_status': 'filled'}])
    result = _open_position_attribution(status, trades)
    assert result['total_open_position_pnl'] == 30
    assert not result['historical_accounting_certified']
    status['positions']['A'] = 2
    assert not _open_position_attribution(status, trades)['data_available']


def test_no_basis_does_not_fall_back_to_incomplete_journal():
    assert not _open_position_attribution({'positions': {'A': 1}}, pd.DataFrame())['data_available']


def test_stage_denominator_includes_zero_and_partial_fills():
    rows = [{'id': str(i), 'client_order_id': f'parent{i}-a1', 'submitted_at': '2026-09-07T14:00:00Z',
             'qty': 10, 'filled_qty': q, 'status': 'canceled'} for i, q in enumerate([0, 4, 10])]
    options = {'start': '2026-09-07T00:00:00Z', 'end': '2026-09-08T00:00:00Z', 'history_complete': True}
    stage = stage_attempt_counts(rows, **options)['stages']['stage1']
    assert stage['orders'] == 3 and stage['filled_orders'] == 1
    assert stage['any_fill_rate'] == 2 / 3 and stage['fill_rate'] == 1 / 3
    assert not stage_attempt_counts(rows + rows[:1], **options)['complete']
    options['history_complete'] = False
    assert not stage_attempt_counts(rows, **options)['complete']


def test_lock_diff_does_not_change_frozen_bytes(tmp_path):
    (tmp_path / 'x.py').write_text('new')
    lock = {'files': {'x.py': 'old'}, 'logic_fingerprint': 'old'}
    path = tmp_path / 'paper_version_lock.json'
    path.write_text(json.dumps(lock))
    before = path.read_bytes()
    result = lock_changes(tmp_path)
    assert result['changes'][0]['file'] == 'x.py'
    assert not result['replacement_approved'] and path.read_bytes() == before


def test_dividend_date_matches_identity_and_exact_amount():
    from audit_evidence_recovery import reconcile_action_dates
    fact = {'symbol': 'SPY', 'cusip': '78462F103', 'ex_date': '2016-06-17',
            'payable_date': '2016-07-29', 'rate': '1.078442', 'reviewed': True,
            'source_url': 'https://example.test/issuer', 'source_sha256': 'a' * 64}
    row = {'symbol': 'SPY', 'cusip': '78462F103', 'ex_date': '2016-06-17',
           'rate': 1.078442, 'source_action_type': 'cash_dividends'}
    report = reconcile_action_dates([row], [fact])
    assert report['supported_missing_dates'] == 1 and not report['full_action_coverage_verified']
    assert 'payable_date' not in row
    row['cusip'] = 'different-security'
    assert reconcile_action_dates([row], [fact])['remaining_missing_dates'] == 1
    with pytest.raises(ValueError):
        reconcile_action_dates([row], [fact, fact])


def test_verified_settlement_retains_parent_and_distributes_child_once():
    from portfolio_ledger import Account
    account = Account(100, {'OLD_IR': 10})
    event = {'event_id': 'settled-1', 'kind': 'security_settlement',
             'share_deltas': {'OLD_IR': -10, 'TT': 10, 'NEW_IR': 8},
             'cash_delta': 5.5, 'settlement_verified': True,
             'source': 'fixture-account-statement', 'source_sha256': 'a' * 64}
    account.action('2020-03-02', event)
    assert account.shares == {'OLD_IR': 0, 'TT': 10, 'NEW_IR': 8}
    assert account.cash == 105.5
    with pytest.raises(ValueError, match='Duplicate'):
        account.action('2020-03-02', event)
    with pytest.raises(ValueError, match='Verified'):
        Account(100, {'OLD_IR': 10}).action('2020-03-02', {**event, 'settlement_verified': False})


def test_invalid_settlement_is_atomic():
    from portfolio_ledger import Account
    account = Account(100, {'OLD': 1})
    event = {'event_id': 'bad', 'kind': 'security_settlement', 'share_deltas': {'NEW': 2, 'OLD': -10},
             'cash_delta': 0, 'settlement_verified': True, 'source': 'fixture', 'source_sha256': 'b' * 64}
    with pytest.raises(ValueError):
        account.action('2020-01-01', event)
    assert account.shares == {'OLD': 1} and account.cash == 100 and not account.seen


def test_cash_movements_preserve_signed_broker_amounts():
    from audit_evidence_recovery import activity_events
    rows = [{'id': 'deposit', 'activity_type': 'CSD', 'date': '2026-09-07', 'net_amount': '100'},
            {'id': 'withdraw', 'activity_type': 'CSW', 'date': '2026-09-07', 'net_amount': '-25'}]
    assert activity_events(rows).amount.tolist() == [100, -25]
    with pytest.raises(ValueError, match='Unmapped'):
        activity_events([{'id': 'stock-transfer', 'activity_type': 'ACATS'}])


def test_prospective_review_requires_both_sessions_and_independent_cohorts():
    from corrected_audit import prospective_status
    from core_satellite_alpha import _nyse_sessions
    sessions = _nyse_sessions('2024-01-03', '2025-12-31')
    frozen = {'frozen_at': '2024-01-02T23:00:00Z', 'strategy_fingerprint': 'same'}
    options = {'now': '2025-12-31T23:00:00Z', 'current_fingerprint': 'same', 'observed_sessions': sessions}
    assert prospective_status(frozen, **options, matured_cohorts=19)['status'] == 'collecting'
    assert prospective_status(frozen, **options, matured_cohorts=20)['status'] == 'ready_for_final_review'
    options['observed_sessions'] = sessions[:251]
    assert prospective_status(frozen, **options, matured_cohorts=20)['status'] == 'collecting'


def test_closure_requires_current_code_and_passing_named_case(tmp_path):
    from audit_gap_register import verified_fixes
    code = tmp_path / 'module.py'; code.write_text('code')
    xml = tmp_path / 'tests.xml'; xml.write_text('<testsuite><testcase classname="tests.fixture" name="test_cash"/></testsuite>')
    proof = {'source_commit': 'abc', 'code_files': {'module.py': hashlib.sha256(code.read_bytes()).hexdigest()},
             'test_report': 'tests.xml', 'test_report_sha256': hashlib.sha256(xml.read_bytes()).hexdigest(),
             'fixes': [{'id': 'code-cash', 'reason': 'cash_conservation', 'tests': ['tests.fixture.test_cash']}]}
    path = tmp_path / 'proof.json'; path.write_text(json.dumps(proof))
    assert verified_fixes(path, tmp_path, 'abc')[0]['id'] == 'code-cash'
    code.write_text('changed')
    with pytest.raises(ValueError, match='fingerprint'):
        verified_fixes(path, tmp_path, 'abc')
    with pytest.raises(ValueError, match='commit'):
        verified_fixes(path, tmp_path, 'other')


def test_register_ids_survive_reworded_next_action():
    report = {'generated_at': 'now', 'audit_code_commit': 'abc',
              'blockers': [{'reason': 'missing', 'next_action': 'old wording'}]}
    first = update_register(report, {'findings': []})
    report['blockers'][0]['next_action'] = 'new wording'
    second = update_register(report, {'findings': []}, first)
    assert len(second['findings']) == 1
    assert second['findings'][0]['id'] == first['findings'][0]['id']



def test_code_closure_cannot_overwrite_original_finding(tmp_path):
    from audit_gap_register import verified_fixes
    code = tmp_path / 'module.py'; code.write_text('code')
    xml = tmp_path / 'tests.xml'; xml.write_text('<testsuite><testcase classname="tests.fixture" name="test_cash"/></testsuite>')
    proof = {'source_commit': 'abc', 'code_files': {'module.py': hashlib.sha256(code.read_bytes()).hexdigest()},
             'test_report': 'tests.xml', 'test_report_sha256': hashlib.sha256(xml.read_bytes()).hexdigest(),
             'fixes': [{'id': 'original-001', 'reason': 'cash', 'tests': ['tests.fixture.test_cash']}]}
    path = tmp_path / 'proof.json'; path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='unique code record'):
        verified_fixes(path, tmp_path, 'abc')
