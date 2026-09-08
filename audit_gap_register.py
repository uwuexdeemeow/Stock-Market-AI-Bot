"""Keep an append-only inventory of evidence gaps without manufacturing closure."""
from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from safe_io import atomic_write_json, atomic_write_text


def code_identity():
    """Identify the checkout that generated a report, including unsaved changes."""
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root).strip())
        return {'source_commit': commit, 'source_worktree_dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'source_commit': None, 'source_worktree_dirty': None}


def fingerprint(value):
    """A stable digest identifies the same finding on later audit runs."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def issue_identity(issue):
    """Changing wording or observation counts does not create a new defect."""
    return fingerprint({key: issue.get(key) for key in
                        ('reason', 'source', 'ticker', 'date', 'file', 'variant', 'period')})


def verified_fixes(path, root, source_commit):
    """Bind closure claims to current code bytes and actual passing test cases."""
    if not path or not Path(path).exists():
        return []
    proof = json.loads(Path(path).read_text())
    if proof.get('source_commit') != source_commit or not proof.get('code_files'):
        raise ValueError('Verification source commit missing or mismatched')
    root = Path(root).resolve()
    for name, expected in proof['code_files'].items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Verification code fingerprint mismatch')
    xml = (root / proof['test_report']).resolve()
    if not xml.is_relative_to(root) or hashlib.sha256(xml.read_bytes()).hexdigest() != proof['test_report_sha256']:
        raise ValueError('Verification test report mismatch')
    try:
        cases = {case.attrib.get('classname', '') + '.' + case.attrib['name']: case
                 for case in ET.fromstring(xml.read_bytes()).iter('testcase')}
    except ET.ParseError as exc:
        raise ValueError('Malformed verification test report') from exc
    output = []
    seen = set()
    for fix in proof.get('fixes', []):
        # Code repair records have their own IDs, so proof cannot overwrite
        # one of the original historical evidence findings.
        if not str(fix.get('id', '')).startswith('code-') or fix['id'] in seen:
            raise ValueError('Closure ID must be a unique code record')
        seen.add(fix['id'])
        if (not fix.get('tests') or any(test not in cases or any(cases[test].find(tag) is not None
                for tag in ('failure', 'error', 'skipped')) for test in fix['tests'])):
            raise ValueError('Closure requires passing named tests')
        output.append({'id': fix['id'], 'reason': fix['reason'], 'tests': fix['tests'],
                       'source_commit': source_commit, 'verification_sha256': fingerprint(proof)})
    return output


def update_register(report, baseline, previous=None):
    """Disappearance is not proof of repair; unresolved history stays visible."""
    records = {r['id']: dict(r) for r in (previous or {}).get('findings', [])}
    original = baseline.get('findings', [])
    for index, issue in enumerate(original, 1):
        identity = f'original-{index:03d}'
        records.setdefault(identity, {'id': identity, 'finding': issue,
            'first_seen': baseline.get('generated_at'), 'status': 'blocked',
            'closure_evidence': [], 'verification_result': 'matching_closure_evidence_required'})
    for issue in report['blockers']:
        matching = [r for r in records.values() if issue_identity(r['finding']) == issue_identity(issue)]
        if not matching:
            identity = 'gap-' + issue_identity(issue)[:20]
            records[identity] = {'id': identity, 'finding': issue,
                'first_seen': report['generated_at'], 'status': 'blocked',
                'closure_evidence': [], 'verification_result': 'not_verified_fixed'}
    current = {issue_identity(b) for b in report['blockers']}
    for row in records.values():
        issue = row['finding']
        if row.get('status') == 'verified_fixed':
            # Historical passing tests remain attached, but a new checkout
            # must supply current proof before repeating the closure claim.
            row['status'] = 'blocked'
            row['verification_result'] = 'current_code_reverification_required'
        row['observed_in_current_audit'] = issue_identity(issue) in current
        row['dependency'] = issue.get('source', 'historical_data_or_broker_replay')
        row['next_action'] = issue.get('next_action', 'Recover independently verified matching evidence.')
        row['affected_security'] = issue.get('ticker')
        row['period'] = {'start': report.get('audit_start'), 'end': report.get('audit_end')}
        row['group'] = issue['reason'].split(':')[0]
        if issue['reason'] == 'normal_session_activity_reconciliation_awaiting_observations':
            row['status'] = 'awaiting_new_observations'
    for fix in report.get('verified_fixes', []):
        records[fix['id']] = {'id': fix['id'], 'finding': {'reason': fix['reason']},
            'first_seen': records.get(fix['id'], {}).get('first_seen', report['generated_at']),
            'status': 'verified_fixed', 'closure_evidence': records.get(fix['id'], {}).get('closure_evidence', []) +
                ([fix] if fix not in records.get(fix['id'], {}).get('closure_evidence', []) else []),
            'verification_result': 'named_tests_pass_current_code',
            'observed_in_current_audit': True, 'dependency': 'code_verification', 'next_action': 'Retain regression checks.',
            'affected_security': None, 'period': None, 'group': 'code_fix'}
    return {'schema_version': 1, 'generated_at': report['generated_at'],
            'source_commit': report['audit_code_commit'], 'baseline_count': len(original),
            'baseline_sha256': fingerprint(baseline), 'findings': list(records.values()),
            'complete': False, 'automatic_closure': False}


def write_register(report, output, baseline_path, previous_path=None):
    """Persist the exact source inventory alongside a readable closure table."""
    baseline = json.loads(Path(baseline_path).read_text())
    target = Path(previous_path) if previous_path else Path(output) / 'gap_register.json'
    previous = json.loads(target.read_text()) if target.exists() else None
    register = update_register(report, baseline, previous)
    atomic_write_json(register, Path(output) / 'gap_register.json')
    lines = ['# Evidence gap register', '',
             'An absent finding is not automatically fixed. Original findings remain recorded.', '',
             '| ID | Finding | Security | Status | Seen now | Next action |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in register['findings']:
        lines.append(f"| {row['id']} | {row['finding']['reason']} | {row['affected_security'] or ''} | {row['status']} | {row['observed_in_current_audit']} | {row['next_action']} |")
    atomic_write_text(Path(output) / 'gap_register.md', '\n'.join(lines) + '\n')
    return register


def lock_changes(root):
    """Compare every frozen file without updating the lock or epoch."""
    root = Path(root)
    lock = json.loads((root / 'paper_version_lock.json').read_text())
    observed, changes = {}, []
    for name, expected in lock.get('files', {}).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Lock path outside project')
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual is not None:
            observed[name] = actual
        if actual != expected:
            changes.append({'file': name, 'expected_sha256': expected, 'observed_sha256': actual})
    actual = hashlib.sha256(json.dumps(observed, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'epoch_id': lock.get('epoch_id'), 'expected_fingerprint': lock.get('logic_fingerprint'),
            'observed_fingerprint': actual, 'changes': changes, 'lock_modified': False,
            'replacement_approved': False}
