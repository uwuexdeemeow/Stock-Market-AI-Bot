"""Evidence integrity regressions using synthetic, non-private reports."""
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from evidence_audit import REQUIRED, approval_summary, snapshot_summary, input_report
from validation_bundle import sha256_value, strategy_config_fingerprint


def identity_fixture():
    config = {'score_source': 'regime_adaptive', 'max_gross_exposure': 1.25, 'deployment_max_gross_exposure': 1.0}
    bundle = {'schema_version': 2, 'config_fingerprint': strategy_config_fingerprint(config),
              'deployment': {'status': 'paper_provisional', 'paper_approved': True}, 'robustness_review': {'pass': True}}
    bundle['validation_bundle_hash'] = sha256_value(bundle)
    live = {'deployment_status': 'paper_provisional', 'paper_approved': True,
            'validation_bundle_hash': bundle['validation_bundle_hash'], 'approvals': {'core-alpha': {'approved': True}},
            'approved_live_configs': {'core-alpha': {'config': config, 'deployment_status': 'paper_provisional',
                'validation_bundle_hash': bundle['validation_bundle_hash']}}}
    return live, bundle


def test_approval_conflicts_never_become_a_pass():
    live, bundle = identity_fixture()
    assert approval_summary(live, bundle)['status'] == 'identity_consistent_pending_runtime_checks'
    live['approved_live_configs']['core-alpha']['deployment_status'] = 'rejected'
    report = approval_summary(live, bundle)
    assert report['status'] == 'blocked'
    assert 'deployment_status_conflict' in report['issues']
    assert not report['real_capital_approved']


def test_deployment_exposure_and_bundle_identity_are_checked():
    live, bundle = identity_fixture()
    live['approved_live_configs']['core-alpha']['config']['deployment_max_gross_exposure'] = 1.25
    assert 'configuration_fingerprint_mismatch' in approval_summary(live, bundle)['issues']
    live['validation_bundle_hash'] = 'old'
    assert 'top_bundle_reference_mismatch' in approval_summary(live, bundle)['issues']
    bundle['robustness_review']['pass'] = False
    assert 'validation_bundle_hash_mismatch' in approval_summary(live, bundle)['issues']


def complete_snapshot():
    files = {n: json.dumps({'run_id': 'run-one', 'cash': 'PRIVATE-CASH'}).encode() for n in REQUIRED}
    manifest = {'status': 'complete', 'run_id': 'run-one', 'generated_at': '2026-09-07T00:00:00+00:00',
                'files': {n: {'sha256': hashlib.sha256(v).hexdigest(), 'required': True} for n, v in files.items()}}
    files['paper_run_manifest.json'] = json.dumps(manifest).encode()
    return files


def test_manifest_is_recomputed_and_raw_balances_not_exposed():
    files = complete_snapshot()
    now = datetime(2026, 9, 7, 1, tzinfo=timezone.utc)
    report = snapshot_summary(files, source='fixture', now=now)
    assert report['complete']
    assert 'PRIVATE-CASH' not in json.dumps(report)
    files[REQUIRED[0]] = b'{"run_id":"other"}'
    changed = snapshot_summary(files, source='fixture', now=now)
    assert not changed['complete']
    assert any('checksum' in i for i in changed['issues'])
    assert any('run_mismatch' in i for i in changed['issues'])


def test_stale_and_partial_snapshots_are_explicit():
    files = complete_snapshot()
    report = snapshot_summary(files, source='fixture', now=datetime(2026, 9, 20, tzinfo=timezone.utc))
    assert 'snapshot_stale_or_future' in report['issues']
    del files[REQUIRED[0]]
    assert not snapshot_summary(files, source='fixture')['complete']
    assert not snapshot_summary({}, source='fixture')['complete']


def test_missing_context_does_not_allow_a_complete_source_report(tmp_path, monkeypatch):
    import corrected_data
    monkeypatch.setattr(corrected_data, 'validate_sources', lambda *a, **k: {'complete': True, 'gaps': []})
    result = input_report(tmp_path, tmp_path / 'membership.csv', {'dated_context': str(tmp_path / 'absent.parquet')},
                          start='2020-01-01', end='2021-01-01')
    assert not result['complete']
    assert result['gaps'][0]['reason'] == 'dated_context_missing_or_unverified'


def test_evidence_mode_rejects_mutating_modes():
    from corrected_audit import main
    for flag in ('--freeze', '--observe', '--audit-only', '--ablations'):
        with pytest.raises(SystemExit) as caught:
            main(['--evidence-report', flag])
        assert caught.value.code == 2


def test_imported_archive_requires_digest_and_avoids_extraction(tmp_path):
    import io
    import zipfile
    from evidence_audit import imported_artifact
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, 'w') as archive:
        archive.writestr('signals/alpaca_execution_scorecard.json', '{"run_id":"test"}')
        archive.writestr('../../private.txt', 'do not extract')
    path = tmp_path / 'artifact.zip'
    path.write_bytes(raw.getvalue())
    metadata = tmp_path / 'metadata.json'
    record = {'digest': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest(),
              'status': 'completed', 'artifact_id': 1, 'run_id': 2, 'head_sha': 'abc',
              'source_url': 'https://api.github.com/repos/example/repo/actions/artifacts/1', 'updated_at': '2026-09-07T00:00:00Z'}
    metadata.write_text(json.dumps(record))
    _, _, files = imported_artifact(path, metadata)
    assert list(files) == ['alpaca_execution_scorecard.json']
    assert not (tmp_path / 'private.txt').exists()
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='digest'):
        imported_artifact(path, metadata)
