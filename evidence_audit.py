"""Read independent evidence snapshots and publish only a small, non-private verdict."""
from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from safe_io import atomic_write_json, atomic_write_text
from validation_bundle import validate_validation_bundle, strategy_config_fingerprint

# Only these files may be read from an operational archive. Their raw contents
# remain in memory: account balances and order identifiers are never published.
NAMES = ('paper_run_manifest.json', 'core_satellite_live_configs.json',
         'core_satellite_validation_bundle.json', 'broker_truth.json',
         'alpaca_execution_scorecard.json', 'paper_validation_epoch_status.json',
         'alpaca_paper_health.json', 'workflow_heartbeat_daily.json',
         'alpaca_paper_log.csv', 'alpaca_submit_outcomes.csv', 'core_satellite_alpha_signal.csv')
NAMES += ('core_satellite_alpha_orders.csv', 'core_satellite_alpha_input_snapshot.csv',
          'workflow_skip.json', 'workflow_heartbeat_execution.json', 'execution_run_manifest.json')
REQUIRED = ('broker_truth.json', 'alpaca_execution_scorecard.json',
            'paper_validation_epoch_status.json', 'alpaca_paper_health.json')


def command(root, *args):
    """Read external state without shell interpolation or displaying credentials."""
    result = subprocess.run(args, cwd=root, capture_output=True, timeout=45)
    if result.returncode:
        raise ValueError('source_unavailable')
    return result.stdout


def payload(files, name):
    """An absent or malformed object is missing evidence, never a passing result."""
    try:
        value = json.loads(files.get(name, b'{}'))
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def snapshot_summary(files, *, source, commit=None, now=None):
    """Recheck manifest bytes and run identities instead of trusting its status."""
    now = now or datetime.now(timezone.utc)
    manifest = payload(files, 'paper_run_manifest.json')
    execution_manifest = payload(files, 'execution_run_manifest.json')
    if execution_manifest.get('profile') == 'execution_observation_only' and execution_manifest.get('run_id'):
        # A post-market refresh is a distinct observation bundle. Select it
        # only when every required report belongs to that exact run.
        if all((payload(files, name).get('run_id') or payload(files, name).get('run_context', {}).get('run_id'))
               == execution_manifest['run_id'] for name in REQUIRED):
            manifest = execution_manifest
    issues = []
    if manifest.get('status') != 'complete' or not manifest.get('run_id'):
        issues.append('complete_run_manifest_missing')
    records = manifest.get('files', {})
    if not isinstance(records, dict) or any(not isinstance(r, dict) for r in records.values()):
        issues.append('manifest_records_malformed')
        records = {}
    for name in sorted(set(REQUIRED) | {n for n, r in records.items() if r.get('required')}):
        data = files.get(name)
        record = records.get(name, {})
        if not data or not record.get('sha256'):
            issues.append('required_evidence_missing:' + name)
        elif hashlib.sha256(data).hexdigest() != record['sha256']:
            issues.append('evidence_checksum_mismatch:' + name)
        if name in REQUIRED:
            item = payload(files, name)
            run = item.get('run_id') or item.get('run_context', {}).get('run_id')
            if not run or run != manifest.get('run_id'):
                issues.append('evidence_run_mismatch:' + name)
    timestamp = manifest.get('generated_at') or payload(files, 'workflow_heartbeat_daily.json').get('completed_at')
    age = None
    try:
        observed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        age = (now - observed).total_seconds() / 3600
        # 96 hours is an observation freshness warning, not a trading calendar gate.
        if age < 0 or age > 96:
            issues.append('snapshot_stale_or_future')
    except (AttributeError, TypeError, ValueError):
        issues.append('snapshot_timestamp_missing')
    scorecard = payload(files, 'alpaca_execution_scorecard.json')
    summary = scorecard.get('summary', {})
    health = payload(files, 'alpaca_paper_health.json')
    epoch = payload(files, 'paper_validation_epoch_status.json')
    operational_issues = []
    if health.get('freshness_ok') is False:
        operational_issues.append('signal_freshness_failed')
    if scorecard.get('decision_eligible') is False:
        operational_issues.append('execution_observations_insufficient_or_failed')
    if epoch.get('paper_version_lock_valid') is False:
        operational_issues.append('reported_paper_version_lock_invalid')
    for check in ('trading_days', 'rebalance_events', 'accepted_orders', 'classified_sessions',
                  'average_slippage', 'bad_slippage_rate', 'stage_comparison_ready', 'two_stage_design'):
        if epoch.get('checks', {}).get(check) is False:
            operational_issues.append('epoch_check_failed:' + check)
    return {'source': source, 'source_commit': commit,
            'profile': manifest.get('profile', 'daily'),
            'run_commit': manifest.get('run_context', {}).get('git_commit'),
            'generated_at': timestamp, 'age_hours': age, 'complete': not issues,
            'issues': sorted(set(issues)), 'measured_fills': summary.get('measured_slippage_count'),
            'sessions': summary.get('trading_sessions'),
            'decision_eligible': scorecard.get('decision_eligible') is True,
            'operational_issues': operational_issues,
            'file_hashes': {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}}


def approval_summary(live, bundle):
    """A derived audit verdict cannot rewrite or override a deployment decision."""
    entry = live.get('approved_live_configs', {}).get('core-alpha', {})
    config = entry.get('config', {})
    _, issues = validate_validation_bundle(bundle)
    issues = list(issues)
    if not config:
        issues.append('deployed_configuration_missing')
    identity = strategy_config_fingerprint(config)
    if identity != bundle.get('config_fingerprint'):
        issues.append('configuration_fingerprint_mismatch')
    expected = bundle.get('validation_bundle_hash')
    for label, record in [('top', live), ('strategy', entry)]:
        if not expected or record.get('validation_bundle_hash') != expected:
            issues.append(label + '_bundle_reference_mismatch')
    statuses = [live.get('deployment_status'), entry.get('deployment_status'), bundle.get('deployment', {}).get('status')]
    if len(set(statuses)) > 1:
        issues.append('deployment_status_conflict')
    if 'rejected' in statuses or not all((live.get('paper_approved') is True,
            bundle.get('deployment', {}).get('paper_approved') is True,
            live.get('approvals', {}).get('core-alpha', {}).get('approved') is True)):
        issues.append('paper_approval_not_unanimous')
    source = config.get('score_source')
    routes = ({'risk_on': 'factor_risk_on_score', 'neutral': 'factor_walkforward_score',
               'risk_off': 'factor_defensive_score'} if source == 'regime_adaptive' else {'configured': source})
    return {'status': 'blocked' if issues else 'identity_consistent_pending_runtime_checks',
            'issues': sorted(set(issues)), 'config_fingerprint': identity,
            'bundle_hash': expected, 'score_source': source, 'score_columns': routes,
            'scoring_implementation': 'core_satellite_alpha._score_col_for_regime',
            'xgboost_attribution_verified': False,
            'research_max_gross': config.get('max_gross_exposure'),
            'deployment_max_gross': config.get('deployment_max_gross_exposure'),
            'real_capital_approved': False}


def archive_files(raw):
    """Read allowed members in memory; never extract ZIP paths onto disk."""
    files = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            if name in NAMES:
                if name in files or member.file_size > 50_000_000:
                    raise ValueError('ambiguous_or_oversized_artifact')
                files[name] = archive.read(member)
    return files


def imported_artifact(path, metadata_path):
    """Bind an externally retrieved archive to its API-reported digest and run."""
    raw = path.read_bytes()
    meta = json.loads(metadata_path.read_text())
    digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
    if (meta.get('digest') != digest or meta.get('status') != 'completed' or
            not all(meta.get(key) for key in ('artifact_id', 'run_id', 'head_sha', 'source_url', 'updated_at'))):
        raise ValueError('artifact_metadata_or_digest_invalid')
    return (f"imported_workflow_artifact:{meta['run_id']}:{meta['artifact_id']}:{meta.get('conclusion')}",
            meta['head_sha'], archive_files(raw))


def iter_api_pages(root, endpoint, field, *, maximum_pages=100):
    """Walk every page with a bound and reject repeated IDs rather than truncate."""
    seen = set()
    for page in range(1, maximum_pages + 1):
        separator = '&' if '?' in endpoint else '?'
        body = json.loads(command(root, 'gh', 'api', f'{endpoint}{separator}per_page=100&page={page}'))
        batch = body[field]
        if not isinstance(batch, list):
            raise ValueError('invalid_api_page')
        for row in batch:
            identity = row.get('id')
            if identity is None or identity in seen:
                raise ValueError('repeated_api_identity')
            seen.add(identity)
        yield from batch
        if len(batch) < 100:
            return
    raise ValueError('api_pagination_limit')


def api_pages(root, endpoint, field, *, maximum_pages=100):
    """Materialize bounded artifact pages; run discovery can stop lazily."""
    return list(iter_api_pages(root, endpoint, field, maximum_pages=maximum_pages))


def collect_snapshots(root, observations=None):
    """Read Git and completed workflow artifacts independently, without checkout."""
    root = Path(root)
    snapshots = []
    local = {n: (root / 'signals' / n).read_bytes() for n in NAMES if (root / 'signals' / n).is_file()}
    snapshots.append(('local_snapshot', None, local))
    errors = []
    observations = observations if observations is not None else []
    try:
        commit = command(root, 'git', 'rev-parse', 'origin/signals/latest').decode().strip()
        files = {}
        for name in NAMES:
            try:
                files[name] = command(root, 'git', 'show', f'{commit}:signals/{name}')
            except ValueError:
                pass
        snapshots.append(('signals/latest', commit, files))
    except (ValueError, OSError, subprocess.TimeoutExpired):
        errors.append('signals_branch_unavailable')
    try:
        # gh resolves the current repository; authentication failures are explicit.
        runs = iter_api_pages(root, 'repos/{owner}/{repo}/actions/workflows/daily_paper_trading.yml/runs?status=completed', 'workflow_runs')
        latest_id = None
        for selected in runs:
            latest_id = latest_id or selected['id']
            artifacts = api_pages(root, f"repos/{{owner}}/{{repo}}/actions/runs/{selected['id']}/artifacts", 'artifacts')
            usable = [a for a in artifacts if not a.get('expired')]
            if not usable:
                # Scheduled guard runs can finish without producing evidence.
                # Keep that fact while looking for a clearly labeled older run.
                errors.append(f"completed_run_has_no_artifact:{selected['id']}")
                continue
            found = False
            for artifact in usable:
                raw = command(root, 'gh', 'api', f"repos/{{owner}}/{{repo}}/actions/artifacts/{artifact['id']}/zip")
                if artifact.get('digest') != 'sha256:' + hashlib.sha256(raw).hexdigest():
                    raise ValueError('artifact_digest_unverified')
                files = archive_files(raw)
                skip = payload(files, 'workflow_skip.json')
                if (skip.get('run_id') == str(selected['id']) and skip.get('reason') in
                        {'market_closed', 'schedule_guard'} and selected.get('conclusion') == 'success'):
                    # Explicit run-bound skip evidence is not a failed trading run.
                    observations.append({'run_id': selected['id'], 'status': 'intentional_skip',
                                         'reason': skip['reason'], 'source_commit': selected['head_sha']})
                    continue
                if not any(name in files for name in REQUIRED):
                    continue
                snapshots.append((f"workflow_artifact:{selected['id']}:{artifact['id']}:{selected['conclusion']}", selected['head_sha'], files))
                found = True
            if found:
                observations.append({'run_id': selected['id'], 'status': 'artifact_selected',
                                     'older_fallback': selected['id'] != latest_id,
                                     'source_commit': selected['head_sha']})
                break
        if latest_id is None:
            errors.append('completed_daily_runs_unavailable')
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired, zipfile.BadZipFile):
        errors.append('workflow_artifacts_unavailable_or_incomplete')
    return snapshots, errors


def review_sections(folder):
    """Read explicitly selected offline reports without copying their raw data."""
    sections = {}
    for name in ('membership_identity_report', 'security_transition_report', 'edge_ablation_comparison'):
        path = Path(folder) / (name + '.json')
        body = payload({'review': path.read_bytes()} if path.exists() else {}, 'review')
        section = {'source_file': name + '.json', 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                   'source_commit': body.get('audit_code_commit') or body.get('source_commit'),
                   'generated_at': body.get('generated_at'), 'complete': body.get('complete') is True,
                   'status': body.get('status', 'missing'), 'issues': [], 'findings': []}
        # Only fixed public summaries cross the publication boundary. Orders,
        # balances, account IDs and arbitrary nested source responses stay out.
        for key in ('candidate_cutoff', 'candidate_symbols', 'candidate_intervals',
                    'verified_full_intervals', 'disagreement_dates', 'disagreement_symbols',
                    'candidate_identity', 'facts_sha256', 'deployed_strategy_equivalent'):
            if key in body:
                section[key] = body[key]
        if not body:
            section['issues'].append('review_report_missing')
        if not section['source_commit'] or not section['generated_at']:
            section['issues'].append('review_source_identity_incomplete')
        if section['status'] != 'complete' or not section['complete']:
            section['issues'].append('review_not_fully_verified')
        for ticker, count in body.get('priority_symbols', []):
            if re.fullmatch(r'[A-Z0-9.-]{1,12}', str(ticker)):
                section['findings'].append({'reason': 'historical_membership_source_disagreement',
                                            'ticker': ticker, 'count': int(count)})
        for event in body.get('transitions', []):
            for check in event.get('checks', []):
                if check.get('result') in {'unseparated_issuer_history', 'candidate_disagrees', 'candidate_coverage_unavailable'}:
                    section['findings'].append({'reason': check['result'], 'ticker': check['ticker'],
                                                'date': check['date']})
        for variant, result in body.get('results', {}).items():
            if result.get('status') != 'complete':
                section['findings'].append({'reason': 'ablation_verified_result_unavailable', 'variant': variant})
        sections[name] = section
    return sections


def input_report(data_dir, membership, spec, *, start, end):
    """Collect source failures even when malformed files stop normal evaluation."""
    from corrected_data import validate_sources, validate_dated_inputs, validate_context_coverage
    import pandas as pd
    try:
        result = validate_sources(data_dir, membership, start=start, end=end)
    except (OSError, ValueError, KeyError, TypeError):
        result = {'complete': False, 'gaps': [{'reason': 'source_manifest_unreadable'}]}
    gaps = result.get('gaps', []).copy()
    context = None
    for key in ('dated_context', 'feature_panel'):
        if spec.get(key):
            try:
                frame = pd.read_parquet(spec[key])
                validate_dated_inputs(frame)
                if frame.duplicated(['date', 'ticker']).any():
                    raise ValueError('duplicate context')
                if key == 'dated_context':
                    context = frame
                if key == 'dated_context' and not {'vix_inverted', 'sector', 'days_to_next_earnings'}.issubset(frame):
                    raise ValueError('required context fields missing')
                if key == 'feature_panel' and spec.get('feature_provenance_verified') is not True:
                    raise ValueError('feature provenance unverified')
            except (OSError, ValueError, KeyError, TypeError):
                gaps.append({'reason': key + '_missing_or_unverified'})
    if result.get('complete') and not gaps:
        try:
            import numpy as np
            from corrected_data import load_raw_panel, eligible_candidates
            from core_satellite_alpha import _nyse_sessions, _session_offset
            bars, _ = load_raw_panel(data_dir, membership, start=start, end=end)
            numeric = bars[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy(dtype=float)
            if (bars.duplicated(['date', 'ticker']).any() or not np.isfinite(numeric).all()
                    or (numeric[:, :4] <= 0).any() or (numeric[:, 4] < 0).any()):
                raise ValueError('invalid raw bars')
            if ((bars.High < bars[['Open', 'Close', 'Low']].max(axis=1)).any() or
                    (bars.Low > bars[['Open', 'Close', 'High']].min(axis=1)).any()):
                raise ValueError('invalid OHLC ordering')
            # Check context on every actual decision date, including the session
            # before each evaluation opens. Merely having a context file is not coverage.
            windows = [window for fold in spec.get('folds', []) for window in [fold, *fold.get('inner', [])]]
            decisions = set()
            for window in windows:
                decisions.update(_nyse_sessions(_session_offset(window['start'], -1), _session_offset(window['end'], -1)))
            candidates = eligible_candidates(bars[['date', 'ticker']], membership)
            validate_context_coverage(context, candidates, spec.get('configurations', []))
            expected = candidates.loc[pd.to_datetime(candidates.date).isin(decisions)]
            needs_context = any(c.get('earnings_blackout_days', 0) or c.get('max_per_sector', 2) or c.get('regime_mode', 'static') != 'static'
                                for c in spec.get('configurations', []))
            if needs_context:
                if context is None or expected.empty:
                    raise ValueError('context coverage missing')
                joined = expected.merge(context, on=['date', 'ticker'], how='left', validate='one_to_one')
                if joined[['vix_inverted', 'sector', 'days_to_next_earnings']].isna().any().any():
                    raise ValueError('context coverage incomplete')
        except (OSError, ValueError, KeyError, TypeError):
            gaps.append({'reason': 'raw_price_or_decision_context_coverage_invalid'})
    # Publish reason codes and ticker coverage only, never arbitrary source text.
    safe_gaps = [{'reason': str(g.get('reason', 'unknown_source_gap')),
                  **{key: g[key] for key in ('ticker', 'count', 'first', 'last') if key in g}}
                 for g in gaps]
    return {'complete': bool(result.get('complete')) and not gaps, 'gaps': safe_gaps,
            'next_action': 'Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.'}


def write_evidence_report(args):
    """One report joins conclusions, not raw account records or mixed snapshots."""
    from corrected_audit import replay_certified
    from paper_validation_epoch import validate_paper_version_lock
    import core_satellite_alpha as core
    root = Path(__file__).resolve().parent
    observations = []
    snapshots, unavailable = collect_snapshots(root, observations)
    imported = None
    if getattr(args, 'workflow_artifact', None):
        try:
            snapshots.append(imported_artifact(args.workflow_artifact, args.artifact_metadata))
            imported = json.loads(args.artifact_metadata.read_text())
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            unavailable.append('imported_workflow_artifact_invalid')
    spec_path = args.spec or root / 'corrected_shadow_spec.json'
    spec = json.loads(spec_path.read_text())
    sources = []
    blockers = [{'reason': issue, 'next_action': 'Restore read-only access and retrieve the latest completed run.'} for issue in unavailable]
    for name, commit, files in snapshots:
        summary = snapshot_summary(files, source=name, commit=commit)
        approval = approval_summary(payload(files, 'core_satellite_live_configs.json'), payload(files, 'core_satellite_validation_bundle.json'))
        sources.append({**summary, 'approval': approval})
        for issue in summary['issues'] + approval['issues'] + summary['operational_issues']:
            blockers.append({'reason': issue, 'source': name,
                             'next_action': 'Recover matching evidence and rerun existing validation; do not edit approval flags.'})
    data = input_report(args.data_dir, args.membership, spec, start=args.start, end=args.end)
    blockers.extend({**g, 'next_action': data['next_action']} for g in data['gaps'])
    # Existing production validators are read-only. They inspect current source
    # files, not a remote snapshot secretly combined with local configuration.
    try:
        runtime = core._load_approved_live_config()
        runtime_ok = runtime.get('approved') is not False and bool(runtime.get('config'))
        runtime_issues = runtime.get('reasons', [])
    except (Exception, SystemExit):
        runtime_ok, runtime_issues = False, ['runtime_validation_unavailable']
    lock_ok, lock_issues = validate_paper_version_lock()
    for issue in runtime_issues + lock_issues:
        # Production messages may contain paths; publish only the reason category.
        blockers.append({'reason': str(issue).split(':')[0], 'source': 'local_runtime',
                         **({'file': str(issue).split(':', 1)[1]} if str(issue).startswith(('locked_file_changed:', 'locked_file_missing:')) else {}),
                         'next_action': 'Review current runtime evidence and version lock; preserve existing freeze.'})
    replay_path = getattr(args, 'reconciliation_report', None) or args.output / 'replay_reconciliation.json'
    replay = payload({'replay': replay_path.read_bytes()} if replay_path.exists() else {}, 'replay')
    certified = replay_certified(replay)
    if not certified:
        blockers.append({'reason': 'recorded_replay_not_certified', 'next_action': 'Recover complete activities, fees and independently verified interval balances; replay to one-cent cash tolerance.'})
    if replay.get('evidence_scope') != 'recorded_activity_interval':
        blockers.append({'reason': 'normal_session_activity_reconciliation_awaiting_observations',
                         'next_action': 'Reconcile a normal-session interval after broker activity and fee postings; never submit orders to manufacture samples.'})
    if replay.get('historical_reconciliation_certified') is not True:
        blockers.append({'reason': 'full_historical_broker_accounting_uncertified',
                         'next_action': 'Obtain independently sourced historical opening balances and all subsequent activity; interval arithmetic is insufficient.'})
    blockers.append({'reason': 'corrected_prospective_validation_requires_explicit_freeze_and_new_observations',
                     'next_action': 'After verified inputs, successful evaluation and certified reconciliation, obtain explicit freeze before 252 new sessions and 20 independent matured cohorts.'})
    recovery_path = getattr(args, 'recovery_report', None)
    recovery = payload({'recovery': recovery_path.read_bytes()} if recovery_path and recovery_path.exists() else {}, 'recovery')
    recovered = {'source_sha256': hashlib.sha256(recovery_path.read_bytes()).hexdigest() if recovery_path and recovery_path.exists() else None,
                 'generated_at': recovery.get('generated_at'),
                 'activity_counts': {k: recovery.get('paper', {}).get('activity_counts', {}).get(k) for k in ('FILL', 'FEE')},
                 'source_candidates': [{k: row.get(k) for k in ('source', 'sha256', 'source_cutoff', 'verified')} for row in (recovery.get('membership_sources', []) if isinstance(recovery.get('membership_sources'), list) else []) if isinstance(row, dict)],
                 'price_probes': [{k: row.get(k) for k in ('ticker', 'rows', 'first', 'last', 'pagination_remaining', 'verified_full_coverage')} for row in recovery.get('price_probes', [])]}
    # The auditor's checkout is distinct from the revision that generated each
    # operational snapshot. Record both rather than asserting they are equal.
    try:
        audit_commit = command(root, 'git', 'rev-parse', 'HEAD').decode().strip()
        audit_dirty = bool(command(root, 'git', 'status', '--porcelain', '--untracked-files=no').strip())
    except (ValueError, OSError, subprocess.TimeoutExpired):
        audit_commit, audit_dirty = None, None
    from audit_gap_register import lock_changes, write_register, verified_fixes
    reviews = review_sections(getattr(args, 'reviews_dir', None) or args.output / 'membership_review')
    for name, section in reviews.items():
        blockers.extend({'reason': issue, 'source': name,
                         'next_action': 'Regenerate the attributed review and resolve its verified-input dependencies.'}
                        for issue in section['issues'])
        blockers.extend({**issue, 'source': name,
                         'next_action': 'Resolve the named security/date or verified-input dependency; retain original observations.'}
                        for issue in section['findings'])
    try:
        lock_detail = lock_changes(root)
    except (ValueError, OSError, KeyError, TypeError):
        lock_detail = {'status': 'unavailable', 'lock_modified': False}
    report = {'schema_version': 2, 'generated_at': datetime.now(timezone.utc).isoformat(),
              'audit_start': args.start, 'audit_end': args.end,
              'audit_code_commit': audit_commit, 'audit_worktree_dirty': audit_dirty,
              'status': 'blocked' if blockers else 'evidence_consistent', 'sources': sources,
              'recovery': recovered,
              'imported_artifact': {k: imported.get(k) for k in ('run_id', 'artifact_id', 'head_sha', 'source_url', 'updated_at', 'latest_completed_run_id', 'latest_completed_run_has_artifacts')} if imported else None,
              'runtime': {'source': 'local_snapshot', 'passed': runtime_ok, 'version_lock_passed': lock_ok},
              'data': data, 'replay': {'certified': certified, 'reconciled': replay.get('reconciled') is True,
                                     'evidence_scope': replay.get('evidence_scope', 'unspecified_in_source'),
                                     'activity_counts': {k: replay.get('activity_counts', {}).get(k) for k in ('FILL', 'FEE')},
                                     'interval_start': replay.get('interval_start'), 'interval_end': replay.get('interval_end'),
                                     'source_sha256': hashlib.sha256(replay_path.read_bytes()).hexdigest() if replay_path.exists() else None,
                                     'arrival_quotes': 'unavailable_unless_recorded_with_order'},
              'shadow_candidate': {'identity': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                                   'scoring': 'fold_local_raw_feature_ranks', 'deployed_strategy_equivalent': False},
              'blockers': blockers, 'real_capital_approved': False, 'freeze_started': False,
              'historical_interpretation': 'superseded_results_require_corrected_rebuild'}
    report['workflow_observations'] = observations
    report['review_sections'] = reviews
    report['version_lock_changes'] = lock_detail
    try:
        report['verified_fixes'] = verified_fixes(getattr(args, 'verification_report', None), root, audit_commit)
    except (ValueError, OSError, KeyError, TypeError):
        report['verified_fixes'] = []
        report['blockers'].append({'reason': 'code_closure_evidence_invalid', 'source': 'verification_report',
                                   'next_action': 'Regenerate named test evidence bound to this exact code revision.'})
        report['status'] = 'blocked'
    args.output.mkdir(parents=True, exist_ok=True)
    register = write_register(report, args.output, root / 'research_evidence/original_gap_baseline.json',
                              getattr(args, 'gap_register', None))
    report['gap_register'] = {'findings': len(register['findings']), 'original_findings': register['baseline_count'],
                              'path': 'gap_register.json', 'automatic_closure': False}
    atomic_write_json(report, args.output / 'evidence_report.json')
    lines = ['# Strategy evidence audit', '', 'Status: **' + report['status'] + '**', '',
             'Historical claims require corrected rebuilding. Shadow candidate is separate from deployed factor scoring.', '',
             '| Source | Complete | Measured fills | Sessions |', '| --- | --- | --- | --- |']
    for source in sources:
        lines.append(f"| {source['source']} | {source['complete']} | {source['measured_fills']} | {source['sessions']} |")
    lines += ['', 'Replay evidence scope: ' + report['replay']['evidence_scope'] + '.',
              'Replay certification concerns interval accounting only; it does not approve the strategy or prove trading performance.',
              '', '## Blockers and next actions', '']
    lines += [f"- {b['reason'].replace('_', ' ')} ({b.get('source', 'data/replay')}): {b['next_action']}" for b in blockers]
    lines += ['', '## Exact version-lock differences', '']
    lines += ['- ' + row['file'] + ': ' + str(row['expected_sha256']) + ' → ' + str(row['observed_sha256'])
              for row in lock_detail.get('changes', [])]
    lines += ['', '## Independently generated reviews', '']
    lines += [f"- {name}: {section['status']}; generated {section['generated_at']}; source commit {section['source_commit']}."
              for name, section in reviews.items()]
    atomic_write_text(args.output / 'evidence_report.md', '\n'.join(lines) + '\n')
    return report
