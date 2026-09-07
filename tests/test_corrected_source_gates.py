"""Counterexamples for historical evidence gates; all source files are synthetic."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from corrected_data import action_gaps, raw_price_gaps, validate_context_coverage, validate_sources


def test_price_hash_cannot_hide_missing_sessions_or_bad_prices(tmp_path):
    path = tmp_path / 'SPY.parquet'
    sessions = pd.DatetimeIndex(['2024-02-01', '2024-02-02', '2024-02-05'])
    frame = pd.DataFrame({'Open': 100., 'High': 101., 'Low': 99., 'Close': 100., 'Volume': 0.}, index=sessions)
    frame.to_parquet(path)
    assert raw_price_gaps(path, 'SPY', sessions) == []
    frame.iloc[[0, 2]].to_parquet(path)
    gaps = raw_price_gaps(path, 'SPY', sessions)
    assert gaps == [{'reason': 'raw_price_sessions_missing', 'ticker': 'SPY', 'count': 1,
                     'first': '2024-02-02', 'last': '2024-02-02'}]
    frame.loc[sessions[1], 'Close'] = np.inf
    frame.to_parquet(path)
    assert raw_price_gaps(path, 'SPY', sessions)[0]['reason'] == 'raw_price_content_invalid'


@pytest.mark.parametrize('change', [
    {'kind': 'spin_off'}, {'kind': 'split', 'value': 0}, {'value': float('nan')},
    {'kind': 'symbol_change'}, {'date': 'bad'}, {'event_id': ''},
])
def test_unsupported_or_invalid_action_cannot_reach_ledger(change):
    row = {'event_id': 'a', 'ticker': 'ABC', 'kind': 'split', 'date': '2024-02-02',
           'value': 2., 'source': 'fixture', **change}
    assert action_gaps(pd.DataFrame([row]))


def test_dividend_payment_cannot_precede_entitlement():
    row = {'event_id': 'a', 'ticker': 'ABC', 'kind': 'dividend', 'date': '2024-02-02',
           'ex_date': '2024-02-05', 'value': 1., 'source': 'fixture'}
    assert action_gaps(pd.DataFrame([row]))[0]['reason'] == 'dividend_entitlement_dates_missing'
    row['date'] = '2024-02-06'
    assert not action_gaps(pd.DataFrame([row]))


def test_context_cannot_use_placeholder_or_partial_history():
    dates = pd.to_datetime(['2024-02-01', '2024-02-02'])
    expected = pd.DataFrame({'date': dates, 'ticker': 'ABC'})
    config = [{'max_per_sector': 2, 'earnings_blackout_days': 5}]
    with pytest.raises(ValueError, match='fields missing'):
        validate_context_coverage(None, expected.assign(sector='OTHER'), config)
    context = expected.assign(sector='Technology', days_to_next_earnings=10.,
                              published_at='2024-01-31T00:00Z', source_url='https://example.test', access_cost='free')
    validate_context_coverage(context, expected, config)
    with pytest.raises(ValueError, match='incomplete'):
        validate_context_coverage(context.iloc[:1], expected, config)
    with pytest.raises(ValueError, match='empty'):
        validate_context_coverage(context.assign(sector=' '), expected, config)
    with pytest.raises(ValueError, match='Invalid'):
        validate_context_coverage(context.assign(days_to_next_earnings=np.inf), expected, config)


def test_bad_manifest_still_delivers_individual_gaps(tmp_path):
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw/manifest.json').write_text('[]')
    report = validate_sources(tmp_path, tmp_path / 'membership.csv', start='2024-02-01', end='2024-02-05')
    reasons = {g['reason'] for g in report['gaps']}
    assert {'source_manifest_unreadable', 'membership_coverage_unverified', 'raw_price_file_missing'} <= reasons


def test_reconstruction_and_overlapping_membership_cannot_be_certified(tmp_path):
    (tmp_path / 'raw').mkdir()
    path = tmp_path / 'membership.csv'
    rows = pd.DataFrame([{'ticker': 'ABC', 'effective_from': '2024-01-01', 'effective_to': '2024-02-02',
                          'status': 'removed', 'source': 'candidate', 'source_url': 'https://example.test',
                          'retrieved_at': '2024-03-01T00:00Z', 'license': 'fixture', 'access_cost': 'free'}] * 2)
    rows.to_csv(path, index=False)
    (tmp_path / 'raw/manifest.json').write_text(json.dumps({'membership': {
        'verified': 'true', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}))
    report = validate_sources(tmp_path, path, start='2024-02-01', end='2024-02-05')
    assert {'membership_coverage_unverified', 'membership_interval_invalid'} <= {g['reason'] for g in report['gaps']}


def test_valid_contract_passes_but_one_missing_bar_blocks(tmp_path, monkeypatch):
    import corrected_data
    monkeypatch.setattr(corrected_data, 'membership_status', lambda *a, **k: {'complete': True, 'reasons': []})
    raw = tmp_path / 'raw'
    raw.mkdir()
    member = tmp_path / 'members.csv'
    pd.DataFrame([{'ticker': 'ABC', 'effective_from': '2024-02-01', 'effective_to': '2024-02-05',
                   'source': 'fixture', 'status': 'active'}]).to_csv(member, index=False)
    proof = {'source_url': 'https://example.test', 'retrieved_at': '2024-02-06T00:00Z',
             'license': 'fixture', 'access_cost': 'free', 'verified': True,
             'start': '2024-02-01', 'end': '2024-02-05'}
    actions = raw / 'actions.csv'
    pd.DataFrame(columns=['event_id', 'ticker', 'kind', 'date', 'value', 'source']).to_csv(actions, index=False)
    manifest = {'membership': {**proof, 'sha256': hashlib.sha256(member.read_bytes()).hexdigest()},
                'actions_sha256': hashlib.sha256(actions.read_bytes()).hexdigest(), 'symbols': {}}
    dates = pd.to_datetime(['2024-02-01', '2024-02-02', '2024-02-05'])
    frame = pd.DataFrame({'Open': 100., 'High': 101., 'Low': 99., 'Close': 100., 'Volume': 10.}, index=dates)
    for ticker in ['ABC', 'SPY', 'QQQ']:
        path = raw / (ticker + '.parquet')
        frame.to_parquet(path)
        manifest['symbols'][ticker] = {**proof, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'adjustment_mode': 'raw_ohlcv', 'feed': 'fixture', 'symbol_mapping': 'none',
                                      'actions_coverage': proof, 'identity': {**proof, 'security_id': ticker,
                                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    (raw / 'manifest.json').write_text(json.dumps(manifest))
    assert validate_sources(tmp_path, member, start='2024-02-01', end='2024-02-05')['complete']
    frame.iloc[:1].to_parquet(raw / 'ABC.parquet')
    # Updating the hash does not repair the missing sessions.
    manifest['symbols']['ABC']['sha256'] = hashlib.sha256((raw / 'ABC.parquet').read_bytes()).hexdigest()
    (raw / 'manifest.json').write_text(json.dumps(manifest))
    result = validate_sources(tmp_path, member, start='2024-02-01', end='2024-02-05')
    assert not result['complete']
    assert next(g for g in result['gaps'] if g['reason'] == 'raw_price_sessions_missing')['count'] == 2


def test_invalid_membership_end_date_is_not_open_ended(tmp_path):
    from universe_membership import load_membership
    path = tmp_path / 'members.csv'
    pd.DataFrame([{'ticker': 'ABC', 'effective_from': '2024-01-01', 'effective_to': 'typo',
                   'status': 'removed', 'source': 'fixture'}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='effective_to'):
        load_membership(path)
