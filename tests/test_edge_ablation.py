"""Counterfactual tests keep comparisons narrow and future outcomes isolated."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from edge_ablation import variants, fit_variant, paired_interval, combine_returns, run_ablations
from causal_research import fit_features, score_features


def specification():
    return json.loads(Path('corrected_shadow_spec.json').read_text())


def test_variants_preserve_baseline_and_single_component_changes():
    spec = specification()
    before = deepcopy(spec)
    choices = variants(spec)
    assert spec == before
    assert choices['full'] == spec
    assert len(choices) == 7
    for name in ('momentum_only', 'without_momentum', 'without_volatility', 'without_dollar_volume'):
        assert choices[name]['configurations'] == spec['configurations']
        assert choices[name]['folds'] == spec['folds']
        assert choices[name]['policy'] == spec['policy']
    assert choices['without_momentum']['features'] == ['causal_volatility_20', 'causal_dollar_volume_20']
    neutral = choices['neutral_allocation']['configurations'][0]
    assert neutral['regime_mode'] == 'static'
    assert neutral['core_weights'] == spec['configurations'][0]['regime_preset']['neutral']['core_weights']
    assert choices['without_stock_overlay']['configurations'][0]['audit_remove_stock_overlay']
    assert not spec['configurations'][0].get('audit_remove_stock_overlay')


def training_panel():
    spec = specification()
    dates = pd.bdate_range('2020-01-01', periods=90)
    return pd.DataFrame([{'date': day, 'ticker': str(i), 'forward_return_20d': i / 100,
         'forward_return_20d_end_date': day + pd.Timedelta(days=28),
         **{f: i + j * .1 for j, f in enumerate(spec['features'])}} for day in dates for i in range(5)])


def test_disabled_ablation_matches_existing_fit_and_future_changes_do_not_leak():
    spec = specification()
    panel = training_panel()
    cutoff = pd.Timestamp('2020-03-30')
    baseline = fit_features(panel, spec['features'], cutoff=cutoff, label=spec['label'])
    assert fit_variant(panel, spec, 'full', cutoff) == baseline
    changed = panel.copy()
    changed.loc[changed.date > cutoff, spec['features'] + [spec['label']]] = -999
    for name, item in variants(spec).items():
        assert fit_variant(panel, item, name, cutoff) == fit_variant(changed, item, name, cutoff)
    removed = variants(spec)['without_momentum']
    assert not any('momentum' in f for f in fit_variant(panel, removed, 'without_momentum', cutoff).features)


def test_momentum_baseline_does_not_learn_negative_direction():
    spec = variants(specification())['momentum_only']
    panel = training_panel()
    panel['forward_return_20d'] *= -1
    artifact = fit_variant(panel, spec, 'momentum_only', '2020-03-30')
    assert artifact.directions == (1.,)
    scored = score_features(panel, artifact)
    assert scored.groupby('ticker').causal_score.mean().is_monotonic_increasing


def test_overlay_removal_keeps_scaled_core_and_cash(monkeypatch):
    import corrected_audit as audit
    import core_satellite_alpha as core
    date = pd.Timestamp('2024-02-01')
    panel = pd.DataFrame({'date': [date], 'ticker': ['A'], 'causal_score': [1.], 'sector': ['fixture']})
    bars = pd.DataFrame({'date': [date, date], 'ticker': ['A', 'QQQ'], 'Close': [100., 100.]})
    monkeypatch.setattr(audit, 'eligible_candidates', lambda frame, _: frame)
    monkeypatch.setattr(core, '_select_sticky_holdings', lambda *a, **k: panel)
    monkeypatch.setattr(core, '_sticky_overlay_weights', lambda *a, **k: pd.Series({'A': .5}))
    config = {'regime_mode': 'static', 'core_weights': {'QQQ': 1.}, 'core_gross': .75, 'overlay_gross': .5,
              'deployment_max_gross_exposure': 1.}
    full = audit.build_daily_targets(panel, config, None, bars=bars)(date, {}, 100000)
    removed = audit.build_daily_targets(panel, {**config, 'audit_remove_stock_overlay': True}, None, bars=bars)(date, {}, 100000)
    assert full == {'QQQ': .6, 'A': .4}
    assert removed == {'QQQ': .6}
    smaller = audit.build_daily_targets(panel, {**config, 'deployment_max_gross_exposure': .8}, None, bars=bars)(date, {}, 100000)
    assert sum(smaller.values()) == pytest.approx(.8)


def test_paired_uncertainty_requires_matched_sufficient_history():
    dates = pd.bdate_range('2020-01-01', periods=450)
    a, b = pd.Series(.002, index=dates), pd.Series(.001, index=dates)
    assert paired_interval(a, b, simulations=50)['status'] == 'useful_in_this_history'
    assert paired_interval(b, a, simulations=50)['status'] == 'harmful_in_this_history'
    assert paired_interval(a, a, simulations=50)['status'] == 'inconclusive'
    assert paired_interval(a.iloc[:20], b.iloc[:20])['status'] == 'inconclusive'
    assert paired_interval(a, b.iloc[:-1])['reason'] == 'unmatched_daily_returns'
    with pytest.raises(ValueError, match='overlap'):
        combine_returns([a, a])


def test_blocked_run_records_all_seven_attempts_without_evaluation(tmp_path, monkeypatch):
    import evidence_audit
    import corrected_audit
    monkeypatch.setattr(evidence_audit, 'input_report', lambda *a, **k: {'complete': False, 'gaps': [{'reason': 'missing_raw'}], 'next_action': 'supply raw'})
    monkeypatch.setattr(corrected_audit, 'load_corrected_inputs', lambda *a: pytest.fail('must not evaluate missing data'))
    args = SimpleNamespace(spec=None, output=tmp_path, data_dir=tmp_path, membership=tmp_path / 'members.csv', start='2020-01-01', end='2025-12-31')
    report = run_ablations(args)
    assert report['status'] == 'blocked'
    assert len(report['results']) == 7
    folder = Path(report['output']).parent
    assert len((folder / 'trials.jsonl').read_text().splitlines()) == 7
    assert (folder / 'comparison.csv').exists()
    assert not list(tmp_path.rglob('prospective_freeze.json'))


def test_end_to_end_seven_variants_use_real_daily_ledger(tmp_path, monkeypatch):
    import corrected_audit
    import evidence_audit
    from core_satellite_alpha import _nyse_sessions
    spec = specification()
    config = spec['configurations'][0]
    config.update(regime_mode='static', core_weights={'SPY': .5, 'QQQ': .5}, core_gross=.5,
                  overlay_gross=.5, earnings_blackout_days=0, max_per_sector=0)
    # Neutral allocation is identical here, providing a real no-change control.
    config['regime_preset']['neutral'] = {'core_weights': {'SPY': .5, 'QQQ': .5}, 'core_gross': .5, 'overlay_gross': .5}
    spec['folds'] = [{'train_end': '2020-05-29', 'start': '2020-06-01', 'end': '2020-06-05',
                     'inner': [{'train_end': '2020-04-30', 'start': '2020-05-01', 'end': '2020-05-05'}]}]
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec))
    dates = _nyse_sessions('2020-01-02', '2020-06-05')
    panel = pd.DataFrame([{'date': d, 'ticker': t, 'sector': 'fixture', 'forward_return_20d': i / 100,
        'forward_return_20d_end_date': d + pd.Timedelta(days=28),
        **{f: i + j * .1 for j, f in enumerate(spec['features'])}}
        for d in dates for i, t in enumerate(['A', 'B', 'C', 'D', 'E'])])
    bars = pd.DataFrame([{'date': d, 'ticker': t, 'Open': 100., 'High': 100., 'Low': 100.,
                         'Close': 100., 'Volume': 1000000.} for d in dates for t in ['SPY', 'QQQ', 'A', 'B', 'C', 'D', 'E']])
    membership = tmp_path / 'members.csv'
    pd.DataFrame([{'ticker': t, 'effective_from': '2019-01-01', 'effective_to': None, 'source': 'fixture', 'status': 'active'}
                  for t in ['A', 'B', 'C', 'D', 'E']]).to_csv(membership, index=False)
    monkeypatch.setattr(evidence_audit, 'input_report', lambda *a, **k: {'complete': True, 'gaps': [], 'next_action': 'none'})
    monkeypatch.setattr(corrected_audit, 'load_corrected_inputs', lambda *a: (panel, bars, pd.DataFrame(), {'adjustment_mode': 'raw_ohlcv', 'actions_verified': True}))
    args = SimpleNamespace(spec=spec_path, output=tmp_path, data_dir=tmp_path, membership=membership, start='2020-01-02', end='2020-06-05')
    report = run_ablations(args)
    assert report['status'] == 'complete', report['results']
    assert len(report['results']) == 7
    full = report['results']['full']
    assert full['combined']['total_return_pct'] < 0  # Constant prices still pay costs.
    assert len(full['folds'][0]['stress_checks']) == 3
    assert full['combined'] == report['results']['neutral_allocation']['combined']
    folder = Path(report['output']).parent
    assert len(list(folder.glob('*/trial_*/events.csv'))) == 35
    assert not list(folder.rglob('prospective_freeze.json'))
