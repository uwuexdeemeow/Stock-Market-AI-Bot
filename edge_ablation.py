"""Seven fixed, offline comparisons of an unapproved corrected shadow candidate."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from causal_research import FoldArtifact, fit_features, score_features, fingerprint
from portfolio_ledger import simulate_daily, daily_metrics
from paper_policy import PaperPolicy
from safe_io import atomic_write_json, atomic_write_csv, atomic_write_text


def variants(spec):
    """Change just one component; never select a winner from a parameter grid."""
    if len(spec.get('configurations', [])) != 1:
        raise ValueError('Ablations require exactly one frozen candidate configuration')
    required = {'causal_momentum_20', 'causal_momentum_60', 'causal_volatility_20', 'causal_dollar_volume_20'}
    if set(spec['features']) != required:
        raise ValueError('Seven-comparison protocol requires the existing four raw features')
    result = {}
    for name in ('full', 'momentum_only', 'without_momentum', 'without_volatility',
                 'without_dollar_volume', 'neutral_allocation', 'without_stock_overlay'):
        item = deepcopy(spec)
        if name == 'momentum_only':
            item['features'] = ['causal_momentum_60']
        elif name.startswith('without_') and name != 'without_stock_overlay':
            token = name.removeprefix('without_')
            item['features'] = [f for f in spec['features'] if token not in f]
        config = item['configurations'][0]
        if name == 'neutral_allocation':
            neutral = config['regime_preset']['neutral']
            config.update(deepcopy(neutral), regime_mode='static')
        if name == 'without_stock_overlay':
            config['audit_remove_stock_overlay'] = True
        result[name] = item
    return result


def fit_variant(panel, spec, name, cutoff):
    """The simple momentum baseline ranks upward; learned variants refit locally."""
    if name != 'momentum_only':
        return fit_features(panel, spec['features'], cutoff=cutoff, label=spec['label'])
    # A plain positive momentum rank has no learned direction, weights or labels.
    # Its identity still records the training-side inputs used by this run.
    training = panel.loc[pd.to_datetime(panel.date) <= pd.Timestamp(cutoff), ['date', 'ticker', 'causal_momentum_60']]
    return FoldArtifact(str(cutoff), spec['label'], ('causal_momentum_60',), (1.,), (1.,),
                        fingerprint(pd.util.hash_pandas_object(training, index=False).tolist()))


def paired_interval(left, right, *, block=20, simulations=2000):
    """Paired blocks estimate annualized mean return differences, not causality."""
    from edge_evidence import _block_samples
    paired = pd.concat([left.rename('left'), right.rename('right')], axis=1)
    if paired.empty or paired.isna().any().any() or not np.isfinite(paired).all().all():
        return {'status': 'inconclusive', 'reason': 'unmatched_daily_returns', 'ci95_annualized_mean_difference_pct': None}
    diff = (paired.left - paired.right).to_numpy()
    out = {'status': 'inconclusive', 'mean_difference_annualized_pct': float(diff.mean() * 25200),
           'paired_sessions': len(diff), 'block_sessions': block,
           'ci95_annualized_mean_difference_pct': None, 'interpretation': 'retrospective_unadjusted_diagnostic'}
    if len(diff) < 20 * block:
        out['reason'] = 'fewer_than_20_blocks'
        return out
    rng = np.random.default_rng(20260907)
    samples = [float(diff[_block_samples(len(diff), block, rng)].mean() * 25200) for _ in range(simulations)]
    ci = np.quantile(samples, [.025, .975]).tolist()
    out['ci95_annualized_mean_difference_pct'] = ci
    out['status'] = 'useful_in_this_history' if ci[0] > 0 else 'harmful_in_this_history' if ci[1] < 0 else 'inconclusive'
    return out


def combine_returns(series):
    """Join disjoint, liquidated folds; reject overlap rather than drop dates."""
    returns = pd.concat(series).sort_index()
    if returns.index.duplicated().any() or returns.isna().any():
        raise ValueError('Fold returns overlap or contain missing observations')
    return returns


def return_metrics(returns):
    """Retain the first day's return by adding an explicit starting value."""
    from core_satellite_alpha import _session_offset
    equity = (1 + returns).cumprod()
    equity = pd.concat([pd.Series([1.], index=[_session_offset(returns.index[0], -1)]), equity])
    return daily_metrics(equity)


def run_ablations(args):
    """Save blocked attempts too; never borrow stale historical result caches."""
    from corrected_audit import load_corrected_inputs, evaluate_corrected, code_fingerprints
    from evidence_audit import input_report
    spec_path = args.spec or Path(__file__).with_name('corrected_shadow_spec.json')
    spec = json.loads(spec_path.read_text())
    choices = variants(spec)
    run = args.output / 'ablations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True, exist_ok=False)
    data = input_report(args.data_dir, args.membership, spec, start=args.start, end=args.end)
    report = {'status': 'blocked', 'output': str(run / 'comparison.json'), 'data': data,
              'candidate_identity': fingerprint(spec), 'code': {**code_fingerprints(),
                  **{n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('edge_ablation.py', 'experiment_ledger.py')}},
              'deployed_strategy_equivalent': False, 'automatic_cutover': False,
              'historical_interpretation': 'retrospective_diagnostic',
              'fold_accounting': 'Each fold starts in cash and liquidates; combined returns reinvest across folds.',
              'results': {}, 'attempted_variants': list(choices)}
    def record(row):
        # The existing experiment ledger's append-only principle is preserved
        # in a dedicated trial journal, so old research records stay untouched.
        with (run / 'trials.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, default=str) + '\n')
    if data['complete']:
        try:
            panel, bars, actions, provenance = load_corrected_inputs(args, spec)
            report['input_identity'] = fingerprint({'bars': pd.util.hash_pandas_object(bars, index=False).tolist(),
                'panel': pd.util.hash_pandas_object(panel, index=False).tolist(), 'actions': actions.to_dict('records'),
                'membership': args.membership.read_text()})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            data['complete'] = False
            data['gaps'].append({'reason': 'verified_input_load_failed', 'error_type': type(exc).__name__})
    all_returns = {}
    fold_returns = {}
    for name, candidate in choices.items():
        if not data['complete']:
            record({'variant': name, 'status': 'blocked', 'gaps': data['gaps']})
            report['results'][name] = {'status': 'blocked', 'reason': 'verified_inputs_required'}
            continue
        config = candidate['configurations'][0]
        policy = PaperPolicy(**candidate.get('policy', {}))
        if float(config.get('cost_stress', 1)) != 1 or policy.maximum_gross != float(config.get('deployment_max_gross_exposure', 1)):
            raise ValueError('Baseline costs and matching policy/deployment gross required')
        folds, returns, references = [], [], {'SPY': [], 'QQQ': []}
        stressed_returns = {stress: [] for stress in (2., 3., 5.)}
        stressed_references = {stress: {'SPY': [], 'QQQ': []} for stress in stressed_returns}
        trial = 0
        try:
            previous_end = None
            for fold in candidate['folds']:
                if not pd.Timestamp(fold['train_end']) < pd.Timestamp(fold['start']) <= pd.Timestamp(fold['end']):
                    raise ValueError('Invalid outer training boundary')
                if previous_end is not None and pd.Timestamp(fold['start']) <= previous_end:
                    raise ValueError('Outer folds must be ordered and disjoint')
                previous_end = pd.Timestamp(fold['end'])
                if not fold.get('inner'):
                    raise ValueError('Inner validation is required')
                windows = [(inner, 'inner') for inner in fold['inner']] + [(fold, 'outer')]
                outer = None
                for window, kind in windows:
                    if not pd.Timestamp(window['train_end']) < pd.Timestamp(window['start']) <= pd.Timestamp(window['end']):
                        raise ValueError('Invalid training boundary')
                    if kind == 'inner' and pd.Timestamp(window['end']) > pd.Timestamp(fold['train_end']):
                        raise ValueError('Inner evaluation crosses outer training boundary')
                    artifact = fit_variant(panel, candidate, name, window['train_end'])
                    scored = score_features(panel.loc[pd.to_datetime(panel.date) <= pd.Timestamp(window['end'])], artifact)
                    stresses = [1.] if kind == 'inner' else [1., 2., 3., 5.]
                    for stress in stresses:
                        trial += 1
                        trial_config = {**config, 'cost_stress': stress}
                        record({'variant': name, 'trial': trial, 'status': 'started', 'window': window,
                                'kind': kind, 'stress': stress, 'artifact': asdict(artifact), 'config': trial_config})
                        result = evaluate_corrected(scored, trial_config, start=window['start'], end=window['end'],
                            bars=bars, actions=actions, provenance=provenance, membership_path=args.membership, policy=policy)
                        folder = run / name / f'trial_{trial:04d}'
                        folder.mkdir(parents=True)
                        atomic_write_csv(result.events, folder / 'events.csv')
                        atomic_write_csv(result.holdings, folder / 'holdings.csv')
                        atomic_write_csv(result.equity.reset_index(), folder / 'daily_equity.csv')
                        metrics = dict(result.metrics)
                        paired_returns = {}
                        for ticker in ('SPY', 'QQQ'):
                            # Passive benchmarks bear the same fill costs and cash
                            # reserve, with no strategy-specific stop or drawdown halt.
                            bench = simulate_daily(bars, lambda date, shares, equity, t=ticker: {t: policy.maximum_gross},
                                start=window['start'], end=window['end'], actions=actions, provenance=provenance,
                                policy=replace(policy, trailing_stop=1., drawdown_halt=1.),
                                base_slippage_pct=metrics['cost_calibration']['base_slippage_pct'], cost_stress=stress)
                            atomic_write_csv(bench.equity.reset_index(), folder / f'{ticker}_equity.csv')
                            atomic_write_csv(bench.events, folder / f'{ticker}_events.csv')
                            paired_returns[ticker] = bench.equity.equity.pct_change(fill_method=None).dropna()
                            metrics[f'excess_vs_{ticker}_pct'] = metrics['total_return_pct'] - bench.metrics['total_return_pct']
                            metrics[f'paired_vs_{ticker}'] = paired_interval(result.equity.equity.pct_change().dropna(), paired_returns[ticker])
                        atomic_write_json({'metrics': metrics, 'artifact': asdict(artifact), 'window': window, 'stress': stress}, folder / 'metrics.json')
                        record({'variant': name, 'trial': trial, 'status': 'complete', 'kind': kind, 'stress': stress, 'metrics': metrics})
                        if kind == 'outer':
                            if stress == 1:
                                outer = {'fold': window, 'metrics': metrics, 'stress_checks': []}
                                returns.append(result.equity.equity.pct_change(fill_method=None).dropna())
                                for ticker in references:
                                    references[ticker].append(paired_returns[ticker])
                            else:
                                outer['stress_checks'].append({'stress': stress, 'metrics': metrics})
                                stressed_returns[stress].append(result.equity.equity.pct_change(fill_method=None).dropna())
                                for ticker in references:
                                    stressed_references[stress][ticker].append(paired_returns[ticker])
                folds.append(outer)
            joined = combine_returns(returns)
            combined = return_metrics(joined)
            combined['sum_fold_turnover_pct'] = sum(f['metrics']['turnover_pct'] for f in folds)
            for ticker, values in references.items():
                benchmark = combine_returns(values)
                combined[f'excess_vs_{ticker}_pct'] = combined['total_return_pct'] - return_metrics(benchmark)['total_return_pct']
                combined[f'paired_vs_{ticker}'] = paired_interval(joined, benchmark)
            all_returns[name] = joined
            fold_returns[name] = returns
            atomic_write_csv(joined.rename('net_return').reset_index(), run / name / 'combined_returns.csv')
            combined_stresses = {}
            for stress, pieces in stressed_returns.items():
                stressed_series = combine_returns(pieces)
                stress_metrics = return_metrics(stressed_series)
                for ticker in references:
                    reference = combine_returns(stressed_references[stress][ticker])
                    stress_metrics[f'excess_vs_{ticker}_pct'] = stress_metrics['total_return_pct'] - return_metrics(reference)['total_return_pct']
                    stress_metrics[f'paired_vs_{ticker}'] = paired_interval(stressed_series, reference)
                stress_metrics['sum_fold_turnover_pct'] = sum(next(s['metrics']['turnover_pct'] for s in f['stress_checks'] if s['stress'] == stress) for f in folds)
                combined_stresses[str(stress)] = stress_metrics
            report['results'][name] = {'status': 'complete', 'folds': folds, 'combined': combined, 'combined_stresses': combined_stresses}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record({'variant': name, 'status': 'blocked', 'error_type': type(exc).__name__, 'reason': str(exc)})
            report['results'][name] = {'status': 'blocked', 'reason': str(exc)}
    if 'full' in all_returns:
        for name, returns in all_returns.items():
            # Positive full-minus-removed means the removed component helped.
            report['results'][name]['full_minus_variant'] = paired_interval(all_returns['full'], returns)
            for index, fold in enumerate(report['results'][name]['folds']):
                fold['full_minus_variant'] = paired_interval(fold_returns['full'][index], fold_returns[name][index])
    report['status'] = 'complete' if all(r['status'] == 'complete' for r in report['results'].values()) else 'blocked'
    from experiment_ledger import append_experiment
    for name, result in report['results'].items():
        append_experiment('corrected_ablation_' + name, choices[name],
                          {'status': result['status'], **result.get('combined', {})},
                          artifacts={'comparison': str(run / 'comparison.json')},
                          notes='Retrospective shadow diagnostic; no automatic promotion', output_dir=str(run))
    atomic_write_json(report, run / 'comparison.json')
    rows = []
    for name, result in report['results'].items():
        values = result.get('combined', {})
        rows.append({'variant': name, 'status': result['status'], **{k: values.get(k) for k in
            ('total_return_pct', 'sharpe', 'max_drawdown_pct', 'excess_vs_SPY_pct', 'excess_vs_QQQ_pct', 'sum_fold_turnover_pct')},
            'component_verdict': result.get('full_minus_variant', {}).get('status', 'inconclusive')})
    atomic_write_csv(pd.DataFrame(rows), run / 'comparison.csv')
    lines = ['# Corrected shadow edge attribution', '', 'Status: ' + report['status'], '',
             'Retrospective diagnostics; no deployed-strategy equivalence or promotion. Confidence intervals are not adjusted for multiple comparisons.', '',
             '| Variant | Status | Component verdict |', '| --- | --- | --- |']
    lines += [f"| {r['variant']} | {r['status']} | {r['component_verdict']} |" for r in rows]
    if data['gaps']:
        lines += ['', 'Required next action: ' + data['next_action']]
        lines += ['- ' + g['reason'] for g in data['gaps']]
    atomic_write_text(run / 'comparison.md', '\n'.join(lines) + '\n')
    return report
