"""Check that paper trading can only restore a coherent research snapshot."""

from pathlib import Path

import yaml


def _steps(filename: str) -> list[dict]:
    """Read the GitHub job steps so the test checks the actual workflow files."""
    workflow = yaml.safe_load(Path('.github/workflows', filename).read_text(encoding='utf-8'))
    job_name = 'paper-trade' if filename == 'daily_paper_trading.yml' else 'refresh-factor-data'
    return workflow['jobs'][job_name]['steps']


def _named(steps: list[dict], name: str) -> dict:
    """Find one human-readable workflow step by its displayed name."""
    return next(step for step in steps if step.get('name') == name)


def test_daily_restores_exactly_the_validated_factor_bundle():
    factor = _steps('factor_data_refresh.yml')
    daily = _steps('daily_paper_trading.yml')
    saved = _named(factor, 'Save validated paper snapshot')
    restored = _named(daily, 'Restore validated factor snapshot')

    # PLAIN ENGLISH: GitHub's cache includes the listed paths in its identity.
    # If these differ, a successful refresh cannot warm the daily run.
    assert saved['with']['path'] == restored['with']['path']
    assert 'data/' in saved['with']['path']
    assert 'signals/research_run_manifest.json' in saved['with']['path']
    assert 'logs/core_satellite_execution_stress.json' in saved['with']['path']
    assert "steps.verify_robustness_evidence.outcome == 'success'" in saved['if']
    assert 'validated-factor-snapshot-v1-' in restored['with']['key']
    assert 'factor-data-parquets-' not in restored['with'].get('restore-keys', '')
    assert 'runtime-state-v4-' not in restored['with'].get('restore-keys', '')


def test_daily_freezes_prices_before_rebuilding_safety_reports():
    daily = _steps('daily_paper_trading.yml')
    names = [step.get('name') for step in daily]
    required_order = [
        'Require validated factor snapshot',
        'Validate restored factor data',
        'Refresh ETF reference prices for daily factor-decay',
        'Record final daily price fingerprint',
        'Refresh approved-config validation for daily snapshot',
        'Refresh execution stress for daily snapshot',
        'Refresh survivorship review for daily snapshot',
        'Refresh factor-decay evidence for daily cache',
        'Verify daily robustness and matching data',
        'Run daily paper trading (Alpaca only)',
    ]
    assert [names.index(name) for name in required_order] == sorted(names.index(name) for name in required_order)
    # A second ETF refresh inside daily_run.py would make these fingerprints
    # stale before signal generation, so the workflow uses its existing skip.
    assert '--skip-refresh' in _named(daily, 'Run daily paper trading (Alpaca only)')['run']
