"""Exercise the real workflow recovery branch without downloading market data."""
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize('fresh,quality', [(True, True), (False, True), (True, False)])
def test_recovery_downloads_only_when_price_or_quality_inputs_need_it(tmp_path, fresh, quality):
    bash = shutil.which('bash')
    if not bash:
        pytest.skip('Bash required to exercise workflow shell')
    workflow = yaml.safe_load(Path('.github/workflows/daily_paper_trading.yml').read_text())
    step = next(s for s in workflow['jobs']['paper-trade']['steps'] if s['name'] == 'Rebuild factor cache if restore is incomplete')
    (tmp_path / 'signals').mkdir()
    (tmp_path / 'logs').mkdir()
    # Report generation is replaced by fixture inputs; branch selection remains
    # the exact shell and embedded Python that GitHub executes.
    (tmp_path / 'factor_data_health.py').write_text(
        'from pathlib import Path\n'
        'def build_factor_data_health():\n'
        f"    return {{'factor_data_fresh': {fresh}, 'feature_quality': {{'ready': {quality}}}}}\n"
        "if __name__ == '__main__': raise SystemExit(0 if Path('profile_refreshed').exists() else 1)\n"
    )
    (tmp_path / 'feature_health.py').write_text("from pathlib import Path\nPath('profile_refreshed').touch()\n")
    (tmp_path / 'research.py').write_text("from pathlib import Path\nPath('research_requested').touch()\nraise SystemExit(7)\n")
    shell = step['run'].replace('python3', shlex.quote(sys.executable))
    result = subprocess.run([bash, '-e', '-c', shell], cwd=tmp_path, capture_output=True, text=True)
    narrow = fresh and quality
    assert (result.returncode == 0) == narrow, result.stdout + result.stderr
    assert (tmp_path / 'profile_refreshed').exists() == narrow
    assert (tmp_path / 'research_requested').exists() != narrow
