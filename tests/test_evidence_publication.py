"""Run the real publication shell blocks against a disposable Git remote."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest
yaml = pytest.importorskip("yaml")


@pytest.mark.parametrize("filename,step_name,output", [
    ("daily_paper_trading.yml", "Commit signals to repo", "alpaca_paper_log.csv"),
    ("shadow_paper_journal.yml", "Commit shadow journal to repo", "shadow_paper_journal.csv"),
    ("post_market_execution_quality.yml", "Publish refreshed evidence", "alpaca_execution_scorecard.json"),
])
def test_publisher_preserves_remote_history_and_other_jobs(tmp_path, filename, step_name, output):
    # Windows Git includes Bash even when it is not on PATH.
    bash = shutil.which("bash")
    if os.name == "nt" and Path("C:/Program Files/Git/bin/bash.exe").exists():
        bash = "C:/Program Files/Git/bin/bash.exe"
    if not bash:
        pytest.skip("Git Bash required for workflow integration test")
    root = tmp_path / "runner"
    root.mkdir()
    remote = tmp_path / "remote.git"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()

    git("init", "--bare", str(remote))
    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (root / "README").write_text("main checkout")
    git("add", ".")
    git("commit", "-m", "main")
    git("remote", "add", "origin", str(remote))
    git("checkout", "-b", "signals/latest")
    (root / "signals").mkdir()
    (root / "signals" / "fractional_shadow_state.json").write_text("preserve me")
    (root / "signals" / output).write_text("old output")
    git("add", ".")
    git("commit", "-m", "previous evidence")
    prior = git("rev-parse", "HEAD")
    git("push", "origin", "signals/latest")
    git("checkout", "main")
    # These untracked generated files reproduce the checkout conflict that
    # previously caused the shadow fallback to replace the remote branch.
    (root / "signals").mkdir(exist_ok=True)
    (root / "logs").mkdir()
    (root / "signals" / output).write_text("new output")
    (root / "signals" / "paper_run_manifest.json").write_text('{"status":"complete","run_id":"test"}')
    (root / "logs" / "daily_run_20260905.json").write_text("{}")
    workflow = yaml.safe_load((Path(".github/workflows") / filename).read_text())
    step = next(step for job in workflow["jobs"].values() for step in job["steps"] if step["name"] == step_name)
    shell = re.sub(r"\$\{\{.*?\}\}", "123", step["run"]).replace("python3 -", "python -")
    result = subprocess.run([bash, "-e", "-o", "pipefail", "-c", shell], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert git("show", "origin/signals/latest:signals/fractional_shadow_state.json") == "preserve me"
    assert git("show", f"origin/signals/latest:signals/{output}") == "new output"
    git("merge-base", "--is-ancestor", prior, "origin/signals/latest")
    if filename == "daily_paper_trading.yml":
        assert git("show", "origin/signals/latest:logs/daily_run_20260905.json") == "{}"
