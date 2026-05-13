import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR_PATH = REPO_ROOT / "Stock picking scripts" / "monitor.py"


def load_monitor_module(name: str = "monitor_under_test"):
    spec = importlib.util.spec_from_file_location(name, MONITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def configure_monitor_paths(monitor, tmp_path):
    monitor._ALERT_LOG = str(tmp_path / "alerts.jsonl")
    monitor._DEDUP_FILE = str(tmp_path / "alert_dedup.json")
    monitor._DEDUP_LOCK_FILE = str(tmp_path / "alert_dedup.json.lock")
    monitor._DRIFT_LOG = str(tmp_path / "drift.jsonl")
    monitor._send_slack = lambda title, body, severity: False
    monitor._send_email = lambda title, body, severity: False


def read_alerts(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_send_alert_dedup_is_process_safe(tmp_path):
    alert_log = tmp_path / "alerts.jsonl"
    dedup_file = tmp_path / "alert_dedup.json"
    lock_file = tmp_path / "alert_dedup.json.lock"
    gate_file = tmp_path / "go"

    child_code = """
import importlib.util
import sys
import time
from pathlib import Path

repo_root, monitor_path, alert_log, dedup_file, lock_file, gate_file = sys.argv[1:]
sys.path.insert(0, repo_root)
spec = importlib.util.spec_from_file_location("monitor_child", monitor_path)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor._ALERT_LOG = alert_log
monitor._DEDUP_FILE = dedup_file
monitor._DEDUP_LOCK_FILE = lock_file
monitor._send_slack = lambda title, body, severity: False
monitor._send_email = lambda title, body, severity: False

gate = Path(gate_file)
deadline = time.time() + 10
while not gate.exists():
    if time.time() > deadline:
        raise SystemExit("timed out waiting for gate")
    time.sleep(0.01)

monitor.send_alert("Race alert", "body", severity=monitor.WARNING, alert_key="race:key")
"""

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                child_code,
                str(REPO_ROOT),
                str(MONITOR_PATH),
                str(alert_log),
                str(dedup_file),
                str(lock_file),
                str(gate_file),
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]

    gate_file.write_text("go")

    for proc in procs:
        stdout, stderr = proc.communicate(timeout=15)
        assert proc.returncode == 0, f"stdout={stdout}\nstderr={stderr}"

    alerts = read_alerts(alert_log)
    assert len(alerts) == 1
    assert alerts[0]["key"] == "race:key"

    cache = json.loads(dedup_file.read_text())
    assert set(cache) == {"race:key"}


def test_check_drift_handles_empty_features(tmp_path):
    monitor = load_monitor_module("monitor_empty_features")
    configure_monitor_paths(monitor, tmp_path)
    Path(monitor._DRIFT_LOG).write_text(
        json.dumps({"status": "drift", "run_at": "2026-05-13T09:30:00Z", "features": {}}) + "\n"
    )

    result = monitor.check_drift()

    assert result["fired"] is True
    assert "Worst feature: unavailable" in result["message"]
    alerts = read_alerts(Path(monitor._ALERT_LOG))
    assert len(alerts) == 1


def test_check_drift_handles_none_features(tmp_path):
    monitor = load_monitor_module("monitor_none_features")
    configure_monitor_paths(monitor, tmp_path)
    Path(monitor._DRIFT_LOG).write_text(
        json.dumps({"status": "drift", "run_at": "2026-05-13T09:30:00Z", "features": None}) + "\n"
    )

    result = monitor.check_drift()

    assert result["fired"] is True
    assert "Worst feature: unavailable" in result["message"]


def test_check_drift_uses_highest_psi_feature(tmp_path):
    monitor = load_monitor_module("monitor_normal_features")
    configure_monitor_paths(monitor, tmp_path)
    Path(monitor._DRIFT_LOG).write_text(
        json.dumps(
            {
                "status": "caution",
                "run_at": "2026-05-13T09:30:00Z",
                "features": {
                    "volume": {"psi": 0.12, "ks_stat": 0.03},
                    "momentum": {"psi": 0.41, "ks_stat": 0.08},
                    "malformed": "ignored",
                },
            }
        )
        + "\n"
    )

    result = monitor.check_drift()

    assert result["fired"] is True
    assert "Worst feature: momentum" in result["message"]
    assert "PSI=0.410" in result["message"]
    assert "KS=0.080" in result["message"]
