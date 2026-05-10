from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

from settings import LOG_DIR


LOGS = Path(LOG_DIR)
REQUIRED_PACKAGES = {
    "yfinance": "==0.2.40",
    "websockets": ">=9,<11",
    "alpaca-trade-api": "==3.2.0",
    "moomoo-api": ">=10",
}


def _package_version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return ""


def _requirement_ok(dist_name: str, spec: str) -> dict:
    version = _package_version(dist_name)
    if not version:
        return {"package": dist_name, "required": spec, "installed": "", "ok": False, "issue": "missing"}
    req = Requirement(f"{dist_name}{spec}")
    ok = Version(version) in req.specifier
    return {
        "package": dist_name,
        "required": spec,
        "installed": version,
        "ok": bool(ok),
        "issue": "" if ok else "version_mismatch",
    }


def _pip_check() -> dict:
    result = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "returncode": result.returncode,
    }


def build_config_health(*, run_pip_check: bool = True) -> dict:
    package_checks = [_requirement_ok(name, spec) for name, spec in REQUIRED_PACKAGES.items()]
    env_checks = {
        "paper_mode_strategy": os.environ.get("PAPER_MODE_STRATEGY", ""),
        "paper_signal_timezone": os.environ.get("PAPER_SIGNAL_TIMEZONE", os.environ.get("TZ", "")),
        "moomoo_unlock_pwd_set": bool(os.environ.get("MOOMOO_UNLOCK_PWD", "")),
        "alpaca_api_key_set": bool(os.environ.get("ALPACA_API_KEY", "")),
    }
    pip = _pip_check() if run_pip_check else {"ok": True, "stdout": "skipped", "stderr": "", "returncode": 0}
    ok = all(item["ok"] for item in package_checks) and bool(pip["ok"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": bool(ok),
        "package_checks": package_checks,
        "pip_check": pip,
        "env_checks": env_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local paper-trading config and dependency health.")
    parser.add_argument("--skip-pip-check", action="store_true", help="Skip python -m pip check.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    report = build_config_health(run_pip_check=not bool(args.skip_pip_check))
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / "config_health.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Config Health")
        print("-" * 72)
        print(f"OK: {report['ok']}")
        for row in report["package_checks"]:
            print(f"{row['package']:18s} installed={row['installed'] or 'missing'} required={row['required']} ok={row['ok']}")
        print(f"pip check ok={report['pip_check']['ok']}")
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
