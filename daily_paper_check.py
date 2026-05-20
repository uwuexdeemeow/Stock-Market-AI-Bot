"""
daily_paper_check.py - Safe Moomoo paper-trading readiness check.

This command does not submit orders. It refreshes read-only broker status,
optionally reconciles fills, rebuilds paper health, runs the gauntlet, and
prints one practical verdict for the current paper setup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from safe_io import run_utf8
from settings import LOG_DIR, SIGNAL_DIR


ROOT = Path(__file__).resolve().parent
SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)
CHECK_PATH = SIGNALS / "daily_paper_check.json"
SNAPSHOT_FILES = {
    "signal": SIGNALS / "core_satellite_alpha_signal.csv",
    "orders": SIGNALS / "core_satellite_alpha_orders.csv",
    "status": SIGNALS / "paper_daily_status.json",
    "health": SIGNALS / "paper_health.json",
    "trades": SIGNALS / "paper_trades.csv",
    "reconciliation": SIGNALS / "paper_reconciliation.csv",
    "equity": SIGNALS / "paper_equity.csv",
}


def _run_step(name: str, cmd: list[str], *, timeout: int) -> dict:
    print(f"\n[{name}] {' '.join(cmd)}")
    started = datetime.now(timezone.utc)
    try:
        result = run_utf8(
            cmd,
            cwd=ROOT,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        print(f"  timeout after {elapsed:.1f}s")
        return {
            "name": name,
            "status": "timeout",
            "elapsed_seconds": round(elapsed, 1),
            "cmd": cmd,
            "stderr_tail": str(exc)[-1000:],
        }

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    stdout_tail = "\n".join(result.stdout.strip().splitlines()[-18:]) if result.stdout else ""
    stderr_tail = "\n".join(result.stderr.strip().splitlines()[-10:]) if result.stderr else ""
    if stdout_tail:
        for line in stdout_tail.splitlines():
            print(f"  {line}")
    if result.returncode == 0:
        print(f"  ok ({elapsed:.1f}s)")
        status = "ok"
    else:
        print(f"  failed exit={result.returncode} ({elapsed:.1f}s)")
        if stderr_tail:
            for line in stderr_tail.splitlines():
                print(f"  err: {line}")
        status = "failed"
    return {
        "name": name,
        "status": status,
        "exit_code": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "cmd": cmd,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _file_digest(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_artifacts(run_id: str) -> dict:
    snapshot_dir = LOGS / "paper_snapshots" / run_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for label, source in SNAPSHOT_FILES.items():
        source = Path(source)
        entry = {
            "source": str(source),
            "exists": source.exists(),
            "snapshot": "",
            "sha256": "",
            "bytes": 0,
        }
        if source.exists() and source.is_file():
            dest = snapshot_dir / source.name
            shutil.copy2(source, dest)
            entry["snapshot"] = str(dest)
            entry["sha256"] = _file_digest(dest)
            entry["bytes"] = int(dest.stat().st_size)
        manifest[label] = entry
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"dir": str(snapshot_dir), "manifest": str(manifest_path), "files": manifest}


def _prune_snapshots(*, keep_days: int, now: datetime | None = None) -> dict:
    root = LOGS / "paper_snapshots"
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=int(keep_days))
    removed: list[str] = []
    kept: list[str] = []
    if not root.exists():
        return {"root": str(root), "keep_days": int(keep_days), "removed": removed, "kept": kept}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        try:
            stamp = datetime.strptime(path.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            kept.append(str(path))
            continue
        if stamp < cutoff:
            shutil.rmtree(path)
            removed.append(str(path))
        else:
            kept.append(str(path))
    return {"root": str(root), "keep_days": int(keep_days), "removed": removed, "kept": kept}


def _blocker_summary(health: dict) -> dict:
    flags = health.get("readiness_flags", {}) or {}
    slippage = health.get("slippage", {}) or {}
    scorecard = health.get("go_live_scorecard", {}) or {}
    return {
        "strategy_ready": flags.get("strategy_ready", False),
        "signal_fresh": flags.get("signal_fresh", False),
        "broker_synced": flags.get("broker_synced", False),
        "open_orders": int(health.get("current_signal_open_orders", 0) or 0),
        "stale_open_orders": int(len(health.get("stale_open_order_alerts", []) or [])),
        "account_aligned": flags.get("account_aligned", False),
        "paper_equity_days": int(health.get("paper_equity_days", 0) or 0),
        "fill_rate": float(health.get("fill_rate", 0.0) or 0.0),
        "cancel_rate": float(health.get("cancel_rate", 0.0) or 0.0),
        "max_drift_abs": health.get("max_drift_abs"),
        "avg_slippage_bps": slippage.get("avg_slippage_bps"),
        "drawdown_ok": flags.get("drawdown_ok", True),
        "concentration_ok": flags.get("concentration_ok", True),
        "approved_for_real_capital": bool(health.get("approved_for_real_capital", False)),
        "go_live_score": f"{scorecard.get('passed', 0)}/{scorecard.get('total', 0)}",
        "gauntlet_reason": health.get("gauntlet_reason", ""),
    }


def _verdict(health: dict, step_results: list[dict]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    failed_steps = [step["name"] for step in step_results if step.get("status") not in {"ok", "skipped"}]
    if failed_steps:
        reasons.append(f"failed_steps={','.join(failed_steps)}")

    flags = health.get("readiness_flags", {}) or {}
    if not flags.get("strategy_ready", False):
        reasons.append("strategy_not_ready")
    if not flags.get("signal_fresh", False):
        reasons.append("signal_not_fresh")
    if not flags.get("broker_synced", False):
        reasons.append("broker_not_synced_to_current_signal")
    open_orders = int(health.get("current_signal_open_orders", 0) or 0)
    if open_orders > 0:
        reasons.append(f"current_signal_open_orders={open_orders}")
    if not flags.get("account_aligned", False):
        reasons.append("account_not_aligned_to_target")
    if not flags.get("slippage_ok", True):
        reasons.append("slippage_over_threshold")
    if not flags.get("drawdown_ok", True):
        reasons.append("paper_drawdown_over_threshold")
    if not flags.get("concentration_ok", True):
        reasons.append("concentration_over_threshold")

    if failed_steps:
        return "BROKER_CHECK_FAILED", reasons
    if not flags.get("strategy_ready", False) or not flags.get("signal_fresh", False):
        return "FIX_SIGNAL_OR_DATA", reasons
    if not flags.get("slippage_ok", True) or not flags.get("drawdown_ok", True) or not flags.get("concentration_ok", True):
        return "RISK_REVIEW", reasons
    if not flags.get("broker_synced", False):
        return "SYNC_OR_WAIT_FOR_FILLS", reasons
    if open_orders > 0:
        return "WAIT_FOR_OPEN_ORDERS", reasons
    if not flags.get("account_aligned", False):
        return "REBALANCE_PENDING", reasons
    if bool(health.get("approved_for_real_capital", False)):
        return "PAPER_GAUNTLET_APPROVED", reasons or ["paper_gauntlet_passed"]
    return "CONTINUE_PAPER", reasons or ["paper_checks_ok_but_gauntlet_not_yet_approved"]


def _after_fill_verdict(health: dict, step_results: list[dict]) -> tuple[str, list[str]]:
    failed_steps = [step["name"] for step in step_results if step.get("status") not in {"ok", "skipped"}]
    if failed_steps:
        return "BROKER_CHECK_FAILED", [f"failed_steps={','.join(failed_steps)}"]
    flags = health.get("readiness_flags", {}) or {}
    open_orders = int(health.get("current_signal_open_orders", 0) or 0)
    if open_orders > 0:
        return "STILL_OPEN", [f"current_signal_open_orders={open_orders}"]
    if not flags.get("broker_synced", False):
        return "SYNC_OR_WAIT_FOR_FILLS", ["broker_not_synced_to_current_signal"]
    if not flags.get("account_aligned", False):
        return "NEEDS_REBALANCE", ["fills_complete_but_account_not_aligned"]
    return "ALIGNED", ["fills_complete_and_account_aligned"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the safe daily Moomoo paper-trading check.")
    parser.add_argument("--after-fill", action="store_true", help="After expected fills, sync and print a fill-focused verdict.")
    parser.add_argument("--prune-snapshots", action="store_true", help="Prune old logs/paper_snapshots folders and exit.")
    parser.add_argument("--keep-days", type=int, default=30, help="Snapshot retention window for --prune-snapshots.")
    parser.add_argument("--skip-status", action="store_true", help="Do not refresh Moomoo status/equity.")
    parser.add_argument("--skip-sync", action="store_true", help="Do not reconcile paper fills from Moomoo.")
    parser.add_argument("--lookback-days", type=int, default=7, help="Order-history lookback for sync.")
    parser.add_argument("--timeout", type=int, default=180, help="Max seconds per command.")
    parser.add_argument("--json", action="store_true", help="Print the final check result as JSON.")
    args = parser.parse_args()
    if args.prune_snapshots:
        result = _prune_snapshots(keep_days=int(args.keep_days))
        print("Snapshot Prune")
        print("-" * 72)
        print(f"Root: {result['root']}")
        print(f"Keep days: {result['keep_days']}")
        print(f"Removed: {len(result['removed'])}")
        print(f"Kept: {len(result['kept'])}")
        return
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    steps: list[tuple[str, list[str]]] = []
    if not args.skip_status:
        steps.append(("status", [sys.executable, "moomoo_paper_trading.py", "--status"]))
    if not args.skip_sync:
        steps.append(("sync", [sys.executable, "moomoo_paper_trading.py", "--sync", "--lookback-days", str(args.lookback_days)]))
    steps.extend(
        [
            ("health", [sys.executable, "paper_health.py"]),
            ("gauntlet", [sys.executable, "paper_gauntlet.py"]),
        ]
    )

    print("Daily Paper Check")
    print("-" * 72)
    print("No orders will be submitted by this command.")

    results = [_run_step(name, cmd, timeout=int(args.timeout)) for name, cmd in steps]
    health = _read_json(SIGNALS / "paper_health.json")
    verdict, reasons = _verdict(health, results)
    after_fill = None
    if args.after_fill:
        after_verdict, after_reasons = _after_fill_verdict(health, results)
        after_fill = {"verdict": after_verdict, "reasons": after_reasons}
    snapshot = _snapshot_artifacts(run_id)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "verdict": verdict,
        "reasons": reasons,
        "after_fill": after_fill,
        "blockers": _blocker_summary(health),
        "health_flags": health.get("readiness_flags", {}),
        "gauntlet_status": health.get("gauntlet_status"),
        "approved_for_real_capital": bool(health.get("approved_for_real_capital", False)),
        "artifact_snapshot": snapshot,
        "steps": results,
    }

    SIGNALS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    CHECK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dated = LOGS / f"daily_paper_check_{datetime.now().strftime('%Y%m%d')}.json"
    dated.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\nVerdict")
        print("-" * 72)
        print(verdict)
        print("; ".join(reasons))
        if after_fill:
            print(f"After-fill: {after_fill['verdict']} - {'; '.join(after_fill['reasons'])}")
        print(f"Snapshot dir -> {snapshot.get('dir')}")
        print(f"Saved -> {CHECK_PATH}")
        print(f"Snapshot -> {dated}")


if __name__ == "__main__":
    main()
