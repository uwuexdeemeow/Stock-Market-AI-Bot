"""Archive old paper evidence and track a clean reliability-validation epoch.

PLAIN ENGLISH: Bot behavior changed, so old and new paper results should not be
judged as one experiment. This script copies the current evidence into a dated
archive and creates a new epoch marker. Live operational files stay in place so
reconciliation and duplicate protection keep working.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from safe_io import atomic_write_json
from settings import LOG_DIR, SIGNAL_DIR


EPOCH_FILE = Path(SIGNAL_DIR) / "paper_validation_epoch.json"
ARCHIVE_ROOT = Path("archive/paper_epochs")
EVIDENCE_FILES = (
    Path(SIGNAL_DIR) / "alpaca_paper_log.csv",
    Path(SIGNAL_DIR) / "alpaca_paper_equity.csv",
    Path(SIGNAL_DIR) / "alpaca_slippage_reversal_report.json",
    Path(SIGNAL_DIR) / "alpaca_execution_scorecard.json",
    Path(SIGNAL_DIR) / "broker_truth.json",
    Path(SIGNAL_DIR) / "alpaca_paper_health.json",
    Path(SIGNAL_DIR) / "alpaca_submit_outcomes.csv",
)


def start_epoch(*, now: datetime | None = None) -> dict:
    """Copy current evidence and begin a new dated paper-validation period."""
    started = now or datetime.now(timezone.utc)
    epoch_id = started.strftime("paper-v%Y%m%dT%H%M%SZ")
    archive_dir = ARCHIVE_ROOT / epoch_id
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived: list[str] = []
    for source in EVIDENCE_FILES:
        if not source.exists():
            continue
        destination = archive_dir / source.name
        shutil.copy2(source, destination)
        archived.append(str(destination))
    payload = {
        "schema_version": 1,
        "epoch_id": epoch_id,
        "started_at": started.isoformat(timespec="seconds"),
        "status": "collecting",
        "archive_dir": str(archive_dir),
        "archived_files": archived,
        "requirements": {
            "minimum_trading_days": 30,
            "minimum_rebalance_events": 3,
            "minimum_accepted_orders": 20,
            "minimum_consecutive_classified_sessions": 10,
            "maximum_duplicate_orders": 0,
            "maximum_unexplained_orders": 0,
            "maximum_target_weight_gap": 0.02,
            "minimum_fill_rate": 0.80,
            "maximum_average_slippage_bps": 10.0,
            "maximum_bad_slippage_rate": 0.60,
        },
        "real_capital_approved": False,
    }
    atomic_write_json(payload, EPOCH_FILE)
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    """Read an optional CSV without making the epoch evaluator crash."""
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    """Read an optional JSON report."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _consecutive_session_count(values: pd.Series) -> int:
    """Count consecutive NYSE sessions working backward from the latest run."""
    dates = sorted(set(pd.to_datetime(values, errors="coerce", utc=True).dropna().dt.tz_convert("America/New_York").dt.normalize()))
    if not dates:
        return 0
    try:
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        sessions = set(pd.DatetimeIndex(calendar.sessions_in_range(dates[0].tz_localize(None), dates[-1].tz_localize(None))).tz_localize(None))
        observed = {date.tz_localize(None) for date in dates}
        count = 0
        for session in sorted(sessions, reverse=True):
            if session in observed:
                count += 1
            else:
                break
        return count
    except Exception:
        return len(dates)


def evaluate_epoch(epoch: dict) -> dict:
    """Measure progress against the fixed operational acceptance rules."""
    start = pd.to_datetime(epoch["started_at"], utc=True)
    journal = _read_csv(Path(SIGNAL_DIR) / "alpaca_submit_outcomes.csv")
    paper_log = _read_csv(Path(SIGNAL_DIR) / "alpaca_paper_log.csv")
    equity = _read_csv(Path(SIGNAL_DIR) / "alpaca_paper_equity.csv")

    if not journal.empty and "generated_at" in journal:
        journal["_time"] = pd.to_datetime(journal["generated_at"], errors="coerce", utc=True)
        journal = journal[journal["_time"] >= start]
    if not paper_log.empty:
        timestamp_col = next((column for column in ("submitted_at", "date") if column in paper_log), None)
        if timestamp_col:
            paper_log["_time"] = pd.to_datetime(paper_log[timestamp_col], errors="coerce", utc=True)
            paper_log = paper_log[paper_log["_time"] >= start]
    # The equity file stores both a calendar ``date`` and an exact ``timestamp``.
    # PLAIN ENGLISH: an epoch can start in the middle of a day.  Reading only
    # ``date`` turns that day's observation into midnight, which can make a real
    # post-start snapshot look older than the epoch.  Prefer the exact clock time
    # and use ``date`` only for older files that do not have it yet.
    if not equity.empty:
        equity_time_col = next((column for column in ("timestamp", "date") if column in equity), None)
        if equity_time_col:
            equity["_time"] = pd.to_datetime(equity[equity_time_col], errors="coerce", utc=True)
    if "_time" in equity:
        equity = equity[equity["_time"] >= start]

    accepted = int(pd.to_numeric(journal.get("accepted_orders", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    classified_sessions = _consecutive_session_count(journal["_time"]) if "_time" in journal else 0
    executed = journal[journal["status"].astype(str) == "executed"] if not journal.empty and "status" in journal else journal.iloc[0:0]
    rebalances = int(executed["_time"].dt.date.nunique()) if "_time" in executed else 0
    trading_days = int(equity["_time"].dt.date.nunique()) if "_time" in equity else 0
    order_ids = paper_log.get("client_order_id", paper_log.get("order_id", pd.Series(dtype=str))).dropna().astype(str)
    order_ids = order_ids[~order_ids.str.startswith(("ERROR", "SKIPPED"))]
    duplicates = int(order_ids.duplicated().sum())
    statuses = paper_log.get("fill_status", pd.Series(dtype=str)).astype(str).str.lower()
    accepted_log_rows = statuses.isin({"accepted", "new", "partially_filled", "filled"})
    filled_log_rows = statuses.eq("filled")
    fill_rate = float(filled_log_rows.sum() / accepted_log_rows.sum()) if int(accepted_log_rows.sum()) else None

    # PLAIN ENGLISH: the execution scorecard owns denominator policy. Reusing
    # its canonical rate prevents this epoch from reporting 14/25 while the
    # scorecard reports 14/19 for the same fills.
    execution_scorecard = _read_json(Path(SIGNAL_DIR) / "alpaca_execution_scorecard.json")
    execution_summary = execution_scorecard.get("summary", {}) or {}
    avg_slippage = execution_summary.get("avg_slippage_bps")
    bad_rate = execution_summary.get("bad_slippage_rate")
    execution_decision_eligible = bool(execution_scorecard.get("decision_eligible", False))

    broker_truth = _read_json(Path(SIGNAL_DIR) / "broker_truth.json")
    truth_summary = broker_truth.get("summary", {}) or {}
    unexplained = int(truth_summary.get("fail_count", 0) or 0)
    weight_gaps = []
    for row in broker_truth.get("rows", []) or []:
        target = pd.to_numeric(row.get("target_weight"), errors="coerce")
        actual = pd.to_numeric(row.get("broker_weight"), errors="coerce")
        if pd.notna(target) and pd.notna(actual):
            weight_gaps.append(abs(float(target) - float(actual)))
    max_weight_gap = max(weight_gaps, default=None)

    requirements = epoch.get("requirements", {}) or {}
    checks = {
        "trading_days": trading_days >= int(requirements.get("minimum_trading_days", 30)),
        "rebalance_events": rebalances >= int(requirements.get("minimum_rebalance_events", 3)),
        "accepted_orders": accepted >= int(requirements.get("minimum_accepted_orders", 20)),
        "classified_sessions": classified_sessions >= int(requirements.get("minimum_consecutive_classified_sessions", 10)),
        "duplicate_orders": duplicates <= int(requirements.get("maximum_duplicate_orders", 0)),
        "unexplained_orders": unexplained <= int(requirements.get("maximum_unexplained_orders", 0)),
        "target_weight_gap": max_weight_gap is not None and max_weight_gap <= float(requirements.get("maximum_target_weight_gap", 0.02)),
        "fill_rate": fill_rate is not None and fill_rate >= float(requirements.get("minimum_fill_rate", 0.80)),
        "average_slippage": execution_decision_eligible and avg_slippage is not None and float(avg_slippage) <= float(requirements.get("maximum_average_slippage_bps", 10.0)),
        "bad_slippage_rate": execution_decision_eligible and bad_rate is not None and float(bad_rate) <= float(requirements.get("maximum_bad_slippage_rate", 0.60)),
    }
    return {
        "epoch_id": epoch.get("epoch_id"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "operational_pass" if all(checks.values()) else "collecting",
        "trading_days": trading_days,
        "rebalance_events": rebalances,
        "accepted_orders": accepted,
        "classified_sessions": classified_sessions,
        "duplicate_orders": duplicates,
        "unexplained_orders": unexplained,
        "maximum_target_weight_gap": max_weight_gap,
        "fill_rate": fill_rate,
        "average_slippage_bps": avg_slippage,
        "bad_slippage_rate": bad_rate,
        "execution_scorecard_decision_eligible": execution_decision_eligible,
        "execution_scorecard_schema_version": execution_scorecard.get("schema_version"),
        "checks": checks,
        "real_capital_approved": False,
    }


def main() -> int:
    """Start a new epoch or print progress for the active one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Evaluate the current epoch without starting a new one.")
    args = parser.parse_args()
    if args.status:
        epoch = json.loads(EPOCH_FILE.read_text(encoding="utf-8"))
        report = evaluate_epoch(epoch)
        atomic_write_json(report, Path(SIGNAL_DIR) / "paper_validation_epoch_status.json")
        print(json.dumps(report, indent=2))
        return 0
    epoch = start_epoch()
    print(f"Started paper validation epoch: {epoch['epoch_id']}")
    print(f"Archived evidence -> {epoch['archive_dir']}")
    print("Operational files remain active; real capital remains blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
