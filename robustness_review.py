"""Evaluate the three robustness reports with one shared set of rules.

PLAIN ENGLISH: walk-forward publishing, validation bundles, and live signals
must answer the same question from the same files.  This module is that single
answer, so an old copied ``pass`` flag cannot overrule a current warning.
"""
from __future__ import annotations

import json
from pathlib import Path

from settings import LOG_DIR


DEFAULT_ROBUSTNESS_REPORT_PATHS = {
    "survivorship": Path(LOG_DIR) / "core_satellite_survivorship_audit.json",
    "execution_stress": Path(LOG_DIR) / "core_satellite_execution_stress.json",
    "factor_decay": Path(LOG_DIR) / "factor_decay_monitor.json",
}

SURVIVORSHIP_MIN_ADJUSTED_SCORE = 0.50
# PLAIN ENGLISH: the audit deliberately adds known failed companies to the
# historical universe. Seeing some of them selected is the test, not itself a
# failure. The strategy fails only when selections are excessive or the
# stressed performance checks below deteriorate too far.
SURVIVORSHIP_MAX_AUDIT_SELECTIONS = 60
SURVIVORSHIP_MIN_RETURN_DELTA_PCT = -5000.0
SURVIVORSHIP_MIN_DRAWDOWN_DELTA_PCT = -5.0
SURVIVORSHIP_MIN_FAILED_NAME_COVERAGE_FOR_CAPITAL = 1.0
SURVIVORSHIP_CAPITAL_MIN_ADJUSTED_SCORE = 0.85
SURVIVORSHIP_CAPITAL_MIN_RETURN_DELTA_PCT = -5.0
SURVIVORSHIP_CAPITAL_MIN_DRAWDOWN_DELTA_PCT = -2.5
EXECUTION_STRESS_MIN_WORST_DRAWDOWN_PCT = -35.0


def read_report(path: Path) -> dict:
    """Read one JSON report; an absent or broken file becomes an empty report."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def evaluate_medium_risk_review(
    *,
    survivorship: dict | None,
    execution: dict | None,
    factor_decay: dict | None,
) -> dict:
    """Return a fail-closed health decision from already-loaded reports."""
    survivorship = survivorship or {}
    execution = execution or {}
    factor_decay = factor_decay or {}
    reasons: list[str] = []

    rows = survivorship.get("rows", []) if isinstance(survivorship, dict) else []
    by_scenario = {str(row.get("scenario")): row for row in rows if isinstance(row, dict)}
    stressed = by_scenario.get("watchlist_plus_failed_audit_tickers", {})
    delta = by_scenario.get("delta_stressed_minus_base", {})
    surv_score = float(survivorship.get("survivorship_adjusted_score", 0.0) or 0.0)
    audit_picks = int(float(stressed.get("audit_rebalance_selections", 0) or 0)) if stressed else 0
    return_delta = float(delta.get("total_return_pct", 0.0) or 0.0) if delta else 0.0
    dd_delta = float(delta.get("max_drawdown_pct", 0.0) or 0.0) if delta else 0.0
    known_failed = survivorship.get("known_audit_tickers", []) or []
    available_failed = survivorship.get("available_audit_tickers", []) or []
    failed_name_coverage = float(
        survivorship.get(
            "failed_name_coverage_rate",
            len(available_failed) / max(len(known_failed), 1) if known_failed else 0.0,
        ) or 0.0
    )
    point_in_time_complete = bool(survivorship.get("point_in_time_universe_complete", False))
    survivorship_pass = bool(
        survivorship
        and stressed
        and bool(stressed.get("paper_ready", False))
        and surv_score > SURVIVORSHIP_MIN_ADJUSTED_SCORE
        and audit_picks <= SURVIVORSHIP_MAX_AUDIT_SELECTIONS
        and return_delta >= SURVIVORSHIP_MIN_RETURN_DELTA_PCT
        and dd_delta >= SURVIVORSHIP_MIN_DRAWDOWN_DELTA_PCT
    )
    if not survivorship:
        reasons.append("survivorship_review_missing")
    elif not survivorship_pass:
        reasons.append("survivorship_review_failed")
    # PLAIN ENGLISH: The partial failed-name stress may still be useful for a
    # paper experiment. It is never enough for capital approval unless every
    # named failure and the full date-effective universe are present.
    survivorship_capital_pass = bool(
        survivorship_pass
        and failed_name_coverage >= SURVIVORSHIP_MIN_FAILED_NAME_COVERAGE_FOR_CAPITAL
        and point_in_time_complete
        and surv_score >= SURVIVORSHIP_CAPITAL_MIN_ADJUSTED_SCORE
        and return_delta >= SURVIVORSHIP_CAPITAL_MIN_RETURN_DELTA_PCT
        and dd_delta >= SURVIVORSHIP_CAPITAL_MIN_DRAWDOWN_DELTA_PCT
    )

    exec_rows = [
        row for row in execution.get("rows", [])
        if isinstance(row, dict) and not str(row.get("scenario", "")).startswith("delta_")
    ]
    exec_failed = [
        row for row in exec_rows
        if not bool(row.get("paper_ready", False))
        or float(row.get("alpha_vs_qqq_pct", -999.0) or -999.0) <= 0.0
        or float(row.get("alpha_vs_blend_pct", -999.0) or -999.0) <= 0.0
    ]
    worst_dd = min((float(row.get("max_drawdown_pct", 0.0) or 0.0) for row in exec_rows), default=0.0)
    execution_pass = bool(exec_rows and not exec_failed and worst_dd >= EXECUTION_STRESS_MIN_WORST_DRAWDOWN_PCT)
    if not execution:
        reasons.append("execution_stress_review_missing")
    elif not execution_pass:
        reasons.append("execution_stress_review_failed")

    edge_status = str(factor_decay.get("edge_health_status", "missing")).lower()
    factor_pass = edge_status in {"pass", "advisory"}
    if not factor_decay:
        reasons.append("factor_decay_review_missing")
    elif not factor_pass:
        reasons.append(f"factor_decay_review_{edge_status}")

    return {
        "pass": not reasons,
        "reasons": reasons,
        "survivorship_review": {
            "pass": survivorship_pass,
            "capital_approval_pass": survivorship_capital_pass,
            "survivorship_adjusted_score": round(surv_score, 4),
            "audit_rebalance_selections": audit_picks,
            "total_return_delta_pct": round(return_delta, 4),
            "max_drawdown_delta_pct": round(dd_delta, 4),
            "failed_name_coverage_rate": round(failed_name_coverage, 4),
            "minimum_failed_name_coverage_for_capital": SURVIVORSHIP_MIN_FAILED_NAME_COVERAGE_FOR_CAPITAL,
            "point_in_time_universe_complete": point_in_time_complete,
            "capital_blockers": [
                reason
                for reason, blocked in (
                    ("failed_name_coverage_incomplete", failed_name_coverage < SURVIVORSHIP_MIN_FAILED_NAME_COVERAGE_FOR_CAPITAL),
                    ("point_in_time_universe_incomplete", not point_in_time_complete),
                    ("survivorship_adjusted_score_below_capital_floor", surv_score < SURVIVORSHIP_CAPITAL_MIN_ADJUSTED_SCORE),
                    ("survivorship_return_delta_below_capital_floor", return_delta < SURVIVORSHIP_CAPITAL_MIN_RETURN_DELTA_PCT),
                    ("survivorship_drawdown_delta_below_capital_floor", dd_delta < SURVIVORSHIP_CAPITAL_MIN_DRAWDOWN_DELTA_PCT),
                    ("survivorship_stress_failed", not survivorship_pass),
                )
                if blocked
            ],
        },
        "execution_stress_review": {
            "pass": execution_pass,
            "failed_scenarios": len(exec_failed),
            "worst_stressed_drawdown_pct": round(worst_dd, 4),
        },
        "factor_decay_review": {
            "pass": factor_pass,
            "edge_health_status": edge_status,
            "reason": factor_decay.get("reason"),
        },
    }


def medium_risk_review_from_reports(
    *,
    survivorship: dict | None = None,
    execution: dict | None = None,
    factor_decay: dict | None = None,
    report_paths: dict[str, Path] | None = None,
) -> dict:
    """Load omitted reports from disk, then apply the canonical rules."""
    paths = DEFAULT_ROBUSTNESS_REPORT_PATHS if report_paths is None else report_paths
    return evaluate_medium_risk_review(
        survivorship=survivorship if survivorship is not None else read_report(Path(paths["survivorship"])),
        execution=execution if execution is not None else read_report(Path(paths["execution_stress"])),
        factor_decay=factor_decay if factor_decay is not None else read_report(Path(paths["factor_decay"])),
    )
