from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from settings import BASE_DIR, LOG_DIR, MODEL_DIR, PAPER_MODE_STRATEGY, SIGNAL_DIR, WATCHLIST


ROOT = Path(BASE_DIR)
MODELS = Path(MODEL_DIR)
SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _file_status(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        exists = path.exists()
        rows.append(
            {
                "path": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
                if exists
                else None,
            }
        )
    return rows


def _latest_matching(directory: Path, pattern: str) -> Path | None:
    matches = list(directory.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _quality_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    quality = _read_csv(MODELS / "model_quality_report.csv")
    if quality.empty:
        findings.append(
            {
                "severity": "high",
                "area": "model_selection",
                "finding": "No model quality report found. Live approval cannot be trusted until backtest.py writes one.",
            }
        )
        return {"rows": 0, "approved": 0, "rejected": 0, "approved_tickers": []}

    approved_mask = quality.get("approved_for_live", pd.Series(False, index=quality.index)).astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    approved = quality[approved_mask].copy()
    rejected = quality[~approved_mask].copy()
    if approved.empty:
        severity = "high" if PAPER_MODE_STRATEGY == "single_name" else "low"
        findings.append(
            {
                "severity": severity,
                "area": "model_selection",
                "finding": (
                    "No single-name ticker is approved for live trading. This blocks single-name paper/live trading; "
                    f"current PAPER_MODE_STRATEGY={PAPER_MODE_STRATEGY}."
                ),
            }
        )

    if "quality_source" in quality.columns:
        preliminary = quality[quality["quality_source"].astype(str).str.contains("train_preliminary", case=False, na=False)]
        if not preliminary.empty:
            findings.append(
                {
                    "severity": "medium",
                    "area": "model_selection",
                    "finding": f"{len(preliminary)} tickers still have preliminary train-only quality rows. Re-run backtest.py for those tickers before live use.",
                }
            )

    return {
        "rows": int(len(quality)),
        "approved": int(len(approved)),
        "rejected": int(len(rejected)),
        "approved_tickers": approved.get("ticker", pd.Series(dtype=str)).astype(str).tolist(),
        "top_rejections": rejected[["ticker", "approval_reason"]].head(8).to_dict("records")
        if {"ticker", "approval_reason"}.issubset(rejected.columns)
        else [],
    }


def _trade_rule_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    report = _read_csv(MODELS / "trade_rule_report.csv")
    rule_files = sorted(MODELS.glob("*_trade_rules.json"))
    if report.empty and not rule_files:
        findings.append(
            {
                "severity": "medium",
                "area": "execution_rules",
                "finding": "No optimized trade rules found. Run optimize_trade_rules.py after each promising backtest.",
            }
        )
    approved_candidates = 0
    if not report.empty and "approved_candidate" in report.columns:
        approved_candidates = int(report["approved_candidate"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    return {
        "rule_files": len(rule_files),
        "report_rows": int(len(report)),
        "approved_rule_candidates": approved_candidates,
        "best_rules": report.head(5).to_dict("records") if not report.empty else [],
    }


def _signals_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    signals = _read_csv(SIGNALS / "signals.csv")
    approved_live = _read_csv(SIGNALS / "approved_live_tickers.csv")
    if signals.empty:
        findings.append(
            {
                "severity": "medium",
                "area": "live_signals",
                "finding": "signals/signals.csv is missing or empty. The live prediction layer has no current candidate set.",
            }
        )
        return {"signals": 0, "actionable": 0, "approved_live_tickers": 0}

    actionable = 0
    if "actionable" in signals.columns:
        actionable = int(signals["actionable"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    approved_count = int(len(approved_live)) if "ticker" in approved_live.columns else 0
    if approved_count == 0:
        severity = "high" if PAPER_MODE_STRATEGY == "single_name" else "low"
        findings.append(
            {
                "severity": severity,
                "area": "live_signals",
                "finding": (
                    "approved_live_tickers.csv has no approved single-name tickers. "
                    f"This is expected while PAPER_MODE_STRATEGY={PAPER_MODE_STRATEGY}."
                ),
            }
        )
    if actionable > 0 and approved_count == 0 and PAPER_MODE_STRATEGY == "single_name":
        findings.append(
            {
                "severity": "high",
                "area": "live_signals",
                "finding": "There are actionable raw signals but zero approved live tickers. Approval gating must remain enforced.",
            }
        )
    return {
        "signals": int(len(signals)),
        "actionable": actionable,
        "approved_live_tickers": approved_count,
        "suppressed_reasons": signals.get("suppressed_reason", pd.Series(dtype=str)).dropna().value_counts().head(8).to_dict(),
    }


def _backtest_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    equity_file = _latest_matching(SIGNALS, "*equity.csv")
    trades_file = _latest_matching(SIGNALS, "*trades.csv")
    trades = _read_csv(trades_file) if trades_file else pd.DataFrame()
    equity = _read_csv(equity_file) if equity_file else pd.DataFrame()
    if trades.empty:
        findings.append(
            {
                "severity": "medium",
                "area": "backtest",
                "finding": "No recent trade ledger found. A quant setup needs a persistent walk-forward trade ledger for model selection.",
            }
        )
    return {
        "latest_trades_file": str(trades_file.relative_to(ROOT)) if trades_file else None,
        "latest_equity_file": str(equity_file.relative_to(ROOT)) if equity_file else None,
        "latest_trade_rows": int(len(trades)),
        "latest_equity_rows": int(len(equity)),
    }


def _paper_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    filtered = _read_csv(SIGNALS / "signals_live_filtered.csv")
    paper_trades = _read_csv(SIGNALS / "paper_trades.csv")
    paper_equity = _read_csv(SIGNALS / "paper_equity.csv")
    core_signal = _read_csv(SIGNALS / "core_satellite_alpha_signal.csv")
    status_path = SIGNALS / "paper_daily_status.json"
    paper_status = _read_json(status_path)
    gauntlet_path = _latest_matching(LOGS, "paper_gauntlet_*.json")
    gauntlet = _read_json(gauntlet_path)
    if paper_trades.empty:
        findings.append(
            {
                "severity": "medium",
                "area": "paper_trading",
                "finding": "No paper trade journal found. Before real capital, log every proposed, accepted, rejected, and closed paper trade.",
            }
        )
    if not core_signal.empty and bool(core_signal.iloc[0].get("paper_ready", False)) and not paper_status:
        findings.append(
            {
                "severity": "medium",
                "area": "paper_trading",
                "finding": "Core-satellite signal is paper-ready but no paper_daily_status.json exists. Run moomoo_paper_trading.py --status.",
            }
        )
    if gauntlet and not bool(gauntlet.get("approved_for_real_capital", False)):
        findings.append(
            {
                "severity": "medium",
                "area": "paper_trading",
                "finding": f"Paper gauntlet blocks real-capital promotion: {gauntlet.get('reason', 'unknown reason')}",
            }
        )
    return {
        "filtered_live_rows": int(len(filtered)),
        "paper_trade_rows": int(len(paper_trades)),
        "paper_equity_rows": int(len(paper_equity)),
        "paper_gauntlet_file": str(gauntlet_path.relative_to(ROOT)) if gauntlet_path else None,
        "paper_gauntlet_status": gauntlet.get("status"),
        "approved_for_real_capital": gauntlet.get("approved_for_real_capital"),
        "paper_gauntlet_reason": gauntlet.get("reason"),
        "paper_benchmark_comparisons": gauntlet.get("paper_benchmark_comparisons", {}),
        "core_satellite_paper_ready": bool(core_signal.iloc[0].get("paper_ready", False)) if not core_signal.empty else False,
        "current_regime": paper_status.get("current_regime"),
        "account_equity": paper_status.get("account_equity"),
        "current_gross_exposure": paper_status.get("current_gross_exposure"),
        "target_gross_exposure": paper_status.get("target_gross_exposure"),
        "max_drift_abs": paper_status.get("max_drift_abs"),
        "submit_allowed": paper_status.get("submit_allowed"),
        "guard_reasons": paper_status.get("guard_reasons", []),
    }


def _survivorship_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    audit_path = LOGS / "core_satellite_survivorship_audit.json"
    audit = _read_json(audit_path)
    rows = audit.get("rows", []) if audit else []
    if not rows:
        findings.append(
            {
                "severity": "medium",
                "area": "survivorship_bias",
                "finding": "No core-satellite survivorship stress report found. Run core_satellite_survivorship_audit.py before trusting long historical alpha.",
            }
        )
        return {"report_file": None, "status": "missing"}

    by_scenario = {str(row.get("scenario")): row for row in rows if isinstance(row, dict)}
    stressed = by_scenario.get("watchlist_plus_failed_audit_tickers", {})
    delta = by_scenario.get("delta_stressed_minus_base", {})
    audit_picks = int(float(stressed.get("audit_rebalance_selections", 0) or 0))
    return_delta = float(delta.get("total_return_pct", 0.0) or 0.0)
    drawdown_delta = float(delta.get("max_drawdown_pct", 0.0) or 0.0)
    stressed_ready = bool(stressed.get("paper_ready", False))
    available = audit.get("available_audit_tickers", [])
    missing = audit.get("missing_audit_tickers", [])

    if not stressed_ready:
        findings.append(
            {
                "severity": "high",
                "area": "survivorship_bias",
                "finding": "Core-satellite strategy fails gates when available failed/delisted audit tickers are added to the historical ranking universe.",
            }
        )
    elif audit_picks > 0 or return_delta < -1000.0 or drawdown_delta < -2.0:
        findings.append(
            {
                "severity": "medium",
                "area": "survivorship_bias",
                "finding": (
                    "Failed-name stress is material: "
                    f"{audit_picks} failed-name selections, total-return delta {return_delta:.2f} pct points, "
                    f"max-DD delta {drawdown_delta:.2f} pct points. Strategy still passes, but alpha is survivor-sensitive."
                ),
            }
        )

    return {
        "report_file": str(audit_path.relative_to(ROOT)),
        "status": "pass_with_warning" if stressed_ready else "fail",
        "available_audit_tickers": available,
        "missing_audit_tickers": missing,
        "audit_rebalance_selections": audit_picks,
        "total_return_delta_pct": return_delta,
        "max_drawdown_delta_pct": drawdown_delta,
        "stressed_paper_ready": stressed_ready,
        "survivorship_adjusted_score": audit.get("survivorship_adjusted_score"),
        "failed_name_stressed_return_pct": audit.get("failed_name_stressed_return_pct"),
    }


def _execution_stress_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    path = LOGS / "core_satellite_execution_stress.json"
    report = _read_json(path)
    rows = [row for row in report.get("rows", []) if isinstance(row, dict) and not str(row.get("scenario", "")).startswith("delta_")]
    if not rows:
        findings.append(
            {
                "severity": "medium",
                "area": "execution_stress",
                "finding": "No execution stress report found. Run core_satellite_execution_stress.py before real-capital promotion.",
            }
        )
        return {"report_file": None, "status": "missing"}
    failed = [
        row for row in rows
        if not bool(row.get("paper_ready", False))
        or float(row.get("alpha_vs_qqq_pct", -999.0) or -999.0) <= 0
        or float(row.get("alpha_vs_blend_pct", -999.0) or -999.0) <= 0
    ]
    worst_drawdown = min(float(row.get("max_drawdown_pct", 0.0) or 0.0) for row in rows)
    worst_alpha_blend = min(float(row.get("alpha_vs_blend_pct", 0.0) or 0.0) for row in rows)
    if failed:
        findings.append(
            {
                "severity": "high",
                "area": "execution_stress",
                "finding": f"{len(failed)} execution stress scenario(s) failed paper gates or benchmark alpha.",
            }
        )
    elif worst_drawdown < -35.0:
        findings.append(
            {
                "severity": "medium",
                "area": "execution_stress",
                "finding": f"Execution stress passes, but worst stressed drawdown is {worst_drawdown:.2f}%.",
            }
        )
    return {
        "report_file": str(path.relative_to(ROOT)),
        "status": "pass" if not failed else "fail",
        "scenarios": len(rows),
        "failed_scenarios": len(failed),
        "worst_drawdown_pct": worst_drawdown,
        "worst_alpha_vs_blend_pct": worst_alpha_blend,
    }


def _drawdown_throttle_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    path = LOGS / "core_satellite_drawdown_throttle.json"
    report = _read_json(path)
    rows = report.get("rows", []) if report else []
    if not rows:
        findings.append(
            {
                "severity": "low",
                "area": "drawdown_throttle",
                "finding": "No drawdown throttle research report found. Run core_satellite_drawdown_throttle.py during weekly robustness checks.",
            }
        )
        return {"report_file": None, "status": "missing"}
    candidate = report.get("best_promotion_candidate")
    return {
        "report_file": str(path.relative_to(ROOT)),
        "status": "candidate_found" if candidate else "no_promotion_candidate",
        "best_promotion_candidate": candidate,
        "tested_rules": [row.get("throttle_rule") for row in rows if isinstance(row, dict)],
    }


def _factor_decay_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    path = LOGS / "factor_decay_monitor.json"
    report = _read_json(path)
    rows = report.get("rows", []) if report else []
    if not rows:
        findings.append(
            {
                "severity": "medium",
                "area": "factor_decay",
                "finding": "No factor decay monitor found. Run factor_decay_monitor.py before real-capital promotion.",
            }
        )
        return {"report_file": None, "status": "missing", "real_capital_block": True}
    warning = bool(report.get("warning", False))
    real_capital_block = bool(report.get("real_capital_block", False))
    if real_capital_block:
        findings.append(
            {
                "severity": "high",
                "area": "factor_decay",
                "finding": f"Factor decay monitor blocks real-capital promotion: {report.get('reason', 'unknown reason')}",
            }
        )
    elif warning:
        findings.append(
            {
                "severity": "medium",
                "area": "factor_decay",
                "finding": "Factor decay warning present: recent rank IC is weak/negative, though recent overlay alpha is not blocking.",
            }
        )
    return {
        "report_file": str(path.relative_to(ROOT)),
        "status": "block" if real_capital_block else "warning" if warning else "pass",
        "warning": warning,
        "real_capital_block": real_capital_block,
        "reason": report.get("reason"),
        "rows": rows,
    }


def _robust_mode_section(findings: list[dict[str, str]]) -> dict[str, Any]:
    primary = _read_json(SIGNALS / "core_satellite_alpha_metrics.json")
    robust = _read_json(SIGNALS / "core_satellite_robust_mode_metrics.json")
    if not primary or not robust:
        findings.append(
            {
                "severity": "low",
                "area": "robust_mode",
                "finding": "Robust-mode comparison is missing. Run core_satellite_robust_mode.py for weekly/monthly comparison.",
            }
        )
        return {"status": "missing"}
    return {
        "status": "available",
        "primary_return_pct": primary.get("total_return_pct"),
        "primary_sharpe": primary.get("sharpe"),
        "primary_max_drawdown_pct": primary.get("max_drawdown_pct"),
        "robust_return_pct": robust.get("total_return_pct"),
        "robust_sharpe": robust.get("sharpe"),
        "robust_max_drawdown_pct": robust.get("max_drawdown_pct"),
        "return_delta_pct": round(float(robust.get("total_return_pct", 0.0) or 0.0) - float(primary.get("total_return_pct", 0.0) or 0.0), 2),
        "sharpe_delta": round(float(robust.get("sharpe", 0.0) or 0.0) - float(primary.get("sharpe", 0.0) or 0.0), 3),
        "max_drawdown_delta_pct": round(float(robust.get("max_drawdown_pct", 0.0) or 0.0) - float(primary.get("max_drawdown_pct", 0.0) or 0.0), 2),
    }


def _summary_section(report: dict[str, Any], findings: list[dict[str, str]]) -> dict[str, Any]:
    primary_metrics = _read_json(SIGNALS / "core_satellite_alpha_metrics.json")
    primary_gates = primary_metrics.get("core_satellite_gate_results", {})
    primary_ok = bool(primary_metrics.get("paper_ready", False) and primary_gates.get("all_pass", False))
    paper = report.get("paper", {})
    real_capital_blocked_reason = None
    if not bool(paper.get("approved_for_real_capital", False)):
        real_capital_blocked_reason = paper.get("paper_gauntlet_reason") or "paper gauntlet has not approved real capital"
    if report.get("factor_decay", {}).get("real_capital_block"):
        real_capital_blocked_reason = report["factor_decay"].get("reason", "factor decay blocks real capital")
    high = [f for f in findings if f.get("severity") == "high"]
    if high:
        largest = high[0]["area"]
    elif real_capital_blocked_reason:
        largest = "paper_execution_history"
    elif report.get("survivorship_bias", {}).get("status") == "pass_with_warning":
        largest = "survivorship_bias"
    elif report.get("factor_decay", {}).get("warning"):
        largest = "factor_decay"
    else:
        largest = "none"
    return {
        "primary_strategy_ok": primary_ok,
        "real_capital_blocked_reason": real_capital_blocked_reason,
        "largest_remaining_risk": largest,
    }


def run_audit() -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    required_files = [
        ROOT / "train.py",
        ROOT / "backtest.py",
        ROOT / "predict.py",
        ROOT / "leakage_audit.py",
        ROOT / "optimize_trade_rules.py",
        ROOT / "model_quality.py",
        ROOT / "trade_rules.py",
        ROOT / "moomoo_paper_trading.py",
        ROOT / "paper_gauntlet.py",
        ROOT / "core_satellite_survivorship_audit.py",
        ROOT / "core_satellite_execution_stress.py",
        ROOT / "core_satellite_drawdown_throttle.py",
        ROOT / "factor_decay_monitor.py",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required_files if not p.exists()]
    if missing:
        findings.append(
            {
                "severity": "high",
                "area": "repo_integrity",
                "finding": f"Missing production files: {', '.join(missing)}",
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "watchlist_size": len(WATCHLIST),
        "files": _file_status(required_files),
        "model_quality": _quality_section(findings),
        "trade_rules": _trade_rule_section(findings),
        "signals": _signals_section(findings),
        "backtest": _backtest_section(findings),
        "paper": _paper_section(findings),
        "survivorship_bias": _survivorship_section(findings),
        "execution_stress": _execution_stress_section(findings),
        "drawdown_throttle": _drawdown_throttle_section(findings),
        "factor_decay": _factor_decay_section(findings),
        "robust_mode": _robust_mode_section(findings),
    }
    report["summary"] = _summary_section(report, findings)
    report["findings"] = sorted(findings, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["severity"], 9))
    LOGS.mkdir(parents=True, exist_ok=True)
    out_path = LOGS / "quant_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    report = run_audit()
    print(f"Quant audit written -> {Path(LOG_DIR) / 'quant_audit.json'}")
    print(
        "Model gate: "
        f"{report['model_quality']['approved']} approved / {report['model_quality']['rejected']} rejected"
    )
    print(
        "Signals: "
        f"{report['signals']['actionable']} actionable / "
        f"{report['signals']['approved_live_tickers']} approved live tickers"
    )
    print(
        "Primary strategy: "
        f"{'OK' if report['summary']['primary_strategy_ok'] else 'NOT OK'} | "
        f"largest risk: {report['summary']['largest_remaining_risk']}"
    )
    if report["summary"].get("real_capital_blocked_reason"):
        print(f"Real capital blocked: {report['summary']['real_capital_blocked_reason']}")
    for finding in report["findings"][:10]:
        print(f"[{finding['severity'].upper()}] {finding['area']}: {finding['finding']}")


if __name__ == "__main__":
    main()
