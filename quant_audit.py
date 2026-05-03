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
        "core_satellite_paper_ready": bool(core_signal.iloc[0].get("paper_ready", False)) if not core_signal.empty else False,
        "current_regime": paper_status.get("current_regime"),
        "account_equity": paper_status.get("account_equity"),
        "current_gross_exposure": paper_status.get("current_gross_exposure"),
        "target_gross_exposure": paper_status.get("target_gross_exposure"),
        "max_drift_abs": paper_status.get("max_drift_abs"),
        "submit_allowed": paper_status.get("submit_allowed"),
        "guard_reasons": paper_status.get("guard_reasons", []),
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
        "findings": sorted(findings, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["severity"], 9)),
    }
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
    for finding in report["findings"][:10]:
        print(f"[{finding['severity'].upper()}] {finding['area']}: {finding['finding']}")


if __name__ == "__main__":
    main()
