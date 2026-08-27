"""Independent daily mark-to-market and bounded shadow strategy audit.

PLAIN ENGLISH: The normal backtest records one profit/loss number per holding
period. That can hide a large loss in the middle of the period and can make
annual statistics look stronger than they are. This script rebuilds every day
from raw Open/Close prices, aligns the result exactly with QQQ, then tests only
the small shadow-only changes approved in the audit plan.

This script never writes the live strategy configuration and never talks to a
broker. Its only outputs are research reports and experiment-ledger rows.
"""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from alpha_factor_backtest import (
    attach_scores,
    load_factor_panel,
    load_feature_specs,
    load_prediction_scores,
)
from backtest import _newey_west_tstat
from core_satellite_alpha import _ensure_robust_score_columns, run_core_satellite
from execution_cost_calibration import calibrated_turnover_cost_pct
from experiment_ledger import append_experiment
from safe_io import atomic_write_csv, atomic_write_json
from settings import DATA_DIR, SIGNAL_DIR
from universe_membership import membership_status


WALKFORWARD_PATH = Path(SIGNAL_DIR) / "core_satellite_nested_walkforward.json"
VALIDATION_BUNDLE_PATH = Path(SIGNAL_DIR) / "core_satellite_validation_bundle.json"
OUTPUT_JSON = Path(SIGNAL_DIR) / "quant_performance_audit.json"
OUTPUT_CSV = Path(SIGNAL_DIR) / "quant_shadow_experiments.csv"
DEAD_FUNDAMENTAL_FEATURES = [
    "fund_pe_sector_z",
    "fund_fcf_yield_sector_z",
    "fund_value_combo_z",
]


class AuditDataError(ValueError):
    """Raised when raw prices cannot support an honest daily calculation."""


@dataclass
class DailyAuditPath:
    """Daily returns plus the evidence needed to audit their construction."""

    returns: pd.Series
    equity: pd.Series
    turnover: float
    rebalance_count: int
    cost_paid_fraction: float
    interval_rows: list[dict]


def _clean_prices(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize one ticker's raw prices without filling historical gaps."""
    required = {"Open", "Close"}
    if not required.issubset(frame.columns):
        raise AuditDataError(f"{ticker}: missing Open/Close columns")
    clean = frame.loc[:, ["Open", "Close"]].copy()
    index = pd.DatetimeIndex(pd.to_datetime(clean.index, errors="coerce"))
    if index.tz is not None:
        index = index.tz_convert(None)
    clean.index = index.normalize()
    clean = clean[~clean.index.isna()].sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    clean["Open"] = pd.to_numeric(clean["Open"], errors="coerce")
    clean["Close"] = pd.to_numeric(clean["Close"], errors="coerce")
    return clean


def parquet_price_loader(data_dir: Path = Path(DATA_DIR)) -> Callable[[str], pd.DataFrame]:
    """Build a cached loader for the project's raw per-ticker parquet files."""
    cache: dict[str, pd.DataFrame] = {}

    def load(ticker: str) -> pd.DataFrame:
        symbol = str(ticker).upper()
        if symbol not in cache:
            path = data_dir / f"{symbol}.parquet"
            if not path.exists():
                raise AuditDataError(f"{symbol}: raw price file missing at {path}")
            cache[symbol] = _clean_prices(pd.read_parquet(path), symbol)
        return cache[symbol]

    return load


def _json_weights(value: object, label: str) -> dict[str, float]:
    """Read a saved target-weight dictionary and reject malformed evidence."""
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = json.loads(str(value or "{}"))
        except json.JSONDecodeError as exc:
            raise AuditDataError(f"invalid {label} JSON") from exc
    if not isinstance(raw, dict):
        raise AuditDataError(f"{label} must be a dictionary")
    return {
        str(key).upper(): float(weight)
        for key, weight in raw.items()
        if abs(float(weight)) > 1e-12
    }


def _target_weights(trade: pd.Series) -> dict[str, float]:
    """Combine ETF core weights and stock overlay weights into one portfolio."""
    core = _json_weights(trade.get("core_weights_json"), "core_weights_json")
    overlay = _json_weights(trade.get("overlay_weights_json"), "overlay_weights_json")
    core_gross = float(trade.get("core_gross", 0.0) or 0.0)
    target = {ticker: core_gross * weight for ticker, weight in core.items()}
    for ticker, weight in overlay.items():
        target[ticker] = target.get(ticker, 0.0) + weight
    return target


def _full_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    """Count every ETF and stock target change, including the first purchase."""
    names = set(previous) | set(current)
    return float(sum(abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in names))


def daily_mark_to_market(
    trades: pd.DataFrame,
    price_loader: Callable[[str], pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    *,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
) -> DailyAuditPath:
    """Rebuild a daily portfolio path from raw prices and saved target weights.

    The signal is known only after its signal-date close, so even a zero-delay
    strategy enters at the next shared market session's Open. A one-day delay
    enters one additional session later. No signal-day price can affect P/L.
    """
    if trades.empty:
        raise AuditDataError("no trades supplied")
    required = {"date", "exit_date", "core_weights_json", "overlay_weights_json"}
    missing = required - set(trades.columns)
    if missing:
        raise AuditDataError("trade evidence missing columns: " + ", ".join(sorted(missing)))

    qqq = _clean_prices(benchmark_prices, "QQQ")
    calendar = qqq.dropna(subset=["Open", "Close"]).index
    rows = trades.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    rows["exit_date"] = pd.to_datetime(rows["exit_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    rows = rows.dropna(subset=["date", "exit_date"]).sort_values("date")

    daily_parts: list[pd.Series] = []
    interval_rows: list[dict] = []
    previous_target: dict[str, float] = {}
    total_turnover = 0.0
    total_cost = 0.0
    last_used_date: pd.Timestamp | None = None

    for _, trade in rows.iterrows():
        signal_date = pd.Timestamp(trade["date"])
        exit_limit = pd.Timestamp(trade["exit_date"])
        delay = max(0, int(float(trade.get("entry_delay_days", 0) or 0)))
        future_sessions = calendar[calendar > signal_date]
        if len(future_sessions) <= delay:
            raise AuditDataError(f"{signal_date.date()}: no executable entry session")
        entry_date = pd.Timestamp(future_sessions[delay])
        holding_calendar = calendar[(calendar >= entry_date) & (calendar <= exit_limit)]
        if len(holding_calendar) == 0:
            raise AuditDataError(f"{signal_date.date()}: no sessions through exit {exit_limit.date()}")
        # Fold segments must not count the same market day twice.
        if last_used_date is not None:
            holding_calendar = holding_calendar[holding_calendar > last_used_date]
        if len(holding_calendar) == 0:
            continue

        target = _target_weights(trade)
        turnover = _full_turnover(previous_target, target)
        transaction_cost = turnover * float(cost_bps) / 10_000.0
        interval_return = pd.Series(0.0, index=holding_calendar, dtype=float)
        for ticker, weight in target.items():
            prices = price_loader(ticker)
            needed = prices.reindex(holding_calendar)
            if needed[["Open", "Close"]].isna().any().any():
                missing_days = needed.index[needed[["Open", "Close"]].isna().any(axis=1)]
                raise AuditDataError(f"{ticker}: missing raw price on {missing_days[0].date()}")
            ticker_returns = needed["Close"].pct_change()
            ticker_returns.iloc[0] = needed["Close"].iloc[0] / needed["Open"].iloc[0] - 1.0
            interval_return = interval_return.add(float(weight) * ticker_returns, fill_value=0.0)

        # Costs hit equity once, on the actual entry day. Gross audits call
        # this function with cost_bps=0; net/stress audits use their own rate.
        interval_return.iloc[0] -= transaction_cost
        if bool((interval_return <= -1.0).any()):
            raise AuditDataError(f"{signal_date.date()}: portfolio loss reached or exceeded 100%")
        daily_parts.append(interval_return)
        interval_rows.append({
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": pd.Timestamp(holding_calendar[-1]).strftime("%Y-%m-%d"),
            "sessions": int(len(holding_calendar)),
            "positions": int(len(target)),
            "gross_exposure": round(float(sum(abs(v) for v in target.values())), 6),
            "cash_weight": round(float(1.0 - sum(target.values())), 6),
            "full_turnover": round(turnover, 8),
            "cost_fraction": round(transaction_cost, 10),
        })
        previous_target = target
        total_turnover += turnover
        total_cost += transaction_cost
        last_used_date = pd.Timestamp(holding_calendar[-1])

    if not daily_parts:
        raise AuditDataError("no auditable daily intervals")
    returns = pd.concat(daily_parts).sort_index()
    if returns.index.has_duplicates:
        raise AuditDataError("daily intervals overlap")
    equity = float(initial_capital) * (1.0 + returns).cumprod()
    return DailyAuditPath(
        returns=returns,
        equity=equity,
        turnover=total_turnover,
        rebalance_count=len(interval_rows),
        cost_paid_fraction=total_cost,
        interval_rows=interval_rows,
    )


def stitch_daily_paths(paths: list[DailyAuditPath], initial_capital: float = 100_000.0) -> DailyAuditPath:
    """Join disjoint OOS fold returns into one chronologically auditable curve."""
    if not paths:
        raise AuditDataError("no OOS paths to stitch")
    returns = pd.concat([path.returns for path in paths]).sort_index()
    if returns.index.has_duplicates:
        duplicates = returns.index[returns.index.duplicated()].unique()
        raise AuditDataError(f"OOS fold paths overlap on {duplicates[0].date()}")
    equity = float(initial_capital) * (1.0 + returns).cumprod()
    return DailyAuditPath(
        returns=returns,
        equity=equity,
        turnover=float(sum(path.turnover for path in paths)),
        rebalance_count=int(sum(path.rebalance_count for path in paths)),
        cost_paid_fraction=float(sum(path.cost_paid_fraction for path in paths)),
        interval_rows=[row for path in paths for row in path.interval_rows],
    )


def _elapsed_years(index: pd.DatetimeIndex) -> float:
    """Convert actual elapsed calendar days to years, never count observations."""
    if len(index) < 2:
        return 0.0
    elapsed = (pd.Timestamp(index[-1]) - pd.Timestamp(index[0])).days / 365.2425
    return max(elapsed, 1.0 / 365.2425)


def calendar_year_block_bootstrap(
    excess_returns: pd.Series,
    *,
    samples: int = 2_000,
    seed: int = 17,
) -> dict:
    """Resample whole calendar years so daily serial patterns stay together."""
    clean = pd.to_numeric(excess_returns, errors="coerce").dropna()
    groups = [group.to_numpy(dtype=float) for _, group in clean.groupby(clean.index.year) if len(group)]
    if len(groups) < 2:
        return {"method": "calendar_year_block_bootstrap", "samples": 0, "lower_95_pct": None, "upper_95_pct": None}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(samples)):
        draw = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        estimates.append(float(np.mean(draw) * 252.0 * 100.0))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "method": "calendar_year_block_bootstrap",
        "samples": int(samples),
        "lower_95_pct": round(float(lower), 4),
        "upper_95_pct": round(float(upper), 4),
        "positive_probability": round(float(np.mean(np.asarray(estimates) > 0.0)), 4),
    }


def audited_metrics(path: DailyAuditPath, qqq_prices: pd.DataFrame, *, bootstrap_samples: int = 2_000) -> dict:
    """Calculate elapsed-time, daily-risk, and exact-overlap OOS statistics."""
    qqq = _clean_prices(qqq_prices, "QQQ")
    qqq_returns = qqq["Close"].pct_change()
    aligned = pd.concat([path.returns.rename("strategy"), qqq_returns.rename("qqq")], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        raise AuditDataError("fewer than two exact strategy/QQQ overlap days")
    strategy = aligned["strategy"]
    benchmark = aligned["qqq"]
    excess = strategy - benchmark
    years = _elapsed_years(aligned.index)
    strategy_total = float((1.0 + strategy).prod() - 1.0)
    qqq_total = float((1.0 + benchmark).prod() - 1.0)
    strategy_cagr = float((1.0 + strategy_total) ** (1.0 / years) - 1.0)
    qqq_cagr = float((1.0 + qqq_total) ** (1.0 / years) - 1.0)
    equity = (1.0 + strategy).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    std = float(strategy.std(ddof=1))
    excess_std = float(excess.std(ddof=1))
    return {
        "start": aligned.index[0].strftime("%Y-%m-%d"),
        "end": aligned.index[-1].strftime("%Y-%m-%d"),
        "elapsed_years": round(years, 6),
        "strategy_days": int(len(path.returns)),
        "benchmark_overlap_days": int(len(aligned)),
        "benchmark_missing_days": int(len(path.returns.index.difference(qqq_returns.dropna().index))),
        "strategy_total_return_pct": round(strategy_total * 100.0, 4),
        "qqq_total_return_pct": round(qqq_total * 100.0, 4),
        "strategy_cagr_pct": round(strategy_cagr * 100.0, 4),
        "qqq_cagr_pct": round(qqq_cagr * 100.0, 4),
        "net_alpha_vs_qqq_pct": round((strategy_cagr - qqq_cagr) * 100.0, 4),
        "sharpe": round(float(strategy.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0, 4),
        "information_ratio_vs_qqq": round(float(excess.mean() / excess_std * math.sqrt(252.0)) if excess_std > 0 else 0.0, 4),
        "max_drawdown_pct": round(float(drawdown.min()) * 100.0, 4),
        "newey_west_alpha_tstat": round(float(_newey_west_tstat(excess)), 4),
        "annualized_turnover_pct": round(path.turnover / years * 100.0, 4),
        "cumulative_turnover_pct": round(path.turnover * 100.0, 4),
        "rebalance_count": int(path.rebalance_count),
        "summed_cost_fraction_pct": round(path.cost_paid_fraction * 100.0, 4),
        "uncertainty": calendar_year_block_bootstrap(excess, samples=bootstrap_samples),
    }


def recent_three_year_metrics(path: DailyAuditPath, qqq_prices: pd.DataFrame) -> dict:
    """Measure the latest three calendar years using the same daily method."""
    last_year = int(path.returns.index.max().year)
    recent = path.returns[path.returns.index.year >= last_year - 2]
    recent_equity = 100_000.0 * (1.0 + recent).cumprod()
    recent_path = DailyAuditPath(recent, recent_equity, 0.0, 0, 0.0, [])
    return audited_metrics(recent_path, qqq_prices, bootstrap_samples=500)


def candidate_gate(candidate: dict, baseline: dict, *, evidence_gates_pass: bool) -> dict:
    """Apply the plan's fixed keep/discard rules without expanding the search."""
    checks = {
        "information_ratio_improvement_at_least_0_10": float(candidate["net"]["information_ratio_vs_qqq"]) - float(baseline["net"]["information_ratio_vs_qqq"]) >= 0.10,
        "full_period_net_alpha_positive": float(candidate["net"]["net_alpha_vs_qqq_pct"]) > 0.0,
        "recent_three_year_net_alpha_positive": float(candidate["recent_three_year"]["net_alpha_vs_qqq_pct"]) > 0.0,
        "drawdown_no_more_than_2pct_worse": float(candidate["net"]["max_drawdown_pct"]) >= float(baseline["net"]["max_drawdown_pct"]) - 2.0,
        "annual_turnover_not_above_baseline": float(candidate["net"]["annualized_turnover_pct"]) <= float(baseline["net"]["annualized_turnover_pct"]) + 1e-9,
        "cost_25bps_not_worse": float(candidate["cost_25bps"]["information_ratio_vs_qqq"]) >= float(baseline["cost_25bps"]["information_ratio_vs_qqq"]),
        "one_day_delay_not_worse": float(candidate["delay_1d"]["information_ratio_vs_qqq"]) >= float(baseline["delay_1d"]["information_ratio_vs_qqq"]),
        "leakage_provenance_fold_robustness_gates": bool(evidence_gates_pass),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _with_overlay_cap(config: dict, gross: float) -> dict:
    """Lower every regime's overlay without accidentally raising risk-off risk."""
    out = deepcopy(config)
    out["overlay_gross"] = min(float(out.get("overlay_gross", gross)), gross)
    preset = out.get("regime_preset")
    if isinstance(preset, dict):
        for regime in ("risk_on", "neutral", "risk_off"):
            if isinstance(preset.get(regime), dict):
                old = float(preset[regime].get("overlay_gross", gross))
                preset[regime]["overlay_gross"] = min(old, gross)
    return out


def shadow_candidate_configs(active: dict) -> dict[str, dict]:
    """Return only the three approved one-change-at-a-time experiments."""
    baseline = deepcopy(active)
    top_five = deepcopy(active)
    top_five["shape"] = "top5"
    overlay_40 = _with_overlay_cap(active, 0.40)
    sticky = deepcopy(active)
    sticky["sticky_blend"] = 0.80
    return {
        "baseline_frozen_active": baseline,
        "top_five": top_five,
        "overlay_40pct": overlay_40,
        "sticky_blend_80pct": sticky,
    }


def _run_trades(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run a non-leveraged shadow config and return its target-position log."""
    if float(config.get("tqqq_weight", 0.0) or 0.0) > 0.0:
        raise AuditDataError("bounded shadow experiments prohibit TQQQ")
    _, trades, _ = run_core_satellite(panel, config)
    return trades


def _run_selected_config(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Recreate one historical fold with its original standard/TQQQ engine."""
    params = config.get("nested_params", {})
    tqqq_weight = float(params.get("tqqq_weight", config.get("tqqq_weight", 0.0)) or 0.0)
    if tqqq_weight <= 0.0:
        return _run_trades(panel, config)

    # Import lazily because normal shadow candidates deliberately prohibit the
    # leveraged ETF. This route exists only to reconstruct the one selected
    # 2013 OOS fold and does not make TQQQ an experiment candidate.
    from core_satellite_tqqq import run_tqqq_backtest

    _, _, trades = run_tqqq_backtest(
        panel,
        tqqq_weight=tqqq_weight,
        cost_stress=float(config.get("cost_stress", 2.0)),
        holding_days=int(params.get("holding_days", config.get("holding_days", 20))),
        preset_name=str(config.get("tqqq_preset", "tqqq_enhanced_cashbuffer")),
        score_source=str(params.get("score_source", config.get("score_source", "regime_adaptive"))),
        shape=str(config.get("shape", params.get("shape", "top5"))),
        weighting=str(config.get("weighting", params.get("weighting", "sticky_score"))),
        overlay_gross=float(params.get("overlay_gross", config.get("overlay_gross", 0.50))),
        regime_ma_window=int(params.get("ma_window", config.get("regime_ma_window", 100))),
        regime_high_vol=float(params.get("high_vol", config.get("regime_high_vol", 0.30))),
        high_vol_mode=str(params.get("high_vol_mode", config.get("high_vol_mode", "fixed"))),
        max_per_sector=int(config.get("max_per_sector", 2)),
        earnings_blackout_days=int(config.get("earnings_blackout_days", 0)),
        drawdown_circuit_breaker=float(config.get("drawdown_circuit_breaker", 0.0)),
        concentration_overlay_mode=str(config.get("concentration_overlay_mode", "off")),
        concentration_overlay_low_gross=float(config.get("concentration_overlay_low_gross", 0.30)),
        concentration_overlay_high_gross=float(config.get("concentration_overlay_high_gross", 0.70)),
        concentration_overlay_threshold=float(config.get("concentration_overlay_threshold", 0.05)),
        concentration_overlay_span=float(config.get("concentration_overlay_span", 0.05)),
        quiet=True,
    )
    return trades


def _filter_oos_trades(trades: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Keep only rebalance signals belonging to declared outer OOS years."""
    dates = pd.to_datetime(trades["date"], errors="coerce")
    return trades.loc[dates.dt.year.isin([int(year) for year in years])].copy()


def _evaluate_shadow_config(
    panel: pd.DataFrame,
    config: dict,
    years: list[int],
    loader: Callable[[str], pd.DataFrame],
    qqq: pd.DataFrame,
) -> dict:
    """Evaluate normal cost, 25-bps cost, and one-session-delay evidence."""
    normal_trades = _filter_oos_trades(_run_trades(panel, config), years)
    gross_path = daily_mark_to_market(normal_trades, loader, qqq, cost_bps=0.0)
    net_path = daily_mark_to_market(normal_trades, loader, qqq, cost_bps=calibrated_turnover_cost_pct() * 10_000.0)
    cost_path = daily_mark_to_market(normal_trades, loader, qqq, cost_bps=25.0)
    delayed_config = deepcopy(config)
    delayed_config["entry_delay_days"] = 1
    delayed_trades = _filter_oos_trades(_run_trades(panel, delayed_config), years)
    delay_path = daily_mark_to_market(delayed_trades, loader, qqq, cost_bps=calibrated_turnover_cost_pct() * 10_000.0)
    return {
        "gross": audited_metrics(gross_path, qqq),
        "net": audited_metrics(net_path, qqq),
        "recent_three_year": recent_three_year_metrics(net_path, qqq),
        "cost_25bps": audited_metrics(cost_path, qqq, bootstrap_samples=500),
        "delay_1d": audited_metrics(delay_path, qqq, bootstrap_samples=500),
        "interval_sample": net_path.interval_rows[:3],
    }


def _evidence_gates(bundle: dict, membership: dict) -> dict:
    """Separate research reproducibility from the blocked survivorship claim."""
    folds = bundle.get("folds", []) if isinstance(bundle, dict) else []
    robustness = bundle.get("robustness_reports", {}) if isinstance(bundle, dict) else {}
    checks = {
        "config_fingerprint_present": bool(bundle.get("config_fingerprint")),
        "all_outer_folds_complete": bool(folds) and all(bool(row.get("valid")) for row in folds),
        "robustness_fingerprints_match": bool(robustness) and all(bool(row.get("match")) for row in robustness.values()),
        "point_in_time_membership_complete": bool(membership.get("complete")),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _selected_fold_audit(
    payload: dict,
    panel: pd.DataFrame,
    loader: Callable[[str], pd.DataFrame],
    qqq: pd.DataFrame,
) -> dict:
    """Re-audit the selected OOS configurations, one outer year at a time."""
    fold_trade_rows: list[pd.DataFrame] = []
    blockers: list[str] = []
    audited_years: list[int] = []
    for item in payload.get("selected_configs", []):
        year = int(item.get("outer_year"))
        config = item.get("config", {})
        trades = _run_selected_config(panel, config)
        fold_trades = _filter_oos_trades(trades, [year])
        if fold_trades.empty:
            blockers.append(f"{year}:no_rebalance_evidence")
            continue
        fold_trade_rows.append(fold_trades)
        audited_years.append(year)
    if not fold_trade_rows:
        raise AuditDataError("no selected outer folds could be audited")
    # Rebuild one continuous portfolio so the first rebalance of a new outer
    # year turns over from the prior year's actual target rather than pretending
    # the account liquidated to cash at every fold boundary.
    selected_trades = pd.concat(fold_trade_rows, ignore_index=True, sort=False)
    stitched_gross = daily_mark_to_market(selected_trades, loader, qqq, cost_bps=0.0)
    stitched_net = daily_mark_to_market(
        selected_trades,
        loader,
        qqq,
        cost_bps=calibrated_turnover_cost_pct() * 10_000.0,
    )
    gross = audited_metrics(stitched_gross, qqq)
    net = audited_metrics(stitched_net, qqq)
    gross["gross_alpha_vs_qqq_pct"] = gross["net_alpha_vs_qqq_pct"]
    existing_return = float(payload.get("compound_oos_return_pct", 0.0) or 0.0)
    existing_sharpe = float(payload.get("mean_oos_sharpe", 0.0) or 0.0)
    existing_drawdown = float(payload.get("worst_oos_max_drawdown_pct", 0.0) or 0.0)
    periodic_to_gross = float(gross["strategy_total_return_pct"]) - existing_return
    gross_to_net = float(net["strategy_total_return_pct"]) - float(gross["strategy_total_return_pct"])
    return {
        "audited_outer_years": audited_years,
        "missing_outer_years": sorted(set(int(x.get("outer_year")) for x in payload.get("selected_configs", [])) - set(audited_years)),
        "gross": gross,
        "net": net,
        "existing_periodic_compound_oos_return_pct": existing_return,
        "net_return_difference_pct_points": round(float(net["strategy_total_return_pct"]) - existing_return, 4),
        "difference_reconciliation": [
            {
                "metric": "compound_oos_return_pct",
                "from": existing_return,
                "to": gross["strategy_total_return_pct"],
                "difference": round(periodic_to_gross, 4),
                "source": "raw daily path plus synchronized next-session Open timing replaces mixed periodic return construction",
            },
            {
                "metric": "compound_oos_return_pct",
                "from": gross["strategy_total_return_pct"],
                "to": net["strategy_total_return_pct"],
                "difference": round(gross_to_net, 4),
                "source": "full ETF-and-stock target turnover charged at calibrated transaction cost",
            },
            {
                "metric": "sharpe",
                "from": existing_sharpe,
                "to": net["sharpe"],
                "difference": round(float(net["sharpe"]) - existing_sharpe, 4),
                "source": "daily returns replace the mean of separately annualized holding-period fold Sharpes",
            },
            {
                "metric": "max_drawdown_pct",
                "from": existing_drawdown,
                "to": net["max_drawdown_pct"],
                "difference": round(float(net["max_drawdown_pct"]) - existing_drawdown, 4),
                "source": "daily marks expose losses hidden between rebalance exits and stitch fold peaks continuously",
            },
        ],
        "material_difference_sources": [
            "audit marks positions every market day instead of only at holding-period exits",
            "audit enters stocks and ETFs together at the next executable Open",
            "audit counts ETF plus stock target changes in turnover and costs",
            "audit annualizes from elapsed calendar time and daily returns",
            "audit uses only exact strategy/QQQ overlap without backfilling benchmark gaps",
        ],
        "blockers": blockers,
        "negative_cash_interval_count": int(
            sum(float(row.get("cash_weight", 0.0) or 0.0) < -1e-9 for row in stitched_net.interval_rows)
        ),
    }


def build_quant_audit(*, run_experiments: bool = True) -> tuple[dict, pd.DataFrame]:
    """Build the independent reference audit and optional bounded experiments."""
    payload = json.loads(WALKFORWARD_PATH.read_text(encoding="utf-8"))
    bundle = json.loads(VALIDATION_BUNDLE_PATH.read_text(encoding="utf-8")) if VALIDATION_BUNDLE_PATH.exists() else {}
    # Build the same score columns used by the strategy, then independently
    # audit only the resulting target positions and raw price path.
    specs = load_feature_specs(write_health_outputs=False)
    panel = _ensure_robust_score_columns(
        attach_scores(load_factor_panel(specs, require_forward_returns=True), specs, load_prediction_scores())
    )
    loader = parquet_price_loader()
    qqq = loader("QQQ")
    membership = membership_status(coverage_end=panel["date"].max())
    evidence = _evidence_gates(bundle, membership)
    selected_audit = _selected_fold_audit(payload, panel, loader, qqq)
    years = sorted(int(row["outer_year"]) for row in payload.get("folds", []) if row.get("valid"))

    experiments: list[dict] = []
    recommendation = {"status": "not_run", "candidate": None, "reason": "shadow_experiments_skipped"}
    if run_experiments:
        active = deepcopy(payload.get("approved_live_config", {}).get("config", {}))
        configs = shadow_candidate_configs(active)
        results: dict[str, dict] = {}
        baseline_name = "baseline_frozen_active"
        for name, config in configs.items():
            evaluated = _evaluate_shadow_config(panel, config, years, loader, qqq)
            gate = {"passed": True, "checks": {"frozen_baseline": True}} if name == baseline_name else candidate_gate(
                evaluated,
                results[baseline_name],
                evidence_gates_pass=evidence["passed"],
            )
            row = {"name": name, "config": config, **evaluated, "gate": gate}
            experiments.append(row)
            results[name] = evaluated
            append_experiment(
                name=f"quant_audit_{name}",
                params={
                    "holding_days": config.get("holding_days"),
                    "overlay_gross": config.get("overlay_gross"),
                    "shape": config.get("shape"),
                    "weighting": config.get("weighting"),
                    "sticky_blend": config.get("sticky_blend", 0.65),
                    "tqqq_weight": config.get("tqqq_weight", 0.0),
                },
                metrics={**evaluated["net"], "gate_pass": gate["passed"]},
                artifacts={"quant_audit": str(OUTPUT_JSON), "experiment_csv": str(OUTPUT_CSV)},
                notes="shadow-only; active paper configuration was not changed",
            )

        passed = [row for row in experiments[1:] if row["gate"]["passed"]]
        # A combined candidate is unavailable unless an isolated change passes.
        if passed:
            combined = deepcopy(active)
            for row in passed:
                if row["name"] == "top_five":
                    combined["shape"] = "top5"
                elif row["name"] == "overlay_40pct":
                    combined = _with_overlay_cap(combined, 0.40)
                elif row["name"] == "sticky_blend_80pct":
                    combined["sticky_blend"] = 0.80
            evaluated = _evaluate_shadow_config(panel, combined, years, loader, qqq)
            gate = candidate_gate(evaluated, results[baseline_name], evidence_gates_pass=evidence["passed"])
            combined_row = {"name": "combined_passed_changes", "config": combined, **evaluated, "gate": gate}
            experiments.append(combined_row)
            append_experiment(
                name="quant_audit_combined_passed_changes",
                params={"passed_components": [row["name"] for row in passed], "config": combined},
                metrics={**evaluated["net"], "gate_pass": gate["passed"]},
                artifacts={"quant_audit": str(OUTPUT_JSON), "experiment_csv": str(OUTPUT_CSV)},
                notes="combined only after isolated passes; shadow-only",
            )
            if gate["passed"]:
                recommendation = {"status": "shadow_candidate", "candidate": "combined_passed_changes", "reason": "all_fixed_guardrails_passed"}
        if recommendation["candidate"] is None:
            if passed:
                best = max(passed, key=lambda row: float(row["net"]["information_ratio_vs_qqq"]))
                recommendation = {"status": "shadow_candidate", "candidate": best["name"], "reason": "best_isolated_candidate_passing_all_guardrails"}
            else:
                recommendation = {"status": "keep_frozen_baseline", "candidate": baseline_name, "reason": "no_candidate_passed_all_fixed_guardrails"}

    reference_blockers = list(selected_audit.get("blockers", []))
    if not membership.get("complete"):
        reference_blockers.append("point_in_time_membership_incomplete")
    if not evidence["checks"].get("robustness_fingerprints_match"):
        reference_blockers.append("robustness_fingerprint_gate_failed")
    if int(selected_audit.get("negative_cash_interval_count", 0) or 0) > 0:
        reference_blockers.append("negative_cash_financing_cost_not_modeled")
    promotion_blockers = [
        *reference_blockers,
        "active_paper_requires_separate_decision_after_minimum_shadow_epoch",
    ]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "independent_daily_mark_to_market_quant_audit",
        "active_paper_configuration_changed": False,
        "production_selection_metrics_replaced": False,
        "reference_audit_status": "blocked" if reference_blockers else "passed",
        "reference_audit_blockers": sorted(set(reference_blockers)),
        "daily_timing": "signal close then next-session Open; one-day stress waits one additional session",
        "cost_method": "full target-weight turnover across ETFs and stocks",
        "dead_fundamental_features_excluded": DEAD_FUNDAMENTAL_FEATURES,
        "point_in_time_membership": membership,
        "validation_evidence_gates": evidence,
        "selected_walkforward_audit": selected_audit,
        "shadow_experiments": experiments,
        "ranked_shadow_recommendation": recommendation,
        "shadow_experiment_interpretation": "retrospective fixed-config comparison on declared outer-OOS calendar years; post-selection and never a live promotion by itself",
        "promotion_blockers": sorted(set(promotion_blockers)),
    }
    flat_rows = []
    for row in experiments:
        flat_rows.append({
            "name": row["name"],
            "gate_pass": bool(row["gate"]["passed"]),
            "net_information_ratio_vs_qqq": row["net"]["information_ratio_vs_qqq"],
            "net_alpha_vs_qqq_pct": row["net"]["net_alpha_vs_qqq_pct"],
            "recent_three_year_net_alpha_vs_qqq_pct": row["recent_three_year"]["net_alpha_vs_qqq_pct"],
            "net_max_drawdown_pct": row["net"]["max_drawdown_pct"],
            "annualized_turnover_pct": row["net"]["annualized_turnover_pct"],
            "cost_25bps_information_ratio": row["cost_25bps"]["information_ratio_vs_qqq"],
            "delay_1d_information_ratio": row["delay_1d"]["information_ratio_vs_qqq"],
            "failed_checks": ",".join(key for key, passed_check in row["gate"]["checks"].items() if not passed_check),
        })
    return report, pd.DataFrame(flat_rows)


def main() -> int:
    """Run the audit locally and write only shadow research artifacts."""
    parser = argparse.ArgumentParser(description="Build independent daily OOS and bounded shadow audit evidence.")
    parser.add_argument("--skip-experiments", action="store_true", help="Build selected-fold audit only; do not run candidate experiments.")
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    report, rows = build_quant_audit(run_experiments=not args.skip_experiments)
    atomic_write_json(report, args.output_json)
    atomic_write_csv(rows, args.output_csv, index=False)
    print(json.dumps({
        "reference_audit_status": report["reference_audit_status"],
        "recommendation": report["ranked_shadow_recommendation"],
        "promotion_blockers": report["promotion_blockers"],
        "json": str(args.output_json),
        "csv": str(args.output_csv),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
