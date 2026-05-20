"""
core_satellite_drawdown_throttle.py — research overlay throttles for the
selected core-satellite strategy.

This script does not change the production signal. It reruns the selected
configuration with simple equity-curve drawdown rules that reduce the overlay
after losses, then reports whether any variant improves drawdown or Sharpe
without losing benchmark alpha.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_backtest import (
    attach_scores,
    benchmark_equity,
    compare_to_benchmarks,
    gate_metrics,
    load_factor_panel,
    load_feature_specs,
    load_prediction_scores,
    portfolio_stats,
    subperiod_metrics,
)
from backtest import INITIAL_CAPITAL
import core_satellite_alpha as core
from settings import SIGNAL_DIR, LOG_DIR, SLIPPAGE_BASE_PCT


OUT_CSV = Path(SIGNAL_DIR) / "core_satellite_drawdown_throttle.csv"
OUT_JSON = Path(LOG_DIR) / "core_satellite_drawdown_throttle.json"

CONFIG_KEYS = (
    "core_preset",
    "regime_mode",
    "core_weights",
    "score_source",
    "shape",
    "weighting",
    "exit_rank_floor",
    "adaptive_exit_mode",
    "max_per_sector",
    "earnings_blackout_days",
    "core_gross",
    "overlay_gross",
    "max_gross_exposure",
    "max_single_name_weight",
    "cost_stress",
    "holding_days",
    "regime_ma_window",
    "regime_high_vol",
)

THROTTLE_RULES = {
    "normal": {"threshold": None, "multiplier": 1.0},
    "soft_throttle_15": {"threshold": -0.15, "multiplier": 0.50},
    "hard_throttle_20": {"threshold": -0.20, "multiplier": 0.0},
}


def _selected_config() -> dict:
    metrics_path = Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json"
    if not metrics_path.exists():
        raise SystemExit("Missing signals/core_satellite_alpha_metrics.json. Run core_satellite_alpha.py first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8", errors="replace"))
    return {key: metrics[key] for key in CONFIG_KEYS if key in metrics}


def _return_col(config: dict) -> str:
    holding_days = int(config.get("holding_days", core.HORIZON_DAYS))
    entry_delay_days = int(config.get("entry_delay_days", 0))
    if entry_delay_days > 0:
        return f"forward_return_delay{entry_delay_days}_{holding_days}d"
    return "forward_return" if holding_days == core.HORIZON_DAYS else f"forward_return_{holding_days}d"


def _run_with_throttle(panel: pd.DataFrame, config: dict, rule_name: str) -> tuple[pd.Series, pd.DataFrame, dict]:
    rule = THROTTLE_RULES[rule_name]
    holding_days = int(config.get("holding_days", core.HORIZON_DAYS))
    entry_delay_days = int(config.get("entry_delay_days", 0))
    return_col = _return_col(config)
    if return_col not in panel.columns:
        raise ValueError(f"Missing return column: {return_col}")

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rebalance_dates = dates[::holding_days]
    entry_dates = pd.DatetimeIndex([pd.Timestamp(dt) + pd.tseries.offsets.BDay(entry_delay_days) for dt in rebalance_dates])
    exit_dates = pd.DatetimeIndex([pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days) for dt in rebalance_dates])
    price_index = pd.DatetimeIndex(sorted(set(rebalance_dates) | set(entry_dates) | set(exit_dates)))
    etf_prices = core._cached_etf_prices(price_index, ["SPY", "QQQ"])
    regime_indicators = None
    if str(config.get("regime_mode", "static")) in core.REGIME_PRESETS:
        regime_indicators = core._load_regime_indicators(rebalance_dates, exit_dates, config)
    day_map = core._panel_day_map(panel)

    equity = INITIAL_CAPITAL
    peak_equity = equity
    held: set[str] = set()
    prev_overlay = pd.Series(dtype=float)
    rows = [{"date": pd.Timestamp(rebalance_dates[0]), "equity": equity, "strategy_ret": 0.0}]
    trade_rows: list[dict] = []
    total_turnover = 0.0
    total_cost = 0.0
    throttle_periods = 0

    for dt in rebalance_dates:
        current_dd = equity / max(peak_equity, 1e-9) - 1.0
        multiplier = 1.0
        threshold = rule["threshold"]
        if threshold is not None and current_dd <= float(threshold):
            multiplier = float(rule["multiplier"])
            throttle_periods += 1

        day = day_map[pd.Timestamp(dt)]
        regime, core_weights, core_gross, overlay_gross_raw = core._resolve_allocation(pd.Timestamp(dt), config, regime_indicators)
        overlay_gross = float(overlay_gross_raw) * multiplier
        score_col = core._score_col_for_regime(str(config["score_source"]), regime)
        selected = core._select_sticky_holdings(
            day,
            held,
            score_col=score_col,
            return_col=return_col,
            shape=config["shape"],
            exit_rank_floor=core._exit_floor_for_regime(config, regime),
            max_per_sector=int(config["max_per_sector"]),
            earnings_blackout_days=int(config.get("earnings_blackout_days", 0)),
        )
        overlay = core._sticky_overlay_weights(
            selected,
            overlay_gross,
            config["weighting"],
            prev_overlay,
            max_single_name_weight=float(config.get("max_single_name_weight", core.MAX_SINGLE_NAME_WEIGHT)),
        )
        held = set(overlay.index.astype(str))
        aligned = pd.concat([prev_overlay.rename("prev"), overlay.rename("now")], axis=1).fillna(0.0)
        turnover = float((aligned["now"] - aligned["prev"]).abs().sum())
        cost = (
            turnover * SLIPPAGE_BASE_PCT * float(config.get("cost_stress", 1.0))
            + turnover * float(config.get("extra_turnover_cost_bps", 0.0)) / 10_000.0
        )
        total_turnover += turnover
        total_cost += cost

        factor_ret = 0.0
        if not overlay.empty:
            selected_by_ticker = selected.set_index("ticker")
            factor_ret = float((selected_by_ticker.loc[overlay.index, return_col] * overlay).sum())

        entry_dt = pd.Timestamp(dt) + pd.tseries.offsets.BDay(entry_delay_days)
        exit_dt = pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days)
        spy_ret = float(etf_prices.loc[exit_dt, "SPY"] / etf_prices.loc[entry_dt, "SPY"] - 1.0)
        qqq_ret = float(etf_prices.loc[exit_dt, "QQQ"] / etf_prices.loc[entry_dt, "QQQ"] - 1.0)
        core_ret = core_gross * (float(core_weights["SPY"]) * spy_ret + float(core_weights["QQQ"]) * qqq_ret)
        strategy_ret = core_ret + factor_ret - cost
        equity *= 1.0 + strategy_ret
        peak_equity = max(peak_equity, equity)

        rows.append({"date": exit_dt, "equity": equity, "strategy_ret": strategy_ret})
        trade_rows.append({
            "date": pd.Timestamp(dt),
            "exit_date": exit_dt,
            "rule": rule_name,
            "regime": regime,
            "drawdown_before_rebalance": current_dd,
            "overlay_multiplier": multiplier,
            "core_gross": float(core_gross),
            "overlay_gross": float(overlay.abs().sum()),
            "gross_exposure": float(core_gross + overlay.abs().sum()),
            "turnover": turnover,
            "cost": cost,
            "core_return": core_ret,
            "factor_overlay_return": factor_ret,
            "period_return": strategy_ret,
        })
        prev_overlay = overlay

    equity_series = pd.DataFrame(rows).drop_duplicates("date").set_index("date")["equity"].sort_index()
    trades = pd.DataFrame(trade_rows)
    extra = {
        "turnover_pct": round(total_turnover * 100.0, 2),
        "estimated_cost_pct": round(total_cost * 100.0, 4),
        "avg_gross_exposure": round(float(trades["gross_exposure"].mean()), 3) if not trades.empty else 0.0,
        "avg_overlay_gross": round(float(trades["overlay_gross"].mean()), 3) if not trades.empty else 0.0,
        "throttle_periods": int(throttle_periods),
        "n_rebalances": int(len(trades)),
    }
    return equity_series, trades, extra


def _evaluate(panel: pd.DataFrame, config: dict, rule_name: str) -> tuple[dict, pd.Series, pd.DataFrame]:
    equity, trades, extra = _run_with_throttle(panel, config, rule_name)
    periods_per_year = 252.0 / int(config.get("holding_days", core.HORIZON_DAYS))
    stats = portfolio_stats(equity, periods_per_year)
    bench = benchmark_equity(pd.DatetimeIndex(equity.index))
    bench_stats = {symbol: portfolio_stats(bench[symbol], periods_per_year) for symbol in bench.columns}
    comps = compare_to_benchmarks(equity, bench)
    subs = subperiod_metrics(equity, bench)
    holdout = core._holdout_comparisons(equity, bench, start="2023-01-01", end="2026-12-31")
    strat_rets = equity.pct_change().fillna(0.0)
    blend_rets = bench["BLEND"].pct_change().reindex(equity.index).fillna(0.0)
    yearly_alpha = (strat_rets - blend_rets).groupby(equity.index.year).sum() * 100.0
    metrics = {
        **config,
        **stats,
        **extra,
        "throttle_rule": rule_name,
        "benchmark_comparisons": comps,
        "benchmark_stats": bench_stats,
        "subperiods": subs,
        "holdout_2023_2026": holdout,
    }
    gates = gate_metrics(metrics, bench_stats, subs, yearly_alpha)
    gates.update(core._core_robust_gate_overrides(metrics, holdout, yearly_alpha))
    gates["all_pass"] = all(v for k, v in gates.items() if k.endswith("_pass"))
    metrics["core_satellite_gate_results"] = gates
    metrics["paper_ready"] = bool(gates["all_pass"])
    return metrics, equity, trades


def _row(metrics: dict, base: dict | None = None) -> dict:
    comps = metrics["benchmark_comparisons"]
    row = {
        "throttle_rule": metrics["throttle_rule"],
        "total_return_pct": metrics["total_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "sharpe": metrics["sharpe"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "alpha_vs_spy_pct": comps["SPY"]["alpha_pct"],
        "alpha_vs_qqq_pct": comps["QQQ"]["alpha_pct"],
        "alpha_vs_blend_pct": comps["BLEND"]["alpha_pct"],
        "turnover_pct": metrics["turnover_pct"],
        "estimated_cost_pct": metrics["estimated_cost_pct"],
        "avg_gross_exposure": metrics["avg_gross_exposure"],
        "avg_overlay_gross": metrics["avg_overlay_gross"],
        "throttle_periods": metrics["throttle_periods"],
        "paper_ready": bool(metrics["paper_ready"]),
    }
    if base is not None:
        row["return_delta_pct"] = round(float(row["total_return_pct"] - base["total_return_pct"]), 4)
        row["sharpe_delta"] = round(float(row["sharpe"] - base["sharpe"]), 4)
        row["max_drawdown_delta_pct"] = round(float(row["max_drawdown_pct"] - base["max_drawdown_pct"]), 4)
        row["improves_risk"] = bool(row["sharpe_delta"] > 0 or row["max_drawdown_delta_pct"] > 0)
        row["promotion_candidate"] = bool(
            row["paper_ready"]
            and row["improves_risk"]
            and row["alpha_vs_qqq_pct"] > 0
            and row["alpha_vs_blend_pct"] > 0
        )
    else:
        row["return_delta_pct"] = 0.0
        row["sharpe_delta"] = 0.0
        row["max_drawdown_delta_pct"] = 0.0
        row["improves_risk"] = False
        row["promotion_candidate"] = False
    return row


def main() -> None:
    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    config = _selected_config()
    specs = load_feature_specs()
    panel = core._ensure_robust_score_columns(attach_scores(load_factor_panel(specs), specs, load_prediction_scores()))

    evaluated: list[tuple[dict, pd.Series, pd.DataFrame]] = []
    for rule_name in THROTTLE_RULES:
        evaluated.append(_evaluate(panel, config, rule_name))

    base_metrics = evaluated[0][0]
    rows = [_row(metrics, None if metrics["throttle_rule"] == "normal" else _row(base_metrics)) for metrics, _eq, _tr in evaluated]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    best_candidates = out[out["promotion_candidate"].astype(bool)].copy()
    payload = {
        "generated_at": datetime.now().isoformat(),
        "purpose": "core_satellite_drawdown_throttle_research",
        "selected_config": config,
        "base_metrics": _row(base_metrics),
        "rows": rows,
        "best_promotion_candidate": best_candidates.head(1).to_dict("records")[0] if not best_candidates.empty else None,
        "pass_condition": "candidate improves Sharpe or max drawdown while retaining positive alpha vs QQQ and BLEND",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    print(f"Drawdown throttle research written -> {OUT_CSV}")
    print(f"Detailed report -> {OUT_JSON}")
    display_cols = [
        "throttle_rule",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "return_delta_pct",
        "sharpe_delta",
        "max_drawdown_delta_pct",
        "throttle_periods",
        "promotion_candidate",
    ]
    print(out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
