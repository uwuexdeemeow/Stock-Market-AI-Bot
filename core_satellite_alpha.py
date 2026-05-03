"""
core_satellite_alpha.py — test factor alpha as a satellite around SPY/QQQ.

The prior alpha_factor_backtest asks whether factors can replace ETF beta. This
script asks a more realistic question: can a lower-turnover factor sleeve add
active return on top of a SPY/QQQ core without exceeding 1.25x gross exposure?
It is research-only and does not enable paper trading.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_backtest import (
    HORIZON_DAYS,
    MAX_GROSS_EXPOSURE,
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
from backtest import INITIAL_CAPITAL, _load_etf_price_frame
from settings import SIGNAL_DIR, SLIPPAGE_BASE_PCT


CORE_PRESETS = {
    "qqq_tilt_40_60": {"SPY": 0.40, "QQQ": 0.60},
    "qqq_heavy_25_75": {"SPY": 0.25, "QQQ": 0.75},
}
REGIME_PRESETS = {
    "qqq_trend_switch": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 1.00,
            "overlay_gross": 0.25,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_overlay35": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.90,
            "overlay_gross": 0.35,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_overlay50": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_fast_core110_overlay15": {
        "ma_window": 75,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 1.10,
            "overlay_gross": 0.15,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.85,
            "overlay_gross": 0.40,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_overlay70_core55": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.65,
            "overlay_gross": 0.35,
        },
    },
}
CORE_OVERLAY_COMBOS = (
    (1.00, 0.25),
    (0.75, 0.25),
    (0.75, 0.50),
)
SCORE_SOURCES = ("factor_walkforward", "regime_adaptive")
SHAPES = ("top3",)
WEIGHTING_MODES = ("score",)
EXIT_RANK_FLOORS = (0.80,)
MAX_PER_SECTOR_OPTIONS = (2,)
COST_STRESS_MULTIPLIERS = (2.0, 3.0)
HOLDING_DAY_OPTIONS = (10, 20)
PERIODS_PER_YEAR = 252.0 / HORIZON_DAYS
_PANEL_DAY_CACHE: dict[int, dict[pd.Timestamp, pd.DataFrame]] = {}
_ETF_PRICE_CACHE: dict[tuple[tuple[pd.Timestamp, ...], tuple[str, ...]], pd.DataFrame] = {}


def _panel_day_map(panel: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    cache_key = id(panel)
    cached = _PANEL_DAY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    mapped = {pd.Timestamp(dt): day for dt, day in panel.groupby("date", sort=True)}
    _PANEL_DAY_CACHE[cache_key] = mapped
    return mapped


def _cached_etf_prices(price_index: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    normalized_dates = tuple(pd.Timestamp(dt) for dt in price_index)
    key = (normalized_dates, tuple(tickers))
    cached = _ETF_PRICE_CACHE.get(key)
    if cached is not None:
        return cached
    prices = _load_etf_price_frame(pd.DatetimeIndex(normalized_dates), tickers)
    _ETF_PRICE_CACHE[key] = prices
    return prices


def _load_regime_indicators(rebalance_dates: pd.DatetimeIndex, exit_dates: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    start = pd.Timestamp(rebalance_dates.min()) - pd.tseries.offsets.BDay(260)
    end = pd.Timestamp(exit_dates.max())
    timeline = pd.date_range(start, end, freq="B")
    prices = _cached_etf_prices(pd.DatetimeIndex(timeline), ["SPY", "QQQ"])
    ma_window = int(config.get("regime_ma_window", 100))
    high_vol = float(config.get("regime_high_vol", 0.30))
    out = pd.DataFrame(index=prices.index)
    out["SPY"] = prices["SPY"]
    out["QQQ"] = prices["QQQ"]
    out["spy_trend_ok"] = prices["SPY"] >= prices["SPY"].rolling(200, min_periods=50).mean()
    out["qqq_trend_ok"] = prices["QQQ"] >= prices["QQQ"].rolling(ma_window, min_periods=50).mean()
    out["qqq_realized_vol"] = prices["QQQ"].pct_change().rolling(20, min_periods=10).std().mul(np.sqrt(252))
    out["high_vol"] = out["qqq_realized_vol"] > high_vol
    return out.ffill().bfill()


def _resolve_allocation(dt: pd.Timestamp, config: dict, regime_indicators: pd.DataFrame | None) -> tuple[str, dict[str, float], float, float]:
    regime_mode = str(config.get("regime_mode", "static"))
    if regime_mode in REGIME_PRESETS and regime_indicators is not None:
        row = regime_indicators.loc[pd.Timestamp(dt)]
        if bool(row["qqq_trend_ok"]) and bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
            regime = "risk_on"
        elif bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
            regime = "neutral"
        else:
            regime = "risk_off"
        preset = REGIME_PRESETS[regime_mode][regime]
        return (
            regime,
            dict(preset["core_weights"]),
            float(preset["core_gross"]),
            float(preset["overlay_gross"]),
        )
    return (
        "static",
        dict(config["core_weights"]),
        float(config["core_gross"]),
        float(config["overlay_gross"]),
    )


def _top_count(n_names: int, shape: str) -> int:
    if shape == "top3":
        return 3
    if shape == "top5":
        return 5
    if shape == "top10":
        return 10
    return max(1, int(np.ceil(n_names * 0.10)))


def _score_col(source: str) -> str:
    if source == "factor_plus_model":
        return "factor_plus_model_score"
    if source == "factor_walkforward":
        return "factor_walkforward_score"
    if source == "regime_adaptive":
        return "factor_walkforward_score"
    return "factor_score"


def _score_col_for_regime(source: str, regime: str) -> str:
    if source != "regime_adaptive":
        return _score_col(source)
    if regime == "risk_on":
        return "factor_risk_on_score"
    if regime == "risk_off":
        return "factor_defensive_score"
    return "factor_walkforward_score"


def _select_sticky_holdings(
    day: pd.DataFrame,
    held: set[str],
    *,
    score_col: str,
    return_col: str | None,
    shape: str,
    exit_rank_floor: float,
    max_per_sector: int,
) -> pd.DataFrame:
    required_cols = [score_col] if return_col is None else [score_col, return_col]
    ranked = day.dropna(subset=required_cols).copy()
    if ranked.empty:
        return ranked
    ranked["_rank_score"] = ranked[score_col].rank(pct=True)
    ranked = ranked.sort_values("_rank_score", ascending=False)
    target_n = _top_count(len(ranked), shape)

    keep = ranked[(ranked["ticker"].isin(held)) & (ranked["_rank_score"] >= exit_rank_floor)]
    selected_rows = []
    sector_counts: dict[str, int] = {}

    def _can_add(row: pd.Series) -> bool:
        if max_per_sector <= 0:
            return True
        sector = str(row.get("sector", "OTHER"))
        return sector_counts.get(sector, 0) < max_per_sector

    def _add(row: pd.Series) -> None:
        selected_rows.append(row)
        sector = str(row.get("sector", "OTHER"))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    for _idx, row in keep.sort_values("_rank_score", ascending=False).iterrows():
        if len(selected_rows) >= target_n:
            break
        if _can_add(row):
            _add(row)
    selected_tickers = {str(row["ticker"]) for row in selected_rows}

    for _idx, row in ranked.iterrows():
        if len(selected_rows) >= target_n:
            break
        ticker = str(row["ticker"])
        if ticker in selected_tickers:
            continue
        if _can_add(row):
            _add(row)
            selected_tickers.add(ticker)

    if not selected_rows:
        return ranked.iloc[0:0]
    return pd.DataFrame(selected_rows)


def _overlay_weights(selected: pd.DataFrame, overlay_gross: float, weighting: str) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    if weighting == "score":
        raw = (selected["_rank_score"] - selected["_rank_score"].min() + 0.01).clip(lower=0.01)
        raw_sum = float(raw.sum())
        weights = raw / raw_sum * overlay_gross if raw_sum > 0 else pd.Series(overlay_gross / len(selected), index=selected.index)
    else:
        weights = pd.Series(overlay_gross / len(selected), index=selected.index)
    weights.index = selected["ticker"].astype(str).to_numpy()
    return weights.astype(float)


def run_core_satellite(panel: pd.DataFrame, config: dict) -> tuple[pd.Series, pd.DataFrame, dict]:
    holding_days = int(config.get("holding_days", HORIZON_DAYS))
    return_col = "forward_return" if holding_days == HORIZON_DAYS else f"forward_return_{holding_days}d"
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rebalance_dates = dates[::holding_days]
    exit_dates = pd.DatetimeIndex([pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days) for dt in rebalance_dates])
    price_index = pd.DatetimeIndex(sorted(set(rebalance_dates) | set(exit_dates)))
    etf_prices = _cached_etf_prices(price_index, ["SPY", "QQQ"])
    regime_indicators = None
    if str(config.get("regime_mode", "static")) in REGIME_PRESETS:
        regime_indicators = _load_regime_indicators(rebalance_dates, exit_dates, config)
    day_map = _panel_day_map(panel)

    equity = INITIAL_CAPITAL
    held: set[str] = set()
    prev_overlay = pd.Series(dtype=float)
    rows = [{"date": pd.Timestamp(rebalance_dates[0]), "equity": equity, "strategy_ret": 0.0}]
    trade_rows: list[dict] = []
    total_turnover = 0.0
    total_cost = 0.0

    for dt in rebalance_dates:
        day = day_map[pd.Timestamp(dt)]
        regime, core_weights, core_gross, overlay_gross = _resolve_allocation(pd.Timestamp(dt), config, regime_indicators)
        score_col = _score_col_for_regime(str(config["score_source"]), regime)
        selected = _select_sticky_holdings(
            day,
            held,
            score_col=score_col,
            return_col=return_col,
            shape=config["shape"],
            exit_rank_floor=config["exit_rank_floor"],
            max_per_sector=int(config["max_per_sector"]),
        )
        overlay = _overlay_weights(selected, overlay_gross, config["weighting"])
        held = set(overlay.index.astype(str))

        aligned = pd.concat([prev_overlay.rename("prev"), overlay.rename("now")], axis=1).fillna(0.0)
        turnover = float((aligned["now"] - aligned["prev"]).abs().sum())
        cost = turnover * SLIPPAGE_BASE_PCT * float(config.get("cost_stress", 1.0))
        total_turnover += turnover
        total_cost += cost

        factor_ret = 0.0
        if not overlay.empty:
            selected_by_ticker = selected.set_index("ticker")
            factor_ret = float((selected_by_ticker.loc[overlay.index, return_col] * overlay).sum())

        exit_dt = pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days)
        spy_ret = float(etf_prices.loc[exit_dt, "SPY"] / etf_prices.loc[dt, "SPY"] - 1.0)
        qqq_ret = float(etf_prices.loc[exit_dt, "QQQ"] / etf_prices.loc[dt, "QQQ"] - 1.0)
        core_ret = core_gross * (float(core_weights["SPY"]) * spy_ret + float(core_weights["QQQ"]) * qqq_ret)
        strategy_ret = core_ret + factor_ret - cost
        equity *= 1.0 + strategy_ret

        rows.append({"date": exit_dt, "equity": equity, "strategy_ret": strategy_ret})
        trade_rows.append({
            "date": pd.Timestamp(dt),
            "exit_date": exit_dt,
            "regime": regime,
            "score_col": score_col,
            "n_overlay_positions": int(len(overlay)),
            "core_spy_weight": float(core_weights["SPY"]),
            "core_qqq_weight": float(core_weights["QQQ"]),
            "core_gross": core_gross,
            "overlay_gross": overlay_gross,
            "gross_exposure": core_gross + float(overlay.abs().sum()),
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
        "avg_overlay_positions": round(float(trades["n_overlay_positions"].mean()), 2) if not trades.empty else 0.0,
        "n_rebalances": int(len(trades)),
    }
    if "regime" in trades.columns:
        extra["regime_counts"] = {str(k): int(v) for k, v in trades["regime"].value_counts().sort_index().items()}
    return equity_series, trades, extra


def evaluate(panel: pd.DataFrame, config: dict) -> tuple[dict, pd.Series, pd.DataFrame]:
    equity, trades, extra = run_core_satellite(panel, config)
    periods_per_year = 252.0 / int(config.get("holding_days", HORIZON_DAYS))
    stats = portfolio_stats(equity, periods_per_year)
    bench = benchmark_equity(pd.DatetimeIndex(equity.index))
    bench_stats = {symbol: portfolio_stats(bench[symbol], periods_per_year) for symbol in bench.columns}
    comps = compare_to_benchmarks(equity, bench)
    subs = subperiod_metrics(equity, bench)
    strat_rets = equity.pct_change().fillna(0.0)
    blend_rets = bench["BLEND"].pct_change().reindex(equity.index).fillna(0.0)
    yearly_alpha = (strat_rets - blend_rets).groupby(equity.index.year).sum() * 100.0
    metrics = {
        **config,
        **stats,
        **extra,
        "benchmark_comparisons": comps,
        "benchmark_stats": bench_stats,
        "subperiods": subs,
        "yearly_alpha_pct": {str(k): round(float(v), 2) for k, v in yearly_alpha.items()},
    }
    gates = gate_metrics(metrics, bench_stats, subs, yearly_alpha)
    gates["all_pass"] = all(v for k, v in gates.items() if k.endswith("_pass"))
    metrics["core_satellite_gate_results"] = gates
    metrics["paper_ready"] = False
    return metrics, equity, trades


def write_paper_signal(panel: pd.DataFrame, metrics: dict) -> Path:
    holding_days = int(metrics.get("holding_days", HORIZON_DAYS))
    latest_date = pd.Timestamp(panel["date"].max())
    regime_indicators = None
    if str(metrics.get("regime_mode", "static")) in REGIME_PRESETS:
        regime_indicators = _load_regime_indicators(
            pd.DatetimeIndex([latest_date]),
            pd.DatetimeIndex([latest_date]),
            metrics,
        )
    current_regime, core_weights, core_gross, overlay_gross = _resolve_allocation(latest_date, metrics, regime_indicators)
    score_col = _score_col_for_regime(str(metrics["score_source"]), current_regime)
    day = panel[panel["date"] == latest_date]
    selected = _select_sticky_holdings(
        day,
        set(),
        score_col=score_col,
        return_col=None,
        shape=str(metrics["shape"]),
        exit_rank_floor=float(metrics["exit_rank_floor"]),
        max_per_sector=int(metrics["max_per_sector"]),
    )
    overlay = _overlay_weights(selected, overlay_gross, str(metrics["weighting"]))
    target_spy = core_gross * float(core_weights["SPY"])
    target_qqq = core_gross * float(core_weights["QQQ"])
    gross = core_gross + float(overlay.abs().sum())
    row = {
        "paper_signal_type": "core_satellite_alpha",
        "paper_ready": bool(metrics.get("paper_ready", False)),
        "core_preset": metrics["core_preset"],
        "regime_mode": str(metrics.get("regime_mode", "static")),
        "current_regime": current_regime,
        "score_source": metrics["score_source"],
        "target_spy_weight": round(target_spy, 4),
        "target_qqq_weight": round(target_qqq, 4),
        "target_cash_weight": round(1.0 - gross, 4),
        "gross_exposure": round(gross, 4),
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "overlay_gross": round(float(overlay.abs().sum()), 4),
        "core_gross": round(core_gross, 4),
        "holding_days": holding_days,
        "overlay_tickers": ",".join(overlay.index.astype(str).tolist()),
        "overlay_weights_json": json.dumps({str(k): round(float(v), 6) for k, v in overlay.items()}, sort_keys=True),
        "single_name_stock_picker_enabled": False,
        "ml_overlay_enabled": bool(metrics["score_source"] == "factor_plus_model"),
        "factor_overlay_enabled": True,
        "latest_factor_date": str(latest_date.date()),
        "cost_stress": float(metrics.get("cost_stress", 1.0)),
        "gates_all_pass": bool(metrics.get("core_satellite_gate_results", {}).get("all_pass", False)),
        "reason": "hardened core-satellite gates pass" if metrics.get("paper_ready") else "hardened core-satellite gates have not passed",
        "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    out = Path(SIGNAL_DIR) / "core_satellite_alpha_signal.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    return out


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):,.2f}%"
    except Exception:
        return "n/a"


def _fmt_num(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "n/a"


def print_run_summary(grid: pd.DataFrame, metrics: dict, signal_path: Path) -> None:
    comps = metrics["benchmark_comparisons"]
    gates = metrics["core_satellite_gate_results"]
    subperiods = metrics["subperiods"]

    print("\nCore-Satellite Alpha Summary")
    print("=" * 30)
    print(f"Status:        {'PAPER READY' if metrics.get('paper_ready') else 'NOT READY'}")
    print(f"Selected:      {metrics.get('core_preset')} ({metrics.get('regime_mode', 'static')})")
    print(f"Holding days:  {metrics.get('holding_days')}")
    print(f"Cost stress:   {_fmt_num(metrics.get('cost_stress'), 1)}x")
    print(f"Regime counts: {metrics.get('regime_counts', {})}")

    print("\nPerformance")
    print(f"  Total return: {_fmt_pct(metrics.get('total_return_pct'))}")
    print(f"  CAGR:         {_fmt_pct(metrics.get('cagr_pct'))}")
    print(f"  Sharpe:       {_fmt_num(metrics.get('sharpe'))}")
    print(f"  Max DD:       {_fmt_pct(metrics.get('max_drawdown_pct'))}")
    print(f"  Turnover:     {_fmt_pct(metrics.get('turnover_pct'))}")
    print(f"  Est. costs:   {_fmt_pct(metrics.get('estimated_cost_pct'))}")

    print("\nBenchmark Alpha")
    for symbol in ("SPY", "QQQ", "BLEND"):
        comp = comps.get(symbol, {})
        print(
            f"  vs {symbol:<5} alpha {_fmt_pct(comp.get('alpha_pct'))}"
            f" | benchmark {_fmt_pct(comp.get('benchmark_return_pct'))}"
            f" | t-stat {_fmt_num(comp.get('nw_tstat_vs_benchmark'))}"
        )

    print("\nSubperiod Alpha vs 60/40")
    for name, row in subperiods.items():
        if not row.get("data_available"):
            print(f"  {name:<9} n/a")
            continue
        print(
            f"  {name:<9} alpha {_fmt_pct(row.get('alpha_pct'))}"
            f" | strategy {_fmt_pct(row.get('return_pct'))}"
            f" | benchmark {_fmt_pct(row.get('benchmark_return_pct'))}"
        )

    failed = [name for name, ok in gates.items() if name.endswith("_pass") and not bool(ok)]
    print("\nGates")
    print(f"  Result: {'PASS' if gates.get('all_pass') else 'FAIL'}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print("  Failed: none")

    display_cols = [
        "paper_ready",
        "core_preset",
        "regime_mode",
        "holding_days",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "alpha_vs_qqq_pct",
        "alpha_vs_blend_pct",
        "subperiod_stability_pass",
    ]
    top = grid.head(5).copy()
    print("\nTop Grid Rows")
    print(top[[c for c in display_cols if c in top.columns]].to_string(index=False))

    print("\nSaved Outputs")
    print(f"  metrics: signals/core_satellite_alpha_metrics.json")
    print(f"  grid:    signals/core_satellite_alpha_grid.csv")
    print(f"  signal:  {signal_path}")


def main() -> None:
    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    specs = load_feature_specs()
    panel = attach_scores(load_factor_panel(specs), specs, load_prediction_scores())

    candidate_configs: list[dict] = []
    for core_preset, core_weights in CORE_PRESETS.items():
        for score_source in SCORE_SOURCES:
            for shape in SHAPES:
                for weighting in WEIGHTING_MODES:
                    for exit_rank_floor in EXIT_RANK_FLOORS:
                        for max_per_sector in MAX_PER_SECTOR_OPTIONS:
                            for core_gross, overlay_gross in CORE_OVERLAY_COMBOS:
                                for cost_stress in COST_STRESS_MULTIPLIERS:
                                    for holding_days in HOLDING_DAY_OPTIONS:
                                        if core_gross + overlay_gross > MAX_GROSS_EXPOSURE + 1e-9:
                                            continue
                                        candidate_configs.append({
                                            "core_preset": core_preset,
                                            "regime_mode": "static",
                                            "core_weights": core_weights,
                                            "score_source": score_source,
                                            "shape": shape,
                                            "weighting": weighting,
                                            "exit_rank_floor": float(exit_rank_floor),
                                            "max_per_sector": int(max_per_sector),
                                            "core_gross": float(core_gross),
                                            "overlay_gross": float(overlay_gross),
                                            "max_gross_exposure": MAX_GROSS_EXPOSURE,
                                            "cost_stress": float(cost_stress),
                                            "holding_days": int(holding_days),
                                        })

    for regime_name, regime_preset in REGIME_PRESETS.items():
        risk_on = regime_preset["risk_on"]
        for score_source in SCORE_SOURCES:
            for shape in SHAPES:
                for weighting in WEIGHTING_MODES:
                    for exit_rank_floor in EXIT_RANK_FLOORS:
                        for max_per_sector in MAX_PER_SECTOR_OPTIONS:
                            for cost_stress in COST_STRESS_MULTIPLIERS:
                                for holding_days in HOLDING_DAY_OPTIONS:
                                    candidate_configs.append({
                                        "core_preset": regime_name,
                                        "regime_mode": regime_name,
                                        "regime_ma_window": int(regime_preset["ma_window"]),
                                        "regime_high_vol": float(regime_preset["high_vol"]),
                                        "core_weights": dict(risk_on["core_weights"]),
                                        "score_source": score_source,
                                        "shape": shape,
                                        "weighting": weighting,
                                        "exit_rank_floor": float(exit_rank_floor),
                                        "max_per_sector": int(max_per_sector),
                                        "core_gross": float(risk_on["core_gross"]),
                                        "overlay_gross": float(risk_on["overlay_gross"]),
                                        "max_gross_exposure": MAX_GROSS_EXPOSURE,
                                        "cost_stress": float(cost_stress),
                                        "holding_days": int(holding_days),
                                    })

    rows: list[dict] = []
    for config in candidate_configs:
        metrics, equity, trades = evaluate(panel, config)
        comps = metrics["benchmark_comparisons"]
        gates = metrics["core_satellite_gate_results"]
        core_weights = config["core_weights"]
        rows.append({
            "core_preset": config["core_preset"],
            "regime_mode": config.get("regime_mode", "static"),
            "regime_ma_window": config.get("regime_ma_window", np.nan),
            "regime_high_vol": config.get("regime_high_vol", np.nan),
            "regime_counts": json.dumps(metrics.get("regime_counts", {}), sort_keys=True),
            "core_spy_weight": core_weights["SPY"],
            "core_qqq_weight": core_weights["QQQ"],
            "score_source": config["score_source"],
            "shape": config["shape"],
            "weighting": config["weighting"],
            "exit_rank_floor": float(config["exit_rank_floor"]),
            "max_per_sector": int(config["max_per_sector"]),
            "core_gross": float(config["core_gross"]),
            "overlay_gross": float(config["overlay_gross"]),
            "cost_stress": float(config["cost_stress"]),
            "holding_days": int(config["holding_days"]),
            "max_gross_exposure": MAX_GROSS_EXPOSURE,
            "total_return_pct": metrics["total_return_pct"],
            "cagr_pct": metrics["cagr_pct"],
            "sharpe": metrics["sharpe"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "alpha_vs_spy_pct": comps["SPY"]["alpha_pct"],
            "alpha_vs_qqq_pct": comps["QQQ"]["alpha_pct"],
            "alpha_vs_blend_pct": comps["BLEND"]["alpha_pct"],
            "nw_tstat_vs_blend": comps["BLEND"]["nw_tstat_vs_benchmark"],
            "turnover_pct": metrics["turnover_pct"],
            "estimated_cost_pct": metrics["estimated_cost_pct"],
            "avg_gross_exposure": metrics["avg_gross_exposure"],
            "avg_overlay_positions": metrics["avg_overlay_positions"],
            "subperiod_stability_pass": gates["subperiod_stability_pass"],
            "paper_ready": gates["all_pass"],
        })

    grid = pd.DataFrame(rows).sort_values(
        ["paper_ready", "alpha_vs_blend_pct", "sharpe", "max_drawdown_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    grid_path = Path(SIGNAL_DIR) / "core_satellite_alpha_grid.csv"
    grid.to_csv(grid_path, index=False)
    if grid.empty:
        raise SystemExit("No core-satellite configs evaluated.")

    hardened = grid[(grid["score_source"].isin(["factor_walkforward", "regime_adaptive"])) & (grid["cost_stress"] >= 2.0)]
    hardened_pass = hardened[hardened["paper_ready"]]
    if not hardened_pass.empty:
        selected = hardened_pass.iloc[0]
        selection_reason = "best_hardened_walkforward_passing_row"
    elif not hardened.empty:
        selected = hardened.iloc[0]
        selection_reason = "best_hardened_walkforward_row"
    else:
        selected = grid.iloc[0]
        selection_reason = "best_available_row"

    selected_config = {
        "core_preset": str(selected["core_preset"]),
        "regime_mode": str(selected.get("regime_mode", "static")),
        "core_weights": {"SPY": float(selected["core_spy_weight"]), "QQQ": float(selected["core_qqq_weight"])},
        "score_source": str(selected["score_source"]),
        "shape": str(selected["shape"]),
        "weighting": str(selected["weighting"]),
        "exit_rank_floor": float(selected["exit_rank_floor"]),
        "max_per_sector": int(selected["max_per_sector"]),
        "core_gross": float(selected["core_gross"]),
        "overlay_gross": float(selected["overlay_gross"]),
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "cost_stress": float(selected["cost_stress"]),
        "holding_days": int(selected["holding_days"]),
    }
    if str(selected_config["regime_mode"]) in REGIME_PRESETS:
        preset = REGIME_PRESETS[str(selected_config["regime_mode"])]
        selected_config["regime_ma_window"] = int(preset["ma_window"])
        selected_config["regime_high_vol"] = float(preset["high_vol"])
    best_metrics, best_equity, best_trades = evaluate(panel, selected_config)
    best_metrics["selected_features"] = specs
    best_metrics["grid_rows"] = int(len(grid))
    best_metrics["best_config_source"] = selection_reason
    best_metrics["paper_ready"] = bool(best_metrics["core_satellite_gate_results"]["all_pass"])
    pd.DataFrame({"equity": best_equity}).to_csv(Path(SIGNAL_DIR) / "core_satellite_alpha_equity.csv")
    best_trades.to_csv(Path(SIGNAL_DIR) / "core_satellite_alpha_trades.csv", index=False)
    signal_path = write_paper_signal(panel, best_metrics)
    with open(Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print_run_summary(grid, best_metrics, signal_path)


if __name__ == "__main__":
    main()
