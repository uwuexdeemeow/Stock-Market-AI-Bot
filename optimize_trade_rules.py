from __future__ import annotations

import argparse
import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd

from backtest import walk_forward_predictions_for_ticker
from execution_model import commission as calc_commission, realistic_fill_price
from settings import DATA_DIR, MODEL_DIR, SIGNAL_DIR, SLIPPAGE_BASE_PCT
from trade_rules import TradeRule, append_rule_report, passes_trade_rule, resolve_rule_exit, save_trade_rule

INITIAL_CAPITAL = 10_000.0
MIN_TRADES = 15

GRID = {
    "confidence_threshold": [52.0, 58.0, 62.5],
    "min_expected_return": [0.0, 0.50, 0.75],
    "allowed_qualities": [("MEDIUM", "HIGH"), ("MEDIUM",)],
    "exit_horizon_days": [5, 10],
    "stop_loss_pct": [0.04, 0.08],
    "take_profit_pct": [0.06, 0.12],
    "max_position_pct": [0.05, 0.15],
}


def _trade_rule_combos(ticker: str):
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, values))
        yield TradeRule(ticker=ticker.upper(), allow_shorts=False, **params)


def simulate_rule(ticker: str, predictions: pd.DataFrame, hist: pd.DataFrame, rule: TradeRule, stress: float = 1.0) -> tuple[pd.DataFrame, dict]:
    cash = INITIAL_CAPITAL
    rows = []
    hist = hist.copy()
    hist.index = pd.DatetimeIndex(hist.index)

    if predictions.empty:
        return pd.DataFrame(), {"score": -1e9, "trades": 0}

    for dt, row in predictions.iterrows():
        passed, _reason = passes_trade_rule(row, rule, mode="long_only")
        if not passed:
            continue
        if str(row.get("signal", "")).upper() != "LONG":
            continue

        entry = float(row["open_next"])
        if entry <= 0:
            continue
        exit_date, exit_px, holding_days, exit_reason = resolve_rule_exit(hist, row, rule)
        current_equity = cash
        trade_value = current_equity * rule.max_position_pct
        shares = trade_value / entry
        adv_entry = float(hist["Volume"].loc[:pd.Timestamp(row["entry_date"])].tail(20).mean()) if "Volume" in hist.columns else 0.0
        adv_exit = float(hist["Volume"].loc[:pd.Timestamp(exit_date)].tail(20).mean()) if "Volume" in hist.columns else adv_entry
        fill_entry = realistic_fill_price(entry, shares, adv_entry, side="buy", base_slippage_pct=SLIPPAGE_BASE_PCT * stress)
        fill_exit = realistic_fill_price(exit_px, shares, adv_exit, side="sell", base_slippage_pct=SLIPPAGE_BASE_PCT * stress)
        fee_in = calc_commission(int(round(shares)))
        fee_out = calc_commission(int(round(shares)))
        net_pnl = shares * (fill_exit - fill_entry) - fee_in - fee_out
        cash += net_pnl
        rows.append({
            "date": pd.Timestamp(dt),
            "ticker": ticker.upper(),
            "signal": "LONG",
            "signal_quality": row.get("signal_quality", ""),
            "confidence": float(row.get("confidence", 0.0)),
            "expected_return": float(row.get("expected_return", 0.0)),
            "entry_date": pd.Timestamp(row["entry_date"]),
            "exit_date": pd.Timestamp(exit_date),
            "entry_price": round(fill_entry, 4),
            "exit_price": round(fill_exit, 4),
            "holding_days": int(holding_days),
            "exit_reason": exit_reason,
            "position_pct": round(rule.max_position_pct * 100.0, 2),
            "net_pnl": round(float(net_pnl), 2),
        })

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, {"score": -1e9, "trades": 0}

    pnl = trades["net_pnl"].astype(float)
    ret = pnl / INITIAL_CAPITAL
    sharpe = 0.0
    if len(ret) > 1 and float(ret.std()) > 0:
        sharpe = float((ret.mean() / ret.std()) * np.sqrt(252 / max(rule.exit_horizon_days, 1)))
    equity = INITIAL_CAPITAL + pnl.cumsum()
    dd = equity / equity.cummax() - 1.0
    total_pnl = float(pnl.sum())
    win_rate = float((pnl > 0).mean())
    max_dd = float(dd.min()) if len(dd) else 0.0
    profit_factor = float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if float(abs(pnl[pnl < 0].sum())) > 0 else 99.0
    score = sharpe * 100.0 + total_pnl / 25.0 + win_rate * 20.0 + min(profit_factor, 5.0) * 5.0 + max_dd * 100.0
    if len(trades) < MIN_TRADES:
        score -= (MIN_TRADES - len(trades)) * 20.0

    metrics = {
        "score": round(score, 4),
        "trades": int(len(trades)),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(float(pnl.mean()), 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "approved_candidate": bool(len(trades) >= MIN_TRADES and total_pnl > 0 and sharpe > 0.5 and max_dd > -0.25),
    }
    return trades, metrics


def optimize_ticker(ticker: str, stress: float = 1.0, force_predictions: bool = False) -> dict:
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    pred_path = os.path.join(SIGNAL_DIR, f"{ticker.upper()}_walkforward_predictions.csv")
    if os.path.exists(pred_path) and not force_predictions:
        predictions = pd.read_csv(pred_path, parse_dates=["date"]).set_index("date")
    else:
        predictions = walk_forward_predictions_for_ticker(ticker.upper())
        if not predictions.empty:
            predictions.reset_index().to_csv(pred_path, index=False)

    hist = pd.read_parquet(os.path.join(DATA_DIR, f"{ticker.upper()}.parquet"))
    hist.index = pd.DatetimeIndex(hist.index)

    best_rule = None
    best_metrics = None
    best_trades = pd.DataFrame()
    rows = []
    for rule in _trade_rule_combos(ticker):
        trades, metrics = simulate_rule(ticker, predictions, hist, rule, stress=stress)
        row = {**rule.to_json_dict(), **metrics}
        rows.append(row)
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_rule = rule
            best_metrics = metrics
            best_trades = trades

    if best_rule is None or best_metrics is None:
        raise RuntimeError(f"No trade rules evaluated for {ticker}")

    best_rule.optimized_at = datetime.now().isoformat(timespec="seconds")
    best_rule.optimization_score = float(best_metrics["score"])
    save_trade_rule(best_rule)

    out_dir = os.path.join(MODEL_DIR, "trade_rule_optimization")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(
        os.path.join(out_dir, f"{ticker.upper()}_trade_rule_grid.csv"),
        index=False,
    )
    best_trades.to_csv(os.path.join(SIGNAL_DIR, f"{ticker.upper()}_optimized_rule_trades.csv"), index=False)
    report_row = {**best_rule.to_json_dict(), **best_metrics}
    append_rule_report([report_row])
    return report_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize per-ticker trading rules from walk-forward predictions.")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--stress", type=float, default=1.0)
    parser.add_argument("--force-predictions", action="store_true")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = sorted(f.replace(".parquet", "") for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))

    rows = []
    for ticker in tickers:
        try:
            row = optimize_ticker(ticker, stress=args.stress, force_predictions=args.force_predictions)
            rows.append(row)
            print(f"{ticker}: score={row['score']} trades={row['trades']} pnl={row['total_pnl']} sharpe={row['sharpe']}")
        except Exception as e:
            print(f"ERROR - {ticker}: {e}")

    if rows:
        print(pd.DataFrame(rows).sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
