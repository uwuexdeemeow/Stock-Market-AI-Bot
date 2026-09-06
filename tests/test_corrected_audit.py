"""Independent accounting examples and causal counterfactuals for the remaining audit."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from broker_history import collect_order_history, implementation_shortfall
from causal_research import fit_features, score_features
from corrected_audit import prospective_status
from edge_evidence import edge_summary
from execution_cost_calibration import causal_cost_parameters
from labels import make_spy_forward_return, make_direction_target
from paper_policy import PaperPolicy, whole_share_target
from portfolio_ledger import Account, simulate_daily, daily_metrics, trailing_outcome, replay_events, LEDGER_VERSION
from trade_rules import TradeRule, resolve_rule_exit


PROVENANCE = {"adjustment_mode": "raw_ohlcv", "actions_verified": True}
POLICY = PaperPolicy(cash_buffer_pct=0, cash_buffer_min=0, minimum_trade=0, etf_drift=0, stock_drift=0)


def bars_for(prices, ticker="SPY"):
    dates = pd.bdate_range("2024-02-01", periods=len(prices))
    return pd.DataFrame({"date": dates, "ticker": ticker, "Open": prices, "High": prices,
                         "Low": prices, "Close": prices, "Volume": 1000000.})


def test_initial_etf_purchase_terminal_sale_and_costs():
    bars = bars_for([100.] * 6)
    result = simulate_daily(bars, lambda *_: {"SPY": .5}, start=bars.date.iloc[1], end=bars.date.iloc[-1], provenance=PROVENANCE, policy=POLICY)
    fills = result.events.query("kind == 'fill'")
    assert fills.iloc[0].quantity == 500
    assert fills.iloc[-1].reason == "terminal"
    assert fills.quantity.sum() == 0
    assert result.metrics["turnover_pct"] >= 99
    assert result.metrics["estimated_cost_pct"] > 0
    assert result.equity.iloc[-1].cash == pytest.approx(100000 - fills.execution_cost.sum() - fills.fee.sum())


def test_drift_rebalances_unchanged_target():
    bars = bars_for([100, 100, 120, 120])
    result = simulate_daily(bars, lambda *_: {"SPY": .5}, start=bars.date.iloc[1], end=bars.date.iloc[-1], provenance=PROVENANCE,
                            policy=POLICY, cost_stress=0, terminal_liquidation=False)
    fills = result.events.query("kind == 'fill'")
    assert fills.quantity.tolist()[:2] == [500, -42]
    assert result.metrics["turnover_pct"] > 50


def test_daily_drawdown_survives_recovery():
    dates = pd.bdate_range("2024-02-01", periods=3)
    result = daily_metrics(pd.Series([100., 60., 100.], index=dates))
    assert result["max_drawdown_pct"] == pytest.approx(-40)
    assert result["total_return_pct"] == 0


def test_conservation_partial_fills_and_duplicate_rejection():
    events = pd.DataFrame([
        {"timestamp": "2024-02-01T14:35:00Z", "kind": "submitted", "order_id": "parent", "quantity": 10},
        {"timestamp": "2024-02-01T14:35:01Z", "kind": "fill", "order_id": "parent", "event_id": "one", "ticker": "SPY", "quantity": 4, "price": 100., "fee": 1.},
        {"timestamp": "2024-02-01T14:35:02Z", "kind": "fill", "order_id": "parent", "event_id": "two", "ticker": "SPY", "quantity": 6, "price": 101., "fee": 1.},
    ])
    result = replay_events(events, opening_cash=2000, opening_holdings={}, expected_cash=992, expected_holdings={"SPY": 10})
    assert result.metrics["reconciled"]
    assert result.metrics["cash"] == 992
    with pytest.raises(ValueError, match="Duplicate fill"):
        replay_events(pd.concat([events, events.iloc[[-1]]]), opening_cash=2000, opening_holdings={})


def test_unknown_fee_or_opening_balance_cannot_reconcile():
    event = pd.DataFrame([{"timestamp": "2024-02-01", "kind": "fill", "event_id": "one", "ticker": "SPY", "quantity": 1, "price": 100., "fee": np.nan}])
    assert not replay_events(event, opening_cash=None, opening_holdings={}).metrics["reconciled"]
    result = replay_events(event, opening_cash=1000, opening_holdings={}, expected_cash=900, expected_holdings={"SPY": 1})
    assert not result.metrics["reconciled"]
    assert any(g["reason"] == "fee_missing" for g in result.data_quality)


def test_replay_values_old_holdings_before_later_sale():
    events = pd.DataFrame([
        {"timestamp": "2024-02-01T14:35:00Z", "kind": "fill", "event_id": "one", "ticker": "SPY", "quantity": 1, "price": 100., "fee": 0.},
        {"timestamp": "2024-02-02T14:35:00Z", "kind": "fill", "event_id": "two", "ticker": "SPY", "quantity": -1, "price": 120., "fee": 0.},
    ])
    marks = pd.DataFrame([{"timestamp": "2024-02-01T21:00:00Z", "ticker": "SPY", "price": 110.},
                          {"timestamp": "2024-02-02T21:00:00Z", "ticker": "SPY", "price": 120.}])
    result = replay_events(events, opening_cash=1000, opening_holdings={}, expected_cash=1020, expected_holdings={}, marks=marks)
    assert result.equity.equity.tolist() == [1010, 1020]
    assert result.metrics["reconciled"]


def test_split_and_dividend_cash_without_double_counting():
    account = Account(1000, {"SPY": 10})
    account.action("2024-02-01", {"event_id": "split", "kind": "split", "ticker": "SPY", "value": 2., "source": "fixture"})
    assert account.shares["SPY"] == 20
    assert account.mark({"SPY": 50.}) == 2000
    account.action("2024-02-02", {"event_id": "dividend", "kind": "dividend", "ticker": "SPY", "value": 1., "entitled_shares": 20, "source": "fixture"})
    assert account.cash == 1020
    with pytest.raises(ValueError, match="Duplicate"):
        account.action("2024-02-02", {"event_id": "dividend", "kind": "dividend", "ticker": "SPY", "value": 1., "entitled_shares": 20, "source": "fixture"})


def test_dividend_entitlement_kept_after_sale():
    bars = bars_for([100.] * 5)
    actions = pd.DataFrame([{"event_id": "d", "kind": "dividend", "ticker": "SPY", "ex_date": bars.date.iloc[2],
                             "date": bars.date.iloc[3], "value": 1., "source": "fixture"}])
    target = lambda date, *_: {"SPY": .5 if date == bars.date.iloc[0] else 0.}
    result = simulate_daily(bars, target, start=bars.date.iloc[1], end=bars.date.iloc[-1], actions=actions,
                            provenance=PROVENANCE, policy=POLICY, cost_stress=0)
    assert result.equity.iloc[-1].cash == 100500


def test_stops_gap_entry_day_and_ambiguous_path():
    fill, _, _ = trailing_outcome(pd.Series({"Open": 80, "High": 90, "Low": 70, "Close": 85}), 100, .08)
    assert fill == 80
    fill, _, ambiguous = trailing_outcome(pd.Series({"Open": 100, "High": 120, "Low": 95, "Close": 115}), 100, .08)
    assert fill == pytest.approx(110.4)
    assert ambiguous


def test_halt_does_not_recover_merely_by_becoming_cash():
    bars = bars_for([100, 100, 70, 70, 70])
    result = simulate_daily(bars, lambda *_: {"SPY": 1.}, start=bars.date.iloc[1], end=bars.date.iloc[-1],
                            provenance=PROVENANCE, policy=POLICY, cost_stress=0, terminal_liquidation=False)
    fills = result.events.query("kind == 'fill'")
    assert fills.iloc[-1].reason == "drawdown_halt"
    assert fills.quantity.tolist() == [1000, -1000]
    assert result.equity.iloc[-1].halted


def test_adjusted_execution_data_is_rejected():
    bars = bars_for([100.] * 3)
    with pytest.raises(ValueError, match="raw prices"):
        simulate_daily(bars, lambda *_: {}, start=bars.date.iloc[1], end=bars.date.iloc[-1], provenance={"adjustment_mode": "adjusted_ohlcv"})


def test_future_perturbation_cannot_change_feature_artifact_or_scores():
    dates = pd.bdate_range("2024-01-02", periods=60)
    panel = pd.DataFrame([{"date": date, "ticker": str(i), "f": i, "g": 4 - i, "ret": i / 100,
                           "ret_end_date": date + pd.Timedelta(days=2)} for date in dates for i in range(5)])
    cutoff = dates[35]
    first = fit_features(panel, ["f", "g"], cutoff=cutoff, label="ret")
    changed = panel.copy()
    changed.loc[changed.date > cutoff, ["f", "g", "ret"]] = -999
    second = fit_features(changed, ["f", "g"], cutoff=cutoff, label="ret")
    assert first == second
    pd.testing.assert_series_equal(score_features(panel, first).loc[panel.date <= cutoff, "causal_score"],
                                   score_features(changed, second).loc[panel.date <= cutoff, "causal_score"])


def test_cost_calibration_ignores_later_fills_and_removes_components():
    rows = [{"filled_at": "2024-01-01", "arrival_shortfall_bps": 12, "half_spread_bps": 2, "modeled_impact_bps": 3}] * 20
    before = causal_cost_parameters({"orders": rows}, "2024-02-01")
    after = causal_cost_parameters({"orders": rows + [{"filled_at": "2025-01-01", "arrival_shortfall_bps": 99999}]}, "2024-02-01")
    assert before == after
    assert before["base_slippage_pct"] >= .0007


def test_positive_raw_return_underperforming_benchmark_is_not_healthy():
    dates = pd.bdate_range("2020-01-01", periods=450)
    result = edge_summary(pd.Series(.001, index=dates), pd.Series(.002, index=dates), pd.Series([.2] * 21), simulations=100)
    assert result["net_portfolio_return_pct"] > 0
    assert result["benchmark_excess_return_pct"] < 0
    assert not result["statistical_healthy"]
    assert result["excess_return_ci95_pct"][1] < 0


def test_missing_benchmark_rejected_but_raw_labels_work():
    frame = pd.DataFrame({"Close": np.arange(100., 110.)})
    with pytest.raises(ValueError, match="benchmark"):
        make_spy_forward_return(frame)
    with pytest.raises(ValueError, match="Benchmark"):
        make_direction_target(frame.Close, prediction_target="excess_return")
    assert len(make_direction_target(frame, prediction_target="raw_return")) == len(frame)


def test_rule_exit_includes_entry_day_and_gap():
    hist = pd.DataFrame({"Open": [100., 80., 90.], "High": [110., 95., 100.], "Low": [90., 75., 80.], "Close": [100., 90., 90.]}, index=pd.bdate_range("2024-02-01", periods=3))
    row = {"entry_date": hist.index[0], "open_next": 100., "signal": "LONG"}
    rule = TradeRule("SPY", exit_horizon_days=2)
    date, price, _, reason = resolve_rule_exit(hist, row, rule)
    assert date == hist.index[0] and price == 94 and reason == "stop_loss"
    hist.loc[hist.index[0], ["High", "Low"]] = [102, 98]
    assert resolve_rule_exit(hist, row, rule)[1] == 80


def test_paginated_history_covers_more_than_one_page_and_detects_stall():
    rows = [SimpleNamespace(id=str(i), submitted_at=pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(seconds=i)) for i in range(250)]
    class API:
        def list_orders(self, **kwargs):
            return sorted([row for row in rows if pd.Timestamp(kwargs["after"]) < row.submitted_at < pd.Timestamp(kwargs["until"])],
                          key=lambda row: row.submitted_at, reverse=True)[:kwargs["limit"]]
    result = collect_order_history(API(), after="2023-01-01T00:00:00Z", until="2025-01-01T00:00:00Z")
    assert result.complete and len(result.orders) == 250
    stalled = collect_order_history(SimpleNamespace(list_orders=lambda **kwargs: rows[:100]), after="2023-01-01T00:00:00Z")
    assert not stalled.complete


def test_history_failure_and_saturated_timestamp_are_incomplete():
    def fail(**kwargs):
        raise RuntimeError("interrupted")
    assert not collect_order_history(SimpleNamespace(list_orders=fail), after="2023-01-01T00:00:00Z").complete
    rows = [SimpleNamespace(id=str(i), submitted_at="2024-01-01T00:00:00Z") for i in range(100)]
    assert not collect_order_history(SimpleNamespace(list_orders=lambda **kwargs: rows), after="2023-01-01T00:00:00Z").complete


def test_arrival_shortfall_does_not_invent_quotes():
    rows = pd.DataFrame([{"parent_order_id": "one", "side": "buy", "filled_qty": 1, "fill_price": 101, "arrival_mid": 100},
                         {"parent_order_id": "two", "side": "buy", "filled_qty": 1, "fill_price": 101, "arrival_mid": np.nan}])
    result = implementation_shortfall(rows)
    assert not result["complete"]
    assert result["fills"][0]["shortfall_bps"] == pytest.approx(100)


def test_whole_shares_match_existing_rounding():
    assert whole_share_target(1051, .5, 100) == round(1051 * .5 / 100)


def test_prospective_period_counts_new_observations_and_restarts():
    frozen = {"frozen_at": "2024-02-01T23:00:00Z", "strategy_fingerprint": "one"}
    result = prospective_status(frozen, now="2024-02-05T23:00:00Z", current_fingerprint="one", observed_sessions=["2024-02-02", "2024-02-05"])
    assert result["observed_sessions"] == 2
    assert result["status"] == "collecting"
    assert prospective_status(frozen, now="2024-02-05T23:00:00Z", current_fingerprint="two", observed_sessions=[])["status"] == "restart_required"


def test_history_includes_both_interval_boundaries():
    stamps = pd.date_range("2024-02-01", periods=4, tz="UTC")
    rows = [SimpleNamespace(id=str(i), submitted_at=stamp) for i, stamp in enumerate(stamps)]
    def page(**kwargs):
        return [row for row in reversed(rows) if pd.Timestamp(kwargs["after"]) < row.submitted_at < pd.Timestamp(kwargs["until"])][:kwargs["limit"]]
    result = collect_order_history(SimpleNamespace(list_orders=page), after=stamps[1], until=stamps[2], page_size=3)
    assert result.complete and {row.id for row in result.orders} == {"1", "2"}


def test_parent_shortfall_deduplicates_children_and_keeps_unfilled_unknown():
    from broker_history import parent_execution_summary
    rows = pd.DataFrame([{"order_id": "a1", "parent_order_id": "a", "side": "buy", "status": "canceled", "filled_qty": 4, "original_requested_quantity": 10},
                         {"order_id": "a2", "parent_order_id": "a", "side": "buy", "status": "filled", "filled_qty": 3, "original_requested_quantity": 10}])
    result = parent_execution_summary(pd.concat([rows, rows.iloc[[0]]]))
    assert result["child_order_count"] == 2
    assert result["parents"][0]["unfilled_quantity"] == 3
    assert not result["complete"]
    rows = rows.assign(arrival_mid=100, opportunity_price=102, opportunity_timestamp="2024-02-02T21:00Z", opportunity_horizon="decision_session_close")
    measured = parent_execution_summary(rows)
    assert measured["complete"] and measured["parents"][0]["opportunity_cost_dollars"] == 6


def test_recovered_paper_fixture_preserves_known_shares_without_invented_cash():
    import json
    from pathlib import Path
    report = json.loads((Path(__file__).parent / "fixtures/execution_20260903/report.json").read_text())
    events = pd.DataFrame([{"kind": "fill", "timestamp": row["filled_at"], "ticker": row["symbol"], "quantity": row["filled_qty"],
                            "price": row["fill_price"], "event_id": row["order_id"]} for row in report["orders"]])
    result = replay_events(events, opening_cash=None, opening_holdings=None)
    assert result.metrics["known_share_changes"] == {"INTC": 205., "FCX": 30.}
    assert not result.metrics["reconciled"]
    assert len([gap for gap in result.data_quality if gap["reason"] == "fee_missing"]) == 3


def test_split_prices_do_not_create_a_false_momentum_crash():
    from corrected_data import build_raw_features
    bars = bars_for([100.] * 22 + [50.] * 3, ticker="ABC")
    actions = pd.DataFrame([{"kind": "split", "ticker": "ABC", "date": bars.date.iloc[22], "value": 2.}])
    panel, raw = build_raw_features(bars, actions, horizon=2)
    assert panel.causal_momentum_20.iloc[-1] == pytest.approx(0.)
    assert raw.Close.iloc[-1] == 50
    assert panel.forward_return_2d.iloc[20] == pytest.approx(0.)


def test_gap_stop_executes_before_buying_more_and_split_moves_stop():
    bars = bars_for([100, 100, 50, 40], ticker="ABC")
    actions = pd.DataFrame([{"event_id": "s", "kind": "split", "ticker": "ABC", "date": bars.date.iloc[2], "value": 2., "source": "fixture"}])
    result = simulate_daily(bars, lambda *_: {"ABC": .5}, start=bars.date.iloc[1], end=bars.date.iloc[-1],
                            provenance=PROVENANCE, policy=POLICY, actions=actions, cost_stress=0, terminal_liquidation=False)
    fills = result.events.query("kind == 'fill'")
    assert fills.quantity.tolist() == [500, -1000]
    assert fills.iloc[-1].price == 40
    assert result.holdings.iloc[-1].stop_price == 46


def test_nested_selection_artifacts_do_not_read_outer_outcomes():
    from causal_research import nested_evaluate
    dates = pd.bdate_range("2024-01-02", periods=80)
    panel = pd.DataFrame([{"date": date, "ticker": str(i), "f": i, "ret": i / 100,
                           "ret_end_date": date + pd.Timedelta(days=2)} for date in dates for i in range(5)])
    folds = [{"train_end": str(dates[55]), "start": str(dates[56]), "end": str(dates[-1]),
              "inner": [{"train_end": str(dates[30]), "start": str(dates[31]), "end": str(dates[50])}]}]
    configs = [{"test_score": 1.}, {"test_score": 2.}]
    callback = lambda scored, config, start, end, artifact: {"total_return_pct": config["test_score"]}
    before = nested_evaluate(panel, ["f"], configs, folds, callback, label="ret", record_trial=lambda *_: None)
    panel.loc[panel.date > dates[55], "ret"] = -99
    after = nested_evaluate(panel, ["f"], configs, folds, callback, label="ret", record_trial=lambda *_: None)
    assert before == after and before[0]["configuration"] == configs[1]


def test_incomplete_ic_cohort_cannot_drop_one_stock():
    from edge_evidence import non_overlapping_ic
    panel = pd.DataFrame({"date": ["2024-02-01"] * 4, "causal_score": [1, 2, 3, 4], "ret": [.1, .2, .3, np.nan],
                          "ret_end_date": ["2024-02-05"] * 4})
    assert non_overlapping_ic(panel, label="ret", as_of="2024-02-06").empty


def test_observed_price_revisions_block_but_new_labels_do_not():
    from corrected_audit import observation_identity
    bars = bars_for([100.] * 4)
    panel = bars.assign(forward_return_20d=np.nan)
    actions = pd.DataFrame(columns=["date", "event_id"])
    before = observation_identity(bars, panel, actions, bars.date.iloc[2])
    panel.forward_return_20d = .1
    assert observation_identity(bars, panel, actions, bars.date.iloc[2]) == before
    bars.loc[1, "Close"] = 99
    assert observation_identity(bars, panel, actions, bars.date.iloc[2]) != before


def test_context_published_after_early_close_is_future_data():
    from corrected_data import validate_dated_inputs
    context = pd.DataFrame([{"date": "2024-11-29", "ticker": "SPY", "published_at": "2024-11-29T19:00Z", "source_url": "https://example.test/source", "access_cost": "free"}])
    with pytest.raises(ValueError, match="future"):
        validate_dated_inputs(context)
    context.published_at = "2024-11-29T17:00Z"
    validate_dated_inputs(context)


def test_corrected_adapter_prices_selected_names_and_costs_benchmark(tmp_path):
    from corrected_audit import evaluate_corrected, paired_benchmark, build_daily_targets
    bars = pd.concat([bars_for([100.] * 4, ticker=t) for t in ["SPY", "QQQ", "ABC", "DEF", "GHI"]], ignore_index=True)
    membership = tmp_path / "members.csv"
    pd.DataFrame([{"ticker": t, "effective_from": "2020-01-01", "effective_to": None, "source": "fixture", "status": "active"} for t in ["ABC", "DEF", "GHI"]]).to_csv(membership, index=False)
    scored = bars.loc[bars.ticker.isin(["ABC", "DEF", "GHI"])].assign(causal_score=lambda frame: frame.ticker.map({"ABC": 3., "DEF": 2., "GHI": 1.}), sector="OTHER", forward_return_20d=.1)
    config = {"regime_mode": "static", "shape": "top3", "weighting": "score", "cost_stress": 2.}
    start, end = bars.date.iloc[1], bars.date.iloc[3]
    selected_before = build_daily_targets(scored, config, membership, bars=bars)(bars.date.iloc[0], {}, 100000)
    scored.forward_return_20d = np.nan
    assert build_daily_targets(scored, config, membership, bars=bars)(bars.date.iloc[0], {}, 100000) == selected_before
    result = evaluate_corrected(scored, config, start=start, end=end, bars=bars, actions=None, provenance=PROVENANCE, membership_path=membership, policy=POLICY)
    benchmark = paired_benchmark(result, bars, None, PROVENANCE, start=start, end=end, policy=POLICY)
    assert benchmark.events.query("kind == 'fill'").iloc[-1].reason == "terminal"
    assert benchmark.metrics["cost_stress"] == 2
    missing = bars.copy()
    missing.loc[(missing.ticker == "ABC") & (missing.date == end), "Close"] = np.nan
    with pytest.raises(ValueError, match="ABC"):
        evaluate_corrected(scored, config, start=start, end=end, bars=missing, actions=None, provenance=PROVENANCE, membership_path=membership, policy=POLICY)


def test_missing_sources_cannot_clear_validation(tmp_path):
    from corrected_data import validate_sources
    result = validate_sources(tmp_path, tmp_path / "members.csv", start="2024-02-01", end="2024-02-05")
    assert not result["complete"]
    assert any(row["reason"] == "corporate_action_file_unverified" for row in result["gaps"])


def test_symbol_change_connects_real_rows_and_transfers_holdings():
    from corrected_data import build_raw_features
    bars = bars_for([100.] * 25, ticker="OLD")
    bars.loc[bars.index >= 22, "ticker"] = "NEW"
    actions = pd.DataFrame([{"event_id": "rename", "kind": "symbol_change", "ticker": "OLD", "new_ticker": "NEW",
                             "date": bars.date.iloc[22], "value": 0., "source": "fixture"}])
    panel, _ = build_raw_features(bars, actions, horizon=2)
    assert panel.causal_momentum_20.iloc[-1] == pytest.approx(0.)
    assert panel.forward_return_2d.iloc[21] == pytest.approx(0.)
    assert panel.ticker.iloc[21] == "OLD" and panel.ticker.iloc[22] == "NEW"
    def target(date, *_):
        return {"OLD" if date < bars.date.iloc[22] else "NEW": .5}
    result = simulate_daily(bars, target, start=bars.date.iloc[21], end=bars.date.iloc[-1], actions=actions,
                            provenance=PROVENANCE, policy=POLICY, cost_stress=0)
    assert result.events.query("kind == 'fill'").quantity.tolist() == [500, -500]
