"""Regression tests for the three September execution-quality audit fixes."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import core_satellite_alpha as alpha
import core_satellite_tqqq as tqqq
import validation_bundle
from order_accounting import classify_logical_orders


def _empty_selection(*_args, **_kwargs) -> pd.DataFrame:
    """Return no stock overlay so schedule behavior can be tested alone."""
    return pd.DataFrame({"ticker": pd.Series(dtype=str)})


def _patch_flat_strategy(monkeypatch, module) -> None:
    """Replace market calculations while leaving the real scheduling loop intact."""
    monkeypatch.setattr(module, "_resolve_allocation", lambda *_args: ("risk_on", {}, 0.0, 0.0))
    monkeypatch.setattr(module, "_select_sticky_holdings", _empty_selection)
    monkeypatch.setattr(module, "_sticky_overlay_weights", lambda *_args, **_kwargs: pd.Series(dtype=float))
    monkeypatch.setattr(module, "_score_col_for_regime", lambda *_args: "score")
    monkeypatch.setattr(module, "_apply_concentration_overlay_target", lambda *_args: (0.0, 0.0, None))


def test_core_fold_starts_flat_and_purges_boundary_crossing_trades(monkeypatch):
    dates = pd.bdate_range("2025-12-01", "2026-02-20")
    panel = pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "forward_return": 0.0,
        "forward_return_10d": 0.0,
    })
    _patch_flat_strategy(monkeypatch, alpha)
    monkeypatch.setattr(
        alpha,
        "_cached_etf_prices",
        lambda index, _tickers: pd.DataFrame(index=pd.DatetimeIndex(index)),
    )

    equity, trades, metrics = alpha.run_core_satellite(
        panel,
        {
            "holding_days": 10,
            "regime_mode": "static",
            "score_source": "raw",
            "shape": "top5",
            "weighting": "score",
            "max_per_sector": 2,
        },
        evaluation_start=pd.Timestamp("2026-01-01"),
        evaluation_end=pd.Timestamp("2026-01-31"),
    )

    assert equity.index[0] == pd.Timestamp("2026-01-02")  # New Year is a market holiday.
    assert float(equity.iloc[0]) == alpha.INITIAL_CAPITAL
    assert (pd.to_datetime(trades["date"]) >= pd.Timestamp("2026-01-01")).all()
    assert (pd.to_datetime(trades["exit_date"]) <= pd.Timestamp("2026-01-31")).all()
    assert metrics["boundary_mode"] == "flat_start_full_periods_only"
    assert metrics["purged_leading_trade_count"] >= 1
    assert metrics["purged_trailing_trade_count"] >= 1


def test_tqqq_fold_uses_same_flat_full_period_boundaries(monkeypatch):
    dates = pd.bdate_range("2025-12-01", "2026-02-20")
    panel = pd.DataFrame({"date": dates, "ticker": "TEST", "forward_return": 0.0})
    _patch_flat_strategy(monkeypatch, tqqq)
    monkeypatch.setattr(tqqq, "_load_regime_indicators", lambda *_args: None)
    monkeypatch.setattr(
        tqqq,
        "_load_etf_prices_with_tqqq",
        lambda index: pd.DataFrame(index=pd.DatetimeIndex(index)),
    )
    monkeypatch.setattr(
        tqqq,
        "benchmark_equity",
        lambda index: pd.DataFrame(
            {"SPY": tqqq.INITIAL_CAPITAL, "QQQ": tqqq.INITIAL_CAPITAL, "BLEND": tqqq.INITIAL_CAPITAL},
            index=index,
        ),
    )

    metrics, equity, trades = tqqq.run_tqqq_backtest(
        panel,
        holding_days=10,
        quiet=True,
        evaluation_start=pd.Timestamp("2026-01-01"),
        evaluation_end=pd.Timestamp("2026-01-31"),
    )

    assert equity.index[0] == pd.Timestamp("2026-01-01")
    assert (pd.to_datetime(trades["date"]) >= pd.Timestamp("2026-01-01")).all()
    assert (pd.to_datetime(trades["exit_date"]) <= pd.Timestamp("2026-01-31")).all()
    assert metrics["boundary_mode"] == "flat_start_full_periods_only"
    assert metrics["purged_leading_trade_count"] >= 1
    assert metrics["purged_trailing_trade_count"] >= 1


def test_logical_order_rates_include_open_canceled_and_partial_denominators():
    frame = pd.DataFrame([
        {"submitted_at": "2026-09-01T14:00:00Z", "parent_order_id": "p1", "order_id": "o1", "ticker": "SPY", "side": "buy", "quantity": 10, "filled_qty": 10, "fill_status": "filled"},
        {"submitted_at": "2026-09-01T14:01:00Z", "parent_order_id": "p2", "order_id": "o2", "ticker": "QQQ", "side": "buy", "quantity": 10, "filled_qty": 0, "fill_status": "pending"},
        {"submitted_at": "2026-09-01T14:02:00Z", "parent_order_id": "p3", "order_id": "o3", "ticker": "SPY", "side": "sell", "quantity": 10, "filled_qty": 0, "fill_status": "canceled"},
        {"submitted_at": "2026-09-01T14:03:00Z", "parent_order_id": "p4", "order_id": "o4", "ticker": "AAPL", "side": "buy", "quantity": 10, "filled_qty": 4, "fill_status": "partially_filled"},
        {"submitted_at": "2026-09-01T14:04:00Z", "order_id": "rejected", "ticker": "MSFT", "side": "buy", "quantity": 10, "fill_status": "rejected"},
        {"submitted_at": "2026-09-01T14:05:00Z", "order_id": "SKIPPED:wide", "ticker": "NVDA", "side": "buy", "quantity": 10, "fill_status": "skipped"},
        {"submitted_at": "2026-09-01T14:06:00Z", "order_id": "stop1", "ticker": "AAPL", "side": "sell", "quantity": 4, "fill_status": "filled", "order_type": "trailing_stop"},
    ])

    result = classify_logical_orders(frame)

    assert result["accepted_logical_orders"] == 4
    assert result["fully_filled_logical_orders"] == 1
    assert result["partially_filled_logical_orders"] == 1
    assert result["open_logical_orders"] == 1
    assert result["canceled_unfilled_logical_orders"] == 1
    assert result["complete_fill_rate"] == 0.25
    assert result["any_fill_rate"] == 0.5


def test_stage_children_group_once_but_true_second_attempt_chain_is_duplicate():
    normal = pd.DataFrame([
        {"submitted_at": "2026-09-01T14:00:00Z", "client_order_id": "parent-a1", "order_id": "broker-a1", "ticker": "SPY", "side": "buy", "quantity": 10, "filled_qty": 4, "fill_status": "canceled"},
        {"submitted_at": "2026-09-01T14:00:20Z", "client_order_id": "parent-a2", "order_id": "broker-a2", "ticker": "SPY", "side": "buy", "quantity": 6, "filled_qty": 6, "fill_status": "filled"},
    ])
    grouped = classify_logical_orders(normal)
    assert grouped["accepted_logical_orders"] == 1
    assert grouped["fully_filled_logical_orders"] == 1
    assert grouped["duplicate_logical_orders"] == 0
    assert grouped["child_attempts"] == 2

    duplicate = pd.concat([
        normal,
        pd.DataFrame([{
            "submitted_at": "2026-09-01T14:00:05Z",
            "client_order_id": "parent-a1",
            "order_id": "unexpected-second-a1",
            "ticker": "SPY",
            "side": "buy",
            "quantity": 10,
            "filled_qty": 0,
            "fill_status": "canceled",
        }]),
    ], ignore_index=True)
    duplicated = classify_logical_orders(duplicate)
    assert duplicated["duplicate_logical_orders"] == 1
    assert duplicated["duplicate_child_attempts"] == 1


def test_missing_requested_quantity_is_visible_and_fails_closed():
    result = classify_logical_orders(pd.DataFrame([{
        "submitted_at": "2026-09-01T14:00:00Z",
        "parent_order_id": "known-parent",
        "order_id": "broker-order",
        "ticker": "SPY",
        "side": "buy",
        "filled_qty": 1,
        "fill_status": "filled",
    }]))

    assert result["accepted_logical_orders"] == 1
    assert result["fully_filled_logical_orders"] == 0
    assert result["unclassifiable_logical_orders"] == 1
    assert result["complete_fill_rate"] == 0.0


def _write_report(path, payload, config_fingerprint: str, dataset_fingerprint: str) -> None:
    payload = dict(payload)
    payload.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_context": {
            "config_fingerprint": config_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
        },
    })
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_current_factor_warning_blocks_bundle_even_when_embedded_review_passed(tmp_path, monkeypatch):
    source = tmp_path / "walkforward.json"
    source.write_text("{}", encoding="utf-8")
    config = {"score_source": "regime_adaptive", "holding_days": 20}
    config_fp = validation_bundle.strategy_config_fingerprint(config)
    dataset_fp = "dataset-current"
    paths = {name: tmp_path / f"{name}.json" for name in validation_bundle.DEFAULT_REPORT_PATHS}
    _write_report(paths["survivorship"], {
        "survivorship_adjusted_score": 0.8,
        "rows": [
            {"scenario": "watchlist_plus_failed_audit_tickers", "paper_ready": True, "audit_rebalance_selections": 0},
            {"scenario": "delta_stressed_minus_base", "total_return_pct": 0.0, "max_drawdown_pct": 0.0},
        ],
    }, config_fp, dataset_fp)
    _write_report(paths["execution_stress"], {"rows": [{
        "scenario": "base", "paper_ready": True, "alpha_vs_qqq_pct": 1.0,
        "alpha_vs_blend_pct": 1.0, "max_drawdown_pct": -10.0,
    }]}, config_fp, dataset_fp)
    _write_report(paths["factor_decay"], {
        "edge_health_status": "warning", "reason": "top bucket edge weak",
    }, config_fp, dataset_fp)
    monkeypatch.setattr(validation_bundle, "membership_status", lambda: {"complete": True})

    bundle = validation_bundle.build_validation_bundle(
        {
            "strategy": "core-alpha",
            "folds": [{"outer_year": 2026}],
            "live_config_approval": {"approved": True},
            "approved_live_config": {
                "config": config,
                "medium_risk_review": {"pass": True, "reasons": []},
            },
        },
        source_json=str(source),
        report_paths=paths,
        dataset_context={"dataset_fingerprint": dataset_fp},
    )

    assert bundle["schema_version"] == 2
    assert bundle["robustness_review"]["pass"] is False
    assert "factor_decay_review_warning" in bundle["robustness_review"]["reasons"]
    assert bundle["deployment"]["paper_approved"] is False
