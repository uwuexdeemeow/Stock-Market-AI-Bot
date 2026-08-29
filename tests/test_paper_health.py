import json
from datetime import datetime, timezone

import paper_health


def test_readiness_flags_require_medium_risk_review():
    flags = paper_health._readiness_flags(
        {
            "paper_ready": True,
            "gates_all_pass": True,
            "medium_risk_review_pass": False,
            "freshness_ok": True,
        }
    )

    assert flags["strategy_ready"] is False


def test_account_alignment_is_tri_state_and_uses_broker_truth():
    """Missing evidence collects; complete weights can pass or fail."""
    missing = paper_health._account_alignment({})
    assert missing["status"] == "collecting"
    assert missing["passed"] is False

    aligned = paper_health._account_alignment(
        {
            "summary": {"target_comparison_enabled": True},
            "rows": [
                {"target_weight": 0.60, "broker_weight": 0.595},
                {"target_weight": 0.20, "broker_weight": 0.205},
            ],
        }
    )
    assert aligned["status"] == "pass"
    assert aligned["max_weight_gap"] == 0.005

    misaligned = paper_health._account_alignment(
        {
            "summary": {"target_comparison_enabled": True},
            "rows": [
                {"target_weight": 0.60, "broker_weight": 0.50},
                {"target_weight": 0.20, "broker_weight": 0.20},
            ],
        }
    )
    assert misaligned["status"] == "fail"
    assert "max_weight_gap" in misaligned["reason"]


def test_account_alignment_rejects_broker_report_for_older_signal():
    """A new target file needs a new Alpaca reconciliation before grading."""
    signal = paper_health.pd.DataFrame(
        [{"predicted_at": "2026-08-30T01:49:00+08:00", "target_qqq_weight": 0.60}]
    )
    alignment = paper_health._account_alignment(
        {
            "summary": {"target_comparison_enabled": True},
            "inputs": {"signal": {"as_of": "2026-08-29T00:00:22+00:00"}},
            "rows": [{"target_weight": 0.0, "broker_weight": 0.60}],
        },
        signal,
    )

    assert alignment["status"] == "collecting"
    assert alignment["max_weight_gap"] is None
    assert alignment["reason"] == "broker_truth_does_not_match_current_signal"


def test_readiness_exposes_collecting_account_alignment():
    flags = paper_health._readiness_flags(
        {
            "paper_ready": True,
            "gates_all_pass": True,
            "medium_risk_review_pass": True,
            "freshness_ok": True,
            "account_alignment": {"status": "collecting"},
        }
    )

    assert flags["account_alignment_status"] == "collecting"
    assert flags["account_aligned"] is False


def test_build_health_reads_submit_gates_from_signal_csv(tmp_path, monkeypatch):
    status_path = tmp_path / "alpaca_daily_status.json"
    signal_path = tmp_path / "core_satellite_alpha_signal.csv"
    trades_path = tmp_path / "alpaca_paper_log.csv"
    equity_path = tmp_path / "alpaca_paper_equity.csv"
    orders_path = tmp_path / "core_satellite_alpha_orders.csv"
    health_path = tmp_path / "alpaca_paper_health.json"
    slippage_report_path = tmp_path / "missing_slippage_report.json"

    # PLAIN ENGLISH: The account snapshot has equity/positions but usually does
    # not carry signal gate fields. The health report must read those from the
    # signal CSV instead.
    status_path.write_text(
        json.dumps({"account_equity": 100000, "positions": {}, "position_values": {}}),
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc)
    signal_path.write_text(
        "\n".join(
            [
                "paper_signal_type,paper_ready,gates_all_pass,medium_risk_review_pass,predicted_at,latest_factor_date",
                f"core-alpha,True,True,True,{now.isoformat()},{now.date().isoformat()}",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(paper_health, "PAPER_STATUS", status_path)
    monkeypatch.setattr(paper_health, "CORE_SIGNAL", signal_path)
    monkeypatch.setattr(paper_health, "PAPER_TRADES", trades_path)
    monkeypatch.setattr(paper_health, "PAPER_EQUITY", equity_path)
    monkeypatch.setattr(paper_health, "CORE_ORDER_PLAN", orders_path)
    monkeypatch.setattr(paper_health, "PAPER_HEALTH", health_path)
    monkeypatch.setattr(paper_health, "SLIPPAGE_REPORT", slippage_report_path)
    monkeypatch.setattr(
        paper_health.alpaca_paper_gauntlet,
        "evaluate_alpaca_paper",
        lambda: {
            "strategy": "core-alpha",
            "status": "watch",
            "approved_for_real_capital": False,
            "reason": "test",
            "current_signal_paper_trades": 0,
            "fill_stats_scope": "test",
            "filled_orders": 0,
            "fill_rate": 0.0,
            "cancel_rate": 0.0,
        },
    )

    health = paper_health.build_health()

    assert health["paper_ready"] is True
    assert health["gates_all_pass"] is True
    assert health["medium_risk_review_pass"] is True
    assert health["strategy_ready"] is True
    assert health["readiness_flags"]["strategy_ready"] is True
    assert health["readiness_source"] == "core_satellite_alpha_signal.csv"
    assert health["freshness_ok"] is True
    assert health["freshness_source"] == "core_satellite_alpha_signal.csv"


def test_signal_freshness_status_flags_stale_signal():
    signal = paper_health.pd.DataFrame(
        [
            {
                "predicted_at": "2020-01-01T00:00:00+00:00",
                "latest_factor_date": "2020-01-01",
            }
        ]
    )

    freshness = paper_health._signal_freshness_status(
        signal,
        {},
        now=datetime(2026, 6, 4, 14, tzinfo=timezone.utc),
    )

    assert freshness["freshness_ok"] is False
    assert any(str(issue).startswith("signal_age_") for issue in freshness["freshness_issues"])
    assert any(str(issue).startswith("factor_age_") for issue in freshness["freshness_issues"])


def test_slippage_summary_prefers_alpaca_api_report():
    trades = paper_health.pd.DataFrame(
        [
            {
                "fill_status": "filled",
                "side": "buy",
                "quantity": 1,
                "price": 100,
                "filled_qty": 1,
                "filled_avg_price": 110,
            }
        ]
    )
    report = {
        "generated_at": "2026-06-12T12:15:28+00:00",
        "summary": {
            "orders_analyzed": 25,
            "avg_slippage_bps": 5.83,
            "median_slippage_bps": 4.16,
            "slippage_bad_count": 19,
            "adverse_15m_count": 9,
            "avg_worst_adverse_60m_bps": 67.58,
            "max_worst_adverse_60m_bps": 327.89,
        },
        "segments": {"limit_orders": {"orders_analyzed": 14, "avg_slippage_bps": 0.79}},
        "orders": [
            {
                "filled_at": "2026-06-11T13:52:36+00:00",
                "slippage_bps": -3.46,
            },
            {
                "filled_at": "2026-06-10T15:37:59+00:00",
                "slippage_bps": 12.47,
            },
        ],
    }

    summary = paper_health._slippage_summary(trades, report)

    assert summary["source"] == "alpaca_slippage_reversal_report.json"
    assert summary["filled_orders_with_slippage"] == 25
    assert summary["avg_slippage_bps"] == 5.83
    assert summary["median_slippage_bps"] == 4.16
    assert summary["worst_slippage_bps"] == 12.47
    assert summary["latest_fill_at"] == "2026-06-11T13:52:36+00:00"
    assert summary["segments"]["limit_orders"]["avg_slippage_bps"] == 0.79
