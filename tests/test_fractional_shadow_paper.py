from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import fractional_shadow_paper as fsp


def _write_signal(path: Path, *, predicted_at: str = "2026-08-27T13:35:00+00:00", weights=None) -> None:
    """Write one beginner-sized active signal for an isolated test account."""
    overlay = weights or {"MU": 0.20, "INTC": 0.170755, "FCX": 0.029245}
    pd.DataFrame([{
        "paper_ready": True,
        "gates_all_pass": True,
        "medium_risk_review_pass": True,
        "latest_factor_date": predicted_at[:10],
        "predicted_at": predicted_at,
        "live_config_hash": "active-config",
        "target_spy_weight": 0.0,
        "target_qqq_weight": 0.6,
        "target_tqqq_weight": 0.0,
        "overlay_weights_json": json.dumps(overlay),
    }]).to_csv(path, index=False)


def _write_prices(data_dir: Path, dates=("2026-08-26", "2026-08-27")) -> None:
    """Create prices where whole shares are too costly but fractions work."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for ticker, values in {
        "QQQ": (700.0, 711.0),
        "MU": (900.0, 938.0),
        "INTC": (87.0, 88.0),
        "FCX": (78.0, 79.0),
    }.items():
        pd.DataFrame({"Close": values}, index=pd.to_datetime(dates)).to_parquet(
            data_dir / f"{ticker}.parquet"
        )


def _paths(tmp_path: Path) -> dict[str, Path]:
    """Keep every generated artifact inside the temporary test folder."""
    return {
        "signal_path": tmp_path / "signal.csv",
        "state_path": tmp_path / "state.json",
        "orders_path": tmp_path / "orders.csv",
        "equity_path": tmp_path / "equity.csv",
        "report_path": tmp_path / "report.json",
        "data_dir": tmp_path / "data",
    }


def test_four_expensive_assets_fit_a_400_fractional_account(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _write_signal(paths["signal_path"])
    _write_prices(paths["data_dir"])
    monkeypatch.setenv("FRACTIONAL_SHADOW_SLIPPAGE_BPS", "10")

    report = fsp.run_fractional_shadow(**paths, initial_equity=400.0)

    orders = pd.read_csv(paths["orders_path"])
    state = json.loads(paths["state_path"].read_text())
    assert report["status"] == "ok"
    assert set(orders["ticker"]) == {"QQQ", "MU", "INTC", "FCX"}
    assert (orders["quantity"] > 0).all()
    assert (orders["quantity"] < 1).all()
    assert state["cash"] >= 0
    assert report["allocation"]["max_absolute_target_weight_gap"] < 0.01
    assert report["execution"]["broker_orders_submitted"] is False
    assert report["safety"]["real_capital_approved"] is False


def test_same_signal_is_idempotent_and_does_not_duplicate_orders(tmp_path):
    paths = _paths(tmp_path)
    _write_signal(paths["signal_path"])
    _write_prices(paths["data_dir"])

    first = fsp.run_fractional_shadow(**paths, initial_equity=400.0)
    state_before = json.loads(paths["state_path"].read_text())
    second = fsp.run_fractional_shadow(**paths, initial_equity=400.0)
    state_after = json.loads(paths["state_path"].read_text())

    orders = pd.read_csv(paths["orders_path"])
    equity = pd.read_csv(paths["equity_path"])
    assert first["execution"]["orders_simulated"] == 4
    assert second["execution"]["orders_simulated"] == 0
    assert second["signal"]["repeated_signal"] is True
    assert len(orders) == 4
    assert len(equity) == 1
    assert state_after["cumulative_slippage_cost"] == state_before["cumulative_slippage_cost"]
    assert state_after["cumulative_regulatory_fees"] == state_before["cumulative_regulatory_fees"]


def test_sell_rebalance_models_slippage_and_rounded_regulatory_fees(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _write_signal(paths["signal_path"])
    _write_prices(paths["data_dir"])
    fsp.run_fractional_shadow(**paths, initial_equity=400.0)

    # A next-day signal removes MU.  The shadow must sell the exact fractional
    # holding, deduct adverse fill movement, and accrue at least one fee cent.
    _write_signal(
        paths["signal_path"],
        predicted_at="2026-08-28T13:35:00+00:00",
        weights={"INTC": 0.370755, "FCX": 0.029245},
    )
    for ticker in ("QQQ", "MU", "INTC", "FCX"):
        frame = pd.read_parquet(paths["data_dir"] / f"{ticker}.parquet")
        frame.loc[pd.Timestamp("2026-08-28"), "Close"] = float(frame.iloc[-1]["Close"])
        frame.to_parquet(paths["data_dir"] / f"{ticker}.parquet")
    monkeypatch.setenv("FRACTIONAL_SHADOW_SEC_SELL_FEE_BPS", "0.206")

    report = fsp.run_fractional_shadow(**paths, initial_equity=400.0)
    orders = pd.read_csv(paths["orders_path"])
    sells = orders[(orders["valuation_date"] == "2026-08-28") & (orders["side"] == "sell")]

    assert "MU" in set(sells["ticker"])
    assert (sells["simulated_fill_price"] < sells["reference_price"]).all()
    assert report["execution"]["regulatory_fees_today"] >= 0.01
    assert report["capital"]["cash"] >= 0


def test_price_reader_never_uses_a_future_close(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {"Close": [100.0]}, index=pd.to_datetime(["2026-08-28"])
    ).to_parquet(data_dir / "QQQ.parquet")

    with pytest.raises(RuntimeError, match="No non-lookahead price"):
        fsp.load_prices({"QQQ"}, "2026-08-27", data_dir=data_dir)


def test_missing_state_with_existing_equity_fails_closed(tmp_path):
    equity = tmp_path / "equity.csv"
    pd.DataFrame([{"valuation_date": "2026-08-27", "equity": 400.0}]).to_csv(equity, index=False)

    with pytest.raises(RuntimeError, match="state is missing"):
        fsp.load_state(tmp_path / "missing.json", equity, initial_equity=400.0)


def test_existing_account_cannot_be_silently_reset_to_new_capital(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(fsp._new_state(400.0)))

    with pytest.raises(RuntimeError, match="different initial equity"):
        fsp.load_state(state_path, tmp_path / "equity.csv", initial_equity=500.0)


def test_script_contains_no_broker_submission_path():
    source = Path(fsp.__file__).read_text(encoding="utf-8").lower()
    forbidden = ("submit_order", "place_order", "cancel_order", "replace_order", "alpaca_trade_api")
    assert not any(token in source for token in forbidden)


def test_shadow_workflow_runs_fractional_mode_under_shared_lock():
    workflow = Path(".github/workflows/shadow_paper_journal.yml").read_text(encoding="utf-8")
    assert "group: signals-latest-publisher" in workflow
    assert "python3 fractional_shadow_paper.py" in workflow
    assert "fractional_shadow_state.json" in workflow
    assert "fractional_shadow_report.json" in workflow
    assert "fractional_shadow_compare.json" in workflow
    assert "alpaca_paper_trading.py --submit" not in workflow
    assert "execution_guard.py" not in workflow


def test_daily_workflow_preserves_fractional_state_on_signals_branch():
    workflow = Path(".github/workflows/daily_paper_trading.yml").read_text(encoding="utf-8")
    for filename in (
        "fractional_shadow_state.json",
        "fractional_shadow_orders.csv",
        "fractional_shadow_equity.csv",
        "fractional_shadow_report.json",
        "fractional_shadow_compare.json",
    ):
        # Each file appears in the artifact, preservation, restoration, and add
        # lists. Requiring several appearances catches accidental state loss.
        assert workflow.count(filename) >= 4
