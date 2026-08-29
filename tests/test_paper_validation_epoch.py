from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import paper_validation_epoch as epoch_module


def test_same_day_equity_snapshot_after_epoch_start_counts_as_trading_day(tmp_path, monkeypatch):
    """An exact post-start timestamp must not be replaced by midnight's date."""
    signal_dir = tmp_path / "signals"
    signal_dir.mkdir()

    # PLAIN ENGLISH: the epoch begins at 08:43 UTC.  The equity row belongs to
    # the same calendar day, but its exact snapshot time is later at 15:31 UTC.
    # Old behavior read only the date as midnight and incorrectly dropped it.
    pd.DataFrame([
        {
            "date": "2026-08-26",
            "timestamp": "2026-08-26 15:31:06+00:00",
            "equity": 106_953.49,
        }
    ]).to_csv(signal_dir / "alpaca_paper_equity.csv", index=False)

    monkeypatch.setattr(epoch_module, "SIGNAL_DIR", str(signal_dir))
    epoch = {
        "epoch_id": "paper-test",
        "started_at": datetime(2026, 8, 26, 8, 43, 15, tzinfo=timezone.utc).isoformat(),
        "requirements": {},
    }

    result = epoch_module.evaluate_epoch(epoch)

    assert result["trading_days"] == 1


def test_legacy_equity_file_without_timestamp_still_uses_date(tmp_path, monkeypatch):
    """Older equity files remain readable when they contain only a date."""
    signal_dir = tmp_path / "signals"
    signal_dir.mkdir()
    pd.DataFrame([
        {"date": "2026-08-27", "equity": 107_000.00}
    ]).to_csv(signal_dir / "alpaca_paper_equity.csv", index=False)

    monkeypatch.setattr(epoch_module, "SIGNAL_DIR", str(signal_dir))
    epoch = {
        "epoch_id": "paper-test",
        "started_at": "2026-08-26T08:43:15+00:00",
        "requirements": {},
    }

    result = epoch_module.evaluate_epoch(epoch)

    assert result["trading_days"] == 1


def test_epoch_slippage_uses_only_post_start_rebalances(tmp_path, monkeypatch):
    signal_dir = tmp_path / "signals"
    signal_dir.mkdir()
    (signal_dir / "alpaca_slippage_reversal_report.json").write_text(json.dumps({
        "orders": [
            {"filled_at": "2026-08-25T14:00:00Z", "order_type": "limit", "slippage_bps": 50.0},
            {"filled_at": "2026-08-27T14:00:00Z", "order_type": "trailing_stop", "slippage_bps": 30.0},
            {"filled_at": "2026-08-27T14:05:00Z", "order_type": "limit", "slippage_bps": 3.0},
        ]
    }), encoding="utf-8")
    monkeypatch.setattr(epoch_module, "SIGNAL_DIR", str(signal_dir))

    result = epoch_module.evaluate_epoch({
        "epoch_id": "paper-test",
        "started_at": "2026-08-26T08:43:15+00:00",
        "requirements": {"bad_slippage_threshold_bps": 2.0},
    })

    assert result["average_slippage_bps"] == 3.0
    assert result["bad_slippage_rate"] == 1.0
