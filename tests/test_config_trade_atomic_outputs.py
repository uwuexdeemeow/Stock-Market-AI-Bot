from __future__ import annotations

import sys

import pandas as pd

import config_health
import trade_rules


def test_config_health_uses_atomic_writer(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(config_health, "LOGS", tmp_path)
    monkeypatch.setattr(
        config_health,
        "build_config_health",
        lambda run_pip_check=True: {"ok": True, "package_checks": [], "pip_check": {"ok": True}},
    )
    monkeypatch.setattr(config_health, "atomic_write_json", lambda data, path, **_kwargs: calls.append((path.name, data["ok"])))
    monkeypatch.setattr(sys, "argv", ["config_health.py"])

    config_health.main()

    assert calls == [("config_health.json", True)]


def test_trade_rules_use_atomic_writers(monkeypatch, tmp_path):
    json_calls = []
    csv_calls = []

    monkeypatch.setattr(trade_rules, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(trade_rules, "TRADE_RULE_REPORT", str(tmp_path / "trade_rule_report.csv"))
    monkeypatch.setattr(trade_rules, "atomic_write_json", lambda data, path, **_kwargs: json_calls.append((str(path), data["ticker"])))
    monkeypatch.setattr(
        trade_rules,
        "atomic_write_csv",
        lambda df, path, **kwargs: csv_calls.append((str(path), len(df), kwargs.get("index"))),
    )

    saved_path = trade_rules.save_trade_rule(trade_rules.TradeRule("QQQ"))
    report = trade_rules.append_rule_report([
        {"ticker": "QQQ", "approved_candidate": True, "score": 1.0},
    ])

    assert saved_path == str(tmp_path / "QQQ_trade_rules.json")
    assert json_calls == [(str(tmp_path / "QQQ_trade_rules.json"), "QQQ")]
    assert isinstance(report, pd.DataFrame)
    assert csv_calls == [(str(tmp_path / "trade_rule_report.csv"), 1, False)]
