from __future__ import annotations

import notifications


def test_warning_alert_has_no_email_channel(monkeypatch):
    monkeypatch.setattr(notifications, "ALERTS_ENABLED", True)
    monkeypatch.setattr(notifications, "send_macos_notification", lambda *_args: True)
    monkeypatch.setattr(notifications, "send_telegram", lambda *_args, **_kwargs: True)

    result = notifications.send_alert("warning", title="Test", priority="warning")

    assert result == {"desktop": True, "telegram": True}


def test_script_telegram_suppression_skips_message(monkeypatch):
    monkeypatch.setenv("STOCKBOT_SCRIPT_TELEGRAM_ENABLED", "0")
    monkeypatch.setattr(notifications, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(notifications, "TELEGRAM_CHAT_ID", "chat")

    assert notifications.send_telegram("hello") is False


def test_script_telegram_suppression_skips_signal_summary(monkeypatch, tmp_path):
    signal = tmp_path / "signal.csv"
    signal.write_text("ticker,target_weight\nSPY,1.0\n")

    monkeypatch.setenv("STOCKBOT_SCRIPT_TELEGRAM_ENABLED", "0")
    monkeypatch.setattr(notifications, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(notifications, "TELEGRAM_CHAT_ID", "chat")

    assert notifications.send_signal_summary_telegram(str(signal), str(tmp_path / "orders.csv")) is False
