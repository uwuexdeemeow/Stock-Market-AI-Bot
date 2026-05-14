"""Centralised notification module for the Stock Market AI Bot.

PLAIN ENGLISH: This is the single place that handles sending alerts
through ALL channels — macOS banners, email, and Telegram.  Every other
script (execution_guard, daily_run, fill_monitor, regime_monitor, etc.)
imports from here instead of duplicating its own notification code.

HOW TO SET UP TELEGRAM:
  1. Open Telegram and message @BotFather
  2. Send /newbot and follow the prompts to create a bot
  3. Copy the bot token (looks like 123456:ABC-DEF...)
  4. Message your new bot, then visit:
     https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
  5. Find your chat_id in the response JSON
  6. Set environment variables:
       export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
       export TELEGRAM_CHAT_ID="your_chat_id"

HOW TO RUN:
  This is a library — import it from other scripts:
    from notifications import send_alert, send_telegram, notify

KEY CONCEPTS:
  - Channel: a way to deliver a message (macOS, email, Telegram)
  - Priority: "info" (routine), "warning" (needs attention), "critical"
    (act now).  Critical messages go to ALL channels; info only goes to
    macOS and Telegram.
"""
from __future__ import annotations

import os

# Load .env file so TELEGRAM_BOT_TOKEN, SMTP_*, etc. are available
# even when this module is imported before settings.py runs.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — env vars must be set manually

import smtplib
import subprocess
from email.mime.text import MIMEText
import ssl
from urllib.request import Request, urlopen
import json

# Build an SSL context using certifi's CA bundle so HTTPS calls
# (Telegram API) work even if the system certs are missing.
def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

# ── Telegram config ──────────────────────────────────────────────────
# Set these env vars to enable Telegram alerts.
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Email config (same env vars that execution_guard / daily_run use) ─
SMTP_HOST: str = os.environ.get("SMTP_HOST", "")
SMTP_PORT: str = os.environ.get("SMTP_PORT", "587")
SMTP_USER: str = os.environ.get("SMTP_USER", "")
SMTP_PASS: str = os.environ.get("SMTP_PASSWORD", "")
ALERT_EMAIL_TO: str = (
    os.environ.get("ALERT_EMAIL_TO")
    or os.environ.get("ALERT_EMAIL")
    or SMTP_USER
)
ALERT_EMAIL_FROM: str = os.environ.get("ALERT_EMAIL_FROM") or SMTP_USER

# ── Master kill switch ───────────────────────────────────────────────
# Set STOCKBOT_ALERTS_ENABLED=0 to silence everything except log prints.
ALERTS_ENABLED: bool = os.environ.get("STOCKBOT_ALERTS_ENABLED", "1") not in ("0", "false", "no")


def _log(message: str) -> None:
    """Print a timestamped log line to stdout."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {message}")


# ── macOS native banner ──────────────────────────────────────────────
def send_macos_notification(title: str, message: str) -> bool:
    """Pop a macOS notification banner.

    PLAIN ENGLISH: Uses the built-in 'osascript' command to show a
    little banner in the top-right corner of your Mac screen.
    Returns True if the notification was sent, False on failure.
    """
    try:
        # Escape double-quotes so the AppleScript string doesn't break
        safe_msg = message.replace('"', '\\"')
        safe_title = title.replace('"', '\\"')
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "{safe_msg}" with title "{safe_title}"',
            ],
            capture_output=True,
            timeout=5,
        )
        return True
    except Exception as exc:
        _log(f"macOS notification failed: {exc}")
        return False


# ── Email ─────────────────────────────────────────────────────────────
def send_email(subject: str, body: str) -> bool:
    """Send an email alert via SMTP.

    PLAIN ENGLISH: If you've configured SMTP environment variables
    (SMTP_HOST, SMTP_USER, SMTP_PASSWORD), this sends an email to
    ALERT_EMAIL_TO.  If any config is missing, it silently skips.
    Returns True if the email was sent, False otherwise.
    """
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO, ALERT_EMAIL_FROM]):
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ALERT_EMAIL_TO
        with smtplib.SMTP(SMTP_HOST, int(SMTP_PORT), timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(ALERT_EMAIL_FROM, [ALERT_EMAIL_TO], msg.as_string())
        return True
    except Exception as exc:
        _log(f"email alert failed: {exc}")
        return False


# ── Telegram ──────────────────────────────────────────────────────────
def send_telegram(message: str, *, parse_mode: str = "HTML") -> bool:
    """Send a Telegram message via the Bot API.

    PLAIN ENGLISH: Makes an HTTPS request to Telegram's server to deliver
    a message to your phone/desktop Telegram app.  Requires TELEGRAM_BOT_TOKEN
    and TELEGRAM_CHAT_ID environment variables.
    Returns True if the message was sent, False otherwise.

    parse_mode: "HTML" (default) lets you use <b>bold</b>, <i>italic</i>,
                <code>monospace</code> in messages.  Set to "" for plain text.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urlopen(req, timeout=10, context=_ssl_context())
        return resp.status == 200
    except Exception as exc:
        _log(f"Telegram alert failed: {exc}")
        return False


# ── Unified send_alert ────────────────────────────────────────────────
def send_alert(
    message: str,
    *,
    title: str = "Stock Bot",
    priority: str = "warning",
) -> dict[str, bool]:
    """Send an alert through all configured channels.

    PLAIN ENGLISH: This is the main function you call when something
    important happens.  It fans out the message to macOS, email, and
    Telegram in one shot.  Returns a dict showing which channels succeeded.

    priority levels:
      - "info"     → macOS + Telegram only (don't spam email for routine stuff)
      - "warning"  → all channels
      - "critical" → all channels (same delivery, but the message prefix
                     makes it clear this is urgent)
    """
    _log(f"ALERT [{priority}]: {message}")

    results: dict[str, bool] = {
        "macos": False,
        "email": False,
        "telegram": False,
    }

    if not ALERTS_ENABLED:
        return results

    # Prefix for Telegram so you can tell severity at a glance
    emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(priority, "⚠️")
    tg_message = f"{emoji} <b>{title}</b>\n{message}"

    # macOS — always send (you might be at your desk)
    results["macos"] = send_macos_notification(title, message)

    # Telegram — always send if configured (works when away from Mac)
    results["telegram"] = send_telegram(tg_message)

    # Email — skip for "info" priority to avoid inbox noise
    if priority in ("warning", "critical"):
        results["email"] = send_email(f"[{title}] {message[:80]}", message)

    return results


# ── Convenience wrappers ──────────────────────────────────────────────
def notify_info(message: str, *, title: str = "Stock Bot") -> dict[str, bool]:
    """Send a low-priority informational alert."""
    return send_alert(message, title=title, priority="info")


def notify_warning(message: str, *, title: str = "Stock Bot") -> dict[str, bool]:
    """Send a warning-level alert."""
    return send_alert(message, title=title, priority="warning")


def notify_critical(message: str, *, title: str = "Stock Bot") -> dict[str, bool]:
    """Send a critical alert — something needs immediate attention."""
    return send_alert(message, title=title, priority="critical")


# ── CLI test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test notification channels.  "
        "Sends a test message through all configured channels."
    )
    parser.add_argument(
        "--message", "-m",
        default="Test alert from Stock Bot notification system",
        help="Custom test message to send",
    )
    parser.add_argument(
        "--priority", "-p",
        choices=["info", "warning", "critical"],
        default="info",
        help="Priority level for the test message (default: info)",
    )
    args = parser.parse_args()

    print("Notification config:")
    print(f"  Alerts enabled:  {ALERTS_ENABLED}")
    print(f"  Telegram token:  {'configured' if TELEGRAM_BOT_TOKEN else 'NOT SET'}")
    print(f"  Telegram chat:   {'configured' if TELEGRAM_CHAT_ID else 'NOT SET'}")
    print(f"  SMTP host:       {SMTP_HOST or 'NOT SET'}")
    print(f"  Email to:        {ALERT_EMAIL_TO or 'NOT SET'}")
    print()

    results = send_alert(args.message, title="Stock Bot Test", priority=args.priority)
    print(f"\nResults: {results}")
    for channel, ok in results.items():
        status = "✓ sent" if ok else "✗ skipped/failed"
        print(f"  {channel}: {status}")
