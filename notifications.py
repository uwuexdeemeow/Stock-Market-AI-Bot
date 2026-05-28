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


def _script_telegram_enabled() -> bool:
    """Return whether Python scripts may send their own Telegram messages.

    PLAIN ENGLISH: GitHub Actions has a final workflow summary message.  When
    `STOCKBOT_SCRIPT_TELEGRAM_ENABLED=0`, individual scripts still log alerts
    and may send email, but they do not create extra Telegram messages.
    """
    value = os.environ.get("STOCKBOT_SCRIPT_TELEGRAM_ENABLED", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


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


# ── Windows native toast ─────────────────────────────────────────────
def send_windows_notification(title: str, message: str) -> bool:
    """Pop a Windows 10/11 toast notification.

    PLAIN ENGLISH: Uses PowerShell's built-in .NET classes to show a
    toast notification in the Windows Action Center (bottom-right corner).
    No third-party packages needed — works on any Windows 10+ machine.
    Returns True if the notification was sent, False on failure.
    """
    try:
        # Escape single-quotes for PowerShell string
        safe_msg = message.replace("'", "''").replace("\n", "\\n")
        safe_title = title.replace("'", "''")
        # PowerShell script that creates a toast notification using
        # Windows.UI.Notifications (built into Windows 10+)
        ps_script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null

$template = @"
<toast>
    <visual>
        <binding template="ToastGeneric">
            <text>{safe_title}</text>
            <text>{safe_msg}</text>
        </binding>
    </visual>
</toast>
"@

$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Stock Bot').Show($toast)
"""
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            timeout=10,
        )
        return True
    except FileNotFoundError:
        # PowerShell not available (not Windows, or stripped install)
        return False
    except Exception as exc:
        _log(f"Windows notification failed: {exc}")
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
    if not _script_telegram_enabled():
        _log("Telegram skipped: STOCKBOT_SCRIPT_TELEGRAM_ENABLED=0")
        return False
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        from http_retry import retry_request
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        # retry_request handles transient 429/500+ errors with backoff.
        # Telegram API rate-limits bots — retrying prevents missed alerts.
        resp = retry_request("POST", url, json=payload, timeout=10)
        return resp is not None and resp.status_code == 200
    except Exception as exc:
        _log(f"Telegram alert failed: {exc}")
        return False


def send_telegram_document(file_path: str, *, caption: str = "") -> bool:
    """Send a file (CSV, JSON, etc.) to Telegram as a document attachment.

    PLAIN ENGLISH: Uploads a file from disk to your Telegram chat via the bot.
    Use this for sending signal CSVs, order logs, or any file you want to
    receive on your phone after a CI run.  Telegram supports files up to 50 MB.

    Args:
        file_path: Path to the file on disk to send.
        caption: Optional text caption shown below the file in the chat.

    Returns True if the file was sent, False otherwise.
    """
    if not _script_telegram_enabled():
        _log("Telegram document skipped: STOCKBOT_SCRIPT_TELEGRAM_ENABLED=0")
        return False
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    from pathlib import Path
    path = Path(file_path)
    if not path.exists():
        _log(f"Telegram document skipped — file not found: {file_path}")
        return False
    try:
        import urllib.request
        import urllib.error
        # Telegram sendDocument API expects multipart/form-data.
        # We build it manually to avoid needing the 'requests' library.
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        boundary = "----PythonFormBoundary7MA4YWxkTrZu0gW"
        file_data = path.read_bytes()
        filename = path.name

        body_parts = []
        # chat_id field
        body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{TELEGRAM_CHAT_ID}".encode())
        # caption field (optional)
        if caption:
            body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}".encode())
        # document file field
        body_parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
            + file_data
        )
        body_parts.append(f"--{boundary}--\r\n".encode())
        body = b"\r\n".join(body_parts)

        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        ctx = _ssl_context()
        with urlopen(req, context=ctx, timeout=30) as resp:
            return resp.status == 200
    except Exception as exc:
        _log(f"Telegram document send failed: {exc}")
        return False


def send_signal_summary_telegram(
    signal_path: str = "signals/core_satellite_alpha_signal.csv",
    orders_path: str = "signals/core_satellite_alpha_orders.csv",
) -> bool:
    """Send today's signal summary + files to Telegram after a daily run.

    PLAIN ENGLISH: Reads the signal CSV, formats a short text summary showing
    target weights and any new orders, sends it as a Telegram message, then
    attaches the full CSV files so you can review details on your phone.

    Called at the end of daily_run.py if the run succeeds.
    """
    from pathlib import Path
    import pandas as pd

    if not _script_telegram_enabled():
        _log("Signal summary Telegram skipped: STOCKBOT_SCRIPT_TELEGRAM_ENABLED=0")
        return False

    signal_file = Path(signal_path)
    orders_file = Path(orders_path)

    if not signal_file.exists():
        _log("No signal file to send via Telegram")
        return False

    # Build a short human-readable summary
    try:
        sig = pd.read_csv(signal_file)
        lines = ["📊 <b>Daily Signal</b>\n"]

        # Show target weights
        if "ticker" in sig.columns and "target_weight" in sig.columns:
            for _, row in sig.iterrows():
                ticker = row["ticker"]
                weight = float(row.get("target_weight", 0))
                if abs(weight) > 0.001:
                    lines.append(f"  <code>{ticker:6s}</code> → {weight*100:.1f}%")

        # Show order count if orders file exists
        if orders_file.exists():
            try:
                orders = pd.read_csv(orders_file)
                if not orders.empty:
                    buys = len(orders[orders.get("side", pd.Series()) == "buy"]) if "side" in orders.columns else 0
                    sells = len(orders[orders.get("side", pd.Series()) == "sell"]) if "side" in orders.columns else 0
                    lines.append(f"\n📋 Planned orders: {buys} buys, {sells} sells")
            except Exception:
                pass

        summary = "\n".join(lines)
    except Exception as exc:
        summary = f"📊 Signal generated (couldn't parse summary: {exc})"

    # Send text summary first
    send_telegram(summary)

    # Then attach the CSV files
    send_telegram_document(str(signal_file), caption="Signal weights")
    if orders_file.exists():
        send_telegram_document(str(orders_file), caption="Orders")

    return True


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
        "desktop": False,
        "email": False,
        "telegram": False,
    }

    if not ALERTS_ENABLED:
        return results

    # Prefix for Telegram so you can tell severity at a glance
    emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(priority, "⚠️")
    tg_message = f"{emoji} <b>{title}</b>\n{message}"

    # Desktop notification — detect OS and use the right method.
    # macOS uses osascript, Windows uses PowerShell toast.
    import sys
    if sys.platform == "darwin":
        results["desktop"] = send_macos_notification(title, message)
    elif sys.platform == "win32":
        results["desktop"] = send_windows_notification(title, message)

    # Telegram — always send if configured (works when away from desk)
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
