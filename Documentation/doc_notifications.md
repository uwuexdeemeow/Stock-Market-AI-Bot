# notifications.py — Shared Alert Sender

## What It Does

`notifications.py` is the shared alert module. Other scripts import it when they
need to send a desktop, email, or Telegram warning.

For GitHub Actions, script-level Telegram can be disabled with:

```bash
STOCKBOT_SCRIPT_TELEGRAM_ENABLED=0
```

That keeps individual scripts quiet while the workflow sends one final
consolidated Telegram summary.

## How To Run It

```bash
python3 notifications.py --message "test" --priority info
python3 notifications.py --message "warning test" --priority warning
```

Inputs:

- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` for Telegram.
- SMTP environment variables for email alerts.
- `STOCKBOT_ALERTS_ENABLED=0` to silence all alert channels.
- `STOCKBOT_SCRIPT_TELEGRAM_ENABLED=0` to silence only script Telegram.

Outputs:

- Telegram messages when enabled.
- Email warnings when SMTP is configured.
- Desktop notifications on supported local machines.

## Key Terms

- **Alert channel** — a delivery path such as Telegram, email, or desktop.
- **Priority** — alert severity: `info`, `warning`, or `critical`.
- **Workflow summary** — the single final Telegram message sent by GitHub
  Actions after a workflow finishes.
