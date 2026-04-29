import os
import re

import requests

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_S = 15


def _to_telegram_markdown(text: str) -> str:
    """
    Convert LLM Markdown to Telegram's legacy Markdown subset.
    Telegram legacy mode supports *bold*, _italic_, `code`, [text](url).
    LLMs typically emit **bold** and ### headings — normalise those.
    """
    # **bold** → *bold*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)
    # ### Heading → *Heading*
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    return text


def send_telegram_alert(message: str) -> bool:
    """
    Send message to the configured Telegram chat.
    Returns True on success, False on any failure (non-fatal — pipeline still completes).
    Requires env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
    """
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("[Telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing — skipping notification.")
        return False

    url     = _TELEGRAM_API.format(token=token)
    payload = {
        "chat_id":    chat_id,
        "text":       _to_telegram_markdown(message),
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=_TIMEOUT_S)
        resp.raise_for_status()
        print("[Telegram] Alert sent successfully.")
        return True
    except requests.RequestException as exc:
        print(f"[Telegram] Failed to send alert: {exc}")
        return False
