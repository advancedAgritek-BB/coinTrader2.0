from __future__ import annotations

import os
from pathlib import Path
import yaml

from crypto_bot.utils.telegram import send_test_message

CONFIG_PATH = Path(__file__).resolve().parents[1] / "crypto_bot" / "config.yaml"


def load_config() -> dict:
    """Load YAML configuration if available."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> None:
    """Send a test Telegram message using config or env variables."""
    cfg = load_config()
    tele_cfg = cfg.get("telegram", {})
    token = os.getenv("TELEGRAM_TOKEN") or tele_cfg.get("token", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or tele_cfg.get("chat_id", "")

    if send_test_message(token, chat_id, "Test message from CoinTrader2.0"):
        print("Telegram credentials valid. Message sent.")
    else:
        print("Failed to send test message. Check token and chat ID.")


if __name__ == "__main__":
    main()
