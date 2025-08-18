from __future__ import annotations

import os
import asyncio

from configy import load_config
from crypto_bot.utils.telegram import send_test_message


def main() -> None:
    """Send a test Telegram message using config or env variables."""
    cfg = load_config()
    tele_cfg = cfg.get("telegram", {})
    token = os.getenv("TELEGRAM_TOKEN") or tele_cfg.get("token", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or tele_cfg.get("chat_id", "")

    try:
        asyncio.run(send_test_message(token, chat_id, "Test message from CoinTrader2.0"))
    except asyncio.TimeoutError:
        print("Timed out contacting Telegram after retries. Please try again later.")
    except ValueError:
        print("Invalid Telegram token or chat ID.")
    except Exception as exc:  # pragma: no cover - network
        print(f"Failed to send test message: {exc}")
    else:
        print("Telegram credentials valid. Message sent.")


if __name__ == "__main__":
    main()
