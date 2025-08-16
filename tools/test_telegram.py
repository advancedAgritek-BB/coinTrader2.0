from __future__ import annotations

import os
from pathlib import Path
import asyncio
import yaml

from crypto_bot.utils.telegram import send_test_message
from crypto_bot.utils.symbol_utils import fix_symbol

CONFIG_PATH = Path(__file__).resolve().parents[1] / "crypto_bot" / "config.yaml"


def load_config() -> dict:
    """Load YAML configuration if available."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    strat_dir = CONFIG_PATH.parent.parent / "config" / "strategies"
    trend_file = strat_dir / "trend_bot.yaml"
    if trend_file.exists():
        with open(trend_file) as sf:
            overrides = yaml.safe_load(sf) or {}
        trend_cfg = data.get("trend_bot", {})
        if isinstance(trend_cfg, dict):
            trend_cfg.update(overrides)
        else:
            trend_cfg = overrides
        data["trend_bot"] = trend_cfg

    if "symbol" in data:
        data["symbol"] = fix_symbol(data["symbol"])
    if "symbols" in data:
        data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]
    return data


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
