import yaml
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).resolve().parent / 'user_config.yaml'


def prompt_user() -> dict:
    """Prompt for API keys, wallet address and trading mode."""
    data: dict = {}

    # Ask the user which exchange they want to trade on
    exchange = input("Select exchange (coinbase/kraken): ").strip().lower()
    if exchange not in {"coinbase", "kraken"}:
        exchange = "coinbase"
    data["exchange"] = exchange

    # Collect API credentials for both exchanges
    data["coinbase_api_key"] = input("Enter Coinbase API key: ")
    data["coinbase_api_secret"] = input("Enter Coinbase API secret: ")
    data["coinbase_passphrase"] = input("Enter Coinbase API passphrase: ")
    data["kraken_api_key"] = input("Enter Kraken API key: ")
    data["kraken_api_secret"] = input("Enter Kraken API secret: ")

    # Telegram and wallet information
    data["telegram_token"] = input("Enter Telegram bot token: ")
    data["telegram_chat_id"] = input("Enter Telegram chat id: ")
    data["wallet_address"] = input("Enter public wallet address: ")

    # Preferred trading mode
    mode = input("Preferred trading mode (cex/onchain/auto): ")
    data["mode"] = mode if mode in {"cex", "onchain", "auto"} else "auto"

    return data


def load_or_create() -> dict:
    """Load credentials from file or prompt the user."""
    if CONFIG_FILE.exists():
        logger.info("Loading user configuration from %s", CONFIG_FILE)
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f)
    creds = prompt_user()
    logger.info("Creating new user configuration at %s", CONFIG_FILE)
    with open(CONFIG_FILE, 'w') as f:
        yaml.safe_dump(creds, f)
    return creds
