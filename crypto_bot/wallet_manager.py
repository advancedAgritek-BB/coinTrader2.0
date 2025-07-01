import os
import yaml
from pathlib import Path

from crypto_bot.utils.logger import setup_logger

try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None

logger = setup_logger(__name__, "crypto_bot/logs/wallet.log")

CONFIG_FILE = Path(__file__).resolve().parent / 'user_config.yaml'

FERNET_KEY = os.getenv("FERNET_KEY")
_fernet = Fernet(FERNET_KEY) if FERNET_KEY and Fernet else None

SENSITIVE_FIELDS = {
    "coinbase_api_key",
    "coinbase_api_secret",
    "coinbase_passphrase",
    "kraken_api_key",
    "kraken_api_secret",
    "telegram_token",
}


def _encrypt(value: str) -> str:
    if _fernet:
        return _fernet.encrypt(value.encode()).decode()
    return value


def _decrypt(value: str) -> str:
    if _fernet:
        try:
            return _fernet.decrypt(value.encode()).decode()
        except Exception:
            pass
    return value


def _env_or_prompt(name: str, prompt: str) -> str:
    """Return environment variable value or prompt the user."""
    return os.getenv(name) or input(prompt)


def prompt_user() -> dict:
    """Prompt for API keys, wallet address and trading mode."""
    data: dict = {}

    exchange = os.getenv("EXCHANGE") or input("Select exchange (coinbase/kraken): ").strip().lower()
    if exchange not in {"coinbase", "kraken"}:
        exchange = "coinbase"
    data["exchange"] = exchange

    # Collect API credentials for both exchanges using env vars when available
    data["coinbase_api_key"] = _env_or_prompt("COINBASE_API_KEY", "Enter Coinbase API key: ")
    data["coinbase_api_secret"] = _env_or_prompt("COINBASE_API_SECRET", "Enter Coinbase API secret: ")
    data["coinbase_passphrase"] = _env_or_prompt("API_PASSPHRASE", "Enter Coinbase API passphrase: ")
    data["kraken_api_key"] = _env_or_prompt("KRAKEN_API_KEY", "Enter Kraken API key: ")
    data["kraken_api_secret"] = _env_or_prompt("KRAKEN_API_SECRET", "Enter Kraken API secret: ")

    data["telegram_token"] = _env_or_prompt("TELEGRAM_TOKEN", "Enter Telegram bot token: ")
    data["telegram_chat_id"] = _env_or_prompt("TELEGRAM_CHAT_ID", "Enter Telegram chat id: ")
    data["wallet_address"] = _env_or_prompt("WALLET_ADDRESS", "Enter public wallet address: ")

    mode = os.getenv("MODE") or input("Preferred trading mode (cex/onchain/auto): ")
    data["mode"] = mode if mode in {"cex", "onchain", "auto"} else "auto"

    return data


def load_or_create() -> dict:
    """Load credentials from file or prompt the user."""
    if CONFIG_FILE.exists():
        logger.info("Loading user configuration from %s", CONFIG_FILE)
        with open(CONFIG_FILE) as f:
            data = yaml.safe_load(f) or {}
        for key in SENSITIVE_FIELDS:
            if key in data:
                data[key] = _decrypt(data[key])
        return data

    creds = prompt_user()
    logger.info("Creating new user configuration at %s", CONFIG_FILE)

    file_data = creds.copy()
    if _fernet is None:
        for key in SENSITIVE_FIELDS:
            file_data.pop(key, None)
    else:
        for key in SENSITIVE_FIELDS:
            if key in file_data:
                file_data[key] = _encrypt(file_data[key])

    with open(CONFIG_FILE, 'w') as f:
        yaml.safe_dump(file_data, f)

    return creds
