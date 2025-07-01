import os
import yaml
from pathlib import Path
from typing import Dict

from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/wallet.log")

CONFIG_FILE = Path(__file__).resolve().parent / 'user_config.yaml'


def load_external_secrets(provider: str, path: str) -> Dict[str, str]:
    """Load secrets from an external provider."""
    try:
        if provider == "aws":
            import boto3

            client = boto3.client("secretsmanager")
            resp = client.get_secret_value(SecretId=path)
            secret = resp.get("SecretString") or resp.get("SecretBinary", "")
            if isinstance(secret, bytes):
                secret = secret.decode()
            return yaml.safe_load(secret)
        if provider == "vault":
            import hvac

            client = hvac.Client()
            secret = client.secrets.kv.v2.read_secret_version(path=path)
            return secret["data"]["data"]
    except Exception as e:  # pragma: no cover - optional providers
        logger.error("Failed to load secrets from %s: %s", provider, e)
    return {}


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
    """Load credentials prioritizing environment variables."""
    creds: Dict[str, str] = {}

    if CONFIG_FILE.exists():
        logger.info("Loading user configuration from %s", CONFIG_FILE)
        with open(CONFIG_FILE) as f:
            creds.update(yaml.safe_load(f))
    else:
        creds.update(prompt_user())
        logger.info("Creating new user configuration at %s", CONFIG_FILE)
        with open(CONFIG_FILE, "w") as f:
            yaml.safe_dump(creds, f)

    provider = os.getenv("SECRETS_PROVIDER")
    if provider:
        path = os.getenv("SECRETS_PATH", "")
        if path:
            creds.update(load_external_secrets(provider, path))

    env_mapping = {key: [key.upper()] for key in creds.keys()}
    env_mapping.setdefault("coinbase_api_key", []).append("API_KEY")
    env_mapping.setdefault("coinbase_api_secret", []).append("API_SECRET")
    env_mapping.setdefault("coinbase_passphrase", []).append("API_PASSPHRASE")
    env_mapping.setdefault("kraken_api_key", []).append("API_KEY")
    env_mapping.setdefault("kraken_api_secret", []).append("API_SECRET")

    for key, env_keys in env_mapping.items():
        for env_key in env_keys:
            val = os.getenv(env_key)
            if val is not None:
                creds[key] = val
                break

    return creds
