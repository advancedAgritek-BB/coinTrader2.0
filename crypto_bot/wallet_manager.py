import yaml
from pathlib import Path

from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/wallet.log")

CONFIG_FILE = Path(__file__).resolve().parent / 'user_config.yaml'


def prompt_user() -> dict:
    """Prompt for API keys, wallet address and trading mode."""
    data = {}
    data['binance_api_key'] = input('Enter Binance API key: ')
    data['binance_api_secret'] = input('Enter Binance API secret: ')
    data['telegram_token'] = input('Enter Telegram bot token: ')
    data['telegram_chat_id'] = input('Enter Telegram chat id: ')
    data['wallet_address'] = input('Enter public wallet address: ')
    mode = input('Preferred trading mode (cex/onchain/auto): ')
    data['mode'] = mode if mode in {'cex', 'onchain', 'auto'} else 'auto'
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
