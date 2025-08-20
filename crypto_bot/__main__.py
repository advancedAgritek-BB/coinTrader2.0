import logging

from crypto_bot.utils.logger import LOG_DIR
from crypto_bot.utils.logging_config import setup_logging

setup_logging(LOG_DIR / "bot.log", level=logging.INFO)

from .cli import main

if __name__ == "__main__":
    main()
