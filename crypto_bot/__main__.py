import logging

from logging_setup import setup_logging

setup_logging(level="INFO", logfile="bot.log", console_level="WARNING")

logger = logging.getLogger("crypto_bot.main")
logger.info("Starting bot")

from .cli import main

if __name__ == "__main__":
    main()
