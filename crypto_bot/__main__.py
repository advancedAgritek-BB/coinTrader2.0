import logging

from logging_setup import setup_logging

# Disable console logging; only log to file so the monitor can display
# the latest entry without a noisy stream in the terminal.
setup_logging(level="INFO", logfile="bot.log", console_level=None)

logger = logging.getLogger("crypto_bot.main")
logger.info("Starting bot")

from .cli import main

if __name__ == "__main__":
    main()
