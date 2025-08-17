import logging
import logging.config
import sys


def setup_logging(level="INFO", logfile="bot.log", console_level=None):
    handlers: dict[str, dict] = {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "console",
            "filename": logfile,
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 3,
            "encoding": "utf8",
        }
    }
    root_handlers = ["file"]
    if console_level is not None:
        handlers["stdout"] = {
            "class": "logging.StreamHandler",
            "level": console_level,
            "formatter": "console",
            "stream": sys.stdout,
        }
        root_handlers.insert(0, "stdout")

    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"},
            "json": {
                "format": '{"t":"%(asctime)s","lv":"%(levelname)s","mod":"%(name)s","thread":"%(threadName)s","msg":"%(message)s"}'
            },
        },
        "handlers": handlers,
        "loggers": {
            "": {"handlers": root_handlers, "level": level},
            "crypto_bot.symbols": {"level": level},
            "crypto_bot.ohlcv": {"level": level},
            "crypto_bot.trader": {"level": level},
            "crypto_bot.onchain": {"level": level},
            "telegram": {"level": "WARNING"},
            "apscheduler": {"level": "INFO"},
        },
    }
    logging.config.dictConfig(LOGGING)
    logging.captureWarnings(True)


def redact(s: str) -> str:
    if not s:
        return s
    return s[:4] + "…" + s[-4:]
