import importlib.util
import sys
from pathlib import Path
import types

# Create dummy package structure to load market_loader without executing package __init__
crypto_bot = types.ModuleType("crypto_bot")
utils_pkg = types.ModuleType("crypto_bot.utils")
crypto_bot.utils = utils_pkg
utils_pkg.__path__ = [str(Path("crypto_bot/utils"))]
sys.modules["crypto_bot"] = crypto_bot
sys.modules["crypto_bot.utils"] = utils_pkg

spec = importlib.util.spec_from_file_location(
    "crypto_bot.utils.market_loader", Path("crypto_bot/utils/market_loader.py")
)
market_loader = importlib.util.module_from_spec(spec)
sys.modules["crypto_bot.utils.market_loader"] = market_loader
spec.loader.exec_module(market_loader)

timeframe_seconds = market_loader.timeframe_seconds


class DummyExchange:
    def parse_timeframe(self, tf: str) -> int:
        raise Exception("fail")


def test_timeframe_seconds_seconds_unit():
    ex = DummyExchange()
    assert timeframe_seconds(ex, "30s") == 30
