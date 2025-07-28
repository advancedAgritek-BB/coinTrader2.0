from crypto_bot.strategy import flash_crash_bot
from crypto_bot import strategy_router


def test_get_strategy_by_name():
    fn = strategy_router.get_strategy_by_name("flash_crash_bot")
    assert callable(fn)
