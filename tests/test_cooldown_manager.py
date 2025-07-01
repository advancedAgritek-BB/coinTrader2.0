from datetime import timedelta

from crypto_bot.cooldown_manager import configure, in_cooldown, mark_cooldown, cooldowns


def test_cooldown_resets(monkeypatch):
    configure(10)
    cooldowns.clear()
    mark_cooldown("BTC/USDT", "trend")
    assert in_cooldown("BTC/USDT", "trend") is True
    cooldowns["BTC/USDT_trend"] -= timedelta(seconds=11)
    assert in_cooldown("BTC/USDT", "trend") is False
