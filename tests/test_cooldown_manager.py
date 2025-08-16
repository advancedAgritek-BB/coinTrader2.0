from datetime import timedelta

from crypto_bot.cooldown_manager import configure, in_cooldown, mark_cooldown, cooldowns


def test_cooldown_resets(monkeypatch):
    configure(10)
    cooldowns.clear()
    mark_cooldown("XBT/USDT", "trend_bot")
    assert in_cooldown("XBT/USDT", "trend_bot") is True
    cooldowns["XBT/USDT_trend_bot"] -= timedelta(seconds=11)
    assert in_cooldown("XBT/USDT", "trend_bot") is False
