import asyncio
from crypto_bot.solana.exit import quick_exit


def test_quick_exit_hits_tp():
    prices = [1.0, 1.11]
    def feed():
        return prices.pop(0) if prices else 1.11
    cfg = {"quick_sell_profit_pct": 10, "quick_sell_timeout_sec": 5, "poll_interval": 0}
    res = asyncio.run(quick_exit(feed, 1.0, cfg))
    assert res["reason"] == "tp"


def test_quick_exit_timeout():
    def feed():
        return 1.0
    cfg = {"quick_sell_profit_pct": 10, "quick_sell_timeout_sec": 0.1, "poll_interval": 0}
    res = asyncio.run(quick_exit(feed, 1.0, cfg))
    assert res["reason"] == "timeout"
