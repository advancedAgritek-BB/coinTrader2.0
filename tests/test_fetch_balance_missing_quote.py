import asyncio
import crypto_bot.main as main


class DummyExchange:
    async def fetch_balance(self):
        return {"BTC": {"free": 1}}


def test_fetch_balance_missing_quote_returns_zero():
    cfg = {"execution_mode": "live", "quote": "USDT"}
    balance = asyncio.run(main.fetch_balance(DummyExchange(), None, cfg))
    assert balance == 0.0
