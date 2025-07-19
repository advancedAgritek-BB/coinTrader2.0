import asyncio

import crypto_bot.main as main
from crypto_bot.phase_runner import BotContext


class DummyExchange:
    async def fetch_balance(self):
        return {"USDT": {"free": 555}}


class DummyNotifier:
    def __init__(self):
        self.sent = []
        self.enabled = True

    def notify(self, text):
        self.sent.append(text)


def test_refresh_balance_updates_context_and_logs():
    ctx = BotContext(positions={}, df_cache={}, regime_cache={}, config={"execution_mode": "live", "telegram": {"balance_updates": True}})
    ctx.exchange = DummyExchange()
    ctx.paper_wallet = None
    ctx.notifier = DummyNotifier()
    ctx.balance = 100.0

    asyncio.run(main.refresh_balance(ctx))

    assert ctx.balance == 555
    assert ctx.notifier.sent == ["Balance changed: 555.00 USDT"]
