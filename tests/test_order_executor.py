import asyncio

from crypto_bot.execution import order_executor


class DummyNotifier:
    def notify(self, msg: str) -> None:
        return None


def test_execute_trade_async_live_calls_cex(monkeypatch):
    calls = {}

    async def fake_cex_execute(*args, **kwargs):
        calls['kwargs'] = kwargs
        return {'ok': True}

    monkeypatch.setattr(order_executor.cex_executor, 'execute_trade_async', fake_cex_execute)

    asyncio.run(
        order_executor.execute_trade_async(
            exchange=object(),
            ws_client=None,
            symbol="BTC/USDT",
            side="buy",
            amount=1.0,
            notifier=DummyNotifier(),
            dry_run=False,
        )
    )

    assert calls['kwargs']['dry_run'] is False
