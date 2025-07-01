import ccxt
from crypto_bot.execution.cex_executor import place_stop_order

class DummyExchange:
    def create_order(self, symbol, type_, side, amount, params=None):
        return {
            "id": "1",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "type": type_,
            "params": params,
        }


def test_place_stop_order_dry_run():
    order = place_stop_order(
        DummyExchange(),
        "BTC/USDT",
        "sell",
        1,
        9000,
        "token",
        "chat",
        dry_run=True,
    )
    assert order["dry_run"] is True
    assert order["stop"] == 9000

