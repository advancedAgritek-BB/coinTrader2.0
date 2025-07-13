import asyncio
from pathlib import Path

import pandas as pd

from crypto_bot.utils.open_trades import get_open_trades
from crypto_bot.console_monitor import trade_stats_lines


class PriceExchange:
    def __init__(self, prices):
        self.prices = prices

    async def fetch_ticker(self, symbol):
        return {"last": self.prices[symbol]}


def test_partial_close_to_short(tmp_path: Path):
    log = tmp_path / "trades.csv"
    data = [
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 100, "timestamp": "t1"},
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1.5, "price": 110, "timestamp": "t2"},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 0.2, "price": 105, "timestamp": "t3"},
    ]
    pd.DataFrame(data).to_csv(log, index=False, header=False)

    open_trades = get_open_trades(log)
    assert open_trades == [
        {
            "symbol": "XBT/USDT",
            "side": "short",
            "amount": 0.3,
            "price": 110.0,
            "entry_time": "t2",
        }
    ]

    ex = PriceExchange({"XBT/USDT": 108})
    lines = asyncio.run(trade_stats_lines(ex, log))
    assert lines == ["XBT/USDT -- 110.00 -- +0.60"]


def test_multiple_buys_partial_sells(tmp_path: Path):
    log = tmp_path / "trades.csv"
    data = [
        {"symbol": "ETH/USDT", "side": "buy", "amount": 1, "price": 50, "timestamp": "a"},
        {"symbol": "ETH/USDT", "side": "buy", "amount": 1, "price": 55, "timestamp": "b"},
        {"symbol": "ETH/USDT", "side": "sell", "amount": 1.5, "price": 60, "timestamp": "c"},
        {"symbol": "ETH/USDT", "side": "sell", "amount": 0.2, "price": 65, "timestamp": "d"},
    ]
    pd.DataFrame(data).to_csv(log, index=False, header=False)

    open_trades = get_open_trades(log)
    assert open_trades == [
        {
            "symbol": "ETH/USDT",
            "side": "long",
            "amount": 0.3,
            "price": 55.0,
            "entry_time": "b",
        }
    ]

    ex = PriceExchange({"ETH/USDT": 70})
    lines = asyncio.run(trade_stats_lines(ex, log))
    assert lines == ["ETH/USDT -- 55.00 -- +4.50"]
