import asyncio
from pathlib import Path

from crypto_bot.console_monitor import trade_stats_lines
from crypto_bot.utils.open_trades import get_open_trades


class PriceExchange:
    def __init__(self, prices):
        self.prices = prices

    async def fetch_ticker(self, symbol):
        return {"last": self.prices[symbol]}


def test_multi_trade_stats(tmp_path: Path):
    log_file = tmp_path / "trades.csv"
    log_file.write_text(
        "BTC/USDT,buy,1,100,t1\nETH/USDT,buy,1,50,t2\nBTC/USDT,sell,0.5,110,t3\n"
    )

    open_trades = get_open_trades(log_file)
    assert len(open_trades) == 2

    ex = PriceExchange({"BTC/USDT": 120, "ETH/USDT": 60})
    lines = asyncio.run(trade_stats_lines(ex, log_file))
    assert "BTC/USDT +10.00" in lines[0] or "BTC/USDT +10.00" in lines[1]
    assert "ETH/USDT +10.00" in lines[0] or "ETH/USDT +10.00" in lines[1]
