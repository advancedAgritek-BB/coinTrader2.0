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
        "XBT/USDT,buy,1,100,t1\n"
        "ETH/USDT,sell,1,60,t2\n"
        "XBT/USDT,sell,0.5,110,t3\n"
    )

    open_trades = get_open_trades(log_file)
    assert len(open_trades) == 2

    ex = PriceExchange({"XBT/USDT": 120, "ETH/USDT": 55})
    lines = asyncio.run(trade_stats_lines(ex, log_file))
    joined = " ".join(lines)
    assert "XBT/USDT -- 100.00 -- +10.00" in joined
    assert "ETH/USDT -- 60.00 -- +5.00" in joined
