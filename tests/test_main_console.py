from pathlib import Path
import asyncio


from crypto_bot import console_monitor


class DummyExchange:
    pass


class DummyWallet:
    pass


def test_console_monitor_outputs_table(monkeypatch, tmp_path):
    trade_file = tmp_path / "trades.csv"
    trade_file.write_text("BTC/USDT,buy,0.1,20000\nETH/USDT,sell,0.05,3000\n")
    monkeypatch.setattr(console_monitor, "TRADE_FILE", trade_file)
    output = console_monitor.display_trades(DummyExchange(), DummyWallet(), trade_file)
    assert "BTC/USDT" in output
    assert "ETH/USDT" in output


class PriceExchange:
    def __init__(self, prices):
        self.prices = prices

    def fetch_ticker(self, symbol):
        return {"last": self.prices[symbol]}


def test_trade_stats_line(tmp_path):
    trade_file = tmp_path / "trades.csv"
    trade_file.write_text("BTC/USDT,buy,1,100,ts\n")
    ex = PriceExchange({"BTC/USDT": 110})
    line = asyncio.run(console_monitor.trade_stats_line(ex, trade_file))
    assert "BTC/USDT" in line
    assert "+10.00" in line

