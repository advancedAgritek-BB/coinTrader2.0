from pathlib import Path
import asyncio
import io
import sys
import pytest


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


class StopLoop(Exception):
    pass


def test_monitor_loop_reads_incremental(monkeypatch, tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("first\n")

    outputs = []

    def fake_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if "[Monitor]" in text:
            outputs.append(text)

    async def fake_stats(*_a, **_kw):
        return ""

    call_count = 0

    async def fake_sleep(_):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            with open(log_file, "a") as fh:
                fh.write("second\n")
        if call_count >= 3:
            raise StopLoop

    monkeypatch.setattr(console_monitor, "trade_stats_line", fake_stats)
    monkeypatch.setattr(console_monitor.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("builtins.print", fake_print)

    ex = type("Ex", (), {"fetch_balance": lambda self: {"USDT": {"free": 0}}})()

    with pytest.raises(StopLoop):
        asyncio.run(console_monitor.monitor_loop(ex, None, log_file))

    assert outputs[0].endswith("first'")
    assert outputs[1].endswith("second'")


def test_monitor_loop_stringio_no_extra_newlines(monkeypatch, tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("start\n")

    async def fake_stats(*_a, **_kw):
        return ""

    call_count = 0

    class StopLoop(Exception):
        pass

    async def fake_sleep(_):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise StopLoop

    monkeypatch.setattr(console_monitor, "trade_stats_line", fake_stats)
    monkeypatch.setattr(console_monitor.asyncio, "sleep", fake_sleep)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)

    ex = type("Ex", (), {"fetch_balance": lambda self: {"USDT": {"free": 0}}})()

    with pytest.raises(StopLoop):
        asyncio.run(console_monitor.monitor_loop(ex, None, log_file))

    # monitor_loop stops before printing on the final iteration
    printed_lines = buf.getvalue().count("\n")
    assert printed_lines == call_count - 1

