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
    trade_file.write_text("XBT/USDT,buy,0.1,20000\nETH/USDT,sell,0.05,3000\n")
    monkeypatch.setattr(console_monitor, "TRADE_FILE", trade_file)
    output = console_monitor.display_trades(DummyExchange(), DummyWallet(), trade_file)
    assert "XBT/USDT" in output
    assert "ETH/USDT" in output


class PriceExchange:
    def __init__(self, prices):
        self.prices = prices

    def fetch_ticker(self, symbol):
        return {"last": self.prices[symbol]}


def test_trade_stats_lines(tmp_path):
    trade_file = tmp_path / "trades.csv"
    trade_file.write_text(
        "XBT/USDT,buy,1,100,ts1\nETH/USDT,sell,2,60,ts2\n"
    )
    ex = PriceExchange({"XBT/USDT": 110, "ETH/USDT": 55})
    lines = asyncio.run(console_monitor.trade_stats_lines(ex, trade_file))
    assert sorted(lines) == sorted(
        [
            "XBT/USDT -- 100.00 -- +10.00",
            "ETH/USDT -- 60.00 -- +10.00",
        ]
    )

    joined = " | ".join(lines)
    assert "XBT/USDT -- 100.00 -- +10.00" in joined
    assert "ETH/USDT -- 60.00 -- +10.00" in joined


class StopLoop(Exception):
    pass


def test_monitor_loop_reads_incremental(monkeypatch, tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("first\n")

    outputs = []

    def fake_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text.startswith("\033"):
            return
        outputs.append(text)

    async def fake_stats(*_a, **_kw):
        return ["stat1", "stat2"]

    call_count = 0

    async def fake_sleep(_):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            with open(log_file, "a") as fh:
                fh.write("second\n")
        if call_count >= 3:
            raise StopLoop

    monkeypatch.setattr(console_monitor, "trade_stats_lines", fake_stats)
    monkeypatch.setattr(console_monitor.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("builtins.print", fake_print)

    ex = type("Ex", (), {"fetch_balance": lambda self: {"USDT": {"free": 0}}})()

    with pytest.raises(StopLoop):
        asyncio.run(console_monitor.monitor_loop(ex, None, log_file))

    assert outputs[0].splitlines()[0] == "first"
    assert outputs[0].splitlines()[1] == "Balance: 0"
    assert outputs[1].splitlines()[0] == "second"
    assert outputs[1].splitlines()[1] == "Balance: 0"
    assert outputs[0].splitlines()[2:] == ["stat1", "stat2"]


def test_monitor_loop_stringio_no_extra_newlines(monkeypatch, tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("start\n")

    async def fake_stats(*_a, **_kw):
        return []

    call_count = 0

    class StopLoop(Exception):
        pass

    async def fake_sleep(_):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise StopLoop

    monkeypatch.setattr(console_monitor, "trade_stats_lines", fake_stats)
    monkeypatch.setattr(console_monitor.asyncio, "sleep", fake_sleep)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)

    ex = type("Ex", (), {"fetch_balance": lambda self: {"USDT": {"free": 0}}})()

    with pytest.raises(StopLoop):
        asyncio.run(console_monitor.monitor_loop(ex, None, log_file))

    # monitor_loop stops before printing on the final iteration and
    # should skip duplicate lines when stdout is not a TTY
    printed_lines = buf.getvalue().count("\n")
    assert printed_lines == 2
    assert buf.getvalue().splitlines() == ["start", "Balance: 0"]


def test_monitor_loop_skips_duplicate_lines(monkeypatch, tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("same\n")

    async def fake_stats(*_a, **_kw):
        return []

    outputs = []

    class StopLoop(Exception):
        pass

    call_count = 0

    async def fake_sleep(_):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            with open(log_file, "a") as fh:
                fh.write("same\n")
        if call_count == 3:
            with open(log_file, "a") as fh:
                fh.write("other\n")
        if call_count >= 4:
            raise StopLoop

    def fake_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text.startswith("\033"):
            return
        outputs.append(text)

    ex = type("Ex", (), {"fetch_balance": lambda self: {"USDT": {"free": 0}}})()

    monkeypatch.setattr(console_monitor, "trade_stats_lines", fake_stats)
    monkeypatch.setattr(console_monitor.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    with pytest.raises(StopLoop):
        asyncio.run(console_monitor.monitor_loop(ex, None, log_file))

    assert len(outputs) == 2
    assert outputs[0].splitlines()[0] == "same"
    assert outputs[0].splitlines()[1] == "Balance: 0"
    assert outputs[1].splitlines()[0] == "other"
    assert outputs[1].splitlines()[1] == "Balance: 0"


def test_monitor_loop_quiet_mode(monkeypatch, tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("first\n")

    async def fake_stats(*_a, **_kw):
        return []

    outputs = []

    class StopLoop(Exception):
        pass

    call_count = 0

    async def fake_sleep(_):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            with open(log_file, "a") as fh:
                fh.write("second\n")
        if call_count >= 3:
            raise StopLoop

    def fake_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text.startswith("\033"):
            return
        outputs.append(text)

    ex = type("Ex", (), {"fetch_balance": lambda self: {"USDT": {"free": 0}}})()

    monkeypatch.setattr(console_monitor, "trade_stats_lines", fake_stats)
    monkeypatch.setattr(console_monitor.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    with pytest.raises(StopLoop):
        asyncio.run(
            console_monitor.monitor_loop(
                ex, None, log_file, quiet_mode=True
            )
        )

    assert len(outputs) == 1
    assert outputs[0].splitlines()[0] == "first"
    assert outputs[0].splitlines()[1] == "Balance: 0"


def test_monitor_loop_async_balance(monkeypatch, tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("start\n")

    async def fake_stats(*_a, **_kw):
        return []

    class AsyncEx:
        async def fetch_balance(self):
            return {"USDT": {"free": 0}}

    class StopLoop(Exception):
        pass

    async def fake_sleep(_):
        raise StopLoop

    monkeypatch.setattr(console_monitor, "trade_stats_lines", fake_stats)
    monkeypatch.setattr(console_monitor.asyncio, "sleep", fake_sleep)

    ex = AsyncEx()
    with pytest.raises(StopLoop):
        asyncio.run(console_monitor.monitor_loop(ex, None, log_file))

