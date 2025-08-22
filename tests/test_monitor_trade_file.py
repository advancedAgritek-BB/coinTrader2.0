import asyncio
import pytest

from crypto_bot import console_monitor


class StopLoop(Exception):
    pass


def test_monitor_loop_custom_trade_file(monkeypatch, tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("start\n")

    trade_file = tmp_path / "trades.csv"
    trade_file.write_text("XBT/USDT,buy,1,100,t1\n")

    outputs = []

    def fake_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text.startswith("\033"):
            return
        outputs.append(text)

    call_count = 0

    async def fake_sleep(_):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            with open(trade_file, "a") as fh:
                fh.write("XBT/USDT,sell,1,120,t2\n")
        if call_count >= 3:
            raise StopLoop

    class PriceEx:
        def fetch_balance(self):
            return {"USDT": {"free": 0}}

        def fetch_ticker(self, symbol):
            return {"last": 110}

    ex = PriceEx()

    monkeypatch.setattr(console_monitor.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("builtins.print", fake_print)

    with pytest.raises(StopLoop):
        asyncio.run(console_monitor.monitor_loop(ex, None, log_file, trade_file))

    first_lines = outputs[0].splitlines()
    second_lines = outputs[1].splitlines()

    assert first_lines[0] == "start"
    assert first_lines[1] == "Balance: 0"
    assert first_lines[2:] == ["XBT/USDT -- 100.00 -- +10.00"]
    assert second_lines[0] == "start"
    assert second_lines[1] == "Balance: 0"
    assert second_lines[2:] == []
    assert first_lines != second_lines
