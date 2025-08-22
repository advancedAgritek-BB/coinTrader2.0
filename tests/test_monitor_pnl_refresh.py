import asyncio
import pytest

from crypto_bot import console_monitor


class StopLoop(Exception):
    pass


def test_monitor_pnl_refresh(monkeypatch, tmp_path):
    log_file = tmp_path / "bot.log"
    log_file.write_text("start\n")

    outputs = []

    def fake_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text.startswith("\033"):
            return
        outputs.append(text)

    call_counts = {"sleep": 0, "stats": 0}

    async def fake_stats(*_a, **_kw):
        call_counts["stats"] += 1
        if call_counts["stats"] == 1:
            return ["XBT/USDT -- 100.00 -- +0.00"]
        return ["XBT/USDT -- 100.00 -- +10.00"]

    async def fake_sleep(_):
        call_counts["sleep"] += 1
        if call_counts["sleep"] >= 3:
            raise StopLoop

    ex = type("Ex", (), {"fetch_balance": lambda self: {"USDT": {"free": 0}}})()

    monkeypatch.setattr(console_monitor, "trade_stats_lines", fake_stats)
    monkeypatch.setattr(console_monitor.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("builtins.print", fake_print)

    with pytest.raises(StopLoop):
        asyncio.run(console_monitor.monitor_loop(ex, None, log_file))

    first_lines = outputs[0].splitlines()
    second_lines = outputs[1].splitlines()

    assert first_lines[0] == "start"
    assert first_lines[1] == "Balance: 0"
    assert first_lines[2:] == ["XBT/USDT -- 100.00 -- +0.00"]
    assert second_lines[0] == "start"
    assert second_lines[1] == "Balance: 0"
    assert second_lines[2:] == ["XBT/USDT -- 100.00 -- +10.00"]
    assert first_lines != second_lines
