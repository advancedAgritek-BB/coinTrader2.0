from pathlib import Path


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
