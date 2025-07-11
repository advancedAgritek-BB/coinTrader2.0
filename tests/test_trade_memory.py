from crypto_bot.utils import trade_memory


def test_should_avoid_after_losses(tmp_path, monkeypatch):
    mem = tmp_path / "mem.json"
    monkeypatch.setattr(trade_memory, "LOG_FILE", mem)
    trade_memory.configure(max_losses=2, slippage_threshold=0.1, lookback_seconds=3600)
    trade_memory.clear()

    trade_memory.record_loss("XBT/USDT", 0.01)
    assert trade_memory.should_avoid("XBT/USDT") is False
    trade_memory.record_loss("XBT/USDT", 0.02)
    assert trade_memory.should_avoid("XBT/USDT") is True


def test_should_avoid_on_slippage(tmp_path, monkeypatch):
    mem = tmp_path / "mem.json"
    monkeypatch.setattr(trade_memory, "LOG_FILE", mem)
    trade_memory.configure(max_losses=5, slippage_threshold=0.05, lookback_seconds=3600)
    trade_memory.clear()

    trade_memory.record_loss("ETH/USDT", 0.1)
    assert trade_memory.should_avoid("ETH/USDT") is True
