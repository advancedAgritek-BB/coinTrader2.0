import asyncio
import pytest

from crypto_bot.solana.exit import monitor_price

class DummyMonitor:
    def __init__(self, suspicious: bool = False):
        self.suspicious = suspicious
        self.calls = 0

    async def is_suspicious(self, threshold: float) -> bool:
        self.calls += 1
        return self.suspicious

@pytest.mark.asyncio
async def test_monitor_price_mempool_spike():
    prices = [100.0, 101.0]
    def feed():
        return prices[0]

    monitor = DummyMonitor(True)
    res = await monitor_price(
        feed,
        100.0,
        {"poll_interval": 0},
        mempool_monitor=monitor,
        mempool_cfg={"enabled": True, "suspicious_fee_threshold": 1.0},
    )
    assert res["reason"] == "mempool_spike"
    assert res["exit_price"] == 100.0
    assert monitor.calls >= 1
