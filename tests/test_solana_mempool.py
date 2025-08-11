import asyncio
import pytest

import asyncio
import pytest
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor


@pytest.mark.asyncio
async def test_is_suspicious_from_env(monkeypatch):
    monkeypatch.setenv("MOCK_PRIORITY_FEE", "150")
    monitor = SolanaMempoolMonitor()
    assert await monitor.is_suspicious(100) is True


@pytest.mark.asyncio
async def test_is_not_suspicious(monkeypatch):
    monkeypatch.setenv("MOCK_PRIORITY_FEE", "10")
    monitor = SolanaMempoolMonitor()
    assert await monitor.is_suspicious(100) is False


class DummyResp:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummySession:
    def __init__(self, data_iter):
        self._data_iter = iter(data_iter)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, timeout=5):
        try:
            data = next(self._data_iter)
        except StopIteration:
            data = {}
        return DummyResp(data)


@pytest.mark.asyncio
async def test_volume_collection(monkeypatch):
    data = [{"volume": 10}, {"volume": 20}]
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr("crypto_bot.execution.solana_mempool.aiohttp", aiohttp_mod)

    monitor = SolanaMempoolMonitor(volume_url="http://dummy")
    monitor.start_volume_collection(0.001)
    await asyncio.sleep(0.003)
    monitor.stop_volume_collection()

    recent = await monitor.get_recent_volume()
    avg = await monitor.get_average_volume()

    assert recent == 20
    assert avg == pytest.approx(15.0)
