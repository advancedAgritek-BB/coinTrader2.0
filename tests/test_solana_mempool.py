import asyncio
import pytest

from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot.execution import solana_executor


def test_is_suspicious_from_env(monkeypatch):
    monkeypatch.setenv("MOCK_PRIORITY_FEE", "150")
    monitor = SolanaMempoolMonitor()
    assert monitor.is_suspicious(100) is True


def test_is_not_suspicious(monkeypatch):
    monkeypatch.setenv("MOCK_PRIORITY_FEE", "10")
    monitor = SolanaMempoolMonitor()
    assert monitor.is_suspicious(100) is False


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
    first = await monitor.get_recent_volume()
    second = await monitor._fetch_volume()
    monitor._volume_history.append(second)
    avg = await monitor.get_average_volume()

    assert first == 10
    assert second == 20
    assert avg == pytest.approx(15.0)


class DummyNotifier:
    def notify(self, text: str):
        return None


class DummyMonitor:
    def fetch_priority_fee(self):
        return 0.0

    def is_suspicious(self, threshold):
        return True


def test_swap_paused(monkeypatch):
    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            1,
            notifier=DummyNotifier(),
            dry_run=False,
            mempool_monitor=DummyMonitor(),
            mempool_cfg={"enabled": True, "action": "pause", "suspicious_fee_threshold": 0},
            config={"confirm_execution": True},
        )
    )
    assert res.get("paused") is True


def test_swap_repriced(monkeypatch):
    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            2,
            notifier=DummyNotifier(),
            dry_run=True,
            mempool_monitor=DummyMonitor(),
            mempool_cfg={"enabled": True, "action": "reprice", "reprice_multiplier": 1.5, "suspicious_fee_threshold": 0},
            config={},
        )
    )
    assert res["amount"] == 3.0
