import asyncio
import sys
import types
import pytest

# Stub optional solana modules so imports succeed without the package
sys.modules.setdefault("solana", types.ModuleType("solana"))
sys.modules.setdefault("solana.rpc", types.ModuleType("solana.rpc"))
sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
sys.modules.setdefault("solana.transaction", types.ModuleType("solana.transaction"))
sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))
if not hasattr(sys.modules["solana.rpc.async_api"], "AsyncClient"):
    class DummyClient:
        pass

    sys.modules["solana.rpc.async_api"].AsyncClient = DummyClient

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


def test_volume_collection(monkeypatch):
    data = [{"volume": 10}, {"volume": 20}]
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr("crypto_bot.execution.solana_mempool.aiohttp", aiohttp_mod)

    monitor = SolanaMempoolMonitor(volume_url="http://dummy")
    first = monitor.get_recent_volume()
    second = monitor._run_coro(monitor._fetch_volume())
    monitor._volume_history.append(second)
    avg = monitor.get_average_volume()

    assert first == 10
    assert second == 20
    assert avg == 15.0


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
