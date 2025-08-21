import importlib.util
import pathlib
import sys
import types
import asyncio
import pandas as pd
import pytest

sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
if not hasattr(sys.modules["solana.rpc.async_api"], "AsyncClient"):
    class DummyClient:
        pass

    sys.modules["solana.rpc.async_api"].AsyncClient = DummyClient

# Stub heavy Solana trading modules to avoid optional dependencies during import
sol_pkg = types.ModuleType("crypto_bot.solana")
sol_pkg.__path__ = []
exit_mod = types.ModuleType("crypto_bot.solana.exit")
exit_mod.monitor_price = lambda *a, **k: {}
trading_mod = types.ModuleType("crypto_bot.solana_trading")
trading_mod.sniper_trade = lambda *a, **k: {}
sol_pkg.exit = exit_mod
sys.modules.setdefault("crypto_bot.solana", sol_pkg)
sys.modules.setdefault("crypto_bot.solana.exit", exit_mod)
sys.modules.setdefault("crypto_bot.solana_trading", trading_mod)
scalping_mod = types.ModuleType("crypto_bot.solana.scalping")
scalping_mod.generate_signal = lambda *a, **k: (0.0, "none")
sys.modules.setdefault("crypto_bot.solana.scalping", scalping_mod)
sniper_mod = types.ModuleType("crypto_bot.strategies.sniper_solana")
sniper_mod.generate_signal = lambda *a, **k: (0.0, "none")
sys.modules.setdefault("crypto_bot.strategies.sniper_solana", sniper_mod)
sys.modules.setdefault("redis", types.ModuleType("redis"))

# Stub out ml_utils to bypass optional dependency issues
ml_utils_stub = types.ModuleType("ml_utils")
ml_utils_stub.is_ml_available = lambda: False
ml_utils_stub.ML_AVAILABLE = False
sys.modules.setdefault("crypto_bot.utils.ml_utils", ml_utils_stub)

# The meme-wave strategy imports Solana helpers which depend on the optional
# ``solana`` package. Tests should run even when that dependency isn't
# installed, so we provide minimal stub modules mirroring the pattern used in
# ``tests/test_solana_executor.py``.
if importlib.util.find_spec("solana") is None:  # pragma: no cover - optional
    sys.modules.setdefault("solana", types.ModuleType("solana"))
    sys.modules.setdefault("solana.rpc", types.ModuleType("solana.rpc"))
    async_mod = types.ModuleType("solana.rpc.async_api")
    setattr(async_mod, "AsyncClient", object)
    sys.modules.setdefault("solana.rpc.async_api", async_mod)
    sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    sys.modules.setdefault("solana.transaction", types.ModuleType("solana.transaction"))
    sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))

path = pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "strategy" / "meme_wave_bot.py"
spec = importlib.util.spec_from_file_location("meme_wave_bot", path)
meme_wave_bot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(meme_wave_bot)


@pytest.fixture(autouse=True)
def stub_solana(monkeypatch):
    """Provide minimal solana modules so imports succeed."""
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    yield


def _make_df(prices, volumes):
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": volumes,
        }
    )


@pytest.fixture
def meme_df():
    """Factory producing DataFrames with custom prices and volumes."""

    def _factory(price=1.0, spike=False):
        prices = [price] * 20
        prices.append(price + (1 if spike else 0))
        volumes = [100] * 20
        volumes.append(400 if spike else 100)
        return _make_df(prices, volumes)

    return _factory


class DummyMonitor:
    def __init__(self, recent, avg):
        self._recent = recent
        self._avg = avg

    def get_recent_volume(self):
        return self._recent

    def get_average_volume(self):
        return self._avg


@pytest.fixture
def high_monitor():
    return DummyMonitor(6000, 1000)


@pytest.fixture
def low_monitor():
    return DummyMonitor(500, 1000)


def test_generate_signal_returns_tuple():
    df = pd.DataFrame({
        "open": [1, 2],
        "high": [1, 2],
        "low": [1, 2],
        "close": [1, 2],
        "volume": [1, 10],
    })
    score, direction = asyncio.run(meme_wave_bot.generate_signal(df))
    assert isinstance(score, float)
    assert isinstance(direction, str)


def test_get_strategy_by_name():
    from crypto_bot import strategy_router

    fn = strategy_router.get_strategy_by_name("meme_wave_bot")
    assert callable(fn)


def test_high_volume_positive_sentiment(meme_df, high_monitor, monkeypatch):
    df = meme_df(spike=True)
    async def fake_sentiment(*a, **k):
        return 80
    monkeypatch.setattr(
        meme_wave_bot, "fetch_twitter_sentiment_async", fake_sentiment
    )
    def good_sentiment(*a, **k):
        return 80
    monkeypatch.setattr(
        meme_wave_bot, "fetch_twitter_sentiment", good_sentiment, raising=False
    )
    monkeypatch.setattr(
        meme_wave_bot, "asyncio", types.SimpleNamespace(run=lambda x: x), raising=False
    )
    cfg = {"meme_wave_bot": {"volume_threshold": 3, "sentiment_threshold": 0.6}}
    score, direction = asyncio.run(
        meme_wave_bot.generate_signal(df, config=cfg, mempool_monitor=high_monitor)
    )
    assert (score, direction) == (1.0, "long")


def test_high_volume_negative_sentiment(meme_df, high_monitor, monkeypatch):
    df = meme_df(spike=True)
    async def fake_sentiment(*a, **k):
        return 20
    monkeypatch.setattr(
        meme_wave_bot, "fetch_twitter_sentiment_async", fake_sentiment
    )
    def bad_sentiment(*a, **k):
        return 20
    monkeypatch.setattr(
        meme_wave_bot, "fetch_twitter_sentiment", bad_sentiment, raising=False
    )
    monkeypatch.setattr(
        meme_wave_bot, "asyncio", types.SimpleNamespace(run=lambda x: x), raising=False
    )
    cfg = {"meme_wave_bot": {"volume_threshold": 3, "sentiment_threshold": 0.6}}
    score, direction = asyncio.run(
        meme_wave_bot.generate_signal(df, config=cfg, mempool_monitor=high_monitor)
    )
    assert (score, direction) == (0.0, "none")


def test_low_volume_any_sentiment(meme_df, low_monitor, monkeypatch):
    df = meme_df(spike=False)
    async def fake_sentiment(*a, **k):
        return 80
    monkeypatch.setattr(
        meme_wave_bot, "fetch_twitter_sentiment_async", fake_sentiment
    )
    def good_sentiment(*a, **k):
        return 80
    monkeypatch.setattr(
        meme_wave_bot, "fetch_twitter_sentiment", good_sentiment, raising=False
    )
    monkeypatch.setattr(
        meme_wave_bot, "asyncio", types.SimpleNamespace(run=lambda x: x), raising=False
    )
    cfg = {"meme_wave_bot": {"volume_threshold": 3, "sentiment_threshold": 0.6}}
    score, direction = asyncio.run(
        meme_wave_bot.generate_signal(df, config=cfg, mempool_monitor=low_monitor)
    )
    assert (score, direction) == (0.0, "none")

