import pytest

from crypto_bot import strategies


class SyncStrategy:
    def __init__(self):
        self.name = "sync"
        self.initialized = False

    def initialize(self, symbols):
        self.initialized = True
        self.symbols = symbols

    def score(self, *, symbols, timeframes):
        return 0.5


class AsyncStrategy:
    def __init__(self):
        self.name = "async"
        self.initialized = False

    async def initialize(self, symbols):
        self.initialized = True
        self.symbols = symbols

    async def generate_signal(self, *, symbols, timeframes):
        return 0.9, "long"


@pytest.mark.asyncio
async def test_initialize_loads_strategies(monkeypatch):
    sync = SyncStrategy()
    async_strat = AsyncStrategy()
    monkeypatch.setattr(
        strategies.loader,
        "load_strategies",
        lambda *a, **k: [sync, async_strat],
    )

    strategies.LOADED_STRATEGIES.clear()
    await strategies.initialize(["BTC/USD"], mode="cex")

    assert set(strategies.LOADED_STRATEGIES) == {"sync", "async"}
    assert sync.initialized and async_strat.initialized


@pytest.mark.asyncio
async def test_score_returns_mapping(monkeypatch):
    sync = SyncStrategy()
    async_strat = AsyncStrategy()
    monkeypatch.setattr(
        strategies.loader,
        "load_strategies",
        lambda *a, **k: [sync, async_strat],
    )

    strategies.LOADED_STRATEGIES.clear()
    await strategies.initialize(["ETH/USD"], mode="cex")
    scores = await strategies.score(symbols=["ETH/USD"], timeframes=["1m"])

    assert scores and scores["sync"] == pytest.approx(0.5) and scores["async"] == pytest.approx(0.9)
