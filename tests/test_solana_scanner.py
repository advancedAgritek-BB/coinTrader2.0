import asyncio
import importlib.util
import pathlib
import sys
import types
from collections import deque

# create minimal package structure
pkg_root = types.ModuleType("crypto_bot")
sol_pkg = types.ModuleType("crypto_bot.solana")
utils_pkg = types.ModuleType("crypto_bot.utils")
regime_pkg = types.ModuleType("crypto_bot.regime")
pkg_root.solana = sol_pkg
pkg_root.utils = utils_pkg
pkg_root.regime = regime_pkg
import importlib.machinery
pkg_root.__spec__ = importlib.machinery.ModuleSpec(
    "crypto_bot", None, is_package=True
)
pkg_root.__spec__.submodule_search_locations = []
pkg_root.__path__ = []
sol_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "crypto_bot.solana", None, is_package=True
)
sol_pkg.__spec__.submodule_search_locations = [str(pathlib.Path("crypto_bot/solana"))]
sol_pkg.__path__ = [str(pathlib.Path("crypto_bot/solana"))]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "crypto_bot.utils", None, is_package=True
)
utils_pkg.__spec__.submodule_search_locations = [str(pathlib.Path("crypto_bot/utils"))]
utils_pkg.__path__ = [str(pathlib.Path("crypto_bot/utils"))]
regime_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "crypto_bot.regime", None, is_package=True
)
regime_pkg.__spec__.submodule_search_locations = [str(pathlib.Path("crypto_bot/regime"))]
regime_pkg.__path__ = [str(pathlib.Path("crypto_bot/regime"))]

sys.modules.setdefault("crypto_bot", pkg_root)
sys.modules.setdefault("crypto_bot.solana", sol_pkg)
sys.modules.setdefault("crypto_bot.utils", utils_pkg)
pkg_root.volatility_filter = types.ModuleType("crypto_bot.volatility_filter")
pkg_root.volatility_filter.calc_atr = lambda *_a, **_k: 0.0
sys.modules.setdefault("crypto_bot.volatility_filter", pkg_root.volatility_filter)
sys.modules.setdefault("crypto_bot.regime", regime_pkg)

# Ensure the real solana_scanner module is loaded even if another test
# inserted a stub under this name.
real_spec = importlib.util.spec_from_file_location(
    "crypto_bot.utils.solana_scanner",
    pathlib.Path(__file__).resolve().parents[1]
    / "crypto_bot" / "utils" / "solana_scanner.py",
)
solana_scanner = importlib.util.module_from_spec(real_spec)
sys.modules["crypto_bot.utils.solana_scanner"] = solana_scanner
real_spec.loader.exec_module(solana_scanner)

sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
sys.modules.setdefault("ccxt.async_support", types.ModuleType("ccxt.async_support"))

spec = importlib.util.spec_from_file_location(
    "crypto_bot.solana.scanner",
    pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "solana" / "scanner.py",
)
scanner = importlib.util.module_from_spec(spec)
sys.modules["crypto_bot.solana.scanner"] = scanner
spec.loader.exec_module(scanner)


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
    def __init__(self, data):
        self._data = data
        self.url = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, timeout=10):
        self.url = url
        return DummyResp(self._data)


def test_get_solana_new_tokens_returns_unique(monkeypatch):
    from crypto_bot.solana.watcher import NewPoolEvent

    events = [
        NewPoolEvent("P1", "A", "C1", 50.0, 3),
        NewPoolEvent("P2", "B", "C1", 60.0, 4),
    ]

    async def watch_stub(self):
        for evt in events:
            yield evt

    monkeypatch.setattr(scanner.PoolWatcher, "watch", watch_stub)

    cfg = {
        "url": "http://x",
        "max_tokens_per_scan": 5,
        "timeout_seconds": 1,
        "min_liquidity": 0,
        "min_tx_count": 0,
    }
    tokens = asyncio.run(scanner.get_solana_new_tokens(cfg))
    assert tokens == ["A", "B"]


async def _scan_once(cfg, queue):
    from crypto_bot.utils import market_loader
    from crypto_bot.regime import regime_classifier

    tokens = await scanner.get_solana_new_tokens(cfg)
    allowed: list[str] = []
    if tokens:
        for sym in tokens:
            try:
                df = await market_loader.fetch_ohlcv_for_token(sym)
                if df is None:
                    continue
                regime, _ = await regime_classifier.classify_regime_cached(
                    sym, "1m", df
                )
                regime = regime.split("_")[-1]
                if regime in {"volatile", "breakout"}:
                    allowed.append(sym)
            except Exception:
                continue
        async with scanner.asyncio.Lock():
            for sym in reversed(allowed):
                queue.appendleft(sym)


def test_solana_scan_loop_enqueues_filtered(monkeypatch):
    async def fake_get(cfg):
        return ["A/USDC", "B/USDC"]

    async def fake_fetch(sym, *a, **k):
        import pandas as pd

        return pd.DataFrame(
            {"timestamp": [0], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}
        )

    async def fake_classify(sym, tf, df):
        return ({"A/USDC": "volatile", "B/USDC": "bear"}[sym], {})

    monkeypatch.setattr(scanner, "get_solana_new_tokens", fake_get)
    from crypto_bot.utils import market_loader
    monkeypatch.setattr(market_loader, "fetch_ohlcv_for_token", fake_fetch)
    from crypto_bot.regime import regime_classifier
    monkeypatch.setattr(regime_classifier, "classify_regime_cached", fake_classify)

    queue = deque()
    asyncio.run(_scan_once({}, queue))
    assert list(queue) == ["A/USDC"]
