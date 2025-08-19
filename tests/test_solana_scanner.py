import asyncio
import importlib.util
import pathlib
import sys
import types
from collections import deque
import pandas as pd

# create minimal package structure
pkg_root = types.ModuleType("crypto_bot")
sol_pkg = types.ModuleType("crypto_bot.solana")
utils_pkg = types.ModuleType("crypto_bot.utils")
pkg_root.solana = sol_pkg
pkg_root.utils = utils_pkg
pkg_root.volatility_filter = types.ModuleType("crypto_bot.volatility_filter")
pkg_root.volatility_filter.calc_atr = lambda *_a, **_k: pd.Series([0.0])
pkg_root.strategy = types.ModuleType("crypto_bot.strategy")
pkg_root.strategy.cross_chain_arb_bot = types.ModuleType(
    "crypto_bot.strategy.cross_chain_arb_bot"
)
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

sys.modules.setdefault("crypto_bot", pkg_root)
sys.modules.setdefault("crypto_bot.solana", sol_pkg)
sys.modules.setdefault("crypto_bot.utils", utils_pkg)
sys.modules.setdefault("crypto_bot.volatility_filter", pkg_root.volatility_filter)
sys.modules.setdefault("crypto_bot.strategy", pkg_root.strategy)

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


def test_get_solana_new_tokens_filters_by_score(monkeypatch):
    data = {"tokens": [{"mint": "A"}, {"mint": "B"}]}
    session = DummySession(data)
    monkeypatch.setattr(scanner, "aiohttp", type("M", (), {"ClientSession": lambda: session}))

    async def fake_search(mint):
        if mint == "A":
            return ("A", 100.0)
        if mint == "B":
            return ("B", 200.0)
        return None

    monkeypatch.setattr(scanner, "search_geckoterminal_token", fake_search)

    class DummyEx:
        async def close(self):
            pass

    async def fake_score(_ex, sym, vol, *_a, **_k):
        return {"A/USDC": 0.6, "B/USDC": 0.4}[sym]

    monkeypatch.setattr(scanner.symbol_scoring, "score_symbol", fake_score)
    monkeypatch.setattr(scanner.kraken_utils, "get_client", lambda *_a, **_k: DummyEx())

    cfg = {
        "url": "http://x", 
        "limit": 5, 
        "min_symbol_score": 0.5,
        "exchange": "kraken",
    }
    tokens = asyncio.run(scanner.get_solana_new_tokens(cfg))
    assert tokens == ["A/USDC"]


async def _scan_once(cfg, queue):
    tokens = await scanner.get_solana_new_tokens(cfg)
    if tokens:
        async with scanner.asyncio.Lock():
            for sym in reversed(tokens):
                queue.appendleft(sym)


def test_solana_scan_loop_enqueues_filtered(monkeypatch):
    async def fake_get(cfg):
        return ["A/USDC"]

    monkeypatch.setattr(scanner, "get_solana_new_tokens", fake_get)
    queue = deque()
    asyncio.run(_scan_once({}, queue))
    assert list(queue) == ["A/USDC"]
