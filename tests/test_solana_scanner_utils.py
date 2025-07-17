import asyncio
import importlib.util
import pathlib
import sys
import types

pkg_root = types.ModuleType("crypto_bot")
utils_pkg = types.ModuleType("crypto_bot.utils")
pkg_root.utils = utils_pkg
utils_pkg.__path__ = [str(pathlib.Path("crypto_bot/utils"))]
sys.modules.setdefault("crypto_bot", pkg_root)
sys.modules.setdefault("crypto_bot.utils", utils_pkg)
sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
sys.modules.setdefault("ccxt.async_support", types.ModuleType("ccxt.async_support"))

spec = importlib.util.spec_from_file_location(
    "crypto_bot.utils.solana_scanner",
    pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "utils" / "solana_scanner.py",
)
solana_scanner = importlib.util.module_from_spec(spec)
sys.modules["crypto_bot.utils.solana_scanner"] = solana_scanner
spec.loader.exec_module(solana_scanner)


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


def test_fetch_new_raydium_pools(monkeypatch):
    data = {
        "data": [
            {"tokenMint": "A", "volumeUsd": 150},
            {"tokenMint": "B", "volumeUsd": 50},
        ]
    }
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(solana_scanner, "aiohttp", aiohttp_mod)
    solana_scanner._MIN_VOLUME_USD = 100
    tokens = asyncio.run(solana_scanner.fetch_new_raydium_pools("k", 5))
    assert tokens == ["A"]
    assert "k" in session.url


def test_get_solana_new_tokens(monkeypatch):
    monkeypatch.setattr(
        solana_scanner,
        "fetch_new_raydium_pools",
        lambda key, limit: ["X", "Y"],
    )
    monkeypatch.setattr(
        solana_scanner,
        "fetch_pump_fun_launches",
        lambda key, limit: ["Y", "Z"],
    )
    async def fake_score(*_a, **_k):
        return 1.0

    class DummyEx:
        async def close(self):
            pass

    monkeypatch.setattr(solana_scanner.symbol_scoring, "score_symbol", fake_score)
    monkeypatch.setattr(solana_scanner.ccxt, "kraken", lambda *_a, **_k: DummyEx(), raising=False)

    cfg = {
        "raydium_api_key": "r",
        "pump_fun_api_key": "p",
        "max_tokens_per_scan": 2,
        "min_volume_usd": 0,
        "gecko_search": False,
    }
    tokens = asyncio.run(solana_scanner.get_solana_new_tokens(cfg))
    assert tokens == ["X/USDC", "Y/USDC"]


def test_search_geckoterminal_token(monkeypatch):
    data = {
        "data": [
            {"attributes": {"address": "M", "volume_usd_h24": "123"}},
        ]
    }
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(solana_scanner, "aiohttp", aiohttp_mod)

    res = asyncio.run(solana_scanner.search_geckoterminal_token("foo"))
    assert res == ("M", 123.0)
    assert "query=foo" in session.url


def test_get_solana_new_tokens_gecko_filter(monkeypatch):
    monkeypatch.setattr(
        solana_scanner,
        "fetch_new_raydium_pools",
        lambda *_a, **_k: ["A", "B"],
    )
    monkeypatch.setattr(
        solana_scanner,
        "fetch_pump_fun_launches",
        lambda *_a, **_k: [],
    )

    async def fake_search(q):
        return (q, 150.0) if q == "A" else (q, 50.0)

    monkeypatch.setattr(solana_scanner, "search_geckoterminal_token", fake_search)

    async def fake_score(_ex, sym, vol, *_a, **_k):
        return 1.0 if sym.startswith("A") else 0.4

    class DummyEx:
        async def close(self):
            pass

    monkeypatch.setattr(solana_scanner.symbol_scoring, "score_symbol", fake_score)
    monkeypatch.setattr(solana_scanner.ccxt, "kraken", lambda *_a, **_k: DummyEx(), raising=False)

    cfg = {
        "raydium_api_key": "r",
        "max_tokens_per_scan": 10,
        "min_volume_usd": 100,
        "gecko_search": True,
        "min_symbol_score": 0.5,
    }
    tokens = asyncio.run(solana_scanner.get_solana_new_tokens(cfg))
    assert tokens == ["A/USDC"]


def test_get_solana_new_tokens_scoring(monkeypatch):
    monkeypatch.setattr(
        solana_scanner,
        "fetch_new_raydium_pools",
        lambda *_a, **_k: ["A", "B"],
    )
    monkeypatch.setattr(
        solana_scanner,
        "fetch_pump_fun_launches",
        lambda *_a, **_k: [],
    )

    async def search(q):
        return (q, 100.0)

    monkeypatch.setattr(
        solana_scanner,
        "search_geckoterminal_token",
        search,
    )

    class DummyEx:
        async def close(self):
            pass

    async def fake_score(_ex, sym, vol, *_a, **_k):
        return {"A/USDC": 0.6, "B/USDC": 0.8}[sym]

    monkeypatch.setattr(solana_scanner.symbol_scoring, "score_symbol", fake_score)
    monkeypatch.setattr(solana_scanner.ccxt, "kraken", lambda *_a, **_k: DummyEx(), raising=False)

    cfg = {
        "raydium_api_key": "r",
        "max_tokens_per_scan": 10,
        "min_volume_usd": 0,
        "gecko_search": True,
        "min_symbol_score": 0.0,
    }
    tokens = asyncio.run(solana_scanner.get_solana_new_tokens(cfg))
    assert tokens == ["B/USDC", "A/USDC"]
