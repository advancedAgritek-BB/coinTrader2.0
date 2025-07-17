import asyncio
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "solana_scanner",
    pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "utils" / "solana_scanner.py",
)
solana_scanner = importlib.util.module_from_spec(spec)
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
    cfg = {"raydium_api_key": "r", "pump_fun_api_key": "p", "max_tokens_per_scan": 2, "min_volume_usd": 0, "gecko_search": False}
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

    cfg = {
        "raydium_api_key": "r",
        "max_tokens_per_scan": 10,
        "min_volume_usd": 100,
        "gecko_search": True,
    }
    tokens = asyncio.run(solana_scanner.get_solana_new_tokens(cfg))
    assert tokens == ["A/USDC"]
