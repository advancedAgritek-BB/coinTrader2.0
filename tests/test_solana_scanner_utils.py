import asyncio
import importlib
import pathlib
import sys
import types
from datetime import datetime
import pathlib

pkg_root = types.ModuleType("crypto_bot")
utils_pkg = types.ModuleType("crypto_bot.utils")
pkg_root.utils = utils_pkg
pkg_root.__path__ = [str(pathlib.Path("crypto_bot"))]
pkg_root.volatility_filter = types.ModuleType("crypto_bot.volatility_filter")
pkg_root.volatility_filter.calc_atr = lambda *_a, **_k: 0.0
utils_pkg.__path__ = [str(pathlib.Path("crypto_bot/utils"))]
market_loader_mod = types.ModuleType("market_loader")
market_loader_mod.get_kraken_listing_date = lambda *_a, **_k: None
token_registry_mod = types.ModuleType("token_registry")
token_registry_mod.TOKEN_MINTS = {}
token_registry_mod.get_mint_from_gecko = lambda *_a, **_k: None
token_registry_mod.fetch_from_helius = lambda *_a, **_k: {}
utils_pkg.market_loader = market_loader_mod
utils_pkg.token_registry = token_registry_mod
sys.modules.setdefault("crypto_bot", pkg_root)
sys.modules.setdefault("crypto_bot.utils", utils_pkg)
sys.modules.setdefault("crypto_bot.volatility_filter", pkg_root.volatility_filter)
sys.modules.setdefault("crypto_bot.utils.market_loader", market_loader_mod)
sys.modules.setdefault("crypto_bot.utils.token_registry", token_registry_mod)
vf_mod = types.ModuleType("volatility_filter")
vf_mod.calc_atr = lambda *a, **k: 0
sys.modules.setdefault("crypto_bot.volatility_filter", vf_mod)

solana_scanner = importlib.import_module("crypto_bot.utils.solana_scanner")


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


def test_fetch_new_raydium_pools(monkeypatch, tmp_path):
    data = [
        {
            "base": {"address": "A"},
            "liquidity": 60000,
            "volume24h": 150000,
            "creationTimestamp": 1000,
            "creation_timestamp": 1000,
            "liquidity_locked": True,
        },
        {
            "base": {"address": "B"},
            "liquidity": 40000,
            "volume24h": 150000,
            "creationTimestamp": 2000,
            "creation_timestamp": 2000,
            "liquidity_locked": True,
        },
        {
            "base": {"address": "C"},
            "liquidity": 60000,
            "volume24h": 150000,
            "creationTimestamp": 3000,
            "creation_timestamp": 3000,
            "liquidity_locked": True,
        },
    ]
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(solana_scanner, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(solana_scanner, "RAYDIUM_TS_FILE", tmp_path / "ts.txt")
    solana_scanner._LAST_RAYDIUM_TS = 1500

    async def fake_extract(_data):
        res = []
        for item in session._data:
            mint = item["base"]["address"]
            liq = item["liquidity"]
            vol = item["volume24h"]
            ts = item["creationTimestamp"]
            if (
                not item["liquidity_locked"]
                or liq < 50_000
                or vol < 100_000
                or ts <= solana_scanner._LAST_RAYDIUM_TS
            ):
                continue
            res.append(mint)
        return res
    monkeypatch.setattr(solana_scanner, "_extract_tokens", fake_extract)

    tokens = asyncio.run(solana_scanner.fetch_new_raydium_pools(10))
    # B is skipped due to low liquidity; A is too old
    assert tokens == ["C"]
    assert session.url == f"{solana_scanner.RAYDIUM_URL}?limit=10"


def test_fetch_new_raydium_pools_skip_old(monkeypatch, tmp_path):
    data = [
        {
            "base": {"address": "A"},
            "liquidity": 60000,
            "volume24h": 150000,
            "creationTimestamp": 1000,
            "creation_timestamp": 1000,
            "liquidity_locked": True,
        }
    ]
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(solana_scanner, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(solana_scanner, "RAYDIUM_TS_FILE", tmp_path / "ts.txt")
    solana_scanner._LAST_RAYDIUM_TS = 1500

    tokens = asyncio.run(solana_scanner.fetch_new_raydium_pools(10))
    assert tokens == []
    assert solana_scanner._LAST_RAYDIUM_TS == 1500
def test_fetch_new_raydium_pools_filters(monkeypatch):
    data = {
        "data": [
            {
                "base": {"address": "A"},
                "liquidity": 150,
                "volume24h": 150,
                "creationTimestamp": 1,
                "creation_timestamp": 1,
                "liquidity_locked": True,
            },
            {
                "base": {"address": "B"},
                "liquidity": 50,
                "volume24h": 150,
                "creationTimestamp": 1,
                "creation_timestamp": 1,
                "liquidity_locked": True,
            },
            {
                "base": {"address": "C"},
                "liquidity": 150,
                "volume24h": 50,
                "creationTimestamp": 1,
                "creation_timestamp": 1,
                "liquidity_locked": True,
            },
            {
                "base": {"address": "D"},
                "liquidity": 150,
                "volume24h": 150,
                "creationTimestamp": 0,
                "creation_timestamp": 0,
                "liquidity_locked": True,
            },
            {
                "base": {"address": "E"},
                "liquidity": 150,
                "volume24h": 150,
                "creationTimestamp": 1,
                "creation_timestamp": 1,
                "liquidity_locked": False,
            },
        ]
    }
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(solana_scanner, "aiohttp", aiohttp_mod)

    solana_scanner._MIN_VOLUME_USD = 100

    async def fake_extract(_data):
        items = session._data["data"]
        res = []
        for item in items:
            mint = item["base"]["address"]
            liq = item["liquidity"]
            vol = item["volume24h"]
            ts = item["creationTimestamp"]
            if (
                not item["liquidity_locked"]
                or liq < 100
                or vol < 100
                or ts <= 0
            ):
                continue
            res.append(mint)
        return res

    monkeypatch.setattr(solana_scanner, "_extract_tokens", fake_extract)

    tokens = asyncio.run(solana_scanner.fetch_new_raydium_pools(5))
    assert tokens == ["A"]
    assert "B" not in tokens
    assert "C" not in tokens
    assert "D" not in tokens
    assert "E" not in tokens


def test_fetch_new_raydium_pools_helius(monkeypatch, tmp_path):
    data = {
        "data": [
            {
                "base": {"address": "A"},
                "liquidity": 150,
                "volume24h": 150,
                "creationTimestamp": 1000,
                "creation_timestamp": 1000,
                "liquidity_locked": True,
            }
        ]
    }
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(solana_scanner, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(solana_scanner, "RAYDIUM_TS_FILE", tmp_path / "ts.txt")
    solana_scanner._LAST_RAYDIUM_TS = 0

    async def fake_extract(_data):
        items = session._data["data"]
        res = []
        latest = solana_scanner._LAST_RAYDIUM_TS
        for item in items:
            mint = item["base"]["address"]
            liq = item["liquidity"]
            vol = item["volume24h"]
            ts = item["creationTimestamp"]
            if not item["liquidity_locked"] or liq < 100 or vol < 100 or ts <= solana_scanner._LAST_RAYDIUM_TS:
                continue
            res.append(mint)
            if ts > latest:
                latest = ts
        if res and latest > solana_scanner._LAST_RAYDIUM_TS:
            solana_scanner._LAST_RAYDIUM_TS = latest
        return res

    monkeypatch.setattr(solana_scanner, "_extract_tokens", fake_extract)

    tokens = asyncio.run(solana_scanner.fetch_new_raydium_pools(5))
    assert tokens == ["A"]
    assert solana_scanner._LAST_RAYDIUM_TS == 1000
    tokens2 = asyncio.run(solana_scanner.fetch_new_raydium_pools(5))
    assert tokens2 == []


def test_fetch_pump_fun_launches_filters(monkeypatch):
    now = datetime.utcnow().isoformat()
    data = [
        {
            "mint": "GOOD",
            "created_at": now,
            "initial_buy": True,
            "market_cap": 1000,
            "twitter": "x",
        },
        {
            "mint": "BAD1",
            "created_at": now,
            "initial_buy": False,
            "market_cap": 1000,
            "twitter": "x",
        },
        {
            "mint": "BAD2",
            "created_at": None,
            "initial_buy": True,
            "market_cap": 1000,
            "twitter": "x",
        },
        {
            "mint": "BAD3",
            "created_at": now,
            "initial_buy": True,
            "market_cap": 0,
            "twitter": "x",
        },
        {
            "mint": "BAD4",
            "created_at": now,
            "initial_buy": True,
            "market_cap": 1000,
            "twitter": None,
        },
    ]

    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(solana_scanner, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(solana_scanner, "TOKEN_MINTS", {})

    async def fake_gecko(base):
        return base

    monkeypatch.setattr(solana_scanner, "get_mint_from_gecko", fake_gecko)
    solana_scanner._MIN_VOLUME_USD = 0

    tokens = asyncio.run(solana_scanner.fetch_pump_fun_launches("k", 10))
    assert tokens == ["GOOD"]


def test_get_solana_new_tokens(monkeypatch):
    monkeypatch.setattr(
        solana_scanner,
        "fetch_new_raydium_pools",
        lambda limit: ["X", "Y"],
    )
    monkeypatch.setattr(
        solana_scanner,
        "fetch_pump_fun_launches",
        lambda limit: ["Y", "Z"],
    )
    async def fake_score(*_a, **_k):
        return 1.0

    class DummyEx:
        async def close(self):
            pass

    monkeypatch.setattr(solana_scanner.symbol_scoring, "score_symbol", fake_score)
    monkeypatch.setattr(solana_scanner.ccxt, "kraken", lambda *_a, **_k: DummyEx(), raising=False)

    cfg = {
        "pump_fun_api_key": "p",
        "raydium_api_key": "r",
        "max_tokens_per_scan": 2,
        "min_volume_usd": 0,
        "gecko_search": False,
    }
    tokens = asyncio.run(solana_scanner.get_solana_new_tokens(cfg))
    assert tokens == ["X/USDC", "Y/USDC"]


def test_search_geckoterminal_token(monkeypatch):
    data = {
        "data": [
            {
                "relationships": {
                    "base_token": {"data": {"id": "solana_M"}}
                },
                "attributes": {"volume_usd_h24": "123"},
            },
        ]
    }
    urls: list[str] = []
    async def fake_req(url, params=None, retries=3):
        urls.append(url)
        return data

    monkeypatch.setattr(
        solana_scanner,
        "gecko_request",
        fake_req,
    )

    res = asyncio.run(solana_scanner.search_geckoterminal_token("foo"))
    assert res == ("M", 123.0)
    assert "query=foo" in urls[0]


def test_get_solana_new_tokens_gecko_filter(monkeypatch):
    monkeypatch.setattr(
        solana_scanner,
        "fetch_new_raydium_pools",
        lambda *_a, **_k: ["A", "B"],
    )
    monkeypatch.setattr(
        solana_scanner,
        "fetch_pump_fun_launches",
        lambda limit: [],
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
        lambda limit: [],
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
        "max_tokens_per_scan": 10,
        "min_volume_usd": 0,
        "gecko_search": True,
        "min_symbol_score": 0.0,
    }
    tokens = asyncio.run(solana_scanner.get_solana_new_tokens(cfg))
    assert tokens == ["B/USDC", "A/USDC"]
