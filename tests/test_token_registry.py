import asyncio
import json
import sys
import logging
import pytest


@pytest.fixture(autouse=True)
def _mock_listing_date(monkeypatch):
    """Override listing date fixture to avoid importing market_loader."""
    yield


class DummyErr(Exception):
    """Minimal exception class for simulating aiohttp errors in tests."""

    pass


def _load_module(monkeypatch, tmp_path):
    import importlib.util, pathlib, sys

    sys.modules.pop("crypto_bot.solana.helius_client", None)
    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot"
        / "utils"
        / "token_registry.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = mod
    spec.loader.exec_module(mod)
    # use temporary cache and clear preloaded tokens
    monkeypatch.setattr(mod, "CACHE_FILE", tmp_path / "token_mints.json", raising=False)
    mod.TOKEN_MINTS.clear()
    mod.TOKEN_DECIMALS.clear()
    return mod


def test_fetch_from_jupiter(monkeypatch, tmp_path):
    data = [{"symbol": "SOL", "address": "So111", "decimals": 9}]

    class DummyResp:
        def __init__(self, d):
            self._d = d
            self.status = 200

        async def json(self, content_type=None):
            return self._d

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class DummySession:
        def __init__(self, d):
            self.d = d
            self.url = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=10):
            self.url = url
            return DummyResp(self.d)

    session = DummySession(data)

    mod = _load_module(monkeypatch, tmp_path)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)

    result = asyncio.run(mod.fetch_from_jupiter())
    assert result == {"SOL": "So111"}
    assert session.url == mod.JUPITER_TOKEN_URL
    assert mod.TOKEN_DECIMALS["So111"] == 9


def test_fetch_from_helius(monkeypatch, tmp_path):
    data = [
        {
            "mint": "mmm",
            "onChainAccountInfo": {"decimals": 5, "supply": 10},
        }
    ]

    class DummyResp:
        def __init__(self, d):
            self._d = d
            self.status = 200

        async def json(self, content_type=None):
            return self._d

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class DummySession:
        def __init__(self, d):
            self.d = d
            self.url = None
            self.payload = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, url, json=None, timeout=10):
            self.url = url
            self.payload = json
            return DummyResp(self.d)

    monkeypatch.setenv("HELIUS_API_KEY", "KEY")
    monkeypatch.setattr(
        "crypto_bot.solana.helius_client.helius_available", lambda: True
    )
    session = DummySession(data)

    mod = _load_module(monkeypatch, tmp_path)
    mod.TOKEN_MINTS["AAA"] = "mmm"
    aiohttp_mod = type(
        "M", (), {"ClientSession": lambda: session, "ClientError": Exception}
    )
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(mod, "helius_available", lambda: True)

    mapping = asyncio.run(mod.fetch_from_helius(["AAA"]))
    assert mapping == {"AAA": "mmm"}
    assert session.url == "https://api.helius.xyz/v0/token-metadata?api-key=KEY"
    assert session.payload == {"mintAccounts": ["mmm"]}


def test_fetch_from_helius_full(monkeypatch, tmp_path):
    data = [
        {
            "mint": "mmm",
            "onChainAccountInfo": {"decimals": 5, "supply": 10},
        }
    ]

    class DummyResp:
        def __init__(self, d):
            self._d = d
            self.status = 200

        async def json(self, content_type=None):
            return self._d

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class DummySession:
        def __init__(self, d):
            self.d = d

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, url, json=None, timeout=10):
            return DummyResp(self.d)

    monkeypatch.setenv("HELIUS_API_KEY", "KEY")
    monkeypatch.setattr(
        "crypto_bot.solana.helius_client.helius_available", lambda: True
    )
    session = DummySession(data)

    mod = _load_module(monkeypatch, tmp_path)
    mod.TOKEN_MINTS["AAA"] = "mmm"
    aiohttp_mod = type(
        "M", (), {"ClientSession": lambda: session, "ClientError": Exception}
    )
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(mod, "helius_available", lambda: True)

    mapping = asyncio.run(mod.fetch_from_helius(["AAA"], full=True))
    assert mapping == {"AAA": {"mint": "mmm", "decimals": 5, "supply": 10}}


def test_fetch_from_helius_sol(monkeypatch, tmp_path):
    monkeypatch.setenv("HELIUS_API_KEY", "KEY")
    monkeypatch.setattr(
        "crypto_bot.solana.helius_client.helius_available", lambda: True
    )
    mod = _load_module(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "helius_available", lambda: True)

    mapping = asyncio.run(mod.fetch_from_helius(["SOL"], full=True))
    assert mapping == {"SOL": {"mint": mod.WSOL_MINT, "decimals": 9, "supply": None}}


def test_fetch_from_helius_4xx(monkeypatch, tmp_path, caplog):
    class DummyResp:
        def __init__(self, status=401):
            self.status = status

        async def text(self):
            return "nope"

        async def json(self, content_type=None):
            return {}

        def raise_for_status(self):
            raise AssertionError("should not be called")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class DummySession:
        def __init__(self):
            self.url = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, url, json=None, timeout=10):
            self.url = url
            return DummyResp()

    monkeypatch.setenv("HELIUS_API_KEY", "KEY")
    monkeypatch.setattr(
        "crypto_bot.solana.helius_client.helius_available", lambda: True
    )
    session = DummySession()
    mod = _load_module(monkeypatch, tmp_path)
    mod.TOKEN_MINTS["AAA"] = "mmm"
    aiohttp_mod = type(
        "M", (), {"ClientSession": lambda: session, "ClientError": Exception}
    )
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(mod, "helius_available", lambda: True)
    caplog.set_level(logging.WARNING)

    mapping = asyncio.run(mod.fetch_from_helius(["AAA"]))
    assert mapping == {}
    assert "Helius lookup failed for AAA [401]" in caplog.text


def test_fetch_from_helius_no_api_key(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.delenv("HELIUS_API_KEY", raising=False)
    monkeypatch.delenv("HELIUS_KEY", raising=False)
    monkeypatch.setattr("crypto_bot.solana.helius_client.HELIUS_API_KEY", "")
    monkeypatch.setattr(
        "crypto_bot.solana.helius_client.helius_available", lambda: False
    )
    mod = _load_module(monkeypatch, tmp_path)

    def raising_session(*_a, **_k):
        raise AssertionError("session should not be created")

    aiohttp_mod = type(
        "M", (), {"ClientSession": raising_session, "ClientError": Exception}
    )
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)

    mapping = asyncio.run(mod.fetch_from_helius(["AAA"]))
    assert mapping == {"AAA": "metadata_unknown"}


def test_periodic_mint_sanity_check(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)
    mod.MANUAL_OVERRIDES.clear()
    mod.MANUAL_OVERRIDES.update({"AAA": "old"})
    mod.TOKEN_MINTS.update(mod.MANUAL_OVERRIDES)

    async def fake_fetch(_symbols, *, full=False):
        return {"AAA": {"mint": "new", "decimals": 9, "supply": 100}}

    monkeypatch.setattr(mod, "fetch_from_helius", fake_fetch)

    called = []

    def fake_write():
        called.append(True)

    monkeypatch.setattr(mod, "_write_cache", fake_write)

    async def fast_sleep(_):
        raise asyncio.CancelledError()

    monkeypatch.setattr(mod.asyncio, "sleep", fast_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mod.periodic_mint_sanity_check(interval_hours=0))

    assert mod.TOKEN_MINTS["AAA"] == "new"
    assert called


def test_periodic_mint_sanity_check_sol(monkeypatch, tmp_path, caplog):
    mod = _load_module(monkeypatch, tmp_path)
    mod.MANUAL_OVERRIDES.clear()
    mod.MANUAL_OVERRIDES.update({"SOL": mod.WSOL_MINT})
    mod.TOKEN_MINTS.update(mod.MANUAL_OVERRIDES)

    async def fake_fetch(_symbols, *, full=False):
        return {"SOL": {"mint": "", "decimals": 9, "supply": None}}

    monkeypatch.setattr(mod, "fetch_from_helius", fake_fetch)
    monkeypatch.setattr(mod, "_write_cache", lambda: None)

    async def fast_sleep(_):
        raise asyncio.CancelledError()

    monkeypatch.setattr(mod.asyncio, "sleep", fast_sleep)
    caplog.set_level(logging.INFO)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mod.periodic_mint_sanity_check(interval_hours=0))

    assert any(
        r.getMessage() == "SOL: using WSOL mint; skipping supply check." and r.levelno == logging.INFO
        for r in caplog.records
    )
    assert mod.TOKEN_MINTS["SOL"] == mod.WSOL_MINT


def test_periodic_mint_sanity_check_missing_logged_once(monkeypatch, tmp_path, caplog):
    mod = _load_module(monkeypatch, tmp_path)
    mod.MANUAL_OVERRIDES.clear()
    mod.MANUAL_OVERRIDES.update({"AAA": "mint"})
    mod.TOKEN_MINTS.update(mod.MANUAL_OVERRIDES)

    async def fake_fetch(_symbols, *, full=False):
        return {}

    monkeypatch.setattr(mod, "fetch_from_helius", fake_fetch)
    monkeypatch.setattr(mod, "_write_cache", lambda: None)

    calls = {"count": 0}

    async def fast_sleep(_):
        calls["count"] += 1
        if calls["count"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(mod.asyncio, "sleep", fast_sleep)
    caplog.set_level(logging.DEBUG)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mod.periodic_mint_sanity_check(interval_hours=0))

    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and r.getMessage() == "No metadata for AAA"
    ]
    debugs = [
        r for r in caplog.records if r.levelno == logging.DEBUG and r.getMessage() == "No metadata for AAA"
    ]
    assert len(warnings) == 1
    assert len(debugs) == 1


def test_load_token_mints(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    async def fake_jup():
        return {"SOL": "So111"}

    monkeypatch.setattr(mod, "fetch_from_jupiter", fake_jup)

    mapping = asyncio.run(mod.load_token_mints(force_refresh=True))
    assert mapping == {"SOL": "So111"}
    assert json.loads(mod.CACHE_FILE.read_text()) == mapping
    assert mod.TOKEN_MINTS["SOL"] == "So111"


def test_load_token_mints_error(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    async def fail_jup():
        raise Exception("boom")

    monkeypatch.setattr(mod, "fetch_from_jupiter", fail_jup)

    class FailingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=10):
            raise Exception("boom")

    aiohttp_mod = type("M", (), {"ClientSession": lambda: FailingSession()})
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)

    async def fast_sleep(_):
        pass

    monkeypatch.setattr(mod.asyncio, "sleep", fast_sleep)

    aiohttp_mod = type(
        "M", (), {"ClientSession": lambda: FailingSession(), "ClientError": DummyErr}
    )

    import importlib.util, pathlib

    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot"
        / "utils"
        / "token_registry.py",
    )
    tr = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = tr
    spec.loader.exec_module(tr)
    tr.TOKEN_MINTS.clear()
    file = mod.CACHE_FILE
    monkeypatch.setattr(tr, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(tr, "CACHE_FILE", file, raising=False)
    file.write_text(json.dumps({"OLD": "M"}))

    mapping = asyncio.run(mod.load_token_mints(force_refresh=True))
    assert mapping == {"OLD": "M"}
    assert mod.TOKEN_MINTS["OLD"] == "M"


def test_load_token_mints_empty(monkeypatch, tmp_path, caplog):
    mod = _load_module(monkeypatch, tmp_path)

    async def fail_jup():
        raise Exception("boom")

    monkeypatch.setattr(mod, "fetch_from_jupiter", fail_jup)

    class FailingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=10):
            raise Exception("boom")

    aiohttp_mod = type("M", (), {"ClientSession": lambda: FailingSession()})
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)

    async def fast_sleep(_):
        pass

    monkeypatch.setattr(mod.asyncio, "sleep", fast_sleep)

    mod._LOADED = False
    caplog.set_level(logging.WARNING)

    mapping = asyncio.run(mod.load_token_mints(force_refresh=True))
    assert mapping == {}
    assert mod._LOADED is False
    assert not mod.CACHE_FILE.exists()
    assert "Token mint mapping is empty" in caplog.text


def test_load_token_mints_force_refresh_creates_dir(monkeypatch, tmp_path):
    data = {"tokens": [{"symbol": "Z", "address": "ZZZ"}]}

    class DummyResp:
        def __init__(self, d):
            self._d = d

        async def json(self, content_type=None):
            return self._d

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class DummySession:
        def __init__(self, d):
            self.d = d

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=10):
            return DummyResp(self.d)

    aiohttp_mod = type(
        "M", (), {"ClientSession": lambda: DummySession(data), "ClientError": Exception}
    )

    import importlib.util, pathlib, shutil

    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot"
        / "utils"
        / "token_registry.py",
    )
    tr = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = tr
    spec.loader.exec_module(tr)
    monkeypatch.setattr(tr, "aiohttp", aiohttp_mod)
    cache_file = tmp_path / "cache" / "token_mints.json"
    monkeypatch.setattr(tr, "CACHE_FILE", cache_file, raising=False)
    tr._LOADED = True

    if cache_file.parent.exists():
        shutil.rmtree(cache_file.parent)

    mapping = asyncio.run(tr.load_token_mints(force_refresh=True))
    assert mapping == {"Z": "ZZZ"}
    assert cache_file.exists()


def test_set_token_mints_updates_cache(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    mod.set_token_mints({"AAA": "mint"})
    assert json.loads(mod.CACHE_FILE.read_text()) == {"AAA": "mint"}

    mod.set_token_mints({"AAA": "mint", "BBB": "mint2"})
    assert json.loads(mod.CACHE_FILE.read_text()) == {"AAA": "mint", "BBB": "mint2"}


def test_get_mint_from_gecko(monkeypatch):
    data = {
        "data": [
            {
                "relationships": {
                    "base_token": {"data": {"id": "solana_M"}}
                }
            }
        ]
    }

    class DummyResp:
        def __init__(self, d):
            self._d = d

        async def json(self):
            return self._d

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class DummySession:
        def __init__(self, d):
            self.d = d
            self.url = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=10):
            self.url = url
            return DummyResp(self.d)

    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    urls: list[str] = []

    import importlib.util, pathlib

    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot"
        / "utils"
        / "token_registry.py",
    )
    tr = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = tr
    spec.loader.exec_module(tr)

    async def fake_req(url, params=None, retries=3):
        urls.append(url)
        return data

    monkeypatch.setattr(tr, "gecko_request", fake_req)

    mint = asyncio.run(tr.get_mint_from_gecko("BONK"))
    assert mint == "M"
    assert "query=BONK" in urls[0]


def test_get_mint_from_gecko_cached(monkeypatch):
    import importlib.util, pathlib

    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot"
        / "utils"
        / "token_registry.py",
    )
    tr = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = tr
    spec.loader.exec_module(tr)

    tr.TOKEN_MINTS["AAA"] = "mint"

    called = {"gecko": False, "helius": False}

    async def fake_req(url, params=None, retries=3):
        called["gecko"] = True
        return {}

    async def fake_hel(symbols):
        called["helius"] = True
        return {}

    monkeypatch.setattr(tr, "gecko_request", fake_req)
    monkeypatch.setattr(tr, "fetch_from_helius", fake_hel)

    mint = asyncio.run(tr.get_mint_from_gecko("AAA"))
    assert mint == "mint"
    assert called["gecko"] is False
    assert called["helius"] is False


def test_get_mint_from_gecko_error(monkeypatch):

    import importlib.util, pathlib

    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot"
        / "utils"
        / "token_registry.py",
    )
    tr = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = tr
    spec.loader.exec_module(tr)

    async def fail_req(url, params=None, retries=3):
        raise DummyErr("boom")

    monkeypatch.setattr(tr, "gecko_request", fail_req)

    async def fake_hel(symbols):
        return {}

    monkeypatch.setattr(tr, "fetch_from_helius", fake_hel)

    mint = asyncio.run(tr.get_mint_from_gecko("AAA"))
    assert mint is None


def test_get_mint_from_gecko_helius_fallback(monkeypatch):
    """Helius fallback is used when Gecko lookups fail."""

    import importlib.util, pathlib

    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot"
        / "utils"
        / "token_registry.py",
    )
    tr = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = tr
    spec.loader.exec_module(tr)

    async def fail_req(url, params=None, retries=3):
        raise DummyErr("boom")

    async def fake_hel(symbols):
        assert symbols == ["AAA"]
        return {"AAA": "mint"}

    monkeypatch.setattr(tr, "gecko_request", fail_req)
    monkeypatch.setattr(tr, "fetch_from_helius", fake_hel)

    mint = asyncio.run(tr.get_mint_from_gecko("AAA"))
    assert mint == "mint"


def test_get_mint_from_gecko_empty_attrs(monkeypatch):
    """fetch_from_helius is used when Gecko returns item without attributes."""

    import importlib.util, pathlib

    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot"
        / "utils"
        / "token_registry.py",
    )
    tr = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = tr
    spec.loader.exec_module(tr)

    async def fake_req(url, params=None, retries=3):
        return {"data": [{}]}

    called = {}

    async def fake_hel(symbols):
        called["symbols"] = symbols
        return {"AAA": "mint"}

    monkeypatch.setattr(tr, "gecko_request", fake_req)
    monkeypatch.setattr(tr, "fetch_from_helius", fake_hel)

    mint = asyncio.run(tr.get_mint_from_gecko("AAA"))
    assert mint == "mint"
    assert called["symbols"] == ["AAA"]


def test_refresh_mints(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    calls = {"load": 0, "cache": 0}

    async def fake_load(*, force_refresh=False):
        calls["load"] += 1
        assert force_refresh is True
        mod.TOKEN_MINTS["SOL"] = "So111"
        return {"SOL": "So111"}

    monkeypatch.setattr(mod, "load_token_mints", fake_load)
    monkeypatch.setattr(
        mod, "_write_cache", lambda: calls.__setitem__("cache", calls["cache"] + 1)
    )
    monkeypatch.setattr(mod, "logger", type("L", (), {"info": lambda *a, **k: None}))

    asyncio.run(mod.refresh_mints())

    assert calls["load"] == 1
    assert calls["cache"] == 1
    assert "SOL" in mod.TOKEN_MINTS


def test_refresh_mints_propagates_failure(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    async def fake_load(*, force_refresh=False):
        assert force_refresh is True
        return {}

    monkeypatch.setattr(mod, "load_token_mints", fake_load)

    with pytest.raises(RuntimeError):
        asyncio.run(mod.refresh_mints())


def test_check_cex_arbitrage_triggers(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    class DummyEx:
        def __init__(self, price):
            self.price = price

        async def fetch_ticker(self, symbol):
            return {"last": self.price}

        async def close(self):
            pass

    import types

    fake_ccxt = types.SimpleNamespace(
        kraken=lambda: DummyEx(100),
        coinbase=lambda: DummyEx(101),
    )
    monkeypatch.setattr(mod, "ccxt", fake_ccxt)

    called = {}

    async def fake_exec(pair, target):
        called["pair"] = pair
        called["target"] = target

    monkeypatch.setattr(
        mod,
        "cross_chain_arb_bot",
        types.SimpleNamespace(execute_arbitrage=fake_exec),
    )

    asyncio.run(mod._check_cex_arbitrage("AAA"))

    assert called == {"pair": "AAA/USD", "target": "BTC"}


def test_check_cex_arbitrage_no_trigger(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    class DummyEx:
        def __init__(self, price):
            self.price = price

        async def fetch_ticker(self, symbol):
            return {"last": self.price}

        async def close(self):
            pass

    import types

    fake_ccxt = types.SimpleNamespace(
        kraken=lambda: DummyEx(100),
        coinbase=lambda: DummyEx(100.4),
    )
    monkeypatch.setattr(mod, "ccxt", fake_ccxt)

    called = {}

    async def fake_exec(pair, target):
        called["pair"] = pair
        called["target"] = target

    monkeypatch.setattr(
        mod,
        "cross_chain_arb_bot",
        types.SimpleNamespace(execute_arbitrage=fake_exec),
    )

    asyncio.run(mod._check_cex_arbitrage("AAA"))

    assert called == {}
def test_monitor_pump_raydium(monkeypatch, tmp_path):
    from datetime import datetime

    mod = _load_module(monkeypatch, tmp_path)

    now = datetime.utcnow()
    pump_data = [
        {
            "symbol": "AAA",
            "mint": "mintAAA",
            "market_cap": 123,
            "created_at": now.isoformat(),
            "initial_buy": True,
            "twitter": "x",
        }
    ]
    ray_data = [
        {
            "baseSymbol": "BBB",
            "baseMint": "mintBBB",
            "liquidity": 60000,
            "created_at": now.isoformat(),
        }
    ]

    class DummyResp:
        def __init__(self, data):
            self._data = data

        async def json(self, content_type=None):
            return self._data

    class DummySession:
        def __init__(self):
            self.calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url, timeout=10):
            self.calls.append(url)
            if url == mod.PUMP_URL:
                return DummyResp(pump_data)
            return DummyResp(ray_data)

    session = DummySession()
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)

    calls: list[tuple] = []

    async def fake_fetch(*args):
        calls.append(args)

    monkeypatch.setattr(mod, "fetch_data_range_async", fake_fetch)

    runs: list[int] = []

    async def fake_run_ml():
        runs.append(1)

    monkeypatch.setattr(mod, "_run_ml_trainer", fake_run_ml)

    writes: list[int] = []

    def fake_write_cache():
        writes.append(1)

    monkeypatch.setattr(mod, "_write_cache", fake_write_cache)

    orig_sleep = asyncio.sleep

    async def fake_sleep(_):
        await orig_sleep(0)
        raise asyncio.CancelledError

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mod.monitor_pump_raydium())

    assert mod.TOKEN_MINTS["AAA"] == "mintAAA"
    assert mod.TOKEN_MINTS["BBB"] == "mintBBB"
    assert len(calls) == 2
    assert len(runs) == 2
    assert len(writes) == 2


def test_to_base_units(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)
    assert mod.to_base_units(1, 9) == 1_000_000_000
    assert mod.to_base_units(1.5, 6) == 1_500_000


def test_get_decimals_cache_and_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("HELIUS_KEY", "KEY")
    monkeypatch.setattr(
        "crypto_bot.solana.helius_client.helius_available", lambda: True
    )

    mod = _load_module(monkeypatch, tmp_path)
    mod.TOKEN_DECIMALS["So111"] = 9
    assert asyncio.run(mod.get_decimals("So111")) == 9

    class DummyHeliusClient:
        def __init__(self):
            pass

        def get_token_metadata(self, mint):  # pragma: no cover - simple
            return type("M", (), {"decimals": 6})()

        def close(self):  # pragma: no cover - simple
            pass

    monkeypatch.setattr(mod, "HeliusClient", DummyHeliusClient)
        def get(self, url, timeout=10):
            self.url = url
            return DummyResp()

    session = DummySession()
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(mod, "helius_available", lambda: True)
    monkeypatch.setenv("HELIUS_KEY", "KEY")

    dec = asyncio.run(mod.get_decimals("m2"))
    assert dec == 6
    assert mod.TOKEN_DECIMALS["m2"] == 6
