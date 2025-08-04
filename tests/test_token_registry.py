import asyncio
import json
import sys
import logging


class DummyErr(Exception):
    """Minimal exception class for simulating aiohttp errors in tests."""

    pass


def _load_module(monkeypatch, tmp_path):
    import importlib.util, pathlib

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
    return mod


def test_fetch_from_jupiter(monkeypatch, tmp_path):
    data = [{"symbol": "SOL", "address": "So111"}]

    class DummyResp:
        def __init__(self, d):
            self._d = d
            self.status = 200
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
    monkeypatch.setattr(tr, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(tr, "CACHE_FILE", tmp_path / "token_mints.json", raising=False)

    mapping = asyncio.run(tr.load_token_mints())
    mapping = asyncio.run(mod.fetch_from_jupiter())
    assert mapping == {"SOL": "So111"}
    assert session.url == mod.JUPITER_TOKEN_URL


def test_fetch_from_helius(monkeypatch, tmp_path):
    data = [{"symbol": "AAA", "mint": "mmm"}]

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
    aiohttp_mod = type(
        "M", (), {"ClientSession": lambda: session, "ClientError": Exception}
    )
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)
    monkeypatch.setenv("HELIUS_KEY", "KEY")

    mapping = asyncio.run(mod.fetch_from_helius(["AAA"]))
    assert mapping == {"AAA": "mmm"}
    assert (
        session.url
        == "https://api.helius.xyz/v0/token-metadata?api-key=KEY&symbol=AAA"
    )


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

        def get(self, url, timeout=10):
            self.url = url
            return DummyResp()

    session = DummySession()
    mod = _load_module(monkeypatch, tmp_path)
    aiohttp_mod = type(
        "M", (), {"ClientSession": lambda: session, "ClientError": Exception}
    )
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)
    caplog.set_level(logging.ERROR)

    mapping = asyncio.run(mod.fetch_from_helius(["AAA"]))
    assert mapping == {}
    assert "Helius lookup failed" in caplog.text


def test_load_token_mints(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    async def fake_jup():
        return {"SOL": "So111"}

    monkeypatch.setattr(mod, "fetch_from_jupiter", fake_jup)

    mapping = asyncio.run(mod.load_token_mints(force_refresh=True))
    assert mapping == {"SOL": "So111"}
    assert json.loads(mod.CACHE_FILE.read_text()) == mapping
    assert mod.TOKEN_MINTS["SOL"] == "So111"


def test_load_token_mints_uses_cache(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    cache = {"AAA": "mint"}
    mod.CACHE_FILE.write_text(json.dumps(cache))

    calls = {"jup": 0}

    async def fake_jup():
        calls["jup"] += 1
        return {"SOL": "So111"}

    monkeypatch.setattr(mod, "fetch_from_jupiter", fake_jup)

    mapping = asyncio.run(mod.load_token_mints())
    assert mapping == {"AAA": "mint"}
    assert calls["jup"] == 0
    assert mod.TOKEN_MINTS["AAA"] == "mint"


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


def test_load_token_mints_retry(monkeypatch, tmp_path, caplog):
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

    caplog.set_level(logging.DEBUG)
    mapping = asyncio.run(mod.load_token_mints(force_refresh=True))
    assert mapping == {}
    assert mod._LOADED is False
    assert "will retry later" in caplog.text


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
