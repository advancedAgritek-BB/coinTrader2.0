import asyncio
import json
import sys


class DummyErr(Exception):
    """Minimal exception class for simulating aiohttp errors in tests."""
    pass


def _load_module(monkeypatch, tmp_path):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot" / "utils" / "token_registry.py",
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
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})

    mod = _load_module(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot" / "utils" / "token_registry.py",
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
            self.body = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, url, json=None, timeout=10):
            self.url = url
            self.body = json
            return DummyResp(self.d)

    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session})

    mod = _load_module(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "aiohttp", aiohttp_mod)
    monkeypatch.setenv("HELIUS_API_KEY", "KEY")
    monkeypatch.setattr(mod, "HELIUS_TOKEN_API", "http://helius")

    mapping = asyncio.run(mod.fetch_from_helius(["AAA"]))
    assert mapping == {"AAA": "mmm"}
    assert session.url == "http://helius?api-key=KEY"
    assert session.body == {"symbols": ["AAA"]}


def test_load_token_mints(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch, tmp_path)

    async def fake_jup():
        return {"SOL": "So111"}

    async def fake_hel(sym):
        return {"BONK": "Mb"}

    monkeypatch.setattr(mod, "fetch_from_jupiter", fake_jup)
    monkeypatch.setattr(mod, "fetch_from_helius", fake_hel)

    mapping = asyncio.run(mod.load_token_mints(force_refresh=True, unknown=["BONK"]))
    assert mapping == {"SOL": "So111", "BONK": "Mb"}
    assert json.loads(mod.CACHE_FILE.read_text()) == mapping
    assert mod.TOKEN_MINTS["SOL"] == "So111"
    assert mod.TOKEN_MINTS["BONK"] == "Mb"


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

    aiohttp_mod = type("M", (), {"ClientSession": lambda: FailingSession(), "ClientError": DummyErr})

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot" / "utils" / "token_registry.py",
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

    aiohttp_mod = type("M", (), {"ClientSession": lambda: DummySession(data), "ClientError": Exception})

    import importlib.util, pathlib, shutil
    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot" / "utils" / "token_registry.py",
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
    data = {"data": [{"attributes": {"address": "M"}}]}

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

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot" / "utils" / "token_registry.py",
    )
    tr = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = tr
    spec.loader.exec_module(tr)
    monkeypatch.setattr(tr, "aiohttp", aiohttp_mod)

    mint = asyncio.run(tr.get_mint_from_gecko("BONK"))
    assert mint == "M"
    assert "query=BONK" in session.url


def test_get_mint_from_gecko_error(monkeypatch):
    class DummyErr(Exception):
        pass

    class FailingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=10):
            raise DummyErr("boom")

    aiohttp_mod = type("M", (), {"ClientSession": lambda: FailingSession(), "ClientError": DummyErr})

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "crypto_bot.utils.token_registry",
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_bot" / "utils" / "token_registry.py",
    )
    tr = importlib.util.module_from_spec(spec)
    sys.modules["crypto_bot.utils.token_registry"] = tr
    spec.loader.exec_module(tr)
    monkeypatch.setattr(tr, "aiohttp", aiohttp_mod)

    mint = asyncio.run(tr.get_mint_from_gecko("AAA"))
    assert mint is None
