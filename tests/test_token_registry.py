import asyncio
import json
import sys


def test_load_token_mints(monkeypatch, tmp_path):
    data = {"tokens": [{"symbol": "SOL", "address": "So111"}]}

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

    holder = {}
    def client():
        session = DummySession(data)
        holder['s'] = session
        return session

    aiohttp_mod = type("M", (), {"ClientSession": client, "ClientError": Exception})

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
    monkeypatch.setattr(tr, "CACHE_FILE", tmp_path / "token_mints.json", raising=False)

    mapping = asyncio.run(tr.load_token_mints())
    assert mapping == {"SOL": "So111"}
    assert tr.TOKEN_MINTS == {}
    assert json.loads(tr.CACHE_FILE.read_text()) == {"SOL": "So111"}
    assert tr.CACHE_FILE.exists()
    assert holder['s'].url == tr.TOKEN_REGISTRY_URL


def test_load_token_mints_text_plain(monkeypatch, tmp_path):
    data = {"tokens": [{"symbol": "A", "address": "AAA"}]}

    class DummyResp:
        def __init__(self, d):
            self._d = d
            self.ct = None

        async def json(self, content_type=None):
            self.ct = content_type
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
            self.resp = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=10):
            self.resp = DummyResp(self.d)
            return self.resp

    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session, "ClientError": Exception})

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
    monkeypatch.setattr(tr, "CACHE_FILE", tmp_path / "token_mints.json", raising=False)

    mapping = asyncio.run(tr.load_token_mints())
    assert mapping == {"A": "AAA"}
    assert session.resp.ct is None


def test_load_token_mints_error(monkeypatch, tmp_path):
    class DummyErr(Exception):
        pass

    class FailingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=10):
            raise DummyErr("boom")

    file = tmp_path / "token_mints.json"
    file.write_text(json.dumps({"OLD": "M"}))

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
    monkeypatch.setattr(tr, "CACHE_FILE", file, raising=False)

    mapping = asyncio.run(tr.load_token_mints())
    assert mapping == {"OLD": "M"}
    assert tr.TOKEN_MINTS == {}


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

