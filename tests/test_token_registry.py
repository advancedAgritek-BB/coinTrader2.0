import asyncio
import json


def test_load_token_mints(monkeypatch, tmp_path):
    data = {"tokens": [{"symbol": "SOL", "address": "So111"}]}

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

    holder = {}
    def client():
        session = DummySession(data)
        holder['s'] = session
        return session

    aiohttp_mod = type("M", (), {"ClientSession": client, "ClientError": Exception})

    import crypto_bot.utils.token_registry as tr
    monkeypatch.setattr(tr, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(tr, "CACHE_FILE", tmp_path / "token_mints.json", raising=False)

    mapping = asyncio.run(tr.load_token_mints())
    assert mapping == {"SOL": "So111"}
    assert tr.TOKEN_MINTS == {}
    assert json.loads(tr.CACHE_FILE.read_text()) == {"SOL": "So111"}
    assert tr.CACHE_FILE.exists()
    assert holder['s'].url == tr.TOKEN_REGISTRY_URL


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

    import crypto_bot.utils.token_registry as tr
    monkeypatch.setattr(tr, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(tr, "CACHE_FILE", file, raising=False)

    mapping = asyncio.run(tr.load_token_mints())
    assert mapping == {"OLD": "M"}
    assert tr.TOKEN_MINTS == {}

