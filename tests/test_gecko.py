import asyncio
from types import SimpleNamespace

import aiohttp
from crypto_bot.utils import gecko


class DummyResponse:
    def __init__(self, status, data=None, url="https://x.com"):
        self.status = status
        self._data = data or {}
        self.headers = {}
        self.request_info = SimpleNamespace(real_url=url)
        self.history = ()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def raise_for_status(self):
        if self.status >= 400 and self.status != 429:
            raise aiohttp.ClientResponseError(
                self.request_info, self.history, status=self.status, message="", headers=self.headers
            )

    async def json(self):
        return self._data


class DummySession:
    def __init__(self, responses):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, params=None, timeout=10):
        return self._responses.pop(0)


def test_gecko_request_retry_backoff(monkeypatch):
    responses = [
        DummyResponse(429),
        DummyResponse(429),
        DummyResponse(200, {"ok": True}),
    ]
    monkeypatch.setattr(gecko, "get_session", lambda: DummySession(responses))

    sleeps: list[float] = []

    async def fake_sleep(delay, *_, **__):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = asyncio.run(gecko.gecko_request("https://x.com"))
    assert result == {"ok": True}
    assert sleeps == [1, 2]
