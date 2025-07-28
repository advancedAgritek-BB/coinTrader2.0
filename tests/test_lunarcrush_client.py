import requests

from crypto_bot.lunarcrush_client import LunarCrushClient


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def test_get_sentiment(monkeypatch):
    def fake_get(self, url, params=None, timeout=5):
        assert params["symbol"] == "BTC"
        return DummyResponse({"data": [{"sentiment": 4}]})

    monkeypatch.setattr(requests.Session, "get", fake_get)
    client = LunarCrushClient()
    score = client.get_sentiment("BTC")
    assert score == 90.0
