from crypto_bot.lunarcrush_client import LunarCrushClient


class DummyResponse:
    def json(self):
        return {"data": [{"sentiment": 0}]}

    def raise_for_status(self):
        pass


class DummySession:
    def __init__(self):
        self.last_params = None

    def get(self, url, params=None, timeout=5):
        self.last_params = params
        return DummyResponse()


def test_env_key_used(monkeypatch):
    session = DummySession()
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "envkey")
    client = LunarCrushClient(session=session)

    client.get_sentiment("BTC")

    assert session.last_params["key"] == "envkey"
