from crypto_bot.execution.kraken_ws import KrakenWSClient, PUBLIC_URL, PRIVATE_URL

class DummyWS:
    def __init__(self):
        self.sent = []
        self.on_close = None

    def send(self, msg):
        self.sent.append(msg)


def test_reconnect_and_resubscribe(monkeypatch):
    client = KrakenWSClient()
    created = []

    def dummy_start_ws(url, conn_type):
        ws = DummyWS()
        created.append((url, conn_type))
        ws.on_close = lambda *_: client.on_close(conn_type)
        return ws

    monkeypatch.setattr(client, "_start_ws", dummy_start_ws)
    monkeypatch.setattr(client, "get_token", lambda: "token")

    client.token = "token"

    client.subscribe_ticker("BTC/USD")
    assert created == [(PUBLIC_URL, "public")]
    assert client.public_ws.sent == [client._public_subs[0]]

    old_ws = client.public_ws
    old_ws.on_close(None, None)

    assert created == [(PUBLIC_URL, "public"), (PUBLIC_URL, "public")]
    assert client.public_ws is not old_ws
    assert client.public_ws.sent == [client._public_subs[0]]

    client.subscribe_orders()
    assert created[-1] == (PRIVATE_URL, "private")
    assert client.private_ws.sent == [client._private_subs[0]]

    old_private = client.private_ws
    old_private.on_close(None, None)

    assert created[-1] == (PRIVATE_URL, "private")
    assert client.private_ws is not old_private
    assert client.private_ws.sent == [client._private_subs[0]]
