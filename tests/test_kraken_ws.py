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
import crypto_bot.execution.kraken_ws as kraken_ws
from crypto_bot.execution.kraken_ws import KrakenWSClient

class DummyWSApp:
    def __init__(self, url, on_message=None, on_error=None, on_close=None, **kwargs):
        self.url = url
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.kwargs = kwargs
        self.run_called = False

    def run_forever(self):
        self.run_called = True

class DummyThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True
        if self.target:
            self.target()

def test_start_ws_passes_callbacks(monkeypatch):
    created = {}

    def fake_ws(url, on_message=None, on_error=None, on_close=None, **kwargs):
        ws = DummyWSApp(url, on_message, on_error, on_close, **kwargs)
        created['ws'] = ws
        return ws

    monkeypatch.setattr(kraken_ws, "WebSocketApp", fake_ws)
    monkeypatch.setattr(kraken_ws.threading, "Thread", DummyThread)

    client = KrakenWSClient()
    on_msg = lambda *a: None
    on_err = lambda *a: None
    on_close = lambda *a: None

    ws = client._start_ws("wss://test", on_message=on_msg, on_error=on_err, on_close=on_close)

    assert created['ws'].on_message is on_msg
    assert created['ws'].on_error is on_err
    assert created['ws'].on_close is on_close
    assert created['ws'].run_called


def test_default_callbacks_log(monkeypatch):
    logs = {"info": [], "error": []}

    class DummyLogger:
        def info(self, msg, *args):
            logs["info"].append(msg % args if args else msg)
        def error(self, msg, *args):
            logs["error"].append(msg % args if args else msg)

    def fake_ws(url, on_message=None, on_error=None, on_close=None, **kwargs):
        return DummyWSApp(url, on_message, on_error, on_close, **kwargs)

    monkeypatch.setattr(kraken_ws, "WebSocketApp", fake_ws)
    monkeypatch.setattr(kraken_ws.threading, "Thread", DummyThread)
    monkeypatch.setattr(kraken_ws, "logger", DummyLogger())

    client = KrakenWSClient()
    ws = client._start_ws("wss://test")

    ws.on_message(None, "hello")
    ws.on_error(None, "oops")

    assert any("hello" in m for m in logs["info"])
    assert any("oops" in m for m in logs["error"])
