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
