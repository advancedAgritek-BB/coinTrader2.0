import json
from datetime import datetime, timedelta, timezone
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

    def dummy_start_ws(url, conn_type=None, **_):
        ws = DummyWS()
        created.append((url, conn_type))
        ws.on_close = lambda *_: client.on_close(conn_type)
        return ws

    monkeypatch.setattr(client, "_start_ws", dummy_start_ws)
    monkeypatch.setattr(client, "get_token", lambda: "token")

    client.token = "token"

    client.subscribe_ticker("BTC/USD")
    assert created == [(PUBLIC_URL, "public")]
    expected = json.dumps(
        {"method": "subscribe", "params": {"channel": "ticker", "symbol": ["BTC/USD"]}}
    )
    assert client.public_ws.sent == [expected]
    assert client._public_subs[0] == expected

    old_ws = client.public_ws
    old_ws.on_close(None, None)

    assert created == [(PUBLIC_URL, "public"), (PUBLIC_URL, "public")]
    assert client.public_ws is not old_ws
    assert client.public_ws.sent == [expected]

    client.subscribe_orders()
    assert created[-1] == (PRIVATE_URL, "private")
    expected_private = json.dumps(
        {
            "method": "subscribe",
            "params": {"channel": "open_orders", "token": client.token},
        }
    )
    assert client.private_ws.sent == [expected_private]
    assert client._private_subs[0] == expected_private

    old_private = client.private_ws
    old_private.on_close(None, None)

    assert created[-1] == (PRIVATE_URL, "private")
    assert client.private_ws is not old_private
    assert client.private_ws.sent == [expected_private]
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

    called = {}

    def on_msg(*a):
        called['msg'] = True

    def on_err(*a):
        called['err'] = True

    def on_close_cb(*a):
        called['user_close'] = True

    def client_on_close(ct):
        called['client_close'] = ct

    monkeypatch.setattr(client, "on_close", client_on_close)

    ws = client._start_ws(
        "wss://test",
        conn_type="public",
        on_message=on_msg,
        on_error=on_err,
        on_close=on_close_cb,
    )

    assert created['ws'].on_message is on_msg
    assert created['ws'].on_error is on_err
    # invoke close handler
    created['ws'].on_close(None, 0, "bye")
    assert called.get('user_close') is True
    assert called.get('client_close') == "public"
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


def test_token_refresh_updates_private_subs(monkeypatch):
    client = KrakenWSClient()
    created = []

    def dummy_start_ws(url, conn_type=None, **_):
        ws = DummyWS()
        created.append((url, conn_type))
        ws.on_close = lambda *_: client.on_close(conn_type)
        return ws

    tokens = ["t1", "t2"]

    def fake_get_token():
        token = tokens.pop(0)
        client.token = token
        client.token_created = datetime.now(timezone.utc)
        return token

    monkeypatch.setattr(client, "_start_ws", dummy_start_ws)
    monkeypatch.setattr(client, "get_token", fake_get_token)

    # initial subscribe obtains first token
    client.subscribe_orders("BTC/USD")
    first_msg = json.dumps(
        {"method": "subscribe", "params": {"channel": "openOrders", "token": "t1"}}
    )
    assert client._private_subs[0] == first_msg

    # expire token and reconnect
    client.token_created -= timedelta(minutes=15)
    client.connect_private()
    second_msg = json.dumps(
        {"method": "subscribe", "params": {"channel": "openOrders", "token": "t2"}}
    )

    assert client._private_subs[0] == second_msg
    assert client.private_ws.sent[-1] == second_msg
def _setup_private_client(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)
    client.token = "token"
    return client, ws


def test_cancel_order(monkeypatch):
    client, ws = _setup_private_client(monkeypatch)
    msg = client.cancel_order("abc")
    expected = {"method": "cancel_order", "params": {"txid": "abc", "token": "token"}}
    assert msg == expected
    assert ws.sent == [json.dumps(expected)]


def test_cancel_all_orders(monkeypatch):
    client, ws = _setup_private_client(monkeypatch)
    msg = client.cancel_all_orders()
    expected = {"method": "cancel_all_orders", "params": {"token": "token"}}
    assert msg == expected
    assert ws.sent == [json.dumps(expected)]


def test_open_orders(monkeypatch):
    client, ws = _setup_private_client(monkeypatch)
    msg = client.open_orders()
    expected = {"method": "open_orders", "params": {"token": "token"}}
    assert msg == expected
    assert ws.sent == [json.dumps(expected)]


def test_parse_ohlc_message_extracts_volume():
    candle = [
        "1712106000",
        "30000.0",
        "30100.0",
        "29900.0",
        "30050.0",
        "30025.0",
        "12.34",
        42,
    ]
    msg = json.dumps([
        1,
        {"channel": "ohlc-1", "symbol": "XBT/USD"},
        candle,
        {"channel": "ohlc", "symbol": "XBT/USD"},
    ])

    result = kraken_ws.parse_ohlc_message(msg)
    assert result == [1712106000000, 30000.0, 30100.0, 29900.0, 30050.0, 12.34]
