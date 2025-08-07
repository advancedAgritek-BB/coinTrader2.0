import json
from datetime import datetime, timedelta, timezone
import sys
import types
import asyncio

sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.stats", types.SimpleNamespace(pearsonr=lambda *a, **k: 0))

sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))
_ccxt_mod = types.ModuleType("ccxt")
_ccxt_pro = types.ModuleType("ccxt.pro")
_ccxt_mod.pro = _ccxt_pro
sys.modules.setdefault("ccxt", _ccxt_mod)
sys.modules.setdefault("ccxt.pro", _ccxt_pro)

from crypto_bot.execution.kraken_ws import KrakenWSClient, PUBLIC_URL, PRIVATE_URL
import pytest


@pytest.fixture(autouse=True)
def _dummy_creds(monkeypatch):
    monkeypatch.setenv("API_KEY", "dummy")
    monkeypatch.setenv("API_SECRET", "dummy")
    monkeypatch.setattr(
        "crypto_bot.execution.kraken_ws.keyring.get_password",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "crypto_bot.execution.kraken_ws.ccxt",
        types.SimpleNamespace(kraken=lambda params: object()),
        raising=False,
    )

class DummyWS:
    def __init__(self):
        self.sent = []
        self.on_close = None
        self.closed = False

    def send(self, msg):
        self.sent.append(msg)

    def close(self):
        self.closed = True


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


def test_reconnect_resubscribes_book(monkeypatch):
    client = KrakenWSClient()
    created = []

    def dummy_start_ws(url, conn_type=None, **_):
        ws = DummyWS()
        created.append((url, conn_type))
        ws.on_close = lambda *_: client.on_close(conn_type)
        return ws

    monkeypatch.setattr(client, "_start_ws", dummy_start_ws)

    client.subscribe_book("BTC/USD")
    sub_msg = json.dumps(
        {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": ["BTC/USD"],
                "depth": 10,
                "snapshot": True,
            },
        }
    )
    assert created == [(PUBLIC_URL, "public")]
    assert client.public_ws.sent == [sub_msg]

    old_ws = client.public_ws
    old_ws.on_close(None, None)

    assert created == [(PUBLIC_URL, "public"), (PUBLIC_URL, "public")]
    assert client.public_ws is not old_ws
    assert client.public_ws.sent == [sub_msg]


def test_reconnect_resubscribes_level3(monkeypatch):
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

    client.subscribe_level3("BTC/USD")
    sub_msg = json.dumps(
        {
            "method": "subscribe",
            "params": {
                "channel": "level3",
                "symbol": ["BTC/USD"],
                "depth": 10,
                "snapshot": True,
                "token": "token",
            },
        }
    )
    assert created == [(PRIVATE_URL, "private")]
    assert client.private_ws.sent == [sub_msg]

    old_ws = client.private_ws
    old_ws.on_close(None, None)

    assert created == [(PRIVATE_URL, "private"), (PRIVATE_URL, "private")]
    assert client.private_ws is not old_ws
    assert client.private_ws.sent == [sub_msg]
def test_subscribe_ticker_with_options(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_ticker(
        "BTC/USD", event_trigger="bbo", snapshot=False, req_id=1
    )

    expected = json.dumps(
        {
            "method": "subscribe",
            "params": {
                "channel": "ticker",
                "symbol": ["BTC/USD"],
                "eventTrigger": "bbo",
                "snapshot": False,
                "req_id": 1,
            },
        }
    )
    assert ws.sent == [expected]
    assert client._public_subs[-1] == expected


def test_unsubscribe_ticker_snapshot_option(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_ticker("ETH/USD", snapshot=False)
    expected_sub = json.dumps(
        {
            "method": "subscribe",
            "params": {"channel": "ticker", "symbol": ["ETH/USD"], "snapshot": False},
        }
    )
    assert ws.sent == [expected_sub]
    assert client._public_subs[-1] == expected_sub

    ws.sent.clear()
    client.unsubscribe_ticker("ETH/USD")
    expected_unsub = json.dumps(
        {
            "method": "unsubscribe",
            "params": {"channel": "ticker", "symbol": ["ETH/USD"]},
        }
    )
    assert ws.sent == [expected_unsub]
    assert client._public_subs == []

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

    def run_forever(self, *args, **kwargs):
        self.run_called = True
        self.run_args = args
        self.run_kwargs = kwargs

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


def test_subscribe_then_unsubscribe(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_ticker("XBT/USD")
    sub_msg = json.dumps(
        {"method": "subscribe", "params": {"channel": "ticker", "symbol": ["XBT/USD"]}}
    )
    assert ws.sent == [sub_msg]
    assert client._public_subs == [sub_msg]

    ws.sent.clear()
    client.unsubscribe_ticker("XBT/USD")
    unsub_msg = json.dumps(
        {
            "method": "unsubscribe",
            "params": {"channel": "ticker", "symbol": ["XBT/USD"]},
        }
    )
    assert ws.sent == [unsub_msg]
    assert client._public_subs == []
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


def test_edit_order_minimal(monkeypatch):
    client, ws = _setup_private_client(monkeypatch)
    msg = client.edit_order("OID", "BTC/USD")
    expected = {
        "method": "edit_order",
        "params": {"order_id": "OID", "symbol": "BTC/USD", "token": "token"},
    }
    assert msg == expected
    assert ws.sent == [json.dumps(expected)]


def test_edit_order_with_options(monkeypatch):
    client, ws = _setup_private_client(monkeypatch)
    msg = client.edit_order(
        "OID",
        "ETH/USD",
        order_qty=1.2,
        limit_price=5.0,
        deadline="2024-01-01T00:00:00Z",
    )
    expected = {
        "method": "edit_order",
        "params": {
            "order_id": "OID",
            "symbol": "ETH/USD",
            "token": "token",
            "order_qty": 1.2,
            "limit_price": 5.0,
            "deadline": "2024-01-01T00:00:00Z",
        },
    }
    assert msg == expected
    assert ws.sent[-1] == json.dumps(expected)
def test_amend_order(monkeypatch):
    client, ws = _setup_private_client(monkeypatch)
    msg = client.amend_order(
        order_id="OID123",
        order_qty=1.1,
        limit_price=30000.5,
        req_id=9,
    )
    expected = {
        "method": "amend_order",
        "params": {
            "token": "token",
            "order_id": "OID123",
            "order_qty": 1.1,
            "limit_price": 30000.5,
        },
        "req_id": 9,
    }
    assert msg == expected
    assert ws.sent == [json.dumps(expected)]


def test_subscribe_and_unsubscribe_orders(monkeypatch):
    client, ws = _setup_private_client(monkeypatch)

    client.subscribe_orders()
    expected_sub = json.dumps(
        {
            "method": "subscribe",
            "params": {"channel": "open_orders", "token": "token"},
        }
    )
    assert ws.sent == [expected_sub]
    assert client._private_subs == [expected_sub]

    ws.sent.clear()
    client.unsubscribe_orders()
    expected_unsub = json.dumps(
        {
            "method": "unsubscribe",
            "params": {"channel": "open_orders", "token": "token"},
        }
    )
    assert ws.sent == [expected_unsub]
    assert client._private_subs == []

    ws.sent.clear()
    client.subscribe_orders("ETH/USD")
    expected_sub = json.dumps(
        {
            "method": "subscribe",
            "params": {"channel": "openOrders", "token": "token"},
        }
    )
    assert ws.sent == [expected_sub]
    assert client._private_subs == [expected_sub]

    ws.sent.clear()
    client.unsubscribe_orders("ETH/USD")
    expected_unsub = json.dumps(
        {
            "method": "unsubscribe",
            "params": {"channel": "openOrders", "token": "token"},
        }
    )
    assert ws.sent == [expected_unsub]
    assert client._private_subs == []


def test_subscribe_and_unsubscribe_executions(monkeypatch):
    client, ws = _setup_private_client(monkeypatch)

    client.subscribe_executions()
    expected_sub = json.dumps(
        {
            "method": "subscribe",
            "params": {"channel": "executions", "token": "token"},
        }
    )
    assert ws.sent == [expected_sub]
    assert client._private_subs == [expected_sub]

    ws.sent.clear()
    client.unsubscribe_executions()
    expected_unsub = json.dumps(
        {
            "method": "unsubscribe",
            "params": {"channel": "executions", "token": "token"},
        }
    )
    assert ws.sent == [expected_unsub]
    assert client._private_subs == []


def test_subscribe_and_unsubscribe_book(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_book("ETH/USD", depth=5)
    sub_msg = json.dumps(
        {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": ["ETH/USD"],
                "depth": 5,
                "snapshot": True,
            },
        }
    )
    assert ws.sent == [sub_msg]
    assert client._public_subs == [sub_msg]

    ws.sent = []
    client.unsubscribe_book("ETH/USD", depth=5)
    unsub_msg = json.dumps(
        {
            "method": "unsubscribe",
            "params": {"channel": "book", "symbol": ["ETH/USD"], "depth": 5},
        }
    )
    assert ws.sent == [unsub_msg]
    assert client._public_subs == []


def test_parse_ohlc_message_extracts_volume(kraken_ohlc_update_msg):
    result = kraken_ws.parse_ohlc_message(kraken_ohlc_update_msg)
    assert result == [1712106000000, 30000.0, 30100.0, 29900.0, 30050.0, 12.34]


def test_parse_ohlc_message_object_format(kraken_ohlc_object_msg):
    result = kraken_ws.parse_ohlc_message(kraken_ohlc_object_msg)
    assert result == [1712016000000, 100.0, 110.0, 90.0, 105.0, 5.5]


def test_ping_sends_correct_json():
    client = KrakenWSClient()
    pub = DummyWS()
    priv = DummyWS()
    client.public_ws = pub
    client.private_ws = priv

    msg = client.ping(42)
    expected = {"method": "ping", "req_id": 42}
    assert msg == expected
    assert priv.sent == [json.dumps(expected)]
    assert not pub.sent


def test_handle_message_records_heartbeat(monkeypatch):
    client = KrakenWSClient()
    pub = DummyWS()
    priv = DummyWS()
    client.public_ws = pub
    client.private_ws = priv

    client._handle_message(pub, json.dumps({"channel": "heartbeat"}))
    assert client.last_public_heartbeat is not None
    assert client.is_alive("public")

    client.last_public_heartbeat -= timedelta(seconds=11)
    assert not client.is_alive("public")

    client._handle_message(priv, json.dumps({"channel": "heartbeat"}))
    assert client.last_private_heartbeat is not None
    assert client.is_alive("private")


def test_subscribe_instruments(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_instruments(snapshot=False)
    expected = json.dumps(
        {"method": "subscribe", "params": {"channel": "instrument", "snapshot": False}}
    )
    assert ws.sent == [expected]
    assert client._public_subs[0] == expected


def test_subscribe_and_unsubscribe_instruments(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_trades("LTC/USD", snapshot=False)
    expected_sub = json.dumps(
        {
            "method": "subscribe",
            "params": {"channel": "trade", "symbol": ["LTC/USD"], "snapshot": False},
        }
    )
    assert ws.sent == [expected_sub]
    assert client._public_subs[-1] == expected_sub

    ws.sent.clear()
    client.unsubscribe_trades("LTC/USD")
    expected_unsub = json.dumps(
        {
            "method": "unsubscribe",
            "params": {"channel": "trade", "symbol": ["LTC/USD"]},
        }
    )
    assert ws.sent == [expected_unsub]
    client.subscribe_instruments(snapshot=False)
    sub_msg = json.dumps(
        {"method": "subscribe", "params": {"channel": "instrument", "snapshot": False}}
    )
    assert ws.sent == [sub_msg]
    assert client._public_subs[0] == sub_msg

    ws.sent.clear()
    client.unsubscribe_instruments()
    unsub_msg = json.dumps(
        {"method": "unsubscribe", "params": {"channel": "instrument"}}
    )
    assert ws.sent == [unsub_msg]
    assert client._public_subs == []


def test_parse_instrument_message_returns_payload():
    msg = json.dumps(
        {
            "channel": "instrument",
            "type": "snapshot",
            "data": {
                "assets": [{"id": "XBT", "status": "enabled"}],
                "pairs": [{"symbol": "BTC/USD", "status": "online"}],
            },
        }
    )

    result = kraken_ws.parse_instrument_message(msg)
    assert result == {
        "assets": [{"id": "XBT", "status": "enabled"}],
        "pairs": [{"symbol": "BTC/USD", "status": "online"}],
    }


def test_subscribe_book_and_unsubscribe(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_book("ETH/USD", depth=25, snapshot=False)
    expected_sub = json.dumps(
        {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": ["ETH/USD"],
                "depth": 25,
                "snapshot": False,
            },
        }
    )
    assert ws.sent == [expected_sub]
    assert client._public_subs[0] == expected_sub

    ws.sent.clear()
    client.unsubscribe_book("ETH/USD")
    expected_unsub = json.dumps(
        {
            "method": "unsubscribe",
            "params": {"channel": "book", "symbol": ["ETH/USD"]},
        }
    )
    assert ws.sent == [expected_unsub]


def test_subscribe_and_unsubscribe_trades(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_trades("ETH/USD")
    expected_sub = json.dumps(
        {
            "method": "subscribe",
            "params": {"channel": "trade", "symbol": ["ETH/USD"]},
        }
    )
    assert ws.sent == [expected_sub]
    assert client._public_subs == [expected_sub]

    ws.sent.clear()
    client.unsubscribe_trades("ETH/USD")
    expected_unsub = json.dumps(
        {
            "method": "unsubscribe",
            "params": {"channel": "trade", "symbol": ["ETH/USD"]},
        }
    )
    assert ws.sent == [expected_unsub]
    assert client._public_subs == []


def test_subscribe_trades_snapshot_option(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_trades("BTC/USD", snapshot=False)
    expected = json.dumps(
        {
            "method": "subscribe",
            "params": {"channel": "trade", "symbol": ["BTC/USD"], "snapshot": False},
        }
    )
    assert ws.sent == [expected]
    assert client._public_subs[0] == expected


def test_subscribe_and_unsubscribe_level3(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)
    client.token = "token"

    client.subscribe_level3("ETH/USD", depth=5, snapshot=False)
    expected_sub = json.dumps(
        {
            "method": "subscribe",
            "params": {
                "channel": "level3",
                "symbol": ["ETH/USD"],
                "depth": 5,
                "snapshot": False,
                "token": "token",
            },
        }
    )
    assert ws.sent == [expected_sub]
    assert client._private_subs[0] == expected_sub

    ws.sent.clear()
    client.unsubscribe_level3("ETH/USD", depth=5)
    expected_unsub = json.dumps(
        {
            "method": "unsubscribe",
            "params": {
                "channel": "level3",
                "symbol": ["ETH/USD"],
                "depth": 5,
                "token": "token",
            },
        }
    )
    assert ws.sent == [expected_unsub]
    assert client._private_subs == []


def test_parse_book_message_snapshot_and_update():
    snap_msg = json.dumps(
        {
            "channel": "book",
            "type": "snapshot",
            "symbol": "BTC/USD",
            "data": {
                "bids": [["30000.1", "1.0"], ["29999.9", "2.0"]],
                "asks": [["30000.2", "1.5"], ["30001.0", "3.0"]],
            },
        }
    )

    upd_msg = json.dumps(
        {
            "channel": "book",
            "type": "update",
            "symbol": "BTC/USD",
            "data": {
                "bids": [["30000.1", "0.5"]],
                "asks": [["30000.2", "1.0"]],
            },
        }
    )

    snap = kraken_ws.parse_book_message(snap_msg)
    upd = kraken_ws.parse_book_message(upd_msg)

    assert snap == {
        "type": "snapshot",
        "bids": [[30000.1, 1.0], [29999.9, 2.0]],
        "asks": [[30000.2, 1.5], [30001.0, 3.0]],
    }

    assert upd == {
        "type": "update",
        "bids": [[30000.1, 0.5]],
        "asks": [[30000.2, 1.0]],
    }


def test_subscribe_and_unsubscribe_ohlc(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    client.subscribe_ohlc("BTC/USD", 1, snapshot=False, req_id=5)
    expected_sub = json.dumps(
        {
            "method": "subscribe",
            "params": {
                "channel": "ohlc",
                "symbol": ["BTC/USD"],
                "interval": 1,
                "snapshot": False,
                "req_id": 5,
            },
        }
    )
    assert ws.sent == [expected_sub]
    assert client._public_subs == [expected_sub]

    ws.sent.clear()
    client.unsubscribe_ohlc("BTC/USD", 1, req_id=6)
    expected_unsub = json.dumps(
        {
            "method": "unsubscribe",
            "params": {
                "channel": "ohlc",
                "symbol": ["BTC/USD"],
                "interval": 1,
                "req_id": 6,
            },
        }
    )
    assert ws.sent == [expected_unsub]
    assert client._public_subs == []
def test_parse_level3_snapshot_and_update():
    snap_msg = json.dumps(
        {
            "channel": "level3",
            "type": "snapshot",
            "data": [
                {
                    "symbol": "ETH/USD",
                    "bids": [
                        {
                            "order_id": "B1",
                            "limit_price": "3000.1",
                            "order_qty": "1.0",
                            "timestamp": "2024-01-01T00:00:00Z",
                        },
                        {
                            "order_id": "B2",
                            "limit_price": "3000.0",
                            "order_qty": "2.0",
                            "timestamp": "2024-01-01T00:00:01Z",
                        },
                    ],
                    "asks": [
                        {
                            "order_id": "A1",
                            "limit_price": "3000.2",
                            "order_qty": "1.5",
                            "timestamp": "2024-01-01T00:00:02Z",
                        }
                    ],
                    "checksum": 111,
                    "timestamp": "2024-01-01T00:00:03Z",
                }
            ],
        }
    )

    upd_msg = json.dumps(
        {
            "channel": "level3",
            "type": "update",
            "data": [
                {
                    "symbol": "ETH/USD",
                    "bids": [
                        {
                            "event": "new",
                            "order_id": "B3",
                            "limit_price": "2999.5",
                            "order_qty": "0.5",
                            "timestamp": "2024-01-01T00:00:04Z",
                        }
                    ],
                    "asks": [
                        {
                            "event": "delete",
                            "order_id": "A1",
                            "limit_price": "3000.2",
                            "order_qty": "0",
                            "timestamp": "2024-01-01T00:00:05Z",
                        }
                    ],
                    "checksum": 112,
                    "timestamp": "2024-01-01T00:00:06Z",
                }
            ],
        }
    )

    snap = kraken_ws.parse_level3_snapshot(snap_msg)
    upd = kraken_ws.parse_level3_update(upd_msg)

    assert snap == {
        "symbol": "ETH/USD",
        "bids": [
            {"order_id": "B1", "limit_price": 3000.1, "order_qty": 1.0},
            {"order_id": "B2", "limit_price": 3000.0, "order_qty": 2.0},
        ],
        "asks": [
            {"order_id": "A1", "limit_price": 3000.2, "order_qty": 1.5},
        ],
        "checksum": 111,
        "timestamp": datetime.fromisoformat("2024-01-01T00:00:03+00:00"),
    }

    assert upd == {
        "symbol": "ETH/USD",
        "bids": [
            {
                "event": "new",
                "order_id": "B3",
                "limit_price": 2999.5,
                "order_qty": 0.5,
                "timestamp": datetime.fromisoformat("2024-01-01T00:00:04+00:00"),
            }
        ],
        "asks": [
            {
                "event": "delete",
                "order_id": "A1",
                "limit_price": 3000.2,
                "order_qty": 0.0,
                "timestamp": datetime.fromisoformat("2024-01-01T00:00:05+00:00"),
            }
        ],
        "checksum": 112,
        "timestamp": datetime.fromisoformat("2024-01-01T00:00:06+00:00"),
    }


def test_level3_invalid_messages():
    assert kraken_ws.parse_level3_snapshot("not json") is None
    bad = json.dumps({"channel": "level3", "type": "snapshot", "data": []})
    assert kraken_ws.parse_level3_snapshot(bad) is None
    bad_upd = json.dumps({"channel": "level3", "type": "update", "data": []})
    assert kraken_ws.parse_level3_update(bad_upd) is None


def test_parse_trade_message_snapshot_update():
    snap_msg = json.dumps(
        {
            "channel": "trade",
            "type": "snapshot",
            "symbol": "BTC/USD",
            "data": [
                {
                    "price": "30000.1",
                    "qty": "0.5",
                    "side": "buy",
                    "ord_type": "limit",
                    "trade_id": "T1",
                    "timestamp": "2024-01-01T00:00:00Z",
                },
                {
                    "price": "30000.2",
                    "qty": "0.6",
                    "side": "sell",
                    "ord_type": "market",
                    "trade_id": "T2",
                    "timestamp": "2024-01-01T00:00:01Z",
                },
            ],
        }
    )

    upd_msg = json.dumps(
        {
            "channel": "trade",
            "type": "update",
            "symbol": "BTC/USD",
            "data": [
                {
                    "price": "30001.0",
                    "qty": "0.2",
                    "side": "buy",
                    "ord_type": "limit",
                    "trade_id": "T3",
                    "timestamp": "2024-01-01T00:00:02Z",
                }
            ],
        }
    )

    snap = kraken_ws.parse_trade_message(snap_msg)
    upd = kraken_ws.parse_trade_message(upd_msg)

    assert snap == [
        {
            "symbol": "BTC/USD",
            "side": "buy",
            "qty": 0.5,
            "price": 30000.1,
            "ord_type": "limit",
            "trade_id": "T1",
            "timestamp_ms": 1704067200000,
        },
        {
            "symbol": "BTC/USD",
            "side": "sell",
            "qty": 0.6,
            "price": 30000.2,
            "ord_type": "market",
            "trade_id": "T2",
            "timestamp_ms": 1704067201000,
        },
    ]

    assert upd == [
        {
            "symbol": "BTC/USD",
            "side": "buy",
            "qty": 0.2,
            "price": 30001.0,
            "ord_type": "limit",
            "trade_id": "T3",
            "timestamp_ms": 1704067202000,
        }
    ]


def test_close_shuts_down_connections(monkeypatch):
    client = KrakenWSClient()
    pub = DummyWS()
    priv = DummyWS()
    client.public_ws = pub
    client.private_ws = priv
    client._public_subs = ["x"]
    client._private_subs = ["y"]

    client.close()

    assert pub.closed
    assert priv.closed
    assert client.public_ws is None
    assert client.private_ws is None
    assert client._public_subs == []
    assert client._private_subs == []


def test_handle_message_enqueues_response():
    client = KrakenWSClient()
    ws = DummyWS()

    msg = {"method": "add_order", "result": {"status": "ok"}}
    client._handle_message(ws, json.dumps(msg))

    err = {"errorMessage": "boom"}
    client._handle_message(ws, json.dumps(err))

    assert list(client._responses)[0] == msg
    assert list(client._responses)[1] == err


def test_add_order_waits_for_response(monkeypatch):
    client, ws = _setup_private_client(monkeypatch)

    response = {"method": "add_order", "result": {"txid": "1"}}

    called = {}

    def fake_wait(method, timeout=5.0):
        called["method"] = method
        called["timeout"] = timeout
        return response

    monkeypatch.setattr(client, "_wait_for_response", fake_wait)

    result = client.add_order("BTC/USD", "buy", 1.0)

    expected = json.dumps(
        {
            "method": "add_order",
            "params": {
                "symbol": "BTC/USD",
                "side": "buy",
                "order_type": "market",
                "order_qty": "1.0",
                "token": "token",
            },
        }
    )
    assert ws.sent == [expected]
    assert result == response
    assert called["method"] == "add_order"


def test_watch_ohlcv_multi_symbol(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    async def run():
        async def feed():
            await asyncio.sleep(0.01)
            candle1 = [
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                1,
            ]
            msg1 = json.dumps(
                [1, {"channel": "ohlc-1", "symbol": "BTC/USD"}, candle1, {"channel": "ohlc", "symbol": "BTC/USD"}]
            )
            client._handle_message(ws, msg1)
            await asyncio.sleep(0.01)
            candle2 = [
                "2",
                "2",
                "2",
                "2",
                "2",
                "2",
                "2",
                1,
            ]
            msg2 = json.dumps(
                [1, {"channel": "ohlc-1", "symbol": "ETH/USD"}, candle2, {"channel": "ohlc", "symbol": "ETH/USD"}]
            )
            client._handle_message(ws, msg2)

        producer = asyncio.create_task(feed())
        res = await client.watch_ohlcv(["BTC/USD", "ETH/USD"], "1m", limit=1)
        await producer
        return res

    result = asyncio.run(run())
    sub = json.dumps(
        {
            "method": "subscribe",
            "params": {
                "channel": "ohlc",
                "symbol": ["BTC/USD", "ETH/USD"],
                "interval": 1,
                "snapshot": True,
            },
        }
    )
    assert ws.sent == [sub]
    assert set(result.keys()) == {"BTC/USD", "ETH/USD"}
    assert len(result["BTC/USD"]) == len(result["ETH/USD"]) == 1


def test_watch_ohlcv_multi_symbol_limit(monkeypatch):
    client = KrakenWSClient()
    ws = DummyWS()
    monkeypatch.setattr(client, "_start_ws", lambda *a, **k: ws)

    async def run():
        async def feed():
            for i in range(2):
                candle_b = [str(i), "1", "1", "1", "1", "1", "1", 1]
                msg_b = json.dumps(
                    [1, {"channel": "ohlc-1", "symbol": "BTC/USD"}, candle_b, {"channel": "ohlc", "symbol": "BTC/USD"}]
                )
                client._handle_message(ws, msg_b)
                candle_e = [str(i), "2", "2", "2", "2", "2", "2", 1]
                msg_e = json.dumps(
                    [1, {"channel": "ohlc-1", "symbol": "ETH/USD"}, candle_e, {"channel": "ohlc", "symbol": "ETH/USD"}]
                )
                client._handle_message(ws, msg_e)
                await asyncio.sleep(0.01)

        prod = asyncio.create_task(feed())
        res = await client.watch_ohlcv(["BTC/USD", "ETH/USD"], "1m", limit=2)
        await prod
        return res

    result = asyncio.run(run())
    assert len(result["BTC/USD"]) == 2
    assert len(result["ETH/USD"]) == 2
