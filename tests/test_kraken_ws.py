import json
from datetime import datetime, timedelta, timezone
import sys
import types

sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.stats", types.SimpleNamespace(pearsonr=lambda *a, **k: 0))

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
                "event_trigger": "bbo",
                "snapshot": False,
                "req_id": 1,
            },
        }
    )
    assert ws.sent == [expected]
    assert client._public_subs[-1] == expected

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


def test_parse_ohlc_message_object_format():
    msg = json.dumps(
        {
            "channel": "ohlc",
            "type": "update",
            "data": [
                {
                    "interval": 1,
                    "interval_begin": "2024-04-02T00:00:00Z",
                    "open": "100.0",
                    "high": "110.0",
                    "low": "90.0",
                    "close": "105.0",
                    "volume": "5.5",
                }
            ],
        }
    )

    result = kraken_ws.parse_ohlc_message(msg)
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
def test_subscribe_trades_snapshot_option(monkeypatch):
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
            "channel": "book",
            "type": "snapshot",
            "symbol": "ETH/USD",
            "data": {
                "bids": [["3000.1", "1.0", "B1"], ["3000.0", "2.0", "B2"]],
                "asks": [["3000.2", "1.5", "A1"]],
                "checksum": 111,
                "timestamp": 1712150000,
            },
        }
    )

    upd_msg = json.dumps(
        {
            "channel": "book",
            "type": "update",
            "symbol": "ETH/USD",
            "data": {
                "bids": [["new", "B3", "2999.5", "0.5", 1712150001]],
                "asks": [["delete", "A1", "3000.2", "0", 1712150002]],
                "checksum": 112,
                "timestamp": 1712150003,
            },
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
        "timestamp": datetime.fromtimestamp(1712150000, timezone.utc),
    }

    assert upd == {
        "symbol": "ETH/USD",
        "bids": [
            {
                "event": "new",
                "order_id": "B3",
                "limit_price": 2999.5,
                "order_qty": 0.5,
                "timestamp": datetime.fromtimestamp(1712150001, timezone.utc),
            }
        ],
        "asks": [
            {
                "event": "delete",
                "order_id": "A1",
                "limit_price": 3000.2,
                "order_qty": 0.0,
                "timestamp": datetime.fromtimestamp(1712150002, timezone.utc),
            }
        ],
        "checksum": 112,
        "timestamp": datetime.fromtimestamp(1712150003, timezone.utc),
    }


def test_level3_invalid_messages():
    assert kraken_ws.parse_level3_snapshot("not json") is None
    bad = json.dumps({"channel": "book", "type": "snapshot", "data": {"bids": []}})
    assert kraken_ws.parse_level3_snapshot(bad) is None
    bad_upd = json.dumps({"channel": "book", "type": "update", "data": {"bids": []}})
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
