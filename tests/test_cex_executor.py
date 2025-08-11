import ccxt
import types
import asyncio
import os
from crypto_bot.execution.cex_executor import place_stop_order
from crypto_bot.utils import trade_logger
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.execution import executor as simple_executor


class DummyStopExchange:
    def create_order(self, symbol, type_, side, amount, params=None):
        return {
            "id": "1",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "type": type_,
            "params": params,
        } 


class DummyNotifier:
    def __init__(self):
        self.messages = []

    def notify(self, text: str):
        self.messages.append(text)
        return None


def test_place_stop_order_dry_run(monkeypatch):
    called = {}

    def fake_log(order, is_stop=False):
        called["flag"] = is_stop

    monkeypatch.setattr("crypto_bot.execution.cex_executor.log_trade", fake_log)
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)

    notifier = DummyNotifier()
    order = place_stop_order(
        DummyStopExchange(),
        "XBT/USDT",
        "sell",
        1,
        9000,
        TelegramNotifier("t", "c"),
        notifier=notifier,
        dry_run=True,
    )
    assert notifier.messages
    assert order["dry_run"] is True
    assert order["stop"] == 9000
    assert called.get("flag") is True


import pytest
from crypto_bot.execution import cex_executor


class DummyMarketExchange:
    def __init__(self):
        self.called = False

    def create_market_order(self, symbol, side, amount):
        self.called = True
        return {"exchange": True, "id": "1", "amount": amount}

    def fetch_order(self, order_id, symbol):
        return {"id": order_id, "status": "closed", "filled": float(self.called and 1)}


class DummyWS:
    def __init__(self):
        self.called = False
        self.msg = None

    def add_order(self, symbol, side=None, amount=None, ordertype="market"):
        self.called = True
        if isinstance(symbol, dict):
            self.msg = symbol
        else:
            self.msg = {
                "method": "add_order",
                "params": {
                    "symbol": symbol,
                    "side": side,
                    "order_qty": amount,
                    "order_type": ordertype,
                },
            }
        return {"ws": True, "id": "5", "amount": amount}


def test_execute_trade_rest_path(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    ex = DummyMarketExchange()
    notifier = DummyNotifier()
    order = cex_executor.execute_trade(
        ex,
        None,
        "XBT/USDT",
        "buy",
        1.0,
        TelegramNotifier("t", "c"),
        notifier=notifier,
        dry_run=False,
    )
    assert order.get("exchange") is True
    assert order.get("status") == "closed"
    assert ex.called


class LimitExchange:
    def __init__(self):
        self.limit_called = False
        self.params = None

    def fetch_ticker(self, symbol):
        return {"bid": 100, "ask": 102}

    def create_limit_order(self, symbol, side, amount, price, params=None):
        self.limit_called = True
        self.params = params
        return {"limit": True, "price": price, "params": params, "id": "2", "amount": amount}

    def create_market_order(self, symbol, side, amount):
        raise AssertionError("market order should not be used")

    def fetch_order(self, order_id, symbol):
        return {"id": order_id, "status": "closed", "filled": float(self.limit_called and 1)}


def test_execute_trade_uses_limit(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    ex = LimitExchange()
    notifier = DummyNotifier()
    order = cex_executor.execute_trade(
        ex,
        None,
        "XBT/USDT",
        "buy",
        1.0,
        TelegramNotifier("t", "c"),
        notifier=notifier,
        dry_run=False,
        config={"hidden_limit": True},
        score=0.9,
    )
    assert ex.limit_called
    assert order["params"]["postOnly"] is True
    assert order["params"].get("hidden") is True
    assert order["status"] == "closed"


def test_execute_trade_ws_path(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    ws = DummyWS()
    notifier = DummyNotifier()
    class PollEx:
        def fetch_order(self, order_id, symbol):
            return {"id": order_id, "status": "closed", "filled": 1.0}

    order = cex_executor.execute_trade(
        PollEx(),
        ws,
        "XBT/USDT",
        "buy",
        1.0,
        TelegramNotifier("t", "c"),
        notifier=notifier,
        dry_run=False,
        use_websocket=True,
    )
    assert order.get("ws") is True
    assert order.get("status") == "closed"
    assert ws.called
    assert ws.msg["method"] == "add_order"
    assert ws.msg["params"]["symbol"] == "XBT/USDT"
    assert ws.msg["params"]["side"] == "buy"
    assert ws.msg["params"]["order_qty"] == 1.0


def test_execute_trade_ws_missing(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    with pytest.raises(ValueError):
        cex_executor.execute_trade(
            object(),
            None,
            "XBT/USDT",
            "buy",
            1.0,
            TelegramNotifier("t", "c"),
            notifier=DummyNotifier(),
            dry_run=False,
            use_websocket=True,
        )


def test_execute_trade_calls_sync(monkeypatch):
    called = {}

    def fake_sync(exchange):
        called["sync"] = True
        return []

    monkeypatch.setattr(cex_executor, "sync_positions", fake_sync)
    monkeypatch.setattr(TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)

    ex = DummyMarketExchange()
    cex_executor.execute_trade(
        ex,
        None,
        "XBT/USDT",
        "buy",
        1.0,
        TelegramNotifier("t", "c"),
        notifier=DummyNotifier(),
        dry_run=False,
    )

    assert called.get("sync") is True


def test_get_exchange_websocket(monkeypatch):
    config = {"exchange": "kraken", "use_websocket": True}

    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setenv("API_SECRET", "sec")
    monkeypatch.setenv("KRAKEN_API_TOKEN", "apitoken")

    monkeypatch.setattr(cex_executor, "get_ws_token", lambda *a, **k: "token")

    def fake_env_or_prompt(name, prompt):
        if name == "KRAKEN_WS_TOKEN":
            return None
        return os.getenv(name, "")

    monkeypatch.setattr(cex_executor, "env_or_prompt", fake_env_or_prompt)

    created = {}

    class DummyWSClient:
        def __init__(
            self, api_key=None, api_secret=None, ws_token=None, api_token=None
        ):
            created["args"] = (api_key, api_secret, ws_token, api_token)

    monkeypatch.setattr(cex_executor, "KrakenWSClient", DummyWSClient)

    class DummyCCXT:
        def __init__(self, params):
            created["params"] = params
            self.options = {}

    monkeypatch.setattr(
        cex_executor.ccxt,
        "kraken",
        lambda params: DummyCCXT(params),
        raising=False,
    )
    exchange, ws = cex_executor.get_exchange(config)
    assert isinstance(ws, DummyWSClient)
    expected = ("key", "sec", "token", "apitoken")
    assert created["args"] == expected


def test_get_exchange_websocket_missing_creds(monkeypatch):
    config = {"exchange": "kraken", "use_websocket": True}
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_SECRET", raising=False)
    monkeypatch.setattr(
        "crypto_bot.execution.kraken_ws.keyring.get_password",
        lambda *a, **k: "dummy",
    )
    monkeypatch.setattr(
        "crypto_bot.execution.kraken_ws.ccxt",
        types.SimpleNamespace(kraken=lambda params: object()),
        raising=False,
    )

    class DummyCCXT2:
        def __init__(self, params):
            self.options = {}

    monkeypatch.setattr(cex_executor.ccxt, "kraken", lambda params: DummyCCXT2(params), raising=False)
    monkeypatch.setattr(cex_executor, "get_ws_token", lambda *a, **k: "token")
    monkeypatch.setattr(cex_executor, "env_or_prompt", lambda *a, **k: None)
    exchange, ws = cex_executor.get_exchange(config)
    assert isinstance(ws, cex_executor.KrakenWSClient)


class SlippageExchange:
    def fetch_order_book(self, symbol, limit=10):
        return {"bids": [[100, 10]], "asks": [[120, 0.5], [150, 0.5]]}

    def create_market_order(self, symbol, side, amount):
        raise AssertionError("should not execute")


def test_execute_trade_skips_on_slippage(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    notifier = DummyNotifier()
    order = cex_executor.execute_trade(
        SlippageExchange(),
        None,
        "XBT/USDT",
        "buy",
        1.0,
        TelegramNotifier("t", "c"),
        notifier=notifier,
        dry_run=False,
        config={"max_slippage_pct": 0.05, "liquidity_depth": 10},
    )
    assert order == {}


class LowBookExchange:
    def fetch_order_book(self, symbol, limit=10):
        return {"bids": [[100, 1]], "asks": [[101, 1]]}

    def create_market_order(self, symbol, side, amount):
        raise AssertionError("should not execute")


def test_execute_trade_skips_on_liquidity_usage(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    notifier = DummyNotifier()
    order = cex_executor.execute_trade(
        LowBookExchange(),
        None,
        "XBT/USDT",
        "buy",
        1.0,
        TelegramNotifier("t", "c"),
        notifier=notifier,
        dry_run=False,
        config={"max_liquidity_usage": 0.5, "liquidity_depth": 10},
    )
    assert order == {}


def test_ws_client_refreshes_expired_token(monkeypatch):
    from crypto_bot.execution.kraken_ws import KrakenWSClient
    from datetime import timedelta, timezone, datetime

    monkeypatch.setattr(
        "crypto_bot.execution.kraken_ws.keyring.get_password",
        lambda *a, **k: "dummy",
    )
    monkeypatch.setattr(
        "crypto_bot.execution.kraken_ws.ccxt",
        types.SimpleNamespace(kraken=lambda params: object()),
        raising=False,
    )

    ws = KrakenWSClient(ws_token="old")
    ws.token_created = datetime.now(timezone.utc) - timedelta(minutes=15)

    refreshed = {}

    def fake_get_token():
        refreshed["called"] = True
        ws.token = "new"
        ws.token_created = datetime.now(timezone.utc)
        return ws.token

    monkeypatch.setattr(ws, "_start_ws", lambda *a, **k: object())
    monkeypatch.setattr(ws, "get_token", fake_get_token)

    ws.connect_private()
    assert refreshed.get("called") is True


def test_execute_trade_dry_run_logs_price(tmp_path, monkeypatch):
    trades = tmp_path / "trades.csv"

    monkeypatch.setattr(trade_logger, "dotenv_values", lambda _: {})
    monkeypatch.setattr(trade_logger, "LOG_DIR", tmp_path)

    class DummyEx:
        def fetch_ticker(self, symbol):
            return {"last": 123}

    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)

    cex_executor.execute_trade(DummyEx(), None, "XBT/USDT", "buy", 1.0, TelegramNotifier("t", "c"), dry_run=True)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    notifier = DummyNotifier()
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)

    cex_executor.execute_trade(
        DummyEx(),
        None,
        "XBT/USDT",
        "buy",
        1.0,
        notifier=notifier,
        dry_run=True,
    )

    row = trades.read_text().strip()
    assert row
    assert float(row.split(",")[3]) > 0


def test_execute_trade_async_dry_run_logs_price(tmp_path, monkeypatch):
    trades = tmp_path / "trades.csv"

    monkeypatch.setattr(trade_logger, "dotenv_values", lambda _: {})
    monkeypatch.setattr(trade_logger, "LOG_DIR", tmp_path)

    class DummyEx:
        async def fetch_ticker(self, symbol):
            return {"last": 321}

    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)

    asyncio.run(
        cex_executor.execute_trade_async(
            DummyEx(), None, "XBT/USDT", "buy", 1.0, TelegramNotifier("t", "c"), dry_run=True
        )
    )
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    notifier = DummyNotifier()
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)

    asyncio.run(
        cex_executor.execute_trade_async(
            DummyEx(), None, "XBT/USDT", "buy", 1.0, notifier=notifier, dry_run=True
        )
    )

    row = trades.read_text().strip()
    assert row
    assert float(row.split(",")[3]) > 0


def test_execute_trade_async_retries_network_error(monkeypatch):
    calls = {"count": 0}
    delays = []

    class DummyEx:
        async def create_market_order(self, symbol, side, amount):
            calls["count"] += 1
            if calls["count"] == 1:
                raise cex_executor.ccxt.NetworkError("fail")
            return {"id": "1"}

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    NetworkError = type("NetworkError", (Exception,), {})
    monkeypatch.setattr(cex_executor.ccxt, "NetworkError", NetworkError, raising=False)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)

    order = asyncio.run(
        cex_executor.execute_trade_async(
            DummyEx(),
            None,
            "XBT/USDT",
            "buy",
            1.0,
            notifier=DummyNotifier(),
            dry_run=False,
            max_retries=2,
        )
    )

    assert order.get("id") == "1"
    assert delays


def test_execute_trade_no_message_when_disabled(monkeypatch):
    calls = {"count": 0}
    from crypto_bot.utils import telegram
    monkeypatch.setattr(
        telegram,
        "send_message_sync",
        lambda *a, **k: calls.__setitem__("count", calls["count"] + 1),
    )

    class DummyExchange:
        def create_market_order(self, symbol, side, amount):
            return {}

    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    cex_executor.execute_trade(
        DummyExchange(), None, "XBT/USDT", "buy", 1.0,
        notifier=TelegramNotifier(False, "t", "c"), dry_run=True
    )

    assert calls["count"] == 0


def test_execute_trade_retries_network_error(monkeypatch):
    class RetryEx:
        def __init__(self):
            self.calls = 0

        def fetch_ticker(self, symbol):
            return {"bid": 1, "ask": 1}

        def create_market_order(self, symbol, side, amount):
            self.calls += 1
            if self.calls == 1:
                raise ccxt.NetworkError("boom")
            return {"ok": True, "id": "3", "amount": amount}

        def fetch_order(self, order_id, symbol):
            return {"id": order_id, "status": "closed", "filled": 1.0}

    sleeps: list[float] = []

    monkeypatch.setattr("time.sleep", lambda x: sleeps.append(x))
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)

    ex = RetryEx()
    order = cex_executor.execute_trade(
        ex, None, "BTC/USD", "buy", 1, notifier=DummyNotifier(), dry_run=False
    )

    assert order.get("ok") is True
    assert order.get("status") == "closed"
    assert ex.calls == 2
    assert sleeps == [1.0]


def test_execute_trade_async_calls_sync(monkeypatch):
    called = {}

    async def fake_sync(exchange):
        called["sync"] = True
        return []

    class DummyEx:
        async def create_market_order(self, symbol, side, amount):
            return {"id": "a1"}

        async def fetch_order(self, oid, symbol):
            return {"id": oid, "status": "closed", "filled": 1.0}

    monkeypatch.setattr(cex_executor, "sync_positions_async", fake_sync)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)

    asyncio.run(
        cex_executor.execute_trade_async(
            DummyEx(),
            None,
            "XBT/USDT",
            "buy",
            1.0,
            notifier=DummyNotifier(),
            dry_run=False,
        )
    )

    assert called.get("sync") is True


def test_execute_trade_async_skips_on_slippage(monkeypatch):
    class AsyncSlippageEx:
        async def fetch_order_book(self, symbol, limit=10):
            return {"asks": [[101, 0.1], [120, 1.0]], "bids": [[100, 1.0]]}

        async def create_market_order(self, symbol, side, amount):
            raise AssertionError("should not execute")

    monkeypatch.setattr(TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)

    order = asyncio.run(
        cex_executor.execute_trade_async(
            AsyncSlippageEx(),
            None,
            "XBT/USDT",
            "buy",
            1.0,
            notifier=DummyNotifier(),
            dry_run=False,
            config={"max_slippage_pct": 0.05},
        )
    )

    assert order == {}


def test_execute_trade_async_insufficient_liquidity(monkeypatch):
    class LowLiquidityEx:
        async def fetch_order_book(self, symbol, limit=10):
            return {"asks": [[110, 0.4]], "bids": [[100, 0.4]]}

        async def create_market_order(self, symbol, side, amount):
            raise AssertionError("should not execute")

    monkeypatch.setattr(TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)

    order = asyncio.run(
        cex_executor.execute_trade_async(
            LowLiquidityEx(),
            None,
            "XBT/USDT",
            "buy",
            1.0,
            notifier=DummyNotifier(),
            dry_run=False,
            config={"liquidity_check": True, "liquidity_depth": 5},
        )
    )

    assert order == {}


def test_execute_trade_async_retries(monkeypatch):
    class RetryEx:
        def __init__(self):
            self.calls = 0

        async def fetch_ticker(self, symbol):
            return {"bid": 1, "ask": 1}

        async def create_market_order(self, symbol, side, amount):
            self.calls += 1
            if self.calls == 1:
                raise ccxt.NetworkError("boom")
            return {"ok": True, "id": "4", "amount": amount}

        async def fetch_order(self, order_id, symbol):
            return {"id": order_id, "status": "closed", "filled": 1.0}

    sleeps: list[float] = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    monkeypatch.setattr(cex_executor.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)

    ex = RetryEx()
    order = asyncio.run(
        cex_executor.execute_trade_async(
            ex, None, "BTC/USD", "buy", 1, notifier=DummyNotifier(), dry_run=False
        )
    )

    assert order.get("ok") is True
    assert order.get("status") == "closed"
    assert ex.calls == 2
    assert sleeps == [1.0]


def test_simple_executor_partial_fill(monkeypatch):
    monkeypatch.setattr(simple_executor, "ccxt", ccxt)
    class PartialEx:
        def __init__(self):
            self.orders = []

        def create_market_order(self, symbol, side, amount):
            oid = f"o{len(self.orders)}"
            self.orders.append((oid, amount))
            return {"id": oid, "symbol": symbol, "amount": amount}

        def fetch_order(self, oid, symbol):
            if oid == "o0":
                return {"id": oid, "status": "closed", "filled": 0.4, "amount": 1.0, "symbol": symbol}
            return {"id": oid, "status": "closed", "filled": 0.6, "amount": 0.6, "symbol": symbol}

    times = {"t": 0.0}

    def fake_sleep(sec):
        times["t"] += sec

    def fake_monotonic():
        return times["t"]

    monkeypatch.setattr(simple_executor.time, "sleep", fake_sleep)
    monkeypatch.setattr(simple_executor.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(simple_executor, "log_trade", lambda order: None)

    ex = PartialEx()
    notifier = DummyNotifier()
    order = simple_executor.execute_trade(ex, "XBT/USDT", "buy", 1.0, {}, notifier, dry_run=False, poll_timeout=10)

    assert order["id"] == "o1"
    assert ex.orders == [("o0", 1.0), ("o1", 0.6)]
    assert any("Partial fill" in m for m in notifier.messages)


def test_simple_executor_timeout(monkeypatch):
    monkeypatch.setattr(simple_executor, "ccxt", ccxt)
    class TimeoutEx:
        def create_market_order(self, symbol, side, amount):
            return {"id": "t1", "symbol": symbol, "amount": amount}

        def fetch_order(self, oid, symbol):
            return {"id": oid, "status": "open", "filled": 0.0, "amount": 1.0, "symbol": symbol}

    times = {"t": 0.0}

    def fake_sleep(sec):
        times["t"] += sec

    def fake_monotonic():
        return times["t"]

    monkeypatch.setattr(simple_executor.time, "sleep", fake_sleep)
    monkeypatch.setattr(simple_executor.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(simple_executor, "log_trade", lambda order: None)

    ex = TimeoutEx()
    notifier = DummyNotifier()
    with pytest.raises(TimeoutError):
        simple_executor.execute_trade(ex, "XBT/USDT", "buy", 1.0, {}, notifier, dry_run=False, poll_timeout=2)

    assert any("timed out" in m for m in notifier.messages)


def test_estimate_book_slippage():
    book = {
        "asks": [[1.0, 2], [1.1, 1]],
        "bids": [[0.9, 2], [0.8, 3]],
    }
    slip = cex_executor.estimate_book_slippage(book, "buy", 3)
    assert slip == pytest.approx(0.033333, rel=1e-3)


def test_estimate_book_slippage_async_sync_exchange():
    class DummySync:
        def __init__(self):
            self.called = False

        def fetch_order_book(self, symbol, limit=2):
            self.called = True
            return {
                "asks": [[1.0, 1], [1.1, 2]],
                "bids": [[0.9, 2], [0.8, 3]],
            }

    ex = DummySync()
    slip = asyncio.run(
        cex_executor.estimate_book_slippage_async(ex, "BTC/USD", "sell", 3, depth=2)
    )
    assert ex.called is True
    assert slip == pytest.approx(0.037037, rel=1e-3)


def test_estimate_book_slippage_async_async_exchange():
    class DummyAsync:
        async def fetch_order_book(self, symbol, limit=2):
            return {
                "asks": [[1.0, 2], [1.1, 1]],
                "bids": [[0.9, 2], [0.8, 3]],
            }

    ex = DummyAsync()
    slip = asyncio.run(
        cex_executor.estimate_book_slippage_async(ex, "BTC/USD", "buy", 3, depth=2)
    )
    assert slip == pytest.approx(0.033333, rel=1e-3)
