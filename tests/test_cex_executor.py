import ccxt
import asyncio
from crypto_bot.execution.cex_executor import place_stop_order
from crypto_bot.utils import trade_logger
from crypto_bot.utils.telegram import TelegramNotifier


class DummyExchange:
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
        DummyExchange(),
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


class DummyExchange:
    def __init__(self):
        self.called = False

    def create_market_order(self, symbol, side, amount):
        self.called = True
        return {"exchange": True}


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
        return {"ws": True}


def test_execute_trade_rest_path(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    ex = DummyExchange()
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
    assert order == {"exchange": True}
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
        return {"limit": True, "price": price, "params": params}

    def create_market_order(self, symbol, side, amount):
        raise AssertionError("market order should not be used")


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


def test_execute_trade_ws_path(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    ws = DummyWS()
    notifier = DummyNotifier()
    order = cex_executor.execute_trade(
        object(),
        ws,
        "XBT/USDT",
        "buy",
        1.0,
        TelegramNotifier("t", "c"),
        notifier=notifier,
        dry_run=False,
        use_websocket=True,
    )
    assert order == {"ws": True}
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


def test_get_exchange_websocket(monkeypatch):
    config = {"exchange": "kraken", "use_websocket": True}

    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setenv("API_SECRET", "sec")
    monkeypatch.setenv("KRAKEN_WS_TOKEN", "token")
    monkeypatch.setenv("KRAKEN_API_TOKEN", "apitoken")

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

    monkeypatch.setattr(cex_executor.ccxt, "kraken", lambda params: DummyCCXT(params))
    if getattr(cex_executor, "ccxtpro", None):
        monkeypatch.setattr(
            cex_executor.ccxtpro, "kraken", lambda params: DummyCCXT(params)
        )

    exchange, ws = cex_executor.get_exchange(config)
    assert isinstance(ws, DummyWSClient)
    assert created["args"] == ("key", "sec", "token", "apitoken")


def test_get_exchange_websocket_missing_creds(monkeypatch):
    config = {"exchange": "kraken", "use_websocket": True}
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_SECRET", raising=False)
    monkeypatch.delenv("KRAKEN_WS_TOKEN", raising=False)

    monkeypatch.setattr(cex_executor.ccxt, "kraken", lambda params: object())
    if getattr(cex_executor, "ccxtpro", None):
        monkeypatch.setattr(cex_executor.ccxtpro, "kraken", lambda params: object())

    exchange, ws = cex_executor.get_exchange(config)
    assert ws is None


class SlippageExchange:
    def fetch_ticker(self, symbol):
        return {"bid": 100, "ask": 110}

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
        config={"max_slippage_pct": 0.05},
    )
    assert order == {}


def test_ws_client_refreshes_expired_token(monkeypatch):
    from crypto_bot.execution.kraken_ws import KrakenWSClient
    from datetime import timedelta, timezone, datetime

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


def test_execute_trade_no_message_when_disabled(monkeypatch):
    calls = {"count": 0}
    monkeypatch.setattr(
        "crypto_bot.utils.telegram.send_message",
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
