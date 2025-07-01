import ccxt
from crypto_bot.execution.cex_executor import place_stop_order


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


def test_place_stop_order_dry_run():
    order = place_stop_order(
        DummyExchange(),
        "BTC/USDT",
        "sell",
        1,
        9000,
        "token",
        "chat",
        dry_run=True,
    )
    assert order["dry_run"] is True
    assert order["stop"] == 9000


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
                    "volume": amount,
                    "ordertype": ordertype,
                },
            }
        return {"ws": True}


def test_execute_trade_rest_path(monkeypatch):
    monkeypatch.setattr(cex_executor, "send_message", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    ex = DummyExchange()
    order = cex_executor.execute_trade(
        ex, None, "BTC/USDT", "buy", 1.0, "t", "c", dry_run=False
    )
    assert order == {"exchange": True}
    assert ex.called


def test_execute_trade_ws_path(monkeypatch):
    monkeypatch.setattr(cex_executor, "send_message", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    ws = DummyWS()
    order = cex_executor.execute_trade(
        object(),
        ws,
        "BTC/USDT",
        "buy",
        1.0,
        "t",
        "c",
        dry_run=False,
        use_websocket=True,
    )
    assert order == {"ws": True}
    assert ws.called
    assert ws.msg["method"] == "add_order"
    assert ws.msg["params"]["symbol"] == "BTC/USDT"
    assert ws.msg["params"]["side"] == "buy"
    assert ws.msg["params"]["volume"] == 1.0


def test_execute_trade_ws_missing(monkeypatch):
    monkeypatch.setattr(cex_executor, "send_message", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    with pytest.raises(ValueError):
        cex_executor.execute_trade(
            object(),
            None,
            "BTC/USDT",
            "buy",
            1.0,
            "t",
            "c",
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
    monkeypatch.setattr(cex_executor, "send_message", lambda *a, **k: None)
    monkeypatch.setattr(cex_executor, "log_trade", lambda order: None)
    order = cex_executor.execute_trade(
        SlippageExchange(),
        None,
        "BTC/USDT",
        "buy",
        1.0,
        "t",
        "c",
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
