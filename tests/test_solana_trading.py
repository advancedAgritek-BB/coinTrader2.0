import asyncio
import types
from typing import Dict

import pytest

# Target module
from crypto_bot import solana_trading


class DummyResp:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummySession:
    def __init__(self, quote, tx):
        self.quote = quote
        self.tx = tx
        self.quote_called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, params=None, timeout=10):
        self.quote_called = True
        return DummyResp({"data": [self.quote]})

    def post(self, url, json=None, timeout=10):
        return DummyResp({"swapTransaction": self.tx})


class DummyKeypair:
    public_key = "PK"

    @staticmethod
    def from_secret_key(b):
        return DummyKeypair()

    def sign(self, tx):
        tx.signed = True


class DummyTx:
    signed = False

    @staticmethod
    def deserialize(raw):
        return DummyTx()

    def sign(self, kp):
        self.signed = True

    def serialize(self):
        return b"signed"


class DummyClient:
    def __init__(self, *a, **k):
        self.tx = None
        self.kp = None

    async def send_transaction(self, tx, kp):
        self.tx = tx
        self.kp = kp
        return {"result": "SIG"}


async def dummy_get_wallet():
    return DummyKeypair(), DummyClient()


@pytest.mark.asyncio
async def test_fetch_price_respects_decimals(monkeypatch):
    captured_params: Dict[str, int] = {}

    class QuoteSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, params=None, timeout=10):
            captured_params.update(params or {})
            return DummyResp({"data": [{"inAmount": "1000000000", "outAmount": "2500000"}]})

    monkeypatch.setattr(
        solana_trading,
        "aiohttp",
        types.SimpleNamespace(ClientSession=lambda: QuoteSession()),
    )

    async def fake_decimals(mint):
        assert mint == "SOL"
        return 9

    monkeypatch.setattr(solana_trading, "get_decimals", fake_decimals)

    price = await solana_trading._fetch_price("SOL", "USDC")
    assert price == pytest.approx(2500000 / 1000000000)
    assert captured_params["amount"] == 1_000_000_000


@pytest.mark.asyncio
async def test_get_swap_quote(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    quote = {"inAmount": 100, "outAmount": 110}
    session = DummySession(quote, "tx")
    monkeypatch.setattr(solana_trading, "aiohttp", types.SimpleNamespace(ClientSession=lambda: session))

    res = await solana_trading.get_swap_quote("SOL", "USDC", 1)
    assert res == quote
    assert session.quote_called


@pytest.mark.asyncio
async def test_execute_swap(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    quote = {"inAmount": 100, "outAmount": 110}
    session = DummySession(quote, "RAW")
    monkeypatch.setattr(solana_trading, "aiohttp", types.SimpleNamespace(ClientSession=lambda: session))
    monkeypatch.setattr(solana_trading, "get_wallet", dummy_get_wallet)
    monkeypatch.setattr(solana_trading, "Transaction", DummyTx)
    monkeypatch.setattr(solana_trading, "Keypair", DummyKeypair)
    monkeypatch.setattr(solana_trading, "AsyncClient", DummyClient)

    sig = await solana_trading.execute_swap("SOL", "USDC", 1)
    assert sig == "SIG"


@pytest.mark.asyncio
async def test_sniper_trade(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    quote = {"inAmount": 100, "outAmount": 110}

    async def fake_quote(*a, **k):
        return quote

    async def fake_exec(*a, **k):
        return "SIG"

    class DummyRM:
        def check(self, *a, **k):
            return True

    rm = DummyRM()
    monkeypatch.setattr(solana_trading, "get_swap_quote", fake_quote)
    monkeypatch.setattr(solana_trading, "execute_swap", fake_exec)

    res = await solana_trading.sniper_trade("SOL", "USDC", 1, rm)
    assert res == "SIG"


class DummyClient:
    def __init__(self):
        self.closed = False

    async def get_confirmed_transaction(self, sig, encoding="jsonParsed"):
        return {
            "result": {
                "meta": {
                    "preTokenBalances": [
                        {"mint": "SOL", "uiTokenAmount": {"uiAmount": 1}},
                        {"mint": "MEME", "uiTokenAmount": {"uiAmount": 0}},
                    ],
                    "postTokenBalances": [
                        {"mint": "SOL", "uiTokenAmount": {"uiAmount": 0}},
                        {"mint": "MEME", "uiTokenAmount": {"uiAmount": 2}},
                    ],
                }
            }
        }

    async def close(self):
        self.closed = True


async def fake_price(_in, _out, max_retries=3):
    assert max_retries == 3
    return 0.65


def test_monitor_profit(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    dummy = DummyClient()
    monkeypatch.setattr(solana_trading, "AsyncClient", lambda url: dummy)
    monkeypatch.setattr(solana_trading, "_fetch_price", fake_price)

    profit = asyncio.run(solana_trading.monitor_profit("tx", threshold=0.2))
    assert profit == pytest.approx(0.6)
    assert dummy.closed


async def fake_execute(*a, **k):
    return {"tx_hash": "tx"}


async def fake_convert(*args, **kwargs):
    return {"tx_hash": "sell"}


async def fake_monitor(tx, threshold):
    assert tx == "tx"
    assert threshold == 0.3
    return 1.0


def test_sniper_trade_profit_conversion(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    monkeypatch.setattr(solana_trading, "execute_swap", fake_execute)
    monkeypatch.setattr(solana_trading, "auto_convert_funds", fake_convert)
    monkeypatch.setattr(solana_trading, "monitor_profit", fake_monitor)

    res = asyncio.run(
        solana_trading.sniper_trade(
            "wallet",
            "USDC",
            "MEME",
            5,
            dry_run=True,
            profit_threshold=0.3,
        )
    )
    assert res == {"tx_hash": "tx"}


