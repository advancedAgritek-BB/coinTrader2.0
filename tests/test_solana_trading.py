import asyncio
import types

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
async def test_get_swap_quote(monkeypatch):
    quote = {"inAmount": 100, "outAmount": 110}
    session = DummySession(quote, "tx")
    monkeypatch.setattr(solana_trading, "aiohttp", types.SimpleNamespace(ClientSession=lambda: session))

    res = await solana_trading.get_swap_quote("SOL", "USDC", 1)
    assert res == quote
    assert session.quote_called


@pytest.mark.asyncio
async def test_execute_swap(monkeypatch):
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

