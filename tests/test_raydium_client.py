import importlib.util
import pathlib
import asyncio
import sys
import types

path = pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "utils" / "raydium_client.py"
logger_mod = types.ModuleType("crypto_bot.utils.logger")
logger_mod.LOG_DIR = pathlib.Path(".")
class DummyLog:
    def info(self, *a, **k):
        pass
    def error(self, *a, **k):
        pass
logger_mod.setup_logger = lambda *a, **k: DummyLog()
sys.modules.setdefault("crypto_bot.utils.logger", logger_mod)

spec = importlib.util.spec_from_file_location("raydium_client", path)
raydium_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raydium_client)


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
    def __init__(self, data):
        self.data = data
        self.url = None
        self.payload = None
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    def get(self, url, params=None, timeout=10):
        self.url = (url, params)
        return DummyResp(self.data)
    def post(self, url, json=None, timeout=10):
        self.url = (url, json)
        tx = base64.b64encode(b"tx").decode()
        return DummyResp({"swapTransaction": tx})


class DummyClient:
    def __init__(self, url):
        self.url = url
        self.sent = None
        self.confirmed = False
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def send_raw_transaction(self, tx, opts=None):
        self.sent = tx
        class R:
            value = '1111111111111111111111111111111111111111111111111111111111111111'
        return R()
    async def confirm_transaction(self, sig, commitment=None, sleep_seconds=0.5, last_valid_block_height=None):
        self.confirmed = True
        return {}


import base64


def test_get_swap_quote(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    session = DummySession({"id": "1", "success": True})
    monkeypatch.setattr(raydium_client.aiohttp, "ClientSession", lambda: session)
    data = asyncio.run(raydium_client.get_swap_quote("A", "B", 1))
    assert data["id"] == "1"
    assert session.url[0] == raydium_client.QUOTE_URL


def test_execute_swap(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    session = DummySession({})
    monkeypatch.setattr(raydium_client.aiohttp, "ClientSession", lambda: session)
    monkeypatch.setattr(raydium_client, "get_wallet", lambda: None)
    class FakeVT:
        def __init__(self, message=None, keys=None):
            self.message = message
        @staticmethod
        def from_bytes(b):
            return FakeVT("m")
        def __bytes__(self):
            return b"bytes"
    monkeypatch.setattr(raydium_client, "VersionedTransaction", FakeVT)
    monkeypatch.setattr(raydium_client, "AsyncClient", lambda url: DummyClient(url))
    res = asyncio.run(
        raydium_client.execute_swap(
            "wallet",
            "in_acc",
            "out_acc",
            {"field": "val"},
            tx_version="V0",
        )
    )
    assert res["tx_hash"]


def test_sniper_trade(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    async def fake_quote(*a, **k):
        return {"data": {"quote": True}}
    monkeypatch.setattr(raydium_client, "get_swap_quote", fake_quote)
    async def fake_exec(*a, **k):
        return {"tx_hash": "sig"}
    monkeypatch.setattr(raydium_client, "execute_swap", fake_exec)
    called = {}
    async def fake_convert(*a, **k):
        called["done"] = True
        return {}
    fm_mod = types.ModuleType("crypto_bot.fund_manager")
    fm_mod.auto_convert_funds = fake_convert
    sys.modules["crypto_bot.fund_manager"] = fm_mod
    rm_mod = types.ModuleType("crypto_bot.risk.risk_manager")
    class DummyRM:
        def __init__(self, cfg):
            pass
        def position_size(self, *a, **k):
            return 1.0
    class DummyCfg:
        def __init__(self, **kw):
            pass
    rm_mod.RiskManager = DummyRM
    rm_mod.RiskConfig = DummyCfg
    sys.modules["crypto_bot.risk.risk_manager"] = rm_mod
    res = asyncio.run(raydium_client.sniper_trade("A", "B", 1, config={"wallet_address": "w"}))
    assert res["tx_hash"] == "sig"
    assert called.get("done")

