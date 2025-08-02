import asyncio
import sys
import types

import pytest

# Target module
from crypto_bot import solana_trading


@pytest.mark.asyncio
async def test_sniper_trade(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")

    async def fake_exec(*a, **k):
        return {"tx_hash": "SIG"}

    async def fake_monitor(*a, **k):
        return 0.0

    monkeypatch.setattr(solana_trading, "execute_swap", fake_exec)
    monkeypatch.setattr(solana_trading, "monitor_profit", fake_monitor)

    res = await solana_trading.sniper_trade("wallet", "SOL", "USDC", 1)
    assert res == {"tx_hash": "SIG"}


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
    calls = {"used": False}

    class DummyModel:
        def predict(self, *_):
            calls["used"] = True
            return [[0.1, 0.1, 0.8]]

    ml_mod = types.SimpleNamespace(load_model=lambda *a, **k: DummyModel())
    sys.modules.setdefault("coinTrader_Trainer", types.ModuleType("coinTrader_Trainer"))
    monkeypatch.setitem(sys.modules, "coinTrader_Trainer.ml_trainer", ml_mod)

    import importlib

    importlib.reload(solana_trading)

    dummy = DummyClient()
    monkeypatch.setattr(solana_trading, "AsyncClient", lambda url: dummy)
    monkeypatch.setattr(solana_trading, "_fetch_price", fake_price)

    profit = asyncio.run(solana_trading.monitor_profit("tx", threshold=0.2))
    assert profit == pytest.approx(0.6)
    assert dummy.closed
    assert calls["used"]


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


