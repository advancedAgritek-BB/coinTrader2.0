import asyncio
import types

import pytest

from crypto_bot import solana_trading


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


async def fake_price(_in, _out):
    return 0.65


def test_monitor_profit(monkeypatch):
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


def test_sniper_trade(monkeypatch):
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


