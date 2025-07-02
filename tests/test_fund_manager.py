import os
import json
import asyncio
from crypto_bot.fund_manager import (
    detect_non_trade_tokens,
    auto_convert_funds,
    check_wallet_balances,
)


def test_detect_non_trade_tokens():
    balances = {"BTC": 1, "ETH": 0.5, "USDC": 10, "SOL": 2}
    result = detect_non_trade_tokens(balances)
    assert set(result) == {"BTC", "ETH"}


def test_auto_convert_funds_returns_dict():
    tx = asyncio.run(auto_convert_funds("wallet", "BTC", "USDC", 1))
    assert tx["from"] == "BTC"
    assert tx["to"] == "USDC"
    assert tx["amount"] == 1


def test_auto_convert_fails_on_unsupported_token():
    tx = asyncio.run(auto_convert_funds("wallet", "DOGE", "USDC", 1))
    assert "error" in tx


def test_check_wallet_balances_env(monkeypatch):
    fake = json.dumps({"BTC": 0.3, "USDC": 50})
    monkeypatch.setenv("FAKE_BALANCES", fake)
    balances = check_wallet_balances("addr")
    assert balances["BTC"] == 0.3
    assert balances["USDC"] == 50


def test_check_wallet_balances_rpc(monkeypatch):
    class Client:
        def __init__(self, url):
            self.url = url

        def get_token_accounts_by_owner(self, owner, opts):
            return {
                "result": {
                    "value": [
                        {
                            "account": {
                                "data": {
                                    "parsed": {
                                        "info": {
                                            "mint": "BTC",
                                            "tokenAmount": {"uiAmount": 1},
                                        }
                                    }
                                }
                            }
                        },
                        {
                            "account": {
                                "data": {
                                    "parsed": {
                                        "info": {
                                            "mint": "USDC",
                                            "tokenAmount": {"uiAmount": 50},
                                        }
                                    }
                                }
                            }
                        },
                    ]
                }
            }

    class PublicKey(str):
        pass

    class TokenAccountOpts:
        def __init__(self, program_id=None, encoding=None):
            self.program_id = program_id
            self.encoding = encoding

    import sys, types

    sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)
    sys.modules.setdefault("solana.publickey", types.ModuleType("solana.publickey"))
    monkeypatch.setattr(sys.modules["solana.publickey"], "PublicKey", PublicKey, raising=False)
    sys.modules.setdefault("solana.rpc.types", types.ModuleType("solana.rpc.types"))
    monkeypatch.setattr(sys.modules["solana.rpc.types"], "TokenAccountOpts", TokenAccountOpts, raising=False)

    import crypto_bot.fund_manager as fm
    monkeypatch.setattr(fm, "Client", Client, raising=False)
    monkeypatch.setattr(fm, "PublicKey", PublicKey, raising=False)
    monkeypatch.setattr(fm, "TokenAccountOpts", TokenAccountOpts, raising=False)
    monkeypatch.delenv("FAKE_BALANCES", raising=False)

    balances = check_wallet_balances("addr")
    assert balances == {"BTC": 1.0, "USDC": 50.0}


def test_detect_zero_balance_tokens_ignored(monkeypatch):
    monkeypatch.setenv("MIN_BALANCE_THRESHOLD", "0.001")
    balances = {"BTC": 0, "ETH": -1, "USDC": 10}
    assert detect_non_trade_tokens(balances) == []
