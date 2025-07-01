import os
import json
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
    tx = auto_convert_funds("wallet", "BTC", "USDC", 1)
    assert tx["from"] == "BTC"
    assert tx["to"] == "USDC"
    assert tx["amount"] == 1


def test_check_wallet_balances_env(monkeypatch):
    fake = json.dumps({"BTC": 0.3, "USDC": 50})
    monkeypatch.setenv("FAKE_BALANCES", fake)
    balances = check_wallet_balances("addr")
    assert balances["BTC"] == 0.3
    assert balances["USDC"] == 50
