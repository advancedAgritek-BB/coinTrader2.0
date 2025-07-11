import asyncio

import crypto_bot.main as main

class DummyExchange:
    def __init__(self):
        self.called = False
    def fetch_balance(self):
        self.called = True
        return {"USDT": {"free": 1234}}

def test_partial_sell_logs_balance_dry_run(monkeypatch):
    wallet = main.PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 2.0, 100.0)
    wallet.close("XBT/USDT", 1.0, 110.0)

    logged = []
    monkeypatch.setattr(main, "log_balance", lambda bal: logged.append(bal))

    cfg = {"execution_mode": "dry_run"}
    asyncio.run(main.fetch_and_log_balance(DummyExchange(), wallet, cfg))

    assert logged == [wallet.balance]

def test_partial_sell_logs_balance_exchange(monkeypatch):
    ex = DummyExchange()
    logged = []
    monkeypatch.setattr(main, "log_balance", lambda bal: logged.append(bal))

    cfg = {"execution_mode": "live"}
    asyncio.run(main.fetch_and_log_balance(ex, None, cfg))

    assert ex.called
    assert logged == [1234]
