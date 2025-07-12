import asyncio
import importlib
import pathlib
import sys
import pandas as pd

# Restore the real crypto_bot package if previous tests replaced it
ROOT = pathlib.Path(__file__).resolve().parents[1] / "crypto_bot"
sys.modules.pop("crypto_bot", None)
sys.modules.pop("crypto_bot.volatility_filter", None)
spec = importlib.util.spec_from_file_location("crypto_bot", ROOT / "__init__.py")
crypto_bot = importlib.util.module_from_spec(spec)
crypto_bot.__path__ = [str(ROOT)]
sys.modules["crypto_bot"] = crypto_bot
spec.loader.exec_module(crypto_bot)
import crypto_bot.main as main
from crypto_bot.phase_runner import BotContext
from crypto_bot.open_position_guard import OpenPositionGuard

class DummyExchange:
    def __init__(self):
        self.calls = []

class DummyRisk:
    def allow_trade(self, df, strategy):
        return True, ""
    def position_size(self, score, balance, df, atr=None, price=None):
        return price or 0
    def can_allocate(self, strategy, size, balance):
        return True
    def allocate_capital(self, strategy, size):
        pass

async def fake_trade(exchange, ws_client, symbol, side, amount, *a, **kw):
    ex = kw.get("exchange_override") or exchange
    if hasattr(ex, "calls"):
        ex.calls.append((symbol, side, amount))
    return {}

def test_execute_signals_dual(monkeypatch):
    buy_ex = DummyExchange()
    sell_ex = DummyExchange()

    df = pd.DataFrame({"close": [100]})
    ctx = BotContext(positions={}, df_cache={"1h": {"BTC/USDT": df}}, regime_cache={}, config={"top_n_symbols": 1, "execution_mode": "dry_run"})
    ctx.risk_manager = DummyRisk()
    ctx.exchange = DummyExchange()
    ctx.ws_client = None
    ctx.notifier = None
    ctx.position_guard = OpenPositionGuard(1)
    ctx.balance = 100
    ctx.analysis_results = [{
        "symbol": "BTC/USDT",
        "df": df,
        "score": 0.5,
        "direction": "long",
        "exchange_buy": buy_ex,
        "exchange_sell": sell_ex,
    }]

    monkeypatch.setattr(main, "cex_trade_async", fake_trade)
    monkeypatch.setattr(main, "log_position", lambda *a, **k: None)

    asyncio.run(main.execute_signals(ctx))

    assert buy_ex.calls == [("BTC/USDT", "buy", 1.0)]
    assert sell_ex.calls == [("BTC/USDT", "sell", 1.0)]
    assert ctx.positions["BTC/USDT"]["exchange_buy"] is buy_ex
    assert ctx.positions["BTC/USDT"]["exchange_sell"] is sell_ex
