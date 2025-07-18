import asyncio
import logging
import pandas as pd
import pytest
import ast
from pathlib import Path
from crypto_bot.phase_runner import BotContext
from crypto_bot.paper_wallet import PaperWallet

class DummyRM:
    def allow_trade(self, *a, **k):
        return True, ""
    def position_size(self, *a, **k):
        return 100.0
    def can_allocate(self, *a, **k):
        return True
    def allocate_capital(self, *a, **k):
        pass
    def register_stop_order(self, *a, **k):
        pass

class DummyPG:
    def can_open(self, positions):
        return True


def load_execute_signals():
    called = {"called": False}
    src = Path("crypto_bot/main.py").read_text()
    module = ast.parse(src)
    funcs = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"direction_to_side", "execute_signals"}:
            funcs[node.name] = ast.get_source_segment(src, node)
    async def _trade(*a, **k):
        called["called"] = True
        return {"id": "1"}

    ns = {
        "asyncio": asyncio,
        "logger": logging.getLogger("test"),
        "cex_trade_async": _trade,
        "fetch_order_book_async": lambda *a, **k: {},
        "_closest_wall_distance": lambda *a, **k: None,
        "datetime": __import__("datetime").datetime,
        "time": __import__("time"),
        "log_position": lambda *a, **k: None,
        "_monitor_micro_scalp_exit": lambda *a, **k: None,
        "BotContext": BotContext,
    }
    exec(funcs["direction_to_side"], ns)
    exec(funcs["execute_signals"], ns)
    return ns["execute_signals"], called


@pytest.mark.asyncio
async def test_execute_signals_respects_allow_short(monkeypatch, caplog):
    df = pd.DataFrame({"close": [100.0]})
    candidate = {
        "symbol": "XBT/USDT",
        "direction": "short",
        "df": df,
        "name": "test",
        "probabilities": {},
        "regime": "bull",
        "score": 1.0,
    }
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"XBT/USDT": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "allow_short": False, "top_n_symbols": 1},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0, allow_short=False),
        position_guard=DummyPG(),
    )
    ctx.balance = 1000.0
    ctx.analysis_results = [candidate]
    ctx.timing = {}

    execute_signals, called = load_execute_signals()
    caplog.set_level(logging.INFO)
    await execute_signals(ctx)

    assert ctx.positions == {}
    assert not called["called"]
    assert "Short selling disabled" in caplog.text
