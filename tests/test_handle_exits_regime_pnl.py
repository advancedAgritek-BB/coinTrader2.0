import ast
import asyncio
from pathlib import Path
import pandas as pd
import pytest
import types
from crypto_bot.phase_runner import BotContext
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.utils import regime_pnl_tracker as rpt

class DummyRM:
    def __init__(self):
        self.deallocated = False

    def deallocate_capital(self, *a, **k):
        self.deallocated = True

    def allocate_capital(self, *a, **k):
        pass

    def can_allocate(self, *a, **k):
        return True


def load_handle_exits_exit():
    src = Path('crypto_bot/main.py').read_text()
    module = ast.parse(src)
    funcs = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"opposite_side", "handle_exits"}:
            funcs[node.name] = ast.get_source_segment(src, node)
    async def _trade(*a, **k):
        pass
    async def _refresh(*a, **k):
        return None
    ns = {
        "asyncio": asyncio,
        "cex_trade_async": _trade,
        "calculate_trailing_stop": lambda *a, **k: 0.0,
        "should_exit": lambda *a, **k: (True, 0.0),
        "dca_bot": type("D", (), {"generate_signal": lambda df:(0.0, "long")}),
        "pd": pd,
        "BotContext": BotContext,
        "log_position": lambda *a, **k: None,
        "refresh_balance": _refresh,
        "regime_pnl_tracker": rpt,
        "state": {},
        "pnl_logger": types.SimpleNamespace(log_pnl=lambda *a, **k: None),
    }
    exec(funcs["opposite_side"], ns)
    exec(funcs["handle_exits"], ns)
    return ns["handle_exits"]


@pytest.mark.asyncio
async def test_handle_exits_logs_regime_pnl(regime_pnl_file, monkeypatch):
    handle_exits = load_handle_exits_exit()
    df = pd.DataFrame({"close": [100.0, 110.0]})
    ctx = BotContext(
        positions={"XBT/USDT": {
            "side": "buy",
            "entry_price": 100.0,
            "entry_time": "t",
            "regime": "trending",
            "strategy": "trend_bot",
            "confidence": 1.0,
            "pnl": 0.0,
            "size": 1.0,
            "trailing_stop": 0.0,
            "highest_price": 100.0,
            "dca_count": 0,
        }},
        df_cache={"1h": {"XBT/USDT": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "timeframe": "1h", "dca": {"max_entries": 0}},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=None,
    )
    ctx.balance = 1000.0

    calls = []
    orig = rpt.log_trade
    def capture(regime, strat, pnl):
        calls.append((regime, strat, pnl))
        orig(regime, strat, pnl)
    monkeypatch.setattr(rpt, "log_trade", capture)

    await handle_exits(ctx)

    assert calls and calls[0][0] == "trending"
    df_logged = pd.read_csv(regime_pnl_file)
    assert len(df_logged) >= 1
    assert (df_logged["regime"] == "trending").all()
    assert (df_logged["strategy"] == "trend_bot").all()
