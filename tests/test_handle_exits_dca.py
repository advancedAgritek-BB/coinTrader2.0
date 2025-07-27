import ast
import asyncio
from pathlib import Path
import pandas as pd
import pytest

from crypto_bot.phase_runner import BotContext
from crypto_bot.paper_wallet import PaperWallet
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dca_bot", Path("crypto_bot/strategy/dca_bot.py")
)
dca_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(dca_module)  # type: ignore[attr-defined]
dca_generate = dca_module.generate_signal

class DummyRM:
    def __init__(self):
        class CT:
            def can_allocate(self, *a, **k):
                return True
        self.capital_tracker = CT()
    def can_allocate(self, *a, **k):
        return True
    def deallocate_capital(self, *a, **k):
        pass
    def allocate_capital(self, *a, **k):
        pass
    def update_stop_order(self, *a, **k):
        pass

def load_handle_exits():
    src = Path('crypto_bot/main.py').read_text()
    module = ast.parse(src)
    funcs = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"opposite_side", "handle_exits"}:
            funcs[node.name] = ast.get_source_segment(src, node)
    calls = {"count": 0}
    async def _trade(*a, **k):
        calls["count"] += 1
    async def _refresh(*a, **k):
        return None
    ns = {
        "asyncio": asyncio,
        "cex_trade_async": _trade,
        "calculate_trailing_stop": lambda *a, **k: 0.0,
        "should_exit": lambda *a, **k: (False, 0.0),
        "dca_bot": type("D", (), {"generate_signal": dca_generate}),
        "pd": pd,
        "BotContext": BotContext,
        "log_position": lambda *a, **k: None,
        "refresh_balance": _refresh,
        "regime_pnl_tracker": type("R", (), {"log_trade": lambda *a, **k: None}),
    }
    exec(funcs["opposite_side"], ns)
    exec(funcs["handle_exits"], ns)
    return ns["handle_exits"], calls

@pytest.mark.asyncio
async def test_handle_exits_triggers_dca_buy():
    df = pd.DataFrame({"close": [100.0]*19 + [80.0]})
    ctx = BotContext(
        positions={"XBT/USDT": {
            "side": "buy",
            "entry_price": 100.0,
            "entry_time": "t",
            "regime": "",
            "strategy": "dca_bot",
            "confidence": 1.0,
            "pnl": 0.0,
            "size": 1.0,
            "trailing_stop": 0.0,
            "highest_price": 100.0,
            "dca_count": 0,
        }},
        df_cache={"1h": {"XBT/USDT": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "timeframe": "1h", "dca": {"max_entries": 2, "size_pct": 1.0}},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=None,
    )
    ctx.balance = 1000.0
    handle_exits, calls = load_handle_exits()
    await handle_exits(ctx)
    assert calls["count"] == 2
    assert ctx.positions["XBT/USDT"]["dca_count"] == 2

@pytest.mark.asyncio
async def test_handle_exits_stops_at_max_entries():
    df = pd.DataFrame({"close": [100.0]*19 + [80.0]})
    ctx = BotContext(
        positions={"XBT/USDT": {
            "side": "buy",
            "entry_price": 100.0,
            "entry_time": "t",
            "regime": "",
            "strategy": "dca_bot",
            "confidence": 1.0,
            "pnl": 0.0,
            "size": 1.0,
            "trailing_stop": 0.0,
            "highest_price": 100.0,
            "dca_count": 2,
        }},
        df_cache={"1h": {"XBT/USDT": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "timeframe": "1h", "dca": {"max_entries": 2, "size_pct": 1.0}},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=None,
    )
    ctx.balance = 1000.0
    handle_exits, calls = load_handle_exits()
    await handle_exits(ctx)
    assert calls["count"] == 0
    assert ctx.positions["XBT/USDT"]["dca_count"] == 2


@pytest.mark.asyncio
async def test_handle_exits_triggers_dca_short():
    df = pd.DataFrame({"close": [100.0]*19 + [120.0]})
    ctx = BotContext(
        positions={"XBT/USDT": {
            "side": "sell",
            "entry_price": 100.0,
            "entry_time": "t",
            "regime": "",
            "strategy": "dca_bot",
            "confidence": 1.0,
            "pnl": 0.0,
            "size": 1.0,
            "trailing_stop": 0.0,
            "highest_price": 100.0,
            "dca_count": 0,
        }},
        df_cache={"1h": {"XBT/USDT": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "timeframe": "1h", "dca": {"max_entries": 2, "size_pct": 1.0}},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=None,
    )
    ctx.balance = 1000.0
    handle_exits, calls = load_handle_exits()
    await handle_exits(ctx)
    assert calls["count"] == 2
    assert ctx.positions["XBT/USDT"]["dca_count"] == 2


@pytest.mark.asyncio
async def test_handle_exits_handles_missing_df():
    ctx = BotContext(
        positions={"XBT/USDT": {
            "side": "buy",
            "entry_price": 100.0,
            "entry_time": "t",
            "regime": "",
            "strategy": "dca_bot",
            "confidence": 1.0,
            "pnl": 0.0,
            "size": 1.0,
            "trailing_stop": 0.0,
            "highest_price": 100.0,
            "dca_count": 0,
        }},
        df_cache={"1h": {}},
        regime_cache={},
        config={"execution_mode": "dry_run", "timeframe": "1h", "dca": {"max_entries": 2, "size_pct": 1.0}},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=None,
    )
    ctx.balance = 1000.0
    handle_exits, calls = load_handle_exits()
    await handle_exits(ctx)
    assert calls["count"] == 0
    assert ctx.positions["XBT/USDT"]["dca_count"] == 0
