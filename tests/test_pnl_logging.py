import ast
import asyncio
from pathlib import Path
import types

import pandas as pd
import pytest

from crypto_bot.phase_runner import BotContext
from crypto_bot.paper_wallet import PaperWallet


class DummyRM:
    def __init__(self) -> None:
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


def load_handle_exits_exit():
    src = Path("crypto_bot/main.py").read_text()
    module = ast.parse(src)
    funcs: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "opposite_side",
            "handle_exits",
        }:
            funcs[node.name] = ast.get_source_segment(src, node)

    calls: dict[str, tuple] = {}

    async def _trade(*a, **k):
        return None

    async def _refresh(*a, **k):
        return None

    def _log_pnl(*a, **k):
        calls["pnl"] = a

    def _log_trade(*a, **k):
        calls.setdefault("trade", a)

    ns = {
        "asyncio": asyncio,
        "cex_trade_async": _trade,
        "calculate_trailing_stop": lambda *a, **k: 0.0,
        "should_exit": lambda *a, **k: (True, 0.0),
        "dca_bot": types.SimpleNamespace(generate_signal=lambda _df: (0.0, None)),
        "pd": pd,
        "BotContext": BotContext,
        "log_position": lambda *a, **k: None,
        "refresh_balance": _refresh,
        "pnl_logger": types.SimpleNamespace(log_pnl=_log_pnl),
        "regime_pnl_tracker": types.SimpleNamespace(log_trade=_log_trade),
        "state": {},
    }

    exec(funcs["opposite_side"], ns)
    exec(funcs["handle_exits"], ns)
    return ns["handle_exits"], calls


def load_monitor_exit():
    src = Path("crypto_bot/main.py").read_text()
    module = ast.parse(src)
    funcs: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "opposite_side",
            "_monitor_micro_scalp_exit",
        }:
            funcs[node.name] = ast.get_source_segment(src, node)

    calls: dict[str, tuple] = {}

    async def _trade(*a, **k):
        return None

    async def _refresh(*a, **k):
        return None

    def _log_pnl(*a, **k):
        calls["pnl"] = a

    def _log_trade(*a, **k):
        calls.setdefault("trade", a)

    async def _monitor(feed, *a, **k):
        return {"exit_price": feed()}

    ns = {
        "asyncio": asyncio,
        "cex_trade_async": _trade,
        "monitor_price": _monitor,
        "BotContext": BotContext,
        "log_position": lambda *a, **k: None,
        "refresh_balance": _refresh,
        "pnl_logger": types.SimpleNamespace(log_pnl=_log_pnl),
        "regime_pnl_tracker": types.SimpleNamespace(log_trade=_log_trade),
        "state": {},
    }

    exec(funcs["opposite_side"], ns)
    exec(funcs["_monitor_micro_scalp_exit"], ns)
    return ns["_monitor_micro_scalp_exit"], calls


def load_force_exit_all():
    src = Path("crypto_bot/main.py").read_text()
    module = ast.parse(src)
    funcs: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "opposite_side",
            "force_exit_all",
        }:
            funcs[node.name] = ast.get_source_segment(src, node)

    calls: dict[str, tuple] = {}

    async def _trade(*a, **k):
        return None

    async def _refresh(ctx):
        return None

    def _log_pnl(*a, **k):
        calls["pnl"] = a

    def _log_trade(*a, **k):
        calls.setdefault("trade", a)

    ns = {
        "asyncio": asyncio,
        "cex_trade_async": _trade,
        "refresh_balance": _refresh,
        "log_position": lambda *a, **k: None,
        "BotContext": BotContext,
        "pnl_logger": types.SimpleNamespace(log_pnl=_log_pnl),
        "regime_pnl_tracker": types.SimpleNamespace(log_trade=_log_trade),
        "logger": __import__("logging").getLogger("test"),
        "state": {},
    }

    exec(funcs["opposite_side"], ns)
    exec(funcs["force_exit_all"], ns)
    return ns["force_exit_all"], calls


@pytest.mark.asyncio
async def test_handle_exits_logs_pnl():
    df = pd.DataFrame({"close": [100.0, 110.0]})
    ctx = BotContext(
        positions={
            "XBT/USDT": {
                "side": "buy",
                "entry_price": 100.0,
                "entry_time": "t",
                "regime": "bull",
                "strategy": "trend_bot",
                "confidence": 1.0,
                "pnl": 0.0,
                "size": 1.0,
                "trailing_stop": 0.0,
                "highest_price": 100.0,
                "dca_count": 0,
            }
        },
        df_cache={"1h": {"XBT/USDT": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "timeframe": "1h", "dca": {}},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=None,
    )
    ctx.balance = 1000.0

    handle_exits, calls = load_handle_exits_exit()
    await handle_exits(ctx)

    assert calls.get("pnl") is not None
    assert calls.get("trade") == ("bull", "trend_bot", 10.0)


@pytest.mark.asyncio
async def test_monitor_micro_scalp_exit_logs_pnl():
    df = pd.DataFrame({"close": [100.0, 105.0]})
    ctx = BotContext(
        positions={
            "SOL/USDC": {
                "side": "buy",
                "entry_price": 100.0,
                "entry_time": "t",
                "regime": "scalp",
                "strategy": "micro_scalp_bot",
                "confidence": 0.5,
                "pnl": 0.0,
                "size": 2.0,
            }
        },
        df_cache={"1m": {"SOL/USDC": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "scalp_timeframe": "1m"},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=None,
    )
    ctx.balance = 1000.0

    monitor_exit, calls = load_monitor_exit()
    await monitor_exit(ctx, "SOL/USDC")

    assert calls.get("pnl") is not None
    assert calls.get("trade") == ("scalp", "micro_scalp_bot", 10.0)


@pytest.mark.asyncio
async def test_force_exit_all_logs_pnl():
    df = pd.DataFrame({"close": [110.0]})
    ctx = BotContext(
        positions={
            "XBT/USDT": {
                "side": "buy",
                "entry_price": 100.0,
                "strategy": "trend_bot",
                "regime": "bull",
                "confidence": 0.9,
                "size": 1.0,
            }
        },
        df_cache={"1h": {"XBT/USDT": df}},
        regime_cache={},
        config={"timeframe": "1h", "execution_mode": "live"},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=None,
        position_guard=None,
    )
    ctx.balance = 1000.0

    force_exit, calls = load_force_exit_all()
    await force_exit(ctx)

    assert calls.get("pnl") is not None
    assert calls.get("trade") == ("bull", "trend_bot", 10.0)

