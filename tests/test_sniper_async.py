import asyncio
import ast
from pathlib import Path
import pandas as pd
import pytest
import logging
from crypto_bot.phase_runner import BotContext
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.utils.logger import LOG_DIR, setup_logger


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
    def sentiment_factor_or_default(self, *a, **k):
        return 1.0


class DummyPG:
    def __init__(self, max_open_trades: int = 2) -> None:
        self.max_open_trades = max_open_trades

    def can_open(self, positions):
        return True


def load_execute_signals(sniper_stub):
    src = Path("crypto_bot/main.py").read_text()
    module = ast.parse(src)
    funcs = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"direction_to_side", "execute_signals"}:
            funcs[node.name] = ast.get_source_segment(src, node)
    async def _trade(*a, **k):
        return {"id": "1"}

    import types, sys
    sys.modules['crypto_bot.solana_trading'] = types.SimpleNamespace(sniper_trade=sniper_stub)
    sys.modules['crypto_bot.solana'] = types.SimpleNamespace(
        sniper_solana=types.SimpleNamespace(generate_signal=lambda _df: (1.0, None))
    )

    from crypto_bot.utils.task_manager import TaskManager

    manager = TaskManager()

    ns = {
        "asyncio": asyncio,
        "logger": logging.getLogger("test"),
        "score_logger": setup_logger(
            "symbol_filter", LOG_DIR / "symbol_filter.log", to_console=False
        ),
        "cex_trade_async": _trade,
        "fetch_order_book_async": lambda *a, **k: {},
        "_closest_wall_distance": lambda *a, **k: None,
        "datetime": __import__("datetime").datetime,
        "time": __import__("time"),
        "log_position": lambda *a, **k: None,
        "_monitor_micro_scalp_exit": lambda *a, **k: None,
        "BotContext": BotContext,
        "refresh_balance": lambda ctx: asyncio.sleep(0),
        "sniper_trade": sniper_stub,
        "sniper_solana": types.SimpleNamespace(
            generate_signal=lambda _df, **k: (1.0, None)
        ),
        "register_task": manager.register,
        "TASK_MANAGER": manager,
        "NEW_SOLANA_TOKENS": set(),
    }
    ns["score_logger"].setLevel(logging.DEBUG)
    exec(funcs["direction_to_side"], ns)
    exec(funcs["execute_signals"], ns)
    return ns["execute_signals"], manager, ns["NEW_SOLANA_TOKENS"]


@pytest.mark.asyncio
async def test_execute_signals_spawns_sniper_task():
    started = asyncio.Event()
    finished = asyncio.Event()

    async def stub(*a, **k):
        started.set()
        await asyncio.sleep(0.05)
        finished.set()

    execute_signals, manager, new_tokens = load_execute_signals(stub)
    sniper_tasks = manager.tasks["sniper"]

    df = pd.DataFrame({"close": [1.0]})
    candidate = {
        "symbol": "SOL/USDC",
        "direction": "long",
        "df": df,
        "name": "test",
        "probabilities": {},
        "regime": "bull",
        "score": 1.0,
        "entry": {"price": 1.0},
        "size": 1.0,
        "valid": True,
    }

    ctx = BotContext(
        positions={},
        df_cache={"1h": {"SOL/USDC": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "top_n_symbols": 1},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=DummyPG(),
    )
    ctx.balance = 1000.0
    ctx.analysis_results = [candidate]
    ctx.timing = {}

    await asyncio.wait_for(execute_signals(ctx), 0.05)
    await asyncio.wait_for(started.wait(), 0.05)
    assert not finished.is_set()
    assert len(sniper_tasks) == 1
    await asyncio.gather(*sniper_tasks)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_new_token_regime_filter():
    called = asyncio.Event()

    async def stub(*a, **k):
        called.set()

    execute_signals, manager, new_tokens = load_execute_signals(stub)
    sniper_tasks = manager.tasks["sniper"]
    new_tokens.add("SOL/USDC")

    df = pd.DataFrame({"close": [1.0]})
    candidate = {
        "symbol": "SOL/USDC",
        "direction": "long",
        "df": df,
        "name": "test",
        "probabilities": {},
        "regime": "bull",
        "score": 1.0,
        "entry": {"price": 1.0},
        "size": 1.0,
        "valid": True,
    }

    ctx = BotContext(
        positions={},
        df_cache={"1h": {"SOL/USDC": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "top_n_symbols": 1},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=DummyPG(),
    )
    ctx.balance = 1000.0
    ctx.analysis_results = [candidate]
    ctx.timing = {}

    await execute_signals(ctx)
    assert not called.is_set()
    assert len(sniper_tasks) == 0


@pytest.mark.asyncio
async def test_new_token_regime_allows_trade():
    started = asyncio.Event()

    async def stub(*a, **k):
        started.set()

    execute_signals, manager, new_tokens = load_execute_signals(stub)
    sniper_tasks = manager.tasks["sniper"]
    new_tokens.add("SOL/USDC")

    df = pd.DataFrame({"close": [1.0]})
    candidate = {
        "symbol": "SOL/USDC",
        "direction": "long",
        "df": df,
        "name": "test",
        "probabilities": {},
        "regime": "volatile",
        "score": 1.0,
        "entry": {"price": 1.0},
        "size": 1.0,
        "valid": True,
    }

    ctx = BotContext(
        positions={},
        df_cache={"1h": {"SOL/USDC": df}},
        regime_cache={},
        config={"execution_mode": "dry_run", "top_n_symbols": 1},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=PaperWallet(1000.0),
        position_guard=DummyPG(),
    )
    ctx.balance = 1000.0
    ctx.analysis_results = [candidate]
    ctx.timing = {}

    await execute_signals(ctx)
    await asyncio.wait_for(started.wait(), 0.05)
    assert len(sniper_tasks) == 1
    await asyncio.gather(*sniper_tasks)
