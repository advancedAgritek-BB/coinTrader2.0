import asyncio
import ast
from pathlib import Path
import pandas as pd
import pytest
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
    sys.modules['crypto_bot.solana'] = types.SimpleNamespace(sniper_solana=types.SimpleNamespace(generate_signal=lambda _df:(1.0, None)))

    ns = {
        "asyncio": asyncio,
        "logger": __import__("logging").getLogger("test"),
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
        "SNIPER_TASKS": set(),
        "NEW_SOLANA_TOKENS": set(),
    }
    exec(funcs["direction_to_side"], ns)
    exec(funcs["execute_signals"], ns)
    return ns["execute_signals"], ns["SNIPER_TASKS"], ns["NEW_SOLANA_TOKENS"]


@pytest.mark.asyncio
async def test_execute_signals_spawns_sniper_task():
    started = asyncio.Event()
    finished = asyncio.Event()

    async def stub(*a, **k):
        started.set()
        await asyncio.sleep(0.05)
        finished.set()

    execute_signals, sniper_tasks, new_tokens = load_execute_signals(stub)

    df = pd.DataFrame({"close": [1.0]})
    candidate = {
        "symbol": "SOL/USDC",
        "direction": "long",
        "df": df,
        "name": "test",
        "probabilities": {},
        "regime": "bull",
        "score": 1.0,
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

    execute_signals, sniper_tasks, new_tokens = load_execute_signals(stub)
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

    execute_signals, sniper_tasks, new_tokens = load_execute_signals(stub)
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
