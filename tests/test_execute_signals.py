import asyncio
import logging
import pandas as pd
import pytest
import ast
from pathlib import Path
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


class DummyPG:
    def __init__(self, max_open_trades: int = 2) -> None:
        self.max_open_trades = max_open_trades

    def can_open(self, positions):
        return True


def load_execute_signals():
    called = {"called": False}
    src = Path("crypto_bot/main.py").read_text()
    module = ast.parse(src)
    funcs = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "direction_to_side",
            "execute_signals",
        }:
            funcs[node.name] = ast.get_source_segment(src, node)

    async def _trade(*a, **k):
        called["called"] = True
        return {"id": "1"}

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
    }
    ns["score_logger"].setLevel(logging.DEBUG)
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
        "entry": {"price": 100.0},
        "size": 1.0,
        "valid": True,
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
    assert "[EVAL] evaluating XBT/USDT" in caplog.text
    assert "[EVAL] XBT/USDT -> short selling disabled" in caplog.text
    assert (
        "Gate summary for XBT/USDT: sentiment=True risk=False budget=True cooldown=True min_score=True"
        in caplog.text
    )
    assert "Trade BLOCKED (short selling disabled)" in caplog.text


@pytest.mark.asyncio
async def test_execute_signals_respects_too_flat(monkeypatch, caplog):
    df = pd.DataFrame({"close": [100.0]})
    candidate = {
        "symbol": "XBT/USDT",
        "direction": "long",
        "df": df,
        "name": "test",
        "probabilities": {},
        "regime": "bull",
        "score": 1.0,
        "too_flat": True,
        "entry": {"price": 100.0},
        "size": 1.0,
        "valid": True,
    }
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"XBT/USDT": df}},
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

    execute_signals, called = load_execute_signals()
    caplog.set_level(logging.INFO)
    await execute_signals(ctx)

    assert ctx.positions == {}
    assert not called["called"]
    assert "[EVAL] XBT/USDT -> atr too flat" in caplog.text
    assert (
        "Gate summary for XBT/USDT: sentiment=True risk=False budget=True cooldown=True min_score=True"
        in caplog.text
    )
    assert "Trade BLOCKED (atr too flat)" in caplog.text


@pytest.mark.asyncio
async def test_execute_signals_logs_execution(monkeypatch, caplog):
    df = pd.DataFrame({"close": [100.0]})
    candidate = {
        "symbol": "XBT/USDT",
        "direction": "long",
        "df": df,
        "name": "test",
        "probabilities": {},
        "regime": "bull",
        "score": 1.0,
        "entry": {"price": 100.0},
        "size": 1.0,
        "valid": True,
    }
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"XBT/USDT": df}},
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

    execute_signals, called = load_execute_signals()
    caplog.set_level(logging.INFO)
    await execute_signals(ctx)

    assert called["called"]
    assert "[EVAL] evaluating XBT/USDT" in caplog.text
    assert (
        "Gate summary for XBT/USDT: sentiment=True risk=True budget=True cooldown=True min_score=True"
        in caplog.text
    )
    assert "DRY-RUN: simulated buy XBT/USDT x1.0 @100.0" in caplog.text
    assert "[EVAL] XBT/USDT -> executed buy 1.0000" in caplog.text


@pytest.mark.asyncio
async def test_execute_signals_no_symbols_qualify(monkeypatch, caplog):
    df = pd.DataFrame({"close": [100.0]})
    candidate = {
        "symbol": "XBT/USDT",
        "direction": "long",
        "df": df,
        "name": "test",
        "probabilities": {},
        "regime": "bull",
        "score": 0.05,
        "min_confidence": 0.1,
        "entry": {"price": 100.0},
        "size": 1.0,
        "valid": True,
    }

    class DummyNotifier:
        def __init__(self):
            self.sent = []

        async def notify_async(self, text):
            self.sent.append(text)

        def notify(self, text):
            self.sent.append(text)

    ctx = BotContext(
        positions={},
        df_cache={"1h": {"XBT/USDT": df}},
        regime_cache={},
        config={
            "execution_mode": "dry_run",
            "top_n_symbols": 1,
            "telegram": {"trade_updates": True},
        },
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=DummyNotifier(),
        paper_wallet=PaperWallet(1000.0),
        position_guard=DummyPG(),
    )
    ctx.balance = 1000.0
    ctx.analysis_results = [candidate]
    ctx.timing = {}

    log_file = LOG_DIR / "symbol_filter.log"
    execute_signals, _ = load_execute_signals()
    caplog.set_level(logging.DEBUG)
    await execute_signals(ctx)

    content = log_file.read_text() if log_file.exists() else ""
    assert ctx.notifier.sent == ["No symbols qualified for trading"]
    assert "Candidate scoring" in content
    assert "nothing actionable" in content
