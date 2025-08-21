import asyncio
import logging
import pandas as pd
import pytest
import ast
from pathlib import Path
from crypto_bot.utils.logger import LOG_DIR, setup_logger

from crypto_bot.phase_runner import BotContext
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig


class DummyRM:
    def allow_trade(self, *a, **k):
        return True, ""

    def position_size(self, *_a, **_k):
        return 50.0

    def can_allocate(self, *a, **k):
        return True

    def allocate_capital(self, *a, **k):
        pass

    def register_stop_order(self, *a, **k):
        pass


class DummyPG:
    max_open_trades = 10
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
        "score_logger": setup_logger("symbol_filter", LOG_DIR / "symbol_filter.log", to_console=False),
        "cex_trade_async": _trade,
        "fetch_order_book_async": lambda *a, **k: {},
        "_closest_wall_distance": lambda *a, **k: None,
        "datetime": __import__("datetime").datetime,
        "time": __import__("time"),
        "log_position": lambda *a, **k: None,
        "_monitor_micro_scalp_exit": lambda *a, **k: None,
        "BotContext": BotContext,
        "refresh_balance": lambda ctx: asyncio.sleep(0),
        "Counter": __import__("collections").Counter,
    }
    ns["score_logger"].setLevel(logging.DEBUG)
    exec(funcs["direction_to_side"], ns)
    exec(funcs["execute_signals"], ns)
    return ns["execute_signals"], called


@pytest.mark.asyncio
async def test_positive_balance_generates_position():
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
    await execute_signals(ctx)

    assert called["called"]
    assert ctx.positions["XBT/USDT"]["size"] > 0


def _simple_cfg():
    return RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        slippage_factor=0.0,
    )


def test_high_win_rate_boosts_position_size(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.risk.risk_manager.get_recent_win_rate", lambda *a, **k: 0.8
    )
    rm = RiskManager(_simple_cfg())
    size = rm.position_size(1.0, 1000, name="trend_bot")
    assert size == pytest.approx(150.0)


def test_high_win_rate_boosts_position_size_short(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.risk.risk_manager.get_recent_win_rate", lambda *a, **k: 0.8
    )
    rm = RiskManager(_simple_cfg())
    size = rm.position_size(1.0, 1000, name="trend_bot", direction="short")
    assert size == pytest.approx(-150.0)


def test_low_win_rate_no_boost(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.risk.risk_manager.get_recent_win_rate", lambda *a, **k: 0.6
    )
    rm = RiskManager(_simple_cfg())
    size = rm.position_size(1.0, 1000, name="trend_bot")
    assert size == pytest.approx(100.0)


def test_custom_win_rate_settings(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.risk.risk_manager.get_recent_win_rate", lambda *a, **k: 0.7
    )
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        win_rate_threshold=0.6,
        win_rate_boost_factor=2.0,
        slippage_factor=0.0,
    )
    rm = RiskManager(cfg)
    size = rm.position_size(1.0, 1000, name="trend_bot")
    assert size == pytest.approx(200.0)
