import asyncio
import types
import pytest
import crypto_bot.main as main


class DummyExchange:
    options = {}

    async def fetch_balance(self):
        return {"USDT": {"free": 0}}

    def fetch_ohlcv(self, *args, **kwargs):
        return []

    def load_markets(self):
        return {}


class DummyWallet:
    def __init__(self, balance, max_trades, short_selling, **kwargs):
        self.total_balance = balance
        self.positions = {}


class DummyNotifier:
    enabled = False
    token = None
    chat_id = None

    def notify(self, msg):
        pass


class DummyRiskManager:
    def __init__(self, cfg):
        pass


class DummyConsole:
    async def control_loop(self, state, ctx, session_state):
        return


class DummyBotContext:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


@pytest.mark.asyncio
async def test_main_impl_hft_ctx(monkeypatch):
    cfg = {
        "hft": True,
        "telegram": {},
        "mempool_monitor": {},
        "scan_markets": False,
        "execution_mode": "dry_run",
        "mode": "cex",
        "scan_in_background": False,
    }

    async def fake_load_config_async():
        return cfg, False

    async def fake_load_mints():
        return {}

    async def fake_initial_scan(*args, **kwargs):
        return None

    async def fake_fetch_and_log_balance(exchange, wallet, config):
        return 0.0

    monkeypatch.setattr(main, "load_config_async", fake_load_config_async)
    monkeypatch.setattr(main, "load_token_mints", fake_load_mints)
    monkeypatch.setattr(main, "set_token_mints", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "load_or_create", lambda **_k: {})
    monkeypatch.setattr(
        main,
        "TelegramNotifier",
        types.SimpleNamespace(from_config=lambda cfg: DummyNotifier()),
        raising=False,
    )
    monkeypatch.setattr(main, "send_test_message", lambda *a, **k: None)
    monkeypatch.setattr(main, "get_exchange", lambda cfg: (DummyExchange(), None))
    monkeypatch.setattr(main, "fetch_balance", lambda *a, **k: 0.0)
    monkeypatch.setattr(main, "initial_scan", fake_initial_scan)
    monkeypatch.setattr(main, "RiskManager", DummyRiskManager)
    monkeypatch.setattr(main, "build_risk_config", lambda c, v: {})
    monkeypatch.setattr(main, "Wallet", DummyWallet, raising=False)
    monkeypatch.setattr(main, "notify_balance_change", lambda *a: 0.0, raising=False)
    monkeypatch.setattr(main, "fetch_and_log_balance", fake_fetch_and_log_balance, raising=False)
    monkeypatch.setattr(main, "load_strategies", lambda *a, **k: [], raising=False)
    monkeypatch.setattr(main, "console_control", DummyConsole(), raising=False)

    async def fake_rotation_loop(*a, **k):
        return None

    monkeypatch.setattr(main, "_rotation_loop", fake_rotation_loop)
    monkeypatch.setattr(main, "log_balance", lambda *a, **k: None)
    monkeypatch.setattr(main, "format_monitor_line", lambda *a, **k: "")
    monkeypatch.setattr(
        main,
        "PortfolioRotator",
        lambda: types.SimpleNamespace(config={}),
        raising=False,
    )
    monkeypatch.setattr(main, "log_ml_status_once", lambda: None, raising=False)
    monkeypatch.setattr(main, "BotContext", DummyBotContext, raising=False)

    result = await main._main_impl()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(result, main.MainResult)
