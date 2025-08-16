import asyncio
import types
import sys
import logging

import pytest

import crypto_bot.main as main


class DummyNotifier:
    def __init__(self):
        self.token = "t"
        self.chat_id = "c"
        self.enabled = True
        self.messages = []

    def notify(self, msg: str) -> None:
        self.messages.append(msg)


@pytest.fixture(autouse=True)
def minimal_environment(monkeypatch):
    mod = types.ModuleType("crypto_bot.utils.token_registry")
    mod.TOKEN_MINTS = {}
    async def refresh_mints():
        pass
    async def periodic_mint_sanity_check():
        pass
    async def monitor_pump_raydium():
        pass
    async def fetch_from_helius(*a, **k):
        return {}
    async def load_token_mints():
        return {}
    async def fetch_from_jupiter():
        return {}
    mod.refresh_mints = refresh_mints
    mod.periodic_mint_sanity_check = periodic_mint_sanity_check
    mod.monitor_pump_raydium = monitor_pump_raydium
    mod.fetch_from_helius = fetch_from_helius
    mod.load_token_mints = load_token_mints
    mod.fetch_from_jupiter = fetch_from_jupiter
    mod.set_token_mints = lambda *a, **k: None
    mod.get_decimals = lambda *a, **k: 9
    mod.to_base_units = lambda *a, **k: 1
    monkeypatch.setitem(sys.modules, "crypto_bot.utils.token_registry", mod)

    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(main, "_ensure_user_setup", lambda: None)
    monkeypatch.setattr(main, "_reload_modules", lambda: None)
    ml_mod = types.ModuleType("crypto_bot.utils.ml_utils")
    ml_mod.init_ml_components = lambda: None
    ml_mod.ML_AVAILABLE = False
    monkeypatch.setitem(sys.modules, "crypto_bot.utils.ml_utils", ml_mod)

    tele_mod = types.ModuleType("crypto_bot.utils.telegram")
    tele_mod.TelegramNotifier = DummyNotifier
    tele_mod.send_test_message = lambda *a, **k: None
    tele_mod.is_admin = lambda *a, **k: True
    monkeypatch.setitem(sys.modules, "crypto_bot.utils.telegram", tele_mod)
    async def _refresh():
        pass
    monkeypatch.setattr(main, "refresh_mints", _refresh, raising=False)


def test_logs_state_stop(monkeypatch, caplog):
    notifier = DummyNotifier()

    async def fake_impl():
        return main.MainResult(notifier, "state['running'] set to False")

    monkeypatch.setattr(main, "_main_impl", fake_impl)

    with caplog.at_level(logging.INFO):
        asyncio.run(main.main())

    assert "Bot shutting down: state['running'] set to False" in caplog.text
    assert notifier.messages[-1] == "Bot shutting down: state['running'] set to False"


def test_logs_exception(monkeypatch, caplog):
    async def fake_impl():
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "_main_impl", fake_impl)

    with caplog.at_level(logging.INFO):
        asyncio.run(main.main())

    assert "Bot shutting down: exception: boom" in caplog.text


def test_logs_cancelled(monkeypatch, caplog):
    async def fake_impl():
        raise asyncio.CancelledError()

    monkeypatch.setattr(main, "_main_impl", fake_impl)

    with caplog.at_level(logging.INFO):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(main.main())

    assert "Bot shutting down: external signal" in caplog.text
