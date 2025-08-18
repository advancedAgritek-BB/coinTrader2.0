import logging
from types import SimpleNamespace

from crypto_bot.utils import market_loader
from crypto_bot.strategy import registry


def test_auto_raise_warmup(monkeypatch, caplog):
    stub = SimpleNamespace(__name__="stub", required_lookback=lambda: {"1m": 1440})
    monkeypatch.setattr(registry, "load_from_config", lambda cfg: [stub])
    cfg = {
        "timeframes": ["1m"],
        "warmup_candles": {"1m": 1000},
        "data": {"auto_raise_warmup": True},
    }
    caplog.set_level(logging.INFO)
    market_loader._ensure_strategy_warmup(cfg)
    assert cfg["warmup_candles"]["1m"] == 1440
    assert "Auto-raising warmup_candles[1m]" in caplog.text


def test_registry_disables_strategy(caplog):
    stub = SimpleNamespace(__name__="stub", required_lookback=lambda: {"1m": 1440})
    cfg = {"warmup_candles": {"1m": 1000}}
    caplog.set_level(logging.WARNING)
    enabled = registry.filter_by_warmup(cfg, [stub])
    assert enabled == []
    assert "Insufficient warmup_candles" in caplog.text


def test_warn_if_strategy_skipped_after_auto_raise(monkeypatch, caplog):
    stub = SimpleNamespace(__name__="stub", required_lookback=lambda: {"1m": 1440})
    monkeypatch.setattr(registry, "load_from_config", lambda cfg: [stub])
    monkeypatch.setattr(registry, "filter_by_warmup", lambda cfg, s: [])
    cfg = {
        "timeframes": ["1m"],
        "warmup_candles": {"1m": 1000},
        "data": {"auto_raise_warmup": True},
    }
    caplog.set_level(logging.WARNING)
    market_loader._ensure_strategy_warmup(cfg)
    assert "Skipping strategies due to insufficient warmup" in caplog.text
