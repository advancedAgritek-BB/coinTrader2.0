import importlib.util
from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / "crypto_bot/strategy/cross_chain_arb_bot.py"
spec = importlib.util.spec_from_file_location("cross_chain_arb_bot", path)
cross_chain_arb_bot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cross_chain_arb_bot)


class DummyMonitor:
    def __init__(self, suspicious=False):
        self._suspicious = suspicious

    async def is_suspicious(self, threshold: float) -> bool:  # pragma: no cover - simple
        return self._suspicious


def make_df(price: float) -> pd.DataFrame:
    return pd.DataFrame({"close": [price]})


def test_spread_generates_signal(monkeypatch):
    df = make_df(10.0)
    async def fake_fetch(symbols):
        return {symbols[0]: 12.0}

    monkeypatch.setattr(cross_chain_arb_bot, "fetch_solana_prices", fake_fetch)
    cfg = {"cross_chain_arb_bot": {"pair": "SOL/USDC", "spread_threshold": 0.05}}
    score, direction = cross_chain_arb_bot.generate_signal(df, cfg, mempool_monitor=DummyMonitor(False), mempool_cfg={"enabled": True, "suspicious_fee_threshold": 30})
    assert direction == "long"
    assert score > 0


def test_below_threshold_no_signal(monkeypatch):
    df = make_df(10.0)
    async def fake_fetch(symbols):
        return {symbols[0]: 10.1}

    monkeypatch.setattr(cross_chain_arb_bot, "fetch_solana_prices", fake_fetch)
    cfg = {"cross_chain_arb_bot": {"pair": "SOL/USDC", "spread_threshold": 0.05}}
    score, direction = cross_chain_arb_bot.generate_signal(df, cfg, mempool_monitor=DummyMonitor(False), mempool_cfg={"enabled": True, "suspicious_fee_threshold": 30})
    assert (score, direction) == (0.0, "none")


def test_fee_blocks_signal(monkeypatch):
    df = make_df(10.0)
    async def fake_fetch(symbols):
        return {symbols[0]: 12.0}

    monkeypatch.setattr(cross_chain_arb_bot, "fetch_solana_prices", fake_fetch)
    cfg = {"cross_chain_arb_bot": {"pair": "SOL/USDC", "spread_threshold": 0.05}}
    score, direction = cross_chain_arb_bot.generate_signal(df, cfg, mempool_monitor=DummyMonitor(True), mempool_cfg={"enabled": True, "suspicious_fee_threshold": 5})
    assert (score, direction) == (0.0, "none")
