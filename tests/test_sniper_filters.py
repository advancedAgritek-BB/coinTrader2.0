import pandas as pd
import pytest

from crypto_bot.solana.sniper_solana import score_new_pool
from crypto_bot.solana.watcher import NewPoolEvent
from crypto_bot.solana.risk import RiskTracker
from crypto_bot.utils import pattern_logger


def make_event(**kw):
    base = dict(
        pool_address="P",
        token_mint="M",
        creator="C",
        liquidity=150.0,
        tx_count=5,
        freeze_authority="",
        mint_authority="",
        timestamp=0.0,
    )
    base.update(kw)
    return NewPoolEvent(**base)


def test_volume_spike_filter(tmp_path, monkeypatch):
    log = tmp_path / "log.csv"
    monkeypatch.setattr(pattern_logger, "LOG_FILE", log)
    log.write_text(
        "timestamp,pattern,strength\n2024-01-01,volume_spike,0.1\n2024-01-02,volume_spike,0.2\n"
    )

    tracker = RiskTracker(tmp_path / "r.json")

    event = make_event()
    cfg = {"safety": {}, "risk": {}, "scoring": {}}

    async def fake_gecko(*a, **k):
        return [[0, 1, 1, 1, 1, 10]], 0.0, 0.0

    monkeypatch.setattr(
        "crypto_bot.solana.sniper_solana.fetch_geckoterminal_ohlcv",
        fake_gecko,
    )
    monkeypatch.setattr(
        "crypto_bot.solana.sniper_solana.detect_patterns",
        lambda df: {"volume_spike": 0.5},
    )
    monkeypatch.setattr(
        "crypto_bot.solana.sniper_solana.classify_regime",
        lambda df: ("volatile", {"volatile": 0.8}),
    )

    score, direction = score_new_pool(event, cfg, tracker)
    assert direction == "long"
    assert score > 0

    monkeypatch.setattr(
        "crypto_bot.solana.sniper_solana.detect_patterns",
        lambda df: {"volume_spike": 0.1},
    )
    score2, direction2 = score_new_pool(event, cfg, tracker)
    assert (score2, direction2) == (0.0, "none")


def test_ml_confidence_filter(tmp_path, monkeypatch):
    log = tmp_path / "log.csv"
    log.write_text("timestamp,pattern,strength\n")
    monkeypatch.setattr(pattern_logger, "LOG_FILE", log)
    tracker = RiskTracker(tmp_path / "r.json")
    event = make_event()
    cfg = {"safety": {}, "risk": {}, "scoring": {}}

    async def fake_gecko2(*a, **k):
        return [[0, 1, 1, 1, 1, 10]], 0.0, 0.0

    monkeypatch.setattr(
        "crypto_bot.solana.sniper_solana.fetch_geckoterminal_ohlcv",
        fake_gecko2,
    )
    monkeypatch.setattr(
        "crypto_bot.solana.sniper_solana.detect_patterns",
        lambda df: {"volume_spike": 1.0},
    )
    # Low confidence -> skip
    monkeypatch.setattr(
        "crypto_bot.solana.sniper_solana.classify_regime",
        lambda df: ("volatile", {"volatile": 0.6}),
    )
    res = score_new_pool(event, cfg, tracker)
    assert res == (0.0, "none")

    # High confidence -> allowed
    monkeypatch.setattr(
        "crypto_bot.solana.sniper_solana.classify_regime",
        lambda df: ("volatile", {"volatile": 0.9}),
    )
    score, direction = score_new_pool(event, cfg, tracker)
    assert direction == "long"
    assert score > 0
