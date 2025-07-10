import pytest

from crypto_bot.solana.sniper_solana import score_new_pool
from crypto_bot.solana.watcher import NewPoolEvent
from crypto_bot.solana.risk import RiskTracker


def make_event(**kwargs):
    base = dict(
        pool_address="P",
        token_mint="M",
        creator="C",
        liquidity=50.0,
        tx_count=5,
        freeze_authority="",
        mint_authority="",
        timestamp=0.0,
    )
    base.update(kwargs)
    return NewPoolEvent(**base)


def test_scoring_with_sentiment(tmp_path):
    tracker = RiskTracker(tmp_path / "risk.json")
    event = make_event()
    cfg = {
        "scoring": {"weight_liquidity": 0.1, "weight_tx": 1.0, "twitter_weight": 0.1},
        "safety": {"min_liquidity": 10},
        "risk": {"max_concurrent": 1},
        "twitter_score": 10,
    }
    score, direction = score_new_pool(event, cfg, tracker)
    assert direction == "long"
    assert score == pytest.approx(11.0)


def test_safety_blocks_pool(tmp_path):
    tracker = RiskTracker(tmp_path / "risk.json")
    event = make_event(liquidity=5)
    cfg = {"safety": {"min_liquidity": 10}, "risk": {"max_concurrent": 1}}
    assert score_new_pool(event, cfg, tracker) == (0.0, "none")


def test_risk_blocks_pool(tmp_path):
    tracker = RiskTracker(tmp_path / "risk.json")
    tracker.add_snipe("M", 1)
    event = make_event(token_mint="N")
    cfg = {"risk": {"max_concurrent": 1}, "safety": {"min_liquidity": 10}}
    assert score_new_pool(event, cfg, tracker) == (0.0, "none")
