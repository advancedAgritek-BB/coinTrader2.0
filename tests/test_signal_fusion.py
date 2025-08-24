import pandas as pd
import pytest

from crypto_bot.signals.signal_fusion import SignalFusionEngine


def strat_a(df):
    return 0.8, "long"


def strat_b(df):
    return 0.2, "long"


def strat_long(df):
    return 0.5, "long"


def strat_short(df):
    return 0.4, "short"


def strat_short2(df):
    return 0.6, "short"


def strat_zero_long(df):
    return 0.0, "long"


def strat_zero_long2(df):
    return 0.0, "long"


def test_weighted_blending():
    df = pd.DataFrame({"close": [1, 2]})
    engine = SignalFusionEngine([(strat_a, 0.75), (strat_b, 0.25)])
    score, direction = engine.fuse(df)
    assert direction == "long"
    assert score == pytest.approx(0.65)


def test_direction_voting_majority():
    df = pd.DataFrame({"close": [1, 2]})
    engine = SignalFusionEngine([(strat_long, 1.0), (strat_short, 1.0), (strat_short2, 1.0)])
    _, direction = engine.fuse(df)
    assert direction == "short"


def test_direction_tie_resolved_by_score():
    df = pd.DataFrame({"close": [1, 2]})
    engine = SignalFusionEngine([(strat_long, 1.0), (strat_short, 0.5)])
    _, direction = engine.fuse(df)
    assert direction == "long"


def test_zero_score_directions_ignored():
    df = pd.DataFrame({"close": [1, 2]})
    engine = SignalFusionEngine(
        [
            (strat_zero_long, 1.0),
            (strat_zero_long2, 1.0),
            (strat_short, 1.0),
        ]
    )
    score, direction = engine.fuse(df)
    assert score == pytest.approx(0.4)
    assert direction == "short"

