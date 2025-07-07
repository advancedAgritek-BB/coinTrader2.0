import json

import pandas as pd
import pytest

from crypto_bot.signals.signal_fusion import SignalFusionEngine
from crypto_bot.signals import weight_optimizer as wo


def test_optimizer_updates_weights(tmp_path, monkeypatch):
    stats = {"a": {"pnl": 1.0}, "b": {"pnl": -0.5}}
    file = tmp_path / "stats.json"
    file.write_text(json.dumps(stats))
    monkeypatch.setattr(wo, "STATS_FILE", file)

    opt = wo.OnlineWeightOptimizer(learning_rate=0.5, min_weight=0.1)
    opt.weights = {"a": 1.0, "b": 1.0}
    opt.update()
    w = opt.get_weights()
    assert pytest.approx(w["a"], rel=1e-3) == 1.5 / 2.25
    assert pytest.approx(w["b"], rel=1e-3) == 0.75 / 2.25


def test_signal_fusion_with_optimizer(tmp_path, monkeypatch):
    stats = {"strat_a": {"pnl": 1.0}, "strat_b": {"pnl": -1.0}}
    file = tmp_path / "stats.json"
    file.write_text(json.dumps(stats))
    monkeypatch.setattr(wo, "STATS_FILE", file)

    opt = wo.OnlineWeightOptimizer(learning_rate=0.1)
    opt.weights = {"strat_a": 1.0, "strat_b": 1.0}

    def strat_a(df):
        return 0.8, "long"

    def strat_b(df):
        return 0.2, "long"

    engine = SignalFusionEngine([(strat_a, 0.5), (strat_b, 0.5)], weight_optimizer=opt)
    cfg = {"signal_weight_optimizer": {"enabled": True, "learning_rate": 0.1, "min_weight": 0.0}}
    df = pd.DataFrame({"close": [1, 2]})
    score, _ = engine.fuse(df, cfg)
    assert score == pytest.approx(0.53, rel=1e-2)


def test_optimizer_persists_weights(tmp_path, monkeypatch):
    stats = {"a": {"pnl": 1.0}}
    stats_file = tmp_path / "stats.json"
    weights_file = tmp_path / "weights.json"
    stats_file.write_text(json.dumps(stats))
    monkeypatch.setattr(wo, "STATS_FILE", stats_file)
    monkeypatch.setattr(wo, "WEIGHTS_FILE", weights_file)

    opt = wo.OnlineWeightOptimizer(learning_rate=0.5)
    opt.weights = {"a": 1.0}
    opt.update()
    assert weights_file.exists()

    opt2 = wo.OnlineWeightOptimizer(learning_rate=0.5)
    # weights loaded from file should match after init
    assert opt2.weights == opt.weights

