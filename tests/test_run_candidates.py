import pytest
pytest.importorskip("pandas")
import asyncio
import pandas as pd
import crypto_bot.utils.market_analyzer as ma


def strat_a(df, cfg=None):
    return 0.4, "long"


def strat_b(df, cfg=None):
    return 0.5, "short"


def strat_c(df, cfg=None):
    return 0.2, "long"


def test_run_candidates_ranking(monkeypatch):
    df = pd.DataFrame({"close": [1, 2]})
    scores = {strat_a: (0.4, "long"), strat_b: (0.5, "short"), strat_c: (0.2, "long")}

    async def fake_eval(strats, df_, cfg_):
        strat = strats[0]
        score, direction = scores[strat]
        return [(score, direction, None)]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    edges = {"strat_a": 0.5, "strat_b": 1.0, "strat_c": 2.0}
    monkeypatch.setattr(ma.perf, "edge", lambda name, sym: edges[name])

    res = asyncio.run(ma.run_candidates(df, [strat_a, strat_b, strat_c], "AAA", {}))
    names = [fn.__name__ for fn, _s, _d in res]
    assert names == ["strat_b", "strat_c", "strat_a"]

