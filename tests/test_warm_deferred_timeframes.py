import pytest

from crypto_bot.main import warm_deferred_timeframes, SessionState


class DummyExchange:
    pass


@pytest.mark.asyncio
async def test_warm_deferred_timeframes_preserves_symbol_count(monkeypatch):
    batches = []

    async def fake_update_multi(exchange, cache, batch, cfg, **kwargs):
        batches.append(list(batch))
        return {}

    async def fake_update_regime(*args, **kwargs):
        return {}

    monkeypatch.setattr(
        "crypto_bot.main.update_multi_tf_ohlcv_cache", fake_update_multi
    )
    monkeypatch.setattr(
        "crypto_bot.main.update_regime_tf_cache", fake_update_regime
    )

    cfg = {
        "ohlcv": {"defer_timeframes": ["4h"]},
        "scan_lookback_limit": 50,
        "symbol_batch_size": 100,
        "regime_timeframes": ["4h"],
    }
    symbols = [f"S{i}/USD" for i in range(60)]
    await warm_deferred_timeframes(DummyExchange(), cfg, SessionState(), symbols)
    assert len(symbols) == 60
    assert batches and len(batches[0]) == 60


@pytest.mark.asyncio
async def test_warm_deferred_timeframes_respects_strategy_lookback(monkeypatch):
    limits: list[int] = []

    async def fake_update_multi(exchange, cache, batch, cfg, limit, **kwargs):
        limits.append(limit)
        return {}

    monkeypatch.setattr(
        "crypto_bot.main.update_multi_tf_ohlcv_cache", fake_update_multi
    )
    monkeypatch.setattr(
        "crypto_bot.main.update_regime_tf_cache", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        "crypto_bot.strategy.registry.load_from_config", lambda cfg: []
    )
    monkeypatch.setattr(
        "crypto_bot.strategy.registry.compute_required_lookback_per_tf",
        lambda _strategies: {"1h": 75},
    )

    cfg = {
        "ohlcv": {"defer_timeframes": ["1h"]},
        "scan_lookback_limit": 50,
    }
    symbols = ["BTC/USDT"]
    await warm_deferred_timeframes(DummyExchange(), cfg, SessionState(), symbols)
    assert limits and limits[0] >= 75
