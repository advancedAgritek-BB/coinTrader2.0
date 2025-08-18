import asyncio
import types
import sys
import pytest

stub = types.ModuleType('solana_scanner')

async def get_solana_new_tokens(*args, **kwargs):
    return []

async def search_geckoterminal_token(*args, **kwargs):
    return None

stub.get_solana_new_tokens = get_solana_new_tokens
stub.search_geckoterminal_token = search_geckoterminal_token
sys.modules['crypto_bot.utils.solana_scanner'] = stub

from crypto_bot.main import initial_scan, SessionState

class DummyExchange:
    pass

@pytest.mark.asyncio
async def test_initial_scan_fetches_200_candles(monkeypatch):
    since_vals = []
    start_vals = []
    limits = []
    frames = []

    async def fake_update_multi(exchange, cache, batch, cfg, limit=0, start_since=None, **kwargs):
        limits.append(limit)
        since_vals.append(start_since)
        start_vals.append(start_since)
        frames.append(cfg.get("timeframes"))
        return {}

    async def fake_update_regime(*args, **kwargs):
        return {}

    monkeypatch.setattr('crypto_bot.main.update_multi_tf_ohlcv_cache', fake_update_multi)

    monkeypatch.setattr('crypto_bot.main.update_regime_tf_cache', fake_update_regime)
    async def fake_get_filtered_symbols(ex, cfg):
        return ([(cfg['symbols'][0], 0.0)], cfg.get('onchain_symbols', []))

    monkeypatch.setattr(
        'crypto_bot.main.get_filtered_symbols',
        fake_get_filtered_symbols,
    )

    cfg = {
        'symbols': ['BTC/USD'],
        'timeframes': ['1h'],
        'scan_lookback_limit': 200,
        'ohlcv': {'bootstrap_timeframes': ['1h']},
    }
    await initial_scan(DummyExchange(), cfg, SessionState())
    assert since_vals
    assert since_vals[0] is not None
    assert limits and limits[0] == 2000
    assert start_vals and isinstance(start_vals[0], int)
    assert frames[0] == ['1h']


@pytest.mark.asyncio
async def test_initial_scan_ws_disabled_and_limit_capped(monkeypatch):
    params = []
    frames = []

    async def fake_update_multi(exchange, cache, batch, cfg, start_since=None, **kwargs):
        params.append({**kwargs, 'start_since': start_since})
        params.append(kwargs)
        frames.append(cfg.get('timeframes'))
        return {}

    async def fake_update_regime(exchange, cache, batch, cfg, **kwargs):
        params.append(kwargs)
        return {}

    monkeypatch.setattr('crypto_bot.main.update_multi_tf_ohlcv_cache', fake_update_multi)
    monkeypatch.setattr('crypto_bot.main.update_regime_tf_cache', fake_update_regime)
    async def fake_get_filtered_symbols(ex, cfg):
        return ([(cfg['symbols'][0], 0.0)], cfg.get('onchain_symbols', []))

    monkeypatch.setattr(
        'crypto_bot.main.get_filtered_symbols',
        fake_get_filtered_symbols,
    )

    cfg = {
        'symbols': ['BTC/USD'],
        'timeframes': ['1h'],
        'scan_lookback_limit': 1000,
        'scan_deep_limit': 700,
        'use_websocket': True,
        'ohlcv': {'bootstrap_timeframes': ['1h']},
    }
    await initial_scan(DummyExchange(), cfg, SessionState())

    assert params
    for kw in params:
        assert kw.get('use_websocket') is False
    assert params[0].get('limit') == 700
    assert params[1].get('limit') == 700
    assert params[-1].get('limit') == 1000
    assert params[0].get('start_since') is not None
    assert frames[0] == ['1h']


@pytest.mark.asyncio
async def test_initial_scan_onchain(monkeypatch):
    updates = []

    def fake_update(cache, tf, sym, df, *args, **kwargs):
        updates.append((tf, sym, df))

    frames = []

    async def fake_update_multi(exchange, cache, batch, cfg, **k):
        frames.append(cfg.get('timeframes'))
        return {}

    async def fake_update_regime(*a, **k):
        return {}

    monkeypatch.setattr('crypto_bot.main.update_multi_tf_ohlcv_cache', fake_update_multi)
    monkeypatch.setattr('crypto_bot.main.update_regime_tf_cache', fake_update_regime)
    monkeypatch.setattr('crypto_bot.main.update_df_cache', fake_update)
    async def fake_get_filtered_symbols(ex, cfg):
        return ([(cfg['symbols'][0], 0.0)], cfg.get('onchain_symbols', []))

    monkeypatch.setattr(
        'crypto_bot.main.get_filtered_symbols',
        fake_get_filtered_symbols,
    )

    cfg = {
        'symbols': ['BTC/USD'],
        'onchain_symbols': ['SOL/USDC'],
        'timeframes': ['1h', '5m'],
        'scan_lookback_limit': 100,
        'ohlcv': {'bootstrap_timeframes': ['1h']},
    }
    await initial_scan(DummyExchange(), cfg, SessionState())

    assert updates == []
    assert frames and frames[0] == ['1h']


@pytest.mark.asyncio
async def test_initial_scan_symbol_filter_overrides(monkeypatch):
    cfg_params = []

    async def fake_update_multi(exchange, cache, batch, cfg, limit=0, **kwargs):
        cfg_params.append((cfg, limit))
        return {}

    async def fake_update_regime(exchange, cache, batch, cfg, limit=0, **kwargs):
        cfg_params.append((cfg, limit))
        return {}

    monkeypatch.setattr('crypto_bot.main.update_multi_tf_ohlcv_cache', fake_update_multi)
    monkeypatch.setattr('crypto_bot.main.update_regime_tf_cache', fake_update_regime)
    async def fake_get_filtered_symbols(ex, cfg):
        return ([(cfg['symbols'][0], 0.0)], cfg.get('onchain_symbols', []))

    monkeypatch.setattr(
        'crypto_bot.main.get_filtered_symbols',
        fake_get_filtered_symbols,
    )

    cfg = {
        'symbols': ['BTC/USD'],
        'timeframes': ['1h'],
        'scan_lookback_limit': 100,
        'symbol_filter': {
            'initial_timeframes': ['5m'],
            'initial_history_candles': 25,
        },
        'ohlcv': {'bootstrap_timeframes': ['5m']},
    }

    await initial_scan(DummyExchange(), cfg, SessionState())

    assert cfg_params
    for c, limit in cfg_params:
        assert set(c['timeframes']) == {'1m', '5m'}
        assert limit == 25


@pytest.mark.asyncio
async def test_initial_scan_filters_tradable_symbols(monkeypatch):
    captured_batches: list[list[str]] = []

    async def fake_update_multi(exchange, cache, batch, cfg, **kwargs):
        captured_batches.append(batch)
        return {}

    async def fake_update_regime(*_args, **_kwargs):
        return {}

    tradable = ['BTC/USD', 'ETH/USD']

    async def fake_get_filtered_symbols(ex, cfg):
        return ([(s, 0.0) for s in tradable], [])

    monkeypatch.setattr(
        'crypto_bot.main.update_multi_tf_ohlcv_cache',
        fake_update_multi,
    )
    monkeypatch.setattr(
        'crypto_bot.main.update_regime_tf_cache',
        fake_update_regime,
    )
    monkeypatch.setattr(
async def test_initial_scan_warms_deferred_timeframes(monkeypatch):
    calls = []

    async def fake_update_multi(exchange, cache, batch, cfg, limit=0, **kwargs):
        calls.append(list(cfg.get('timeframes', [])))
        return {}

    async def fake_update_regime(*args, **kwargs):
        return {}

    monkeypatch.setattr('crypto_bot.main.update_multi_tf_ohlcv_cache', fake_update_multi)
    monkeypatch.setattr('crypto_bot.main.update_regime_tf_cache', fake_update_regime)

    async def fake_get_filtered_symbols(ex, cfg):
        return ([(cfg['symbols'][0], 0.0)], [])

    monkeypatch.setattr(
        'crypto_bot.main.get_filtered_symbols',
        fake_get_filtered_symbols,
    )

    cfg = {
        'symbols': tradable + ['DOGE/USD'],
        'timeframes': ['1h'],
        'scan_lookback_limit': 50,
        'symbol_batch_size': 10,
    }

    await initial_scan(DummyExchange(), cfg, SessionState())

    assert captured_batches == [tradable]
        'symbols': ['BTC/USD'],
        'ohlcv': {
            'bootstrap_timeframes': ['1h'],
            'defer_timeframes': ['4h'],
        },
        'scan_lookback_limit': 100,
    }

    await initial_scan(DummyExchange(), cfg, SessionState())
    await asyncio.sleep(0)

    assert calls and set(calls[0]) == {'1m', '5m', '1h'}
    assert any(set(call) == {'4h'} for call in calls[1:])
