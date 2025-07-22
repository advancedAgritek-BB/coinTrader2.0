import types
import sys
import pandas as pd
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

    async def fake_update_multi(exchange, cache, batch, cfg, limit=0, start_since=None, **kwargs):
        limits.append(limit)
        since_vals.append(start_since)
        start_vals.append(start_since)
        return {}

    async def fake_update_regime(*args, **kwargs):
        return {}

    monkeypatch.setattr('crypto_bot.main.update_multi_tf_ohlcv_cache', fake_update_multi)
    monkeypatch.setattr('crypto_bot.main.update_regime_tf_cache', fake_update_regime)

    cfg = {'symbols': ['BTC/USD'], 'timeframes': ['1h'], 'scan_lookback_limit': 200}
    await initial_scan(DummyExchange(), cfg, SessionState())
    assert since_vals
    assert since_vals[0] is not None
    assert limits and limits[0] == 2000
    assert start_vals and isinstance(start_vals[0], int)


@pytest.mark.asyncio
async def test_initial_scan_ws_disabled_and_limit_capped(monkeypatch):
    params = []

    async def fake_update_multi(exchange, cache, batch, cfg, start_since=None, **kwargs):
        params.append({**kwargs, 'start_since': start_since})
        params.append(kwargs)
        return {}

    async def fake_update_regime(exchange, cache, batch, cfg, **kwargs):
        params.append(kwargs)
        return {}

    monkeypatch.setattr('crypto_bot.main.update_multi_tf_ohlcv_cache', fake_update_multi)
    monkeypatch.setattr('crypto_bot.main.update_regime_tf_cache', fake_update_regime)

    cfg = {
        'symbols': ['BTC/USD'],
        'timeframes': ['1h'],
        'scan_lookback_limit': 1000,
        'scan_deep_limit': 700,
        'use_websocket': True,
    }
    await initial_scan(DummyExchange(), cfg, SessionState())

    assert params
    for kw in params:
        assert kw.get('use_websocket') is False
        assert kw.get('limit') == 700
    assert params[0].get('start_since') is not None


@pytest.mark.asyncio
async def test_initial_scan_onchain(monkeypatch):
    calls = []

    async def fake_fetch(symbol, timeframe="1h", limit=0):
        calls.append((symbol, timeframe, limit))
        return pd.DataFrame({"timestamp": [1], "open": [0], "high": [0], "low": [0], "close": [0], "volume": [0]})

    updates = []

    def fake_update(cache, tf, sym, df, *args, **kwargs):
        updates.append((tf, sym, df))

    async def fake_update_multi(*a, **k):
        return {}

    async def fake_update_regime(*a, **k):
        return {}

    monkeypatch.setattr('crypto_bot.main.update_multi_tf_ohlcv_cache', fake_update_multi)
    monkeypatch.setattr('crypto_bot.main.update_regime_tf_cache', fake_update_regime)
    monkeypatch.setattr('crypto_bot.solana.fetch_solana_historical', fake_fetch, raising=False)
    monkeypatch.setattr('crypto_bot.main.update_df_cache', fake_update)

    cfg = {
        'symbols': ['BTC/USD'],
        'onchain_symbols': ['SOL/USDC'],
        'timeframes': ['1h', '5m'],
        'scan_lookback_limit': 100,
    }
    await initial_scan(DummyExchange(), cfg, SessionState())

    assert set(calls) == {('SOL/USDC', '1h', 100), ('SOL/USDC', '5m', 100)}
    assert [(c[0], c[1]) for c in updates] == [('1h', 'SOL/USDC'), ('5m', 'SOL/USDC')]


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

    cfg = {
        'symbols': ['BTC/USD'],
        'timeframes': ['1h'],
        'scan_lookback_limit': 100,
        'symbol_filter': {
            'initial_timeframes': ['5m'],
            'initial_history_candles': 25,
        },
    }

    await initial_scan(DummyExchange(), cfg, SessionState())

    assert cfg_params
    for c, limit in cfg_params:
        assert c['timeframes'] == ['5m']
        assert limit == 25
