import asyncio
import types
import sys
import pytest

stub = types.ModuleType('solana_scanner')
async def get_solana_new_tokens(*args, **kwargs):
    return []
stub.get_solana_new_tokens = get_solana_new_tokens
def search_geckoterminal_token(*_a, **_k):
    return None
stub.search_geckoterminal_token = search_geckoterminal_token
sys.modules['crypto_bot.utils.solana_scanner'] = stub

from crypto_bot.main import initial_scan, SessionState

class DummyExchange:
    pass

@pytest.mark.asyncio
async def test_initial_scan_fetches_200_candles(monkeypatch):
    since_vals = []

    async def fake_update_multi(exchange, cache, batch, cfg, start_since=None, **kwargs):
        since_vals.append(start_since)
        return {}

    async def fake_update_regime(*args, **kwargs):
        return {}

    monkeypatch.setattr('crypto_bot.main.update_multi_tf_ohlcv_cache', fake_update_multi)
    monkeypatch.setattr('crypto_bot.main.update_regime_tf_cache', fake_update_regime)

    cfg = {'symbols': ['BTC/USD'], 'timeframes': ['1h'], 'scan_lookback_limit': 200}
    await initial_scan(DummyExchange(), cfg, SessionState())
    assert since_vals
    assert since_vals[0] is not None


@pytest.mark.asyncio
async def test_initial_scan_ws_disabled_and_limit_capped(monkeypatch):
    params = []

    async def fake_update_multi(exchange, cache, batch, cfg, start_since=None, **kwargs):
        params.append({**kwargs, 'start_since': start_since})
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
        'use_websocket': True,
    }
    await initial_scan(DummyExchange(), cfg, SessionState())

    assert params
    for kw in params:
        assert kw.get('use_websocket') is False
        assert kw.get('start_since') is not None
