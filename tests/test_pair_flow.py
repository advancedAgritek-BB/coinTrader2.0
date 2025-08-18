import asyncio
import ast
from pathlib import Path
from collections import deque
import pandas as pd
import numpy as np
import logging

from crypto_bot.phase_runner import BotContext


def load_funcs():
    src = Path('crypto_bot/main.py').read_text()
    module = ast.parse(src)
    funcs = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            'fetch_candidates',
            'update_caches',
            'analyse_batch',
        }:
            funcs[node.name] = ast.get_source_segment(src, node)

    ns = {
        'asyncio': asyncio,
        'pd': pd,
        'np': np,
        'time': __import__('time'),
        'logger': logging.getLogger('test'),
        'pipeline_logger': logging.getLogger('pipeline'),
        'deque': deque,
        'BotContext': BotContext,
        'symbol_priority_queue': deque(),
        'QUEUE_LOCK': asyncio.Lock(),
        'OHLCV_LOCK': asyncio.Lock(),
        'is_market_pumping': lambda *a, **k: False,
        'get_filtered_symbols': lambda ex, cfg: asyncio.sleep(0, result=([(s, 1.0) for s in cfg.get('symbols', [])], [])),
        'compute_average_atr': lambda *a, **k: 0.01,
        'calc_atr': lambda df, window=14: 0.01,
        'get_market_regime': lambda ctx: 'unknown',
        'scan_cex_arbitrage': lambda *a, **k: [],
        'scan_arbitrage': lambda *a, **k: [],
        'scan_and_enqueue_solana_tokens': lambda *a, **k: [],
        'build_priority_queue': lambda scores: deque([s for s, _ in scores]),
        'enqueue_solana_tokens': lambda tokens, mark_new=False: None,
        'recent_solana_set': set(),
        'recent_solana_tokens': deque(),
        'no_data_symbols': set(),
        'symbol_cache_guard': lambda: type(
            'G',
            (),
            {
                '__aenter__': lambda self: asyncio.sleep(0),
                '__aexit__': lambda self, exc_type, exc, tb: asyncio.sleep(0),
            },
        )(),
        '_update_caches_impl': lambda ctx: asyncio.sleep(0),
        '_analyse_batch_impl': lambda ctx: asyncio.sleep(0),
        'TOTAL_ANALYSES': 0,
        'UNKNOWN_COUNT': 0,
    }

    preload_df = pd.DataFrame({
        'timestamp': [1, 2],
        'open': [1, 1],
        'high': [1, 1],
        'low': [1, 1],
        'close': [1, 1],
        'volume': [1, 1],
    })

    async def fake_update_multi(*args, **kwargs):
        batch = args[2]
        return {'1h': {sym: preload_df for sym in batch}}

    async def fake_update_regime(*args, **kwargs):
        return {}

    ns['update_multi_tf_ohlcv_cache'] = fake_update_multi
    ns['update_regime_tf_cache'] = fake_update_regime
    async def fake_analyze_symbol(sym, df_map, mode, config, notifier, **kw):
        return {'symbol': sym, 'regime': 'bull', 'score': 1.0}

    ns['analyze_symbol'] = fake_analyze_symbol

    for name, code in funcs.items():
        exec(code, ns)
    return ns['fetch_candidates'], ns['update_caches'], ns['analyse_batch']


fetch_candidates, update_caches, analyse_batch = load_funcs()


def test_pair_flow():
    df = pd.DataFrame({'high': [1, 2], 'low': [0, 1], 'close': [1, 2]})

    class DummyExchange:
        async def watchOHLCV(self, symbol, timeframe="1h", **kwargs):
            return []

    ctx = BotContext(
        positions={},
        df_cache={'1h': {'BTC/USD': df, 'ETH/USD': df}},
        regime_cache={},
        config={'timeframe': '1h', 'symbols': ['BTC/USD', 'ETH/USD'], 'symbol_batch_size': 5},
    )
    ctx.exchange = DummyExchange()

    asyncio.run(fetch_candidates(ctx))
    asyncio.run(update_caches(ctx))
    asyncio.run(analyse_batch(ctx))

    symbols = {res['symbol'] for res in ctx.analysis_results}
    assert {'BTC/USD', 'ETH/USD'} <= symbols
