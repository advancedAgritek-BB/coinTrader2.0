import asyncio
import logging

from crypto_bot.utils import market_loader


def test_update_multi_tf_ohlcv_cache_clamps(monkeypatch, caplog, tmp_path):
    async def fake_load(exchange, symbol, timeframe="1m", limit=100, mode="rest", since=None, **kwargs):
        return [[(since or 0) + i * 60000, 0, 0, 0, 0, 0] for i in range(limit)]

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_load)
    async def _listing(_s):
        return 0
    monkeypatch.setattr(market_loader, "get_kraken_listing_date", _listing)
    async def _split(exchange, syms):
        return syms, syms
    monkeypatch.setattr(market_loader, "split_symbols_by_timeframe", _split)
    now_ms = 1_000_000_000_000
    monkeypatch.setattr(market_loader, "utc_now_ms", lambda: now_ms)

    class Ex:
        id = "dummy"
        timeframes = {"1m": "1m", "5m": "5m"}
        symbols = ["BTC/USD"]

    cfg = {
        "timeframes": ["1m", "5m"],
        "backfill_days": {"1m": 2, "5m": 3},
        "warmup_candles": {"1m": 2000, "5m": 2000},
    }
    import crypto_bot.main as main
    monkeypatch.setattr(main, "update_df_cache", lambda cache, tf, sym, df: cache.setdefault(tf, {}).update({sym: df}))
    monkeypatch.setattr(market_loader, "BOOTSTRAP_STATE_FILE", tmp_path / "state.json")

    log_file = market_loader.LOG_DIR / "bot.log"
    before = log_file.read_text() if log_file.exists() else ""
    asyncio.run(
        market_loader.update_multi_tf_ohlcv_cache(
            Ex(),
            {},
            ["BTC/USD"],
            cfg,
            limit=5000,
            start_since=0,
        )
    )
    log_text = log_file.read_text()[len(before):]
    assert "starting from" in log_text
    assert "dropping start_since" not in log_text
    assert "Clamping warmup candles for 1m" in log_text
    assert "Clamping warmup candles for 5m" in log_text
