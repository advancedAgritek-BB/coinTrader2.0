import asyncio
import pytest

from crypto_bot.utils import market_loader


def test_update_multi_tf_ohlcv_cache_logs_progress(monkeypatch, caplog):
    """Capture progress lines emitted during a simulated bootstrap."""

    async def fake_update_multi(exchange, cache, symbols, config, **kwargs):
        for i, sym in enumerate(symbols, 1):
            market_loader.logger.info("bootstrap %d/%d %s", i, len(symbols), sym)
        return cache

    logs: list[str] = []

    def capture(msg, *args):
        logs.append(msg % args)

    monkeypatch.setattr(market_loader, "update_multi_tf_ohlcv_cache", fake_update_multi)
    monkeypatch.setattr(market_loader.logger, "info", capture)

    asyncio.run(
        market_loader.update_multi_tf_ohlcv_cache(
            None, {}, ["BTC/USD", "ETH/USD"], {"timeframes": ["1h"]}
        )
    )

    assert "bootstrap 1/2 BTC/USD" in logs
    assert "bootstrap 2/2 ETH/USD" in logs


def test_update_multi_tf_ohlcv_cache_resume(monkeypatch, progress_path, caplog):
    """Simulate resuming a bootstrap using a progress file."""

    async def fake_update_multi(exchange, cache, symbols, config, **kwargs):
        existing = set()
        if progress_path.exists():
            existing = set(progress_path.read_text().splitlines())
        for sym in symbols:
            if sym in existing:
                market_loader.logger.info("skip %s", sym)
            else:
                market_loader.logger.info("process %s", sym)
                with progress_path.open("a") as fh:
                    fh.write(f"{sym}\n")
        return cache

    monkeypatch.setattr(market_loader, "update_multi_tf_ohlcv_cache", fake_update_multi)
    caplog.set_level("INFO")

    # Pre-populate with one symbol already processed
    progress_path.write_text("BTC/USD\n")

    asyncio.run(
        market_loader.update_multi_tf_ohlcv_cache(
            None, {}, ["BTC/USD", "ETH/USD"], {"timeframes": ["1h"]}
        )
    )

    assert "skip BTC/USD" in caplog.text
    assert "process ETH/USD" in caplog.text

    caplog.clear()

    # Second run should skip both symbols
    asyncio.run(
        market_loader.update_multi_tf_ohlcv_cache(
            None, {}, ["BTC/USD", "ETH/USD"], {"timeframes": ["1h"]}
        )
    )

    assert caplog.text.count("skip BTC/USD") == 1
    assert "skip ETH/USD" in caplog.text
