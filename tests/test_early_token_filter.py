import asyncio

import pytest

from crypto_bot.early_token_filter import assess_early_token


async def _run(symbol, mint, cfg):
    return await assess_early_token(symbol, mint, cfg)


def test_assess_early_token_scoring(monkeypatch):
    async def fake_sentiment(query: str):
        assert query == "AAA"
        return 80

    async def fake_gecko(pair: str, limit: int = 5):
        assert pair == "AAA/USDC"
        return [[0, 1, 1, 1, 1, 20_000]]

    def fake_load_model(name: str):
        assert name == "xrpusd_regime_lgbm"
        return object()

    def fake_predict_regime(df, model):
        return {"volatile_pump": 0.5}

    monkeypatch.setattr(
        "crypto_bot.early_token_filter.fetch_twitter_sentiment_async", fake_sentiment
    )
    monkeypatch.setattr(
        "crypto_bot.early_token_filter.fetch_geckoterminal_ohlcv", fake_gecko
    )
    monkeypatch.setattr("crypto_bot.early_token_filter.load_model", fake_load_model)
    monkeypatch.setattr("crypto_bot.early_token_filter.predict_regime", fake_predict_regime)

    score = asyncio.run(_run("AAA", "mint", {}))

    expected = (80 / 100) * 0.3 + 0.2 + 0.5 * 0.4
    assert score == pytest.approx(expected)


def test_assess_early_token_rejects_onchain(monkeypatch):
    class DummyResp:
        ok = True

        def json(self):
            return {
                "mint": {
                    "dev_holding": 30,
                    "total_supply": 100,
                    "initial_liquidity_usd": 10_000,
                }
            }

    def fake_get(url, timeout=10):
        return DummyResp()

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr(
        "crypto_bot.early_token_filter.fetch_twitter_sentiment_async",
        lambda *a, **k: 50,
    )
    monkeypatch.setattr(
        "crypto_bot.early_token_filter.fetch_geckoterminal_ohlcv",
        lambda *a, **k: [[0, 1, 1, 1, 1, 0]],
    )
    monkeypatch.setattr("crypto_bot.early_token_filter.load_model", lambda *a, **k: None)
    monkeypatch.setattr(
        "crypto_bot.early_token_filter.predict_regime", lambda df, model: {"volatile_pump": 0}
    )

    score = asyncio.run(_run("AAA", "mint", {"helius_api_key": "k"}))
    assert score == 0.0
