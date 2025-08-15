import httpx
import pytest

from crypto_bot.solana.raydium_client import RaydiumClient
from crypto_bot.solana.pump_fun_client import PumpFunClient


def test_raydium_get_pairs_fallback_and_normalization():
    """RaydiumClient should try multiple endpoints and normalize data."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "v2/main/pairs" in url:
            return httpx.Response(500)
        if "api.raydium.io/pairs" in url:
            data = {
                "data": [
                    {
                        "ammId": "pool1",
                        "baseMint": "BASE",
                        "quoteMint": "USDC",
                        "liquidityUsd": "123.4",
                        "volume24hUsd": "45.6",
                        "price": "1.0",
                    }
                ]
            }
            return httpx.Response(200, json=data)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = RaydiumClient(client=httpx.Client(transport=transport))
    try:
        pairs = client.get_pairs()
    finally:
        client.close()
    assert pairs == [
        {
            "address": "pool1",
            "baseMint": "BASE",
            "quoteMint": "USDC",
            "liquidityUsd": 123.4,
            "volume24hUsd": 45.6,
            "price": 1.0,
        }
    ]


def test_raydium_best_pool_for_mint(monkeypatch):
    """Select the pool with highest volume and liquidity above threshold."""

    client = RaydiumClient(client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))))
    monkeypatch.setattr(
        client,
        "get_pairs",
        lambda: [
            {
                "address": "a",
                "baseMint": "TOKEN",
                "quoteMint": "USDC",
                "liquidityUsd": 3000,
                "volume24hUsd": 100,
            },
            {
                "address": "b",
                "baseMint": "USDC",
                "quoteMint": "TOKEN",
                "liquidityUsd": 6000,
                "volume24hUsd": 200,
            },
            {
                "address": "c",
                "baseMint": "USDC",
                "quoteMint": "TOKEN",
                "liquidityUsd": 5000,
                "volume24hUsd": 50,
            },
        ],
    )
    try:
        best = client.best_pool_for_mint("TOKEN", min_liquidity_usd=5000)
    finally:
        client.close()
    assert best["address"] == "b"


@pytest.mark.asyncio
async def test_pump_fun_trending(monkeypatch):
    """PumpFunClient.trending should parse list from wrapped result."""

    data = {"data": [{"mint": "abc"}]}

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = PumpFunClient()
    client._c = httpx.AsyncClient(transport=transport)
    try:
        items = await client.trending()
    finally:
        await client.aclose()
    assert items == data["data"]
