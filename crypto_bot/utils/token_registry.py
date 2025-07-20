"""Mapping of token symbols to Solana mint addresses."""

TOKEN_MINTS: dict[str, str] = {
    "BTC": "So11111111111111111111111111111111111111112",
    "ETH": "2NdXGW7dpwye9Heq7qL3gFYYUUDewfxCUUDq36zzfrqD",
    "USDC": "EPjFWdd5AufqSSqeM2q6ksjLpaEweidnGj9n92gtQgNf",
    "SOL": "So11111111111111111111111111111111111111112",
}


def set_token_mints(mapping: dict[str, str]) -> None:
    """Replace ``TOKEN_MINTS`` with ``mapping`` after normalizing keys."""
    TOKEN_MINTS.clear()
    TOKEN_MINTS.update({k.upper(): v for k, v in mapping.items()})

