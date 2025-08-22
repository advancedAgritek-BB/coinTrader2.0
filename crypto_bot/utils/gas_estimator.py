"""Gas fee estimation helpers."""

from __future__ import annotations

import os
from typing import Optional

import requests
from crypto_bot.utils.market_loader import get_http_session

LAMPORTS_PER_SOL = 1_000_000_000
DEFAULT_SOL_FEE_URL = "https://quote-api.jup.ag/v6/estimate"


def _fetch_solana_fee_lamports(url: str = DEFAULT_SOL_FEE_URL) -> float:
    """Return estimated swap fee in lamports.

    The value may be overridden by the ``MOCK_SOLANA_FEE_LAMPORTS``
    environment variable for testing purposes. When the network request
    fails a fee of ``0.0`` is returned.
    """
    mock = os.getenv("MOCK_SOLANA_FEE_LAMPORTS")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0
    try:
        session = get_http_session()
        resp = session.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return float(data.get("fee", 0.0))
    except Exception:
        pass
    return 0.0


def _fetch_eth_gas_price_wei(web3: Optional[object] = None) -> float:
    """Return the current Ethereum gas price in wei.

    If ``web3`` is provided ``web3.eth.gas_price`` is used. Otherwise the
    ``MOCK_ETH_GAS_PRICE_WEI`` environment variable is checked. Failure to
    obtain a price returns ``0.0``.
    """
    if web3 is not None:
        try:
            return float(web3.eth.gas_price)
        except Exception:
            return 0.0
    mock = os.getenv("MOCK_ETH_GAS_PRICE_WEI")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0
    return 0.0


def fetch_priority_fee_gwei(web3: Optional[object] = None) -> float:
    """Return the current Ethereum priority fee in gwei.

    When ``web3`` is supplied the value is retrieved from
    ``web3.eth.max_priority_fee``. Otherwise the
    ``MOCK_ETH_PRIORITY_FEE_GWEI`` environment variable is checked. If
    neither source is available a fee of ``0.0`` is returned.
    """
    if web3 is not None:
        try:
            wei = web3.eth.max_priority_fee
            return float(wei) / 1_000_000_000
        except Exception:
            return 0.0
    mock = os.getenv("MOCK_ETH_PRIORITY_FEE_GWEI")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0
    return 0.0


def estimate_gas_fee_usd(
    chain: str,
    gas_units: int,
    token_price_usd: float,
    web3: Optional[object] = None,
) -> float:
    """Estimate the USD value of gas for ``chain``.

    ``gas_units`` represents the approximate number of gas units the
    transaction will consume. ``token_price_usd`` should be the current
    price of the chain's native token in USD.
    """
    if gas_units <= 0 or token_price_usd <= 0:
        return 0.0
    if chain.lower() == "solana":
        lamports = _fetch_solana_fee_lamports()
        return lamports / LAMPORTS_PER_SOL * token_price_usd
    if chain.lower() == "ethereum":
        wei = _fetch_eth_gas_price_wei(web3)
        return wei * gas_units / 1_000_000_000_000_000_000 * token_price_usd
    return 0.0


def gas_fee_too_high(
    chain: str,
    trade_amount_usd: float,
    gas_units: int,
    limit_pct: float,
    token_price_usd: float,
    web3: Optional[object] = None,
) -> bool:
    """Return ``True`` if the estimated gas fee exceeds ``limit_pct`` of the trade.

    ``trade_amount_usd`` is the notional value of the trade in USD.
    ``limit_pct`` is expressed as a percentage (e.g. ``0.5`` for 0.5%).
    """
    fee_usd = estimate_gas_fee_usd(chain, gas_units, token_price_usd, web3)
    if trade_amount_usd <= 0:
        return False
    return fee_usd > trade_amount_usd * (limit_pct / 100)

