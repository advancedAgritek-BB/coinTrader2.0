from typing import Dict, List
import json
import os
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

SUPPORTED_FUNDING = ["BTC", "ETH", "XRP"]
REQUIRED_TOKENS = ["USDC", "SOL"]

def check_wallet_balances(wallet_address: str) -> Dict[str, float]:
    """Return token balances for the given wallet.

    This is a simplified placeholder that checks the environment variable
    ``FAKE_BALANCES`` for test data. In a production environment this
    would query on-chain or exchange APIs.
    """
    env_balances = os.getenv("FAKE_BALANCES")
    if env_balances:
        try:
            return json.loads(env_balances)
        except json.JSONDecodeError:
            logger.error("Invalid FAKE_BALANCES JSON")
    return {}

def detect_non_trade_tokens(balances: Dict[str, float]) -> List[str]:
    """Return list of tokens that should be converted for trading."""
    non_trade = []
    for token, amount in balances.items():
        if token in SUPPORTED_FUNDING and token not in REQUIRED_TOKENS and amount > 0:
            non_trade.append(token)
    return non_trade

def auto_convert_funds(
    wallet: str,
    from_token: str,
    to_token: str,
    amount: float,
    dry_run: bool = True,
) -> Dict:
    """Convert funds using the Solana Jupiter aggregator."""

    logger.info("Converting %s %s to %s", amount, from_token, to_token)

    if dry_run:
        tx_hash = "DRYRUN"
    else:
        from crypto_bot.execution.solana_executor import execute_swap

        try:
            result = execute_swap(from_token, to_token, amount, "", "", dry_run=False)
            tx_hash = result["tx_hash"]
        except Exception as e:
            logger.error("Conversion failed: %s", e)
            tx_hash = "ERROR"

    result = {
        "wallet": wallet,
        "from": from_token,
        "to": to_token,
        "amount": amount,
        "tx_hash": tx_hash,
    }

    logger.info("Conversion result: %s", result)
    return result
