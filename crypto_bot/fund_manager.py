from typing import Dict, List
import json
import os
from pathlib import Path

try:
    from solana.rpc.api import Client  # type: ignore
    from solana.publickey import PublicKey  # type: ignore
    from solana.rpc.types import TokenAccountOpts  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    Client = None  # type: ignore
    PublicKey = None  # type: ignore
    TokenAccountOpts = None  # type: ignore

from crypto_bot.execution.solana_executor import execute_swap
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.telegram import TelegramNotifier


logger = setup_logger(__name__, LOG_DIR / "fund_manager.log")

SUPPORTED_FUNDING = ["BTC", "ETH", "XRP"]
REQUIRED_TOKENS = ["USDC", "SOL"]

# Map common symbols to Solana token mints used by Jupiter
TOKEN_MINTS = {
    "BTC": "So11111111111111111111111111111111111111112",
    "ETH": "2NdXGW7dpwye9Heq7qL3gFYYUUDewfxCUUDq36zzfrqD",
    "USDC": "EPjFWdd5AufqSSqeM2q6ksjLpaEweidnGj9n92gtQgNf",
    "SOL": "So11111111111111111111111111111111111111112",
}

MIN_BALANCE_THRESHOLD = float(os.getenv("MIN_BALANCE_THRESHOLD", "0.001"))

def check_wallet_balances(wallet_address: str) -> Dict[str, float]:
    """Return token balances for ``wallet_address``.

    The ``FAKE_BALANCES`` environment variable still takes precedence so
    tests can supply dummy data.  When not set the Solana RPC is queried
    using ``SOLANA_RPC_URL`` to obtain SPL token account balances.  The
    amounts are returned as human readable floats keyed by token mint.
    """

    env_balances = os.getenv("FAKE_BALANCES")
    if env_balances:
        try:
            return json.loads(env_balances)
        except json.JSONDecodeError:
            logger.error("Invalid FAKE_BALANCES JSON")
            return {}

    if not wallet_address:
        return {}

    if Client is None:
        logger.error("Solana RPC client not available")
        return {}

    try:
        rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        client = Client(rpc_url)

        owner = PublicKey(wallet_address)
        opts = TokenAccountOpts(
            program_id=PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"),
            encoding="jsonParsed",
        )
        resp = client.get_token_accounts_by_owner(owner, opts)
        accounts = resp.get("result", {}).get("value", [])
        balances: Dict[str, float] = {}
        for acc in accounts:
            info = (
                acc.get("account", {})
                .get("data", {})
                .get("parsed", {})
                .get("info", {})
            )
            mint = info.get("mint")
            amount_info = info.get("tokenAmount", {})
            amount = amount_info.get("uiAmount")
            if amount is None:
                try:
                    amount = float(amount_info.get("uiAmountString", 0))
                except (ValueError, TypeError):
                    amount = 0.0
            balances[mint] = balances.get(mint, 0.0) + float(amount)

        return balances
    except Exception as e:  # pragma: no cover - network
        logger.error("Failed to fetch wallet balances: %s", e)
        return {}

def detect_non_trade_tokens(balances: Dict[str, float]) -> List[str]:
    """Return list of tokens that should be converted for trading."""
    non_trade = []
    threshold = MIN_BALANCE_THRESHOLD
    for token, amount in balances.items():
        if (
            token in SUPPORTED_FUNDING
            and token not in REQUIRED_TOKENS
            and amount >= threshold
        ):
            non_trade.append(token)
    return non_trade

async def auto_convert_funds(
    wallet: str,
    from_token: str,
    to_token: str,
    amount: float,
    dry_run: bool = True,
    slippage_bps: int = 50,
    notifier: TelegramNotifier | None = None,
) -> Dict:
    """Convert funds using the Solana Jupiter aggregator."""

    logger.info("Converting %s %s to %s", amount, from_token, to_token)

    from_mint = TOKEN_MINTS.get(from_token)
    to_mint = TOKEN_MINTS.get(to_token)
    if from_mint is None or to_mint is None:
        logger.error("Unsupported token conversion: %s -> %s", from_token, to_token)
        return {"error": "unsupported pair"}

    if dry_run:
        result = {
            "wallet": wallet,
            "from": from_token,
            "to": to_token,
            "amount": amount,
            "tx_hash": "DRYRUN",
            "status": "simulated",
        }
        logger.info("Conversion result: %s", result)
        return result
    else:
        try:
            result = await execute_swap(
                from_mint,
                to_mint,
                amount,
                notifier=notifier if notifier else TelegramNotifier(),
                slippage_bps=slippage_bps,
                dry_run=False,
            )
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
