import json
import logging
from pathlib import Path

import aiohttp

TOKEN_REGISTRY_URL = (
    "https://raw.githubusercontent.com/solana-labs/token-list/main/src/tokens/solana.tokenlist.json"
)

CACHE_FILE = Path(__file__).resolve().parents[2] / "cache" / "token_mints.json"

logger = logging.getLogger(__name__)


async def load_token_mints() -> dict[str, str]:
    """Return a mapping of token symbol to mint address."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(TOKEN_REGISTRY_URL, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
        tokens = data.get("tokens") or []
        results: dict[str, str] = {}
        for item in tokens:
            sym = item.get("symbol")
            mint = item.get("address") or item.get("mint") or item.get("tokenMint")
            if sym and mint:
                results[str(sym)] = str(mint)
        if results:
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_FILE, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as exc:  # pragma: no cover - optional cache
                logger.error("Failed to write %s: %s", CACHE_FILE, exc)
        return results
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Token registry fetch failed: %s", exc)
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE) as f:
                    cached = json.load(f)
                if isinstance(cached, dict):
                    return {str(k): str(v) for k, v in cached.items()}
            except Exception as err:  # pragma: no cover - best effort
                logger.error("Failed to read cache: %s", err)
        return {}


TOKEN_MINTS: dict[str, str] = {}
