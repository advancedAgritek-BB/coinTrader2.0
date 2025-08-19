import base64
import json
import os
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import set_key

from crypto_bot.utils.env import env_or_prompt
from crypto_bot.utils.logger import LOG_DIR, setup_logger


try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None

logger = setup_logger(__name__, LOG_DIR / "wallet.log")

CONFIG_FILE = Path(__file__).resolve().parent / 'user_config.yaml'
ENV_FILE = Path(__file__).resolve().parent / '.env'

FERNET_KEY = os.getenv("FERNET_KEY")
_fernet = Fernet(FERNET_KEY) if FERNET_KEY and Fernet else None


def _encrypt(value: str) -> str:
    if _fernet:
        return _fernet.encrypt(value.encode()).decode()
    return value


def _decrypt(value: str) -> str:
    if _fernet:
        try:
            return _fernet.decrypt(value.encode()).decode()
        except Exception:
            pass
    return value


def _sanitize_secret(secret: str) -> str:
    """Return a base64 padded secret string."""
    secret = secret.strip()
    pad = len(secret) % 4
    if pad:
        secret += "=" * (4 - pad)
    try:
        base64.b64decode(secret)
    except Exception:
        logger.warning("API secret does not look base64 encoded")
    return secret


def _persist_supabase_env(url: str, key: str) -> None:
    """Store Supabase credentials in ``.env`` and log without leaking secrets."""
    if not url and not key:
        return
    try:
        if url:
            set_key(str(ENV_FILE), "SUPABASE_URL", url)
        if key:
            set_key(str(ENV_FILE), "SUPABASE_KEY", key)
        logger.info(
            "Wrote Supabase creds to %s (url=%s, key_len=%d)",
            ENV_FILE,
            bool(url),
            len(key),
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Failed to write Supabase creds to %s: %s", ENV_FILE, exc)


def load_external_secrets(provider: str, path: str) -> Dict[str, str]:
    """Load secrets from an external provider."""
    try:
        if provider == "aws":
            import boto3

            client = boto3.client("secretsmanager")
            resp = client.get_secret_value(SecretId=path)
            secret = resp.get("SecretString") or resp.get("SecretBinary", "")
            if isinstance(secret, bytes):
                secret = secret.decode()
            return yaml.safe_load(secret)
        if provider == "vault":
            import hvac

            client = hvac.Client()
            secret = client.secrets.kv.v2.read_secret_version(path=path)
            return secret["data"]["data"]
    except Exception as e:  # pragma: no cover - optional providers
        logger.error("Failed to load secrets from %s: %s", provider, e)
    return {}


def prompt_user() -> dict:
    """Prompt for API keys, wallet address and trading mode."""
    data: dict = {}

    default_ex = os.getenv("EXCHANGE", "coinbase")
    resp = input(f"Select exchange (coinbase/kraken) [{default_ex}]: ").strip().lower()
    exchange = resp or default_ex
    if exchange not in {"coinbase", "kraken"}:
        exchange = "coinbase"
    data["exchange"] = exchange

    if exchange == "coinbase":
        data["coinbase_api_key"] = env_or_prompt(
            "COINBASE_API_KEY", "Enter Coinbase API key: "
        )
        data["coinbase_api_secret"] = env_or_prompt(
            "COINBASE_API_SECRET", "Enter Coinbase API secret: "
        )
        data["coinbase_passphrase"] = env_or_prompt(
            "API_PASSPHRASE", "Enter Coinbase API passphrase: "
        )
        data["kraken_api_key"] = os.getenv("KRAKEN_API_KEY", "")
        data["kraken_api_secret"] = os.getenv("KRAKEN_API_SECRET", "")
    else:
        data["kraken_api_key"] = env_or_prompt(
            "KRAKEN_API_KEY", "Enter Kraken API key: "
        )
        data["kraken_api_secret"] = env_or_prompt(
            "KRAKEN_API_SECRET", "Enter Kraken API secret: "
        )
        data["coinbase_api_key"] = os.getenv("COINBASE_API_KEY", "")
        data["coinbase_api_secret"] = os.getenv("COINBASE_API_SECRET", "")
        data["coinbase_passphrase"] = os.getenv("API_PASSPHRASE", "")

    data["telegram_token"] = env_or_prompt(
        "TELEGRAM_TOKEN", "Enter Telegram bot token: "
    )
    data["telegram_chat_id"] = env_or_prompt(
        "TELEGRAM_CHAT_ID", "Enter Telegram chat id: "
    )
    data["wallet_address"] = env_or_prompt(
        "WALLET_ADDRESS", "Enter public wallet address: "
    )
    data["solana_private_key"] = env_or_prompt(
        "SOLANA_PRIVATE_KEY", "Enter Solana private key: "
    )
    data["helius_api_key"] = env_or_prompt(
        "HELIUS_KEY", "Enter Helius API key: "
    )
    data["supabase_url"] = env_or_prompt(
        "SUPABASE_URL", "Enter Supabase URL: "
    )
    existing_key = os.getenv("SUPABASE_KEY")
    if existing_key:
        os.environ["SUPABASE_KEY"] = existing_key
    data["supabase_key"] = env_or_prompt(
        "SUPABASE_KEY", "Enter Supabase API key: "
    )
    key = data["supabase_key"]
    os.environ["SUPABASE_KEY"] = key
    _persist_supabase_env(data["supabase_url"], key)
    data["lunarcrush_api_key"] = env_or_prompt(
        "LUNARCRUSH_API_KEY", "Enter LunarCrush API key: "
    )

    mode = os.getenv("MODE") or input("Preferred trading mode (cex/onchain/auto): ")
    data["mode"] = mode if mode in {"cex", "onchain", "auto"} else "auto"

    return data


def load_or_create(interactive: bool = False) -> dict:
    """Load credentials prioritizing environment variables.

    When ``interactive`` is ``False`` (the default), missing configuration values
    are not prompted for and only environment variables are consulted. This
    prevents library code from blocking on ``input()`` when running under
    non-interactive environments such as systemd or Docker.
    """

    creds: Dict[str, str] = {}

    if CONFIG_FILE.exists():
        logger.info("Loading user configuration from %s", CONFIG_FILE)
        with open(CONFIG_FILE) as f:
            creds.update(yaml.safe_load(f))
    elif interactive:
        logger.info("user_config.yaml not found; prompting for credentials")
        creds.update(prompt_user())
        logger.info("Creating new user configuration at %s", CONFIG_FILE)
        with open(CONFIG_FILE, "w") as f:
            yaml.safe_dump(creds, f)
    else:
        logger.info(
            "user_config.yaml not found; running in non-interactive mode"
        )

    provider = os.getenv("SECRETS_PROVIDER")
    if provider:
        path = os.getenv("SECRETS_PATH", "")
        if path:
            creds.update(load_external_secrets(provider, path))

    fields = {
        "exchange",
        "coinbase_api_key",
        "coinbase_api_secret",
        "coinbase_passphrase",
        "kraken_api_key",
        "kraken_api_secret",
        "telegram_token",
        "telegram_chat_id",
        "wallet_address",
        "solana_private_key",
        "helius_api_key",
        "supabase_url",
        "supabase_key",
        "lunarcrush_api_key",
        "mode",
    }
    fields.update(creds.keys())
    env_mapping: Dict[str, List[str]] = {key: [key.upper()] for key in fields}

    aliases = {
        "coinbase_api_key": ["API_KEY"],
        "coinbase_api_secret": ["API_SECRET"],
        "coinbase_passphrase": ["COINBASE_API_PASSPHRASE", "API_PASSPHRASE"],
        "kraken_api_key": ["API_KEY"],
        "kraken_api_secret": ["API_SECRET"],
        "helius_api_key": ["HELIUS_API_KEY", "HELIUS_KEY"],
    }

    for key, env_keys in aliases.items():
        env_mapping.setdefault(key, [])
        for env_key in env_keys:
            if env_key not in env_mapping[key]:
                env_mapping[key].append(env_key)

    for key, env_keys in env_mapping.items():
        for env_key in env_keys:
            val = os.getenv(env_key)
            if val is not None:
                creds[key] = val
                break

    for sec in ("coinbase_api_secret", "kraken_api_secret"):
        if sec in creds and creds[sec]:
            creds[sec] = _sanitize_secret(str(creds[sec]))

    # expose credentials via dedicated env vars
    os.environ["COINBASE_API_KEY"] = creds.get("coinbase_api_key", "")
    os.environ["COINBASE_API_SECRET"] = creds.get("coinbase_api_secret", "")
    os.environ["COINBASE_API_PASSPHRASE"] = creds.get("coinbase_passphrase", "")
    os.environ["KRAKEN_API_KEY"] = creds.get("kraken_api_key", "")
    os.environ["KRAKEN_API_SECRET"] = creds.get("kraken_api_secret", "")
    helius_val = creds.get("helius_api_key", "")
    os.environ["HELIUS_API_KEY"] = helius_val
    os.environ["HELIUS_KEY"] = helius_val
    os.environ["SUPABASE_URL"] = creds.get("supabase_url", "")
    supabase_key = creds.get("supabase_key", "")
    os.environ["SUPABASE_KEY"] = supabase_key
    try:  # refresh ML availability after setting credentials
        from crypto_bot.utils.ml_utils import init_ml_components

        init_ml_components()
    except Exception:  # pragma: no cover - optional dependency
        pass
    os.environ["LUNARCRUSH_API_KEY"] = creds.get("lunarcrush_api_key", "")

    # expose selected exchange credentials via generic env vars for ccxt
    exch = creds.get("primary_exchange") or creds.get("exchange")
    if exch == "coinbase":
        os.environ["API_KEY"] = os.environ.get("COINBASE_API_KEY", "")
        os.environ["API_SECRET"] = os.environ.get("COINBASE_API_SECRET", "")
        os.environ["API_PASSPHRASE"] = os.environ.get("COINBASE_API_PASSPHRASE", "")
    elif exch == "kraken":
        os.environ["API_KEY"] = os.environ.get("KRAKEN_API_KEY", "")
        os.environ["API_SECRET"] = os.environ.get("KRAKEN_API_SECRET", "")
        os.environ.pop("API_PASSPHRASE", None)

    return creds


def get_wallet() -> "Keypair":
    """Return a Keypair loaded from ``SOLANA_PRIVATE_KEY`` env variable."""
    from solana.keypair import Keypair

    private_key = env_or_prompt(
        "SOLANA_PRIVATE_KEY", "Enter Solana private key: "
    )
    try:
        key_bytes = bytes(json.loads(private_key))
    except Exception as exc:  # pragma: no cover - should be rare
        raise ValueError("Invalid SOLANA_PRIVATE_KEY") from exc
    return Keypair.from_secret_key(key_bytes)


__all__ = [
    "load_or_create",
    "prompt_user",
    "load_external_secrets",
    "get_wallet",
]


if __name__ == "__main__":  # pragma: no cover - CLI helper
    load_or_create(interactive=True)
