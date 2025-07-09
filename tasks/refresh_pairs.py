from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import logging

import ccxt
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "crypto_bot" / "config.yaml"
CACHE_DIR = Path("cache")
PAIR_FILE = CACHE_DIR / "liquid_pairs.json"

DEFAULT_MIN_VOLUME_USD = 1_000_000
DEFAULT_TOP_K = 40
DEFAULT_REFRESH_INTERVAL = 6 * 3600  # 6 hours

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load YAML configuration if available."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_exchange(config: dict) -> ccxt.Exchange:
    """Instantiate the configured ccxt exchange."""
    name = config.get("exchange", "kraken").lower()
    if not hasattr(ccxt, name):
        raise ValueError(f"Unsupported exchange: {name}")
    return getattr(ccxt, name)({"enableRateLimit": True})


def refresh_pairs(min_volume_usd: float, top_k: int, config: dict) -> list[str]:
    """Fetch tickers and update the cached liquid pairs list."""
    old_pairs: list[str] = []
    if PAIR_FILE.exists():
        try:
            with open(PAIR_FILE) as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    old_pairs = loaded
        except Exception as exc:  # pragma: no cover - corrupted cache
            logger.error("Failed to read %s: %s", PAIR_FILE, exc)

    exchange = get_exchange(config)
    try:
        tickers = exchange.fetch_tickers()
        if not isinstance(tickers, dict):
            raise TypeError("fetch_tickers returned invalid data")
    except Exception as exc:
        logger.error("Failed to fetch tickers: %s", exc)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PAIR_FILE, "w") as f:
            json.dump(old_pairs, f, indent=2)
        return old_pairs

    pairs: list[tuple[str, float]] = []
    for symbol, data in tickers.items():
        vol = data.get("quoteVolume")
        if vol is None:
            continue
        pairs.append((symbol, float(vol)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = [sym for sym, vol in pairs if vol >= min_volume_usd][:top_k]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAIR_FILE, "w") as f:
        json.dump(top_pairs, f, indent=2)

    return top_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh liquid trading pairs")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--min-quote-volume-usd", type=float, default=DEFAULT_MIN_VOLUME_USD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=DEFAULT_REFRESH_INTERVAL,
        help="Refresh interval in seconds",
    )
    args = parser.parse_args()

    cfg = load_config()

    while True:
        try:
            pairs = refresh_pairs(args.min_quote_volume_usd, args.top_k, cfg)
            print(f"Updated {PAIR_FILE} with {len(pairs)} pairs")
        except Exception as exc:  # pragma: no cover - network failures
            print(f"Failed to refresh pairs: {exc}")
        if args.once:
            break
        time.sleep(args.refresh_interval)


if __name__ == "__main__":
    main()
