import argparse
import os
import shutil
from pathlib import Path

from crypto_bot.utils.bootstrap_progress import reset_bootstrap_progress

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
BOOTSTRAP_FILE = CACHE_DIR / "ohlcv_bootstrap_state.json"


def select_exchange(args):
    if args.exchange:
        return args.exchange
    # Fall back to env or default without prompting
    return os.getenv("EXCHANGE", "coinbase")


def cache_purge(_args):
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Purged cache directory {CACHE_DIR}")
    else:
        print("No cache directory found")


def cache_reset_bootstrap(_args):
    if BOOTSTRAP_FILE.exists():
        BOOTSTRAP_FILE.unlink()
        print(f"Removed progress file {BOOTSTRAP_FILE}")
    else:
        print("No progress file found")


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    cache_p = sub.add_parser("cache", help="Cache utilities")
    cache_sub = cache_p.add_subparsers(dest="cache_command")
    purge_p = cache_sub.add_parser("purge", help="Delete cache files")
    purge_p.set_defaults(func=cache_purge)
    reset_p = cache_sub.add_parser(
        "reset-bootstrap", help="Remove OHLCV bootstrap progress"
    )
    reset_p.set_defaults(func=cache_reset_bootstrap)

    p.add_argument(
        "--exchange", choices=["coinbase", "kraken"], help="Exchange to use"
    )
    p.add_argument(
        "--paper", action="store_true", help="Run in paper trading mode"
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick fetch/strategy smoke test then exit",
    )
    p.add_argument(
        "--ohlcv-chunk-size",
        type=int,
        help="Symbols per chunk when fetching OHLCV during bootstrap",
    )
    p.add_argument(
        "--reset-bootstrap-progress",
        action="store_true",
        help="Reset bootstrap progress tracker and start over",
    )
    p.add_argument(
        "--min-score",
        type=float,
        help="Override minimum score threshold for signal execution",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        help="Override minimum confidence threshold for signal execution",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "func", None):
        args.func(args)
        return
    exchange = select_exchange(args)
    os.environ["EXCHANGE"] = exchange
    if args.paper:
        os.environ["EXECUTION_MODE"] = "dry_run"
    if args.smoke_test:
        # In smoke test just output selection and exit quickly
        print(f"Selected exchange {exchange} (paper={args.paper})")
        return
    if args.ohlcv_chunk_size is not None:
        os.environ["OHLCV_CHUNK_SIZE"] = str(args.ohlcv_chunk_size)
    if args.min_score is not None:
        os.environ["MIN_SCORE"] = str(args.min_score)
    if args.min_confidence is not None:
        os.environ["MIN_CONFIDENCE"] = str(args.min_confidence)
    if args.reset_bootstrap_progress:
        reset_bootstrap_progress()
    from crypto_bot.main import main as bot_main
    import asyncio

    asyncio.run(bot_main())


if __name__ == "__main__":
    main()
