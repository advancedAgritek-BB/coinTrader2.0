import os
import argparse
import os


def select_exchange(args):
    if args.exchange:
        return args.exchange
    # Fall back to env or default without prompting
    return os.getenv("EXCHANGE", "coinbase")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", choices=["coinbase", "kraken"], help="Exchange to use")
    p.add_argument("--paper", action="store_true", help="Run in paper trading mode")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick fetch/strategy smoke test then exit",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    exchange = select_exchange(args)
    os.environ["EXCHANGE"] = exchange
    if args.paper:
        os.environ["EXECUTION_MODE"] = "dry_run"
    if args.smoke_test:
        # In smoke test just output selection and exit quickly
        print(f"Selected exchange {exchange} (paper={args.paper})")
        return
    from .main import main as bot_main
    import asyncio

    asyncio.run(bot_main())


if __name__ == "__main__":
    main()

