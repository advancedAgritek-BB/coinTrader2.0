import argparse
from crypto_bot.backtest.backtest_runner import BacktestConfig, BacktestRunner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run BacktestRunner")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument(
        "--offline",
        metavar="PATH",
        help="Load OHLCV data from local directory instead of CCXT",
    )
    args = parser.parse_args(argv)

    cfg = BacktestConfig(symbol=args.symbol, timeframe=args.timeframe, since=0)

    if args.offline:
        try:
            import bulk_loader  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(f"Offline mode requires bulk_loader: {exc}")
        try:
            from inspect import signature

            sig = signature(bulk_loader.load_dir)
            if len(sig.parameters) >= 3:
                df = bulk_loader.load_dir(args.offline, args.symbol, args.timeframe)
            else:
                df = bulk_loader.load_dir(args.offline)
        except Exception as exc:  # pragma: no cover - runtime errors
            raise SystemExit(f"Failed to load offline data: {exc}")
        runner = BacktestRunner(cfg, df=df)
    else:
        runner = BacktestRunner(cfg)

    metrics = runner.run_grid()
    print(metrics)


if __name__ == "__main__":  # pragma: no cover - manual use
    main()
