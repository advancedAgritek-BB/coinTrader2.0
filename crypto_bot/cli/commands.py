"""Auxiliary CLI commands for diagnostics.

Currently only implements ``diagnose eval <symbol>`` which prints a
compact readiness report for the evaluation engine.  The command is kept
lightweight so it can be executed from tests or a running web interface.
"""

from __future__ import annotations

import argparse

from crypto_bot.engine import evaluation_engine


def _diagnose_eval(symbol: str) -> int:
    """Print evaluation readiness information for ``symbol``.

    Returns 0 on success and non-zero if the evaluation engine is not
    configured.
    """

    try:
        engine = evaluation_engine.get_engine()
    except Exception as exc:  # pragma: no cover - safety fallback
        print(str(exc))
        return 1

    report = engine.diagnose(symbol)
    print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crypto-bot-cli")
    sub = parser.add_subparsers(dest="command")

    diag = sub.add_parser("diagnose", help="diagnostic helpers")
    diag_sub = diag.add_subparsers(dest="diag_cmd")

    eval_p = diag_sub.add_parser("eval", help="show evaluation readiness")
    eval_p.add_argument("symbol", help="trading pair, e.g. BTC/USDT")
    eval_p.set_defaults(func=lambda a: _diagnose_eval(a.symbol))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return int(args.func(args))
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
