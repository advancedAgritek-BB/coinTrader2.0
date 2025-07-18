"""Split INFO and WARNING lines from a log file."""

from pathlib import Path
import argparse


def split_logs(log_path: Path, info_path: Path, warning_path: Path) -> None:
    """Read ``log_path`` and write INFO lines to ``info_path`` and
    WARNING lines to ``warning_path``."""
    with open(log_path) as src, open(info_path, "w") as info_f, open(
        warning_path, "w"
    ) as warn_f:
        for line in src:
            if " - INFO - " in line:
                info_f.write(line)
            if " - WARNING - " in line:
                warn_f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split bot.log by level")
    parser.add_argument("--log", default="crypto_bot/logs/bot.log",
                        help="Input log file path")
    parser.add_argument("--info", default="info.log",
                        help="Output file for INFO lines")
    parser.add_argument("--warning", default="warning.log",
                        help="Output file for WARNING lines")
    args = parser.parse_args()

    split_logs(Path(args.log), Path(args.info), Path(args.warning))


if __name__ == "__main__":
    main()
