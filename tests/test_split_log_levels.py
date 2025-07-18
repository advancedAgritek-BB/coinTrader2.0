from pathlib import Path
from tools.split_log_levels import split_logs


def test_split_logs(tmp_path: Path) -> None:
    src = tmp_path / "bot.log"
    info = tmp_path / "info.log"
    warn = tmp_path / "warning.log"
    lines = [
        "2024-01-01 00:00:00 - INFO - start\n",
        "2024-01-01 00:00:01 - WARNING - caution\n",
        "2024-01-01 00:00:02 - DEBUG - detail\n",
    ]
    src.write_text("".join(lines))

    split_logs(src, info, warn)

    assert info.read_text() == lines[0]
    assert warn.read_text() == lines[1]

