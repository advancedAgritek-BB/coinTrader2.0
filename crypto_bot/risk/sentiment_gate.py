import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "bot.log")

SENTIMENT_FILE = LOG_DIR / "sentiment.json"


def load_sentiment(path: Path = SENTIMENT_FILE) -> Optional[Tuple[float, float]]:
    """Return the timestamp and sentiment score.

    The score is loaded from ``path`` if it exists. During testing the
    ``MOCK_TWITTER_SENTIMENT`` or ``MOCK_FNG_VALUE`` environment variables can
    provide a temporary score so that sentiment checks proceed without an
    external service.

    Parameters
    ----------
    path : Path, optional
        Location of the sentiment data file. Defaults to :data:`SENTIMENT_FILE`.

    Returns
    -------
    tuple[float, float] | None
        ``(timestamp, score)`` if available, otherwise ``None``.
    """
    if os.getenv("MOCK_TWITTER_SENTIMENT") or os.getenv("MOCK_FNG_VALUE"):
        score = float(os.getenv("MOCK_TWITTER_SENTIMENT", 50))
        return time.time(), score
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - corrupted file
        logger.error("Failed to load sentiment data: %s", exc)
        return None
    ts = float(data.get("timestamp") or data.get("ts") or 0.0)
    score = float(data.get("score") or data.get("factor") or 1.0)
    return ts, score


def sentiment_factor_or_default(
    now_ts: float, require_sentiment: bool, max_age_s: float
) -> float:
    """Return sentiment score or ``1.0`` when data is missing or stale.

    Parameters
    ----------
    now_ts : float
        Current timestamp used for age comparison.
    require_sentiment : bool
        Whether sentiment data is required. If ``False`` or when the global
        configuration disables the sentiment filter, the function immediately
        returns ``1.0``.
    max_age_s : float
        Maximum allowed age in seconds for the sentiment data.
    """

    env_val = os.getenv("CT_REQUIRE_SENTIMENT")
    if env_val is not None:
        require_sentiment = env_val.lower() in ("1", "true", "yes")
    else:
        try:  # Check top-level filters in config.yaml
            from crypto_bot.main import load_config  # type: ignore

            cfg = load_config()
            enabled = cfg.get("filters", {}).get("sentiment", {}).get("enabled", True)
            require_sentiment = require_sentiment and enabled
        except Exception:  # pragma: no cover - config loading issues
            pass

    if not require_sentiment:
        return 1.0

    data = load_sentiment()
    if not data:
        logger.warning("No sentiment data available; using default factor 1.0")
        return 1.0
    ts, score = data
    if now_ts - ts > max_age_s:
        logger.warning(
            "Sentiment data too old (age %.0fs > %.0fs); using default factor 1.0",
            now_ts - ts,
            max_age_s,
        )
        return 1.0
    return float(score)


__all__ = ["sentiment_factor_or_default"]
