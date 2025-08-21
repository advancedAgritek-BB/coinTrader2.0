import logging
from datetime import datetime, timedelta, timezone

import pytest

from crypto_bot import sentiment_filter


@pytest.mark.parametrize("result", [None, (None, None)])
def test_sentiment_factor_absent(monkeypatch, caplog, result):
    """Return neutral factor when no sentiment data is present."""

    def fake_load():
        return result

    monkeypatch.setattr(sentiment_filter, "load_sentiment", fake_load, raising=False)
    with caplog.at_level(logging.WARNING):
        factor = sentiment_filter.sentiment_factor_or_default(require_sentiment=True)
    assert factor == 1.0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_sentiment_factor_stale(monkeypatch, caplog):
    """Return neutral factor when sentiment is stale."""

    stale_time = datetime.now(timezone.utc) - timedelta(hours=2)

    def fake_load():
        return 0.5, stale_time

    monkeypatch.setattr(sentiment_filter, "load_sentiment", fake_load, raising=False)
    with caplog.at_level(logging.WARNING):
        factor = sentiment_filter.sentiment_factor_or_default(require_sentiment=True)
    assert factor == 1.0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_sentiment_not_required(monkeypatch, caplog):
    """Trading proceeds with neutral sentiment when not required."""

    def fake_load():
        return None

    monkeypatch.setattr(sentiment_filter, "load_sentiment", fake_load, raising=False)
    with caplog.at_level(logging.WARNING):
        factor = sentiment_filter.sentiment_factor_or_default(require_sentiment=False)
    assert factor == 1.0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
