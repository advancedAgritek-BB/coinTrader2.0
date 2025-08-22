"""Core utilities and shared state for crypto_bot."""

from .queues import TradeCandidate, trade_queue

__all__ = ["TradeCandidate", "trade_queue"]
