"""Utility to inspect the Solana mempool for high priority fees.

This module provides a helper that queries a priority fee API to gauge
network congestion. It falls back to the environment variable
``MOCK_PRIORITY_FEE`` so tests and offline runs can control the value.
"""

from __future__ import annotations

import os
from typing import Optional

import requests


class SolanaMempoolMonitor:
    """Simple monitor for Solana priority fees."""

    def __init__(self, priority_fee_url: Optional[str] = None) -> None:
        self.priority_fee_url = priority_fee_url or os.getenv(
            "SOLANA_PRIORITY_FEE_URL",
            "https://mempool.solana.com/api/v0/fees/priority_fee",
        )

    def fetch_priority_fee(self) -> float:
        """Return the current priority fee per compute unit in micro lamports."""
        mock_fee = os.getenv("MOCK_PRIORITY_FEE")
        if mock_fee is not None:
            try:
                return float(mock_fee)
            except ValueError:
                return 0.0
        try:
            resp = requests.get(self.priority_fee_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return float(data.get("priorityFee", 0.0))
        except Exception:
            pass
        return 0.0

    def is_suspicious(self, threshold: float) -> bool:
        """Return True when the priority fee exceeds ``threshold``."""
        fee = self.fetch_priority_fee()
        return fee >= threshold
