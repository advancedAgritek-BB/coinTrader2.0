import os
from typing import Optional

import requests


def get_pyth_price(symbol: str, config: Optional[dict] = None) -> float:
    """Return the latest price for ``symbol`` from the Pyth network."""
    mock = os.getenv("MOCK_PYTH_PRICE")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0

    # Placeholder: normally we'd query the Pyth API.
    url = config.get("pyth_url") if config else None
    if url:
        try:
            resp = requests.get(f"{url}/{symbol}", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return float(data.get("price", 0.0))
        except Exception:
            pass
    return 0.0
