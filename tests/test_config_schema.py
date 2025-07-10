import pytest
from pydantic import ValidationError

from schema.scanner import ScannerConfig


def test_symbols_required_when_not_scanning():
    with pytest.raises(ValidationError):
        ScannerConfig(scan_markets=False, symbols=[])


def test_invalid_market_type():
    with pytest.raises(ValidationError):
        ScannerConfig(scan_markets=True, exchange_market_types=["invalid"])


def test_positive_values_enforced():
    with pytest.raises(ValidationError):
        ScannerConfig(scan_markets=True, scan_lookback_limit=0)

