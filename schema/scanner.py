from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError, validator


class ScannerConfig(BaseModel):
    """Configuration for market scanning."""

    scan_markets: bool = Field(False, description="Load all exchange pairs")
    symbols: list[str] | None = Field(default_factory=list, description="Symbols to trade")
    excluded_symbols: list[str] = Field(default_factory=list, description="Symbols to skip")
    exchange_market_types: list[str] = Field(default_factory=lambda: ["spot"], description="Market types")
    min_symbol_age_days: int = 0
    symbol_batch_size: int = 10
    scan_lookback_limit: int = 50
    cycle_lookback_limit: int | None = Field(
        default=None,
        description="Override per-cycle candle load (default min(150, timeframe_minutes * 2))",
    )
    max_spread_pct: float = 1.0

    class Config:
        extra = "allow"

    @validator("symbols", pre=True)
    def _default_symbols(cls, v):
        return v or []

    @validator("symbols")
    def _require_symbols_if_not_scanning(cls, v, values):
        if not values.get("scan_markets") and not v:
            raise ValueError("symbols must be provided when scan_markets is false")
        return v

    @validator("exchange_market_types", each_item=True)
    def _validate_market_type(cls, v):
        allowed = {"spot", "margin", "futures"}
        if v not in allowed:
            raise ValueError(f"invalid market type: {v}")
        return v

    @validator("symbol_batch_size", "scan_lookback_limit", "cycle_lookback_limit")
    def _positive_int(cls, v, values, config, field):
        if v is not None and v <= 0:
            raise ValueError(f"{field.name} must be > 0")
        return v

