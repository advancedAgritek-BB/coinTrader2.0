from typing import Dict, Optional, Tuple
import pandas as pd


def generate_signal(
    df: pd.DataFrame,
    config: Optional[Dict[str, float | int | str]] = None,
    *,
    breakout_pct: float = 0.1,
    volume_multiple: float = 3.0,
    max_history: int = 50,
    initial_window: int = 3,
    min_volume: float = 100.0,
    direction: str = "auto",
) -> Tuple[float, str]:
    """Detect pumps for newly listed tokens using early price and volume action.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data ordered oldest -> newest.
    config : dict, optional
        Configuration values overriding the keyword defaults.
    breakout_pct : float, optional
        Minimum percent change from the first close considered a breakout.
    volume_multiple : float, optional
        Minimum multiple of the average volume of the first ``initial_window``
        candles considered abnormal.
    max_history : int, optional
        Maximum history length still considered a new listing.
    initial_window : int, optional
        Number of early candles used to compute baseline volume.
    min_volume : float, optional
        Minimum trade volume for the latest candle to consider a signal.
    direction : {"auto", "long", "short"}, optional
        Force a trade direction or infer automatically.

    Returns
    -------
    Tuple[float, str]
        Score between 0 and 1 and trade direction.
    """
    if config:
        breakout_pct = config.get("breakout_pct", breakout_pct)
        volume_multiple = config.get("volume_multiple", volume_multiple)
        max_history = config.get("max_history", max_history)
        initial_window = config.get("initial_window", initial_window)
        min_volume = config.get("min_volume", min_volume)
        direction = config.get("direction", direction)

    if len(df) < initial_window:
        return 0.0, "none"

    price_change = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    base_volume = df["volume"].iloc[:initial_window].mean()
    vol_ratio = df["volume"].iloc[-1] / base_volume if base_volume > 0 else 0

    if df["volume"].iloc[-1] < min_volume:
        return 0.0, "none"

    if (
        len(df) <= max_history
        and price_change >= breakout_pct
        and vol_ratio >= volume_multiple
    ):
        price_score = min(price_change / breakout_pct, 1.0)
        vol_score = min(vol_ratio / volume_multiple, 1.0)
        score = (price_score + vol_score) / 2
        if direction not in {"auto", "long", "short"}:
            direction = "auto"
        trade_direction = direction if direction != "auto" else "long"
        return score, trade_direction

    return 0.0, "none"
