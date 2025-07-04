from typing import Tuple

import pandas as pd
import ta

from crypto_bot.utils.logger import setup_logger

# Use the main bot log for exit messages
logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


def calculate_trailing_stop(
    price_series: pd.Series, trail_pct: float = 0.1
) -> float:
    """Return a trailing stop from the high of ``price_series``.

    Parameters
    ----------
    price_series : pd.Series
        Series of closing prices.
    trail_pct : float, optional
        Percentage to trail below the maximum price.

    Returns
    -------
    float
        Calculated trailing stop value.
    """
    highest = price_series.max()
    stop = highest * (1 - trail_pct)
    logger.info("Calculated trailing stop %.4f from high %.4f", stop, highest)
    return stop


def momentum_timer(price_series: pd.Series) -> Tuple[int, bool]:
    """Measure momentum of consecutive closes.

    Parameters
    ----------
    price_series : pd.Series
        Series of closing prices.

    Returns
    -------
    Tuple[int, bool]
        Number of consecutive up closes and whether price has stalled
        with three or more consecutive down closes.
    """
    diffs = price_series.diff().dropna()
    up_streak = 0
    for change in reversed(diffs):
        if change > 0:
            up_streak += 1
        else:
            break
    down_streak = 0
    for change in reversed(diffs):
        if change < 0:
            down_streak += 1
        else:
            break
    stall = down_streak >= 3
    return up_streak, stall


def momentum_healthy(df: pd.DataFrame) -> bool:
    """Check RSI, MACD and volume to gauge trend health.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data used to compute indicators.

    Returns
    -------
    bool
        ``True`` if the momentum indicators confirm strength.
    """
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    vol_avg = df['volume'].rolling(3).mean()
    # Ensure at least two non-null volume averages exist before comparing
    if vol_avg.dropna().shape[0] < 2:
        return False
    vol_rising = vol_avg.iloc[-1] > vol_avg.iloc[-2]

    latest = df.iloc[-1]
    # Verify momentum indicators have valid values
    if (
        pd.isna(latest.get('rsi'))
        or pd.isna(latest.get('macd'))
        or pd.isna(latest.get('macd_signal'))
    ):
        return False

    return bool(
        latest['rsi'] > 55
        and latest['macd'] > latest['macd_signal']
        and vol_rising
    )


def fib_extensions(
    last_swing_low: float, last_swing_high: float
) -> Tuple[float, float]:
    """Return 1.618 and 2.618 fib extension levels.

    Parameters
    ----------
    last_swing_low : float
        Most recent swing low in price.
    last_swing_high : float
        Most recent swing high in price.

    Returns
    -------
    Tuple[float, float]
        The 1.618 and 2.618 extension levels.
    """
    diff = last_swing_high - last_swing_low
    ext_1618 = last_swing_low + diff * 1.618
    ext_2618 = last_swing_low + diff * 2.618
    return ext_1618, ext_2618


def should_exit(
    df: pd.DataFrame,
    current_price: float,
    trailing_stop: float,
    config: dict,
    risk_manager=None,
) -> Tuple[bool, float]:
    """Determine whether to exit a position and update trailing stop.

    Parameters
    ----------
    df : pd.DataFrame
        Recent market data.
    current_price : float
        Latest traded price.
    trailing_stop : float
        Current trailing stop value.
    config : dict
        Strategy configuration.

    Returns
    -------
    Tuple[bool, float]
        Flag indicating whether to exit and the updated stop price.
    """
    exit_signal = False
    new_stop = trailing_stop
    if current_price < trailing_stop:
        if not momentum_healthy(df):
            logger.info(
                "Price %.4f hit trailing stop %.4f",
                current_price,
                trailing_stop,
            )
            exit_signal = True
            if risk_manager and getattr(risk_manager, "stop_order", None):
                order = risk_manager.stop_order
                entry = order.get("entry_price")
                direction = order.get("direction")
                strategy = order.get("strategy", "")
                symbol = order.get("symbol", config.get("symbol", ""))
                confidence = order.get("confidence", 0.0)
                if entry is not None and direction:
                    pnl = (current_price - entry) * (
                        1 if direction == "buy" else -1
                    )
                    from crypto_bot.utils.pnl_logger import log_pnl

                    log_pnl(
                        strategy,
                        symbol,
                        entry,
                        current_price,
                        pnl,
                        confidence,
                        direction,
                    )
    else:
        trailed = calculate_trailing_stop(
            df['close'],
            config['exit_strategy']['trailing_stop_pct'],
        )
        if trailed > trailing_stop:
            new_stop = trailed
            logger.info("Trailing stop moved to %.4f", new_stop)
    return exit_signal, new_stop


def get_partial_exit_percent(pnl_pct: float) -> int:
    """Return percent of position to close based on profit.

    Parameters
    ----------
    pnl_pct : float
        Unrealized profit or loss percentage.

    Returns
    -------
    int
        Portion of the position to close expressed as a percentage.
    """
    if pnl_pct > 100:
        return 50
    if pnl_pct > 50:
        return 30
    if pnl_pct > 25:
        return 20
    return 0
