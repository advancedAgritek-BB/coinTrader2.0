import pandas as pd
import ta

from crypto_bot.utils.logger import setup_logger
from crypto_bot.execution.cex_executor import place_trailing_stop_ws, KrakenWebsocketClient

logger = setup_logger(__name__, "crypto_bot/logs/exit.log")


def calculate_trailing_stop(price_series: pd.Series, trail_pct: float = 0.1) -> float:
    """Calculate trailing stop based on highest price in series."""
    highest = price_series.max()
    stop = highest * (1 - trail_pct)
    logger.info("Calculated trailing stop %.4f from high %.4f", stop, highest)
    return stop


def momentum_timer(price_series: pd.Series) -> tuple:
    """Return consecutive up closes count and stall flag for 3+ down closes."""
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
    """Check RSI, MACD and volume to gauge trend health."""
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    vol_avg = df['volume'].rolling(3).mean()
    vol_rising = vol_avg.iloc[-1] > vol_avg.iloc[-2]
    latest = df.iloc[-1]
    return (
        latest['rsi'] > 55
        and latest['macd'] > latest['macd_signal']
        and vol_rising
    )


def fib_extensions(last_swing_low: float, last_swing_high: float) -> tuple:
    """Return 1.618 and 2.618 fib extension levels."""
    diff = last_swing_high - last_swing_low
    ext_1618 = last_swing_low + diff * 1.618
    ext_2618 = last_swing_low + diff * 2.618
    return ext_1618, ext_2618


def should_exit(df: pd.DataFrame, current_price: float, trailing_stop: float, config: dict) -> tuple:
    """Determine exit decision and updated stop."""
    exit_signal = False
    new_stop = trailing_stop
    if current_price < trailing_stop:
        if not momentum_healthy(df):
            logger.info("Price %.4f hit trailing stop %.4f", current_price, trailing_stop)
            exit_signal = True
    else:
        highest = df['close'].max()
        trailed = calculate_trailing_stop(df['close'], config['exit_strategy']['trailing_stop_pct'])
        if trailed > trailing_stop:
            new_stop = trailed
            logger.info("Trailing stop moved to %.4f", new_stop)
    return exit_signal, new_stop


def get_partial_exit_percent(pnl_pct: float) -> int:
    """Return percent of position to close based on profit."""
    if pnl_pct > 100:
        return 50
    if pnl_pct > 50:
        return 30
    if pnl_pct > 25:
        return 20
    return 0


def send_trailing_stop_order(
    ws_client: KrakenWebsocketClient, symbol: str, side: str, qty: float, pct: float
) -> None:
    """Convenience wrapper to submit a trailing-stop order over WebSocket."""
    try:
        place_trailing_stop_ws(ws_client, symbol, side, qty, pct)
        logger.info("Submitted trailing stop via WebSocket")
    except Exception as exc:
        logger.warning("Failed to submit trailing stop: %s", exc)

