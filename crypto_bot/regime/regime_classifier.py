from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import ta
import yaml


CONFIG_PATH = Path(__file__).with_name("regime_config.yaml")


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = _load_config(CONFIG_PATH)


def classify_regime(df: pd.DataFrame, *, config_path: Optional[str] = None) -> str:
    """Classify market regime.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    config_path : Optional[str], default None
        Optional path to override the default configuration. Primarily used for
        testing.
    """

    if df is None or df.empty or len(df) < 20:
        return "unknown"

    cfg = CONFIG if config_path is None else _load_config(Path(config_path))

    df = df.copy()

    df['ema20'] = (
        ta.trend.ema_indicator(df['close'], window=cfg['ema_fast'])
        if len(df) >= cfg['ema_fast']
        else np.nan
    )
    df['ema50'] = (
        ta.trend.ema_indicator(df['close'], window=cfg['ema_slow'])
        if len(df) >= cfg['ema_slow']
        else np.nan
    )

    if len(df) >= cfg['indicator_window']:
        try:
            df['adx'] = ta.trend.adx(
                df['high'], df['low'], df['close'], window=cfg['indicator_window']
            )
            df['rsi'] = ta.momentum.rsi(df['close'], window=cfg['indicator_window'])
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=cfg['indicator_window']
            )
        except IndexError:
            df['adx'] = np.nan
            df['rsi'] = np.nan
            df['atr'] = np.nan
            return "unknown"
    else:
        df['adx'] = np.nan
        df['rsi'] = np.nan
        df['atr'] = np.nan

    if len(df) >= cfg['bb_window']:
        bb = ta.volatility.BollingerBands(df['close'], window=cfg['bb_window'])
        df['bb_width'] = bb.bollinger_wband()
    else:
        df['bb_width'] = np.nan

    volume_ma20 = (
        df['volume'].rolling(cfg['ma_window']).mean()
        if len(df) >= cfg['ma_window']
        else pd.Series(np.nan, index=df.index)
    )
    atr_ma20 = (
        df['atr'].rolling(cfg['ma_window']).mean()
        if len(df) >= cfg['ma_window']
        else pd.Series(np.nan, index=df.index)
    )

    latest = df.iloc[-1]
    regime = 'sideways'

    # Trending
    if latest['adx'] > cfg['adx_trending_min'] and latest['ema20'] > latest['ema50']:
        regime = 'trending'
    # Sideways
    elif latest['adx'] < cfg['adx_sideways_max'] and latest['bb_width'] < cfg['bb_width_sideways_max']:
        regime = 'sideways'
    # Breakout
    elif (
        latest['bb_width'] < cfg['bb_width_breakout_max']
        and not np.isnan(volume_ma20.iloc[-1])
        and latest['volume'] > volume_ma20.iloc[-1] * cfg['breakout_volume_mult']
    ):
        regime = 'breakout'
    # Mean-reverting
    elif (
        cfg['rsi_mean_rev_min'] <= latest['rsi'] <= cfg['rsi_mean_rev_max']
        and abs(latest['close'] - latest['ema20']) / latest['close'] < cfg['ema_distance_mean_rev_max']
    ):
        regime = 'mean-reverting'
    # Volatile
    elif (
        not np.isnan(atr_ma20.iloc[-1])
        and latest['atr'] > atr_ma20.iloc[-1] * cfg['atr_volatility_mult']
    ):
        regime = 'volatile'

    return regime
