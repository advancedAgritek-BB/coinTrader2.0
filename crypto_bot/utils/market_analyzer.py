import pandas as pd
from typing import Dict

from crypto_bot.regime.regime_classifier import classify_regime_async
from crypto_bot.strategy_router import strategy_for, strategy_name
from crypto_bot.signals.signal_scoring import evaluate_async


async def analyze_symbol(symbol: str, df: pd.DataFrame, mode: str, config: Dict) -> Dict:
    """Classify the market regime and evaluate the trading signal for ``symbol``.

    Parameters
    ----------
    symbol : str
        Trading pair to analyze.
    df : pd.DataFrame
        OHLCV data for the pair.
    mode : str
        Execution mode of the bot ("cex", "onchain" or "auto").
    config : Dict
        Bot configuration.
    """
    regime = await classify_regime_async(df)

    period = int(config.get("regime_return_period", 5))
    future_return = 0.0
    if len(df) > period:
        start = df["close"].iloc[-period - 1]
        end = df["close"].iloc[-1]
        future_return = (end - start) / start * 100

    result = {
        "symbol": symbol,
        "df": df,
        "regime": regime,
        "future_return": future_return,
    }

    if regime != "unknown":
        env = mode if mode != "auto" else "cex"
        strategy_fn = strategy_for(regime)
        name = strategy_name(regime, env)
        score, direction = await evaluate_async(strategy_fn, df, config)
        result.update({
            "env": env,
            "name": name,
            "score": score,
            "direction": direction,
        })
    return result

