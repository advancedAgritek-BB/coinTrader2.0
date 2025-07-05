import pandas as pd
from typing import Dict

from crypto_bot.regime.regime_classifier import classify_regime_async
from crypto_bot.strategy_router import route, strategy_name
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.signals.signal_scoring import evaluate_async


async def analyze_symbol(
    symbol: str,
    df_map: Dict[str, pd.DataFrame],
    mode: str,
    config: Dict,
    notifier: TelegramNotifier | None = None,
) -> Dict:
    """Classify the market regime and evaluate the trading signal for ``symbol``.

    Parameters
    ----------
    symbol : str
        Trading pair to analyze.
    df_map : Dict[str, pd.DataFrame]
        Mapping of timeframe to OHLCV data.
    mode : str
        Execution mode of the bot ("cex", "onchain" or "auto").
    config : Dict
        Bot configuration.
    notifier : TelegramNotifier | None
        Optional notifier used to send a message when the strategy is invoked.
    """
    base_tf = config.get("timeframe", "1h")
    df = df_map.get(base_tf)
    higher_df = df_map.get("1d")
    regime = await classify_regime_async(df, higher_df)

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
        strategy_fn = route(regime, env, config, notifier)
        name = strategy_name(regime, env)
        cfg = {**config, "symbol": symbol}
        score, direction = await evaluate_async(strategy_fn, df, cfg)
        result.update({
            "env": env,
            "name": name,
            "score": score,
            "direction": direction,
        })
    return result

