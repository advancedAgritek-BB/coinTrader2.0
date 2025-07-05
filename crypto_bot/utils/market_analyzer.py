import pandas as pd
from typing import Dict

from crypto_bot.regime.regime_classifier import classify_regime_async
from crypto_bot.strategy_router import (
    route,
    strategy_name,
    get_strategies_for_regime,
)
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.signals.signal_scoring import evaluate_async, evaluate_strategies


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
    regime, patterns = await classify_regime_async(df, higher_df)

    regime_counts: Dict[str, int] = {}
    regime_tfs = config.get("regime_timeframes", [base_tf])
    min_agree = config.get("min_consistent_agreement", 1)

    for tf in regime_tfs:
        tf_df = df_map.get(tf)
        if tf_df is None:
            continue
        higher_df = df_map.get("1d") if tf != "1d" else None
        r = await classify_regime_async(tf_df, higher_df)
        regime_counts[r] = regime_counts.get(r, 0) + 1

    if regime_counts:
        regime, votes = max(regime_counts.items(), key=lambda kv: kv[1])
    else:
        regime, votes = "unknown", 0
    confidence = votes / max(len(regime_tfs), 1)
    if votes < min_agree:
        regime = "unknown"

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
        "patterns": patterns,
        "future_return": future_return,
        "confidence": confidence,
    }

    if regime != "unknown":
        env = mode if mode != "auto" else "cex"
        eval_mode = config.get("strategy_evaluation_mode", "mapped")
        cfg = {**config, "symbol": symbol}

        if eval_mode == "best":
            strategies = get_strategies_for_regime(regime)
            res = evaluate_strategies(strategies, df, cfg)
            name = res.get("name", strategy_name(regime, env))
            score = float(res.get("score", 0.0))
            direction = res.get("direction", "none")
        else:
            strategy_fn = route(regime, env, config, notifier)
            name = strategy_name(regime, env)
            score, direction = await evaluate_async(strategy_fn, df, cfg)

        weights = config.get("scoring_weights", {})
        final = (
            weights.get("strategy_score", 1.0) * score
            + weights.get("regime_confidence", 0.0) * confidence
            + weights.get("volume_score", 0.0) * 1.0
            + weights.get("symbol_score", 0.0) * 1.0
            + weights.get("spread_penalty", 0.0) * 0.0
            + weights.get("strategy_regime_strength", 0.0) * 1.0
        )

        result.update({
            "env": env,
            "name": name,
            "score": final,
            "direction": direction,
        })
    return result

