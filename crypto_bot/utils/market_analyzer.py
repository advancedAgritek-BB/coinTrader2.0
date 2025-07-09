import pandas as pd
from typing import Dict, Iterable, Tuple, List

from .logger import LOG_DIR, setup_logger
from . import perf

from crypto_bot.regime.pattern_detector import detect_patterns
from crypto_bot.regime.regime_classifier import (
    classify_regime_async,
    classify_regime_cached,
)
from crypto_bot.strategy_router import (
    route,
    strategy_name,
    get_strategies_for_regime,
    get_strategy_by_name,
    strategy_for,
    RouterConfig,
    evaluate_regime,
)
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot import meta_selector
from crypto_bot.signals.signal_scoring import evaluate_async, evaluate_strategies
from crypto_bot.volatility_filter import calc_atr


analysis_logger = setup_logger("strategy_rank", LOG_DIR / "strategy_rank.log")


async def run_candidates(
    df: pd.DataFrame,
    strategies: Iterable,
    symbol: str,
    cfg: Dict,
) -> List[Tuple[callable, float, str]]:
    """Evaluate ``strategies`` and rank them by score times edge."""

    results: List[Tuple[float, callable, float, str]] = []
    for strat in strategies:
        try:
            score, direction, _ = (await evaluate_async([strat], df, cfg))[0]
        except Exception as exc:  # pragma: no cover - safety
            analysis_logger.warning("Strategy %s failed: %s", strat.__name__, exc)
            continue
        try:
            gross_edge = perf.edge(strat.__name__, symbol)
        except Exception:  # pragma: no cover - if perf fails use neutral edge
            gross_edge = 1.0
        rank = score * gross_edge
        results.append((rank, strat, score, direction))

    results.sort(key=lambda x: x[0], reverse=True)
    return [(s, sc, d) for _r, s, sc, d in results]


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
    router_cfg = RouterConfig.from_dict(config)
    base_tf = router_cfg.timeframe
    higher_tf = config.get("higher_timeframe", "1d")
    df = df_map.get(base_tf)
    higher_df = df_map.get("1d")
    regime, probs = await classify_regime_async(df, higher_df)
    patterns = detect_patterns(df)
    base_conf = float(probs.get(regime, 0.0))
    profile = bool(config.get("profile_regime", False))
    regime, _ = await classify_regime_cached(
        symbol,
        base_tf,
        df,
        higher_df,
        profile,
    )
    higher_df = df_map.get(higher_tf)

    if df is not None:
        regime, info = await classify_regime_async(df, higher_df)
        if isinstance(info, dict):
            patterns = info
        elif isinstance(info, set):
            patterns = {p: 1.0 for p in info}
        else:
            base_conf = float(info)

    regime_counts: Dict[str, int] = {}
    regime_tfs = config.get("regime_timeframes", [base_tf])
    min_agree = config.get("min_consistent_agreement", 1)

    vote_map: Dict[str, pd.DataFrame] = {}
    for tf in regime_tfs:
        tf_df = df_map.get(tf)
        if tf_df is None:
            continue
        higher_df = df_map.get("1d") if tf != "1d" else None
        r, _ = await classify_regime_cached(
            symbol,
            tf,
            tf_df,
            higher_df,
            profile,
        )
        regime_counts[r] = regime_counts.get(r, 0) + 1
        if tf_df is not None:
            vote_map[tf] = tf_df
    if higher_tf in df_map:
        vote_map.setdefault(higher_tf, df_map[higher_tf])

    if vote_map:
        labels = await classify_regime_async(df_map=vote_map)
        if isinstance(labels, tuple):
            label_map = dict(zip(vote_map.keys(), labels))
        else:
            label_map = labels
        for tf in regime_tfs:
            r = label_map.get(tf)
            if r:
                regime_counts[r] = regime_counts.get(r, 0) + 1

    if regime_counts:
        regime, votes = max(regime_counts.items(), key=lambda kv: kv[1])
    else:
        regime, votes = "unknown", 0
    confidence = votes / max(len(regime_tfs), 1)
    confidence *= base_conf
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

        atr = None
        if eval_mode == "best":
            strategies = get_strategies_for_regime(regime, router_cfg)
            res = evaluate_strategies(strategies, df, cfg)
            name = res.get("name", strategy_name(regime, env))
            score = float(res.get("score", 0.0))
            direction = res.get("direction", "none")
        elif eval_mode == "ensemble":
            min_conf = float(config.get("ensemble_min_conf", 0.15))
            candidates = [strategy_for(regime, router_cfg)]
            extra = meta_selector._scores_for(regime)
            for strat_name, val in extra.items():
                if val >= min_conf:
                    fn = get_strategy_by_name(strat_name)
                    if fn and fn not in candidates:
                        candidates.append(fn)
            ranked = await run_candidates(df, candidates, symbol, cfg)
            if ranked:
                best_fn, raw_score, raw_dir = ranked[0]
                name = best_fn.__name__
                score = raw_score
                direction = raw_dir if raw_score >= min_conf else "none"
                if len(ranked) > 1:
                    second = ranked[1]
                    analysis_logger.info(
                        "%s second %s %.4f %s",
                        symbol,
                        second[0].__name__,
                        second[1],
                        second[2],
                    )
            else:
                name = strategy_name(regime, env)
                score = 0.0
                direction = "none"
        else:
            strategy_fn = route(regime, env, router_cfg, notifier, df=df)
            name = strategy_name(regime, env)
            score, direction, atr = (await evaluate_async([strategy_fn], df, cfg))[0]

        atr_period = int(config.get("risk", {}).get("atr_period", 14))
        if direction != "none" and {"high", "low", "close"}.issubset(df.columns):
            atr = calc_atr(df, window=atr_period)

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
            "atr": atr,
        })

        votes = []
        voting = config.get("voting_strategies", [])
        if isinstance(voting, list):
            for strat_name in voting:
                fn = get_strategy_by_name(strat_name)
                if fn is None:
                    continue
                try:
                    dir_vote = (await evaluate_async([fn], df, cfg))[0][1]
                except Exception:  # pragma: no cover - safety
                    continue
                votes.append(dir_vote)

        if votes:
            counts = {}
            for d in votes:
                counts[d] = counts.get(d, 0) + 1
            best_dir, n = max(counts.items(), key=lambda kv: kv[1])
            min_votes = int(config.get("min_agreeing_votes", 1))
            if n >= min_votes:
                result["direction"] = best_dir
            else:
                result["direction"] = "none"
    return result

