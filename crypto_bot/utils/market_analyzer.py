import asyncio
import importlib
import functools
import pandas as pd
from typing import Dict, Iterable, Tuple, List

from .logger import LOG_DIR, setup_logger
from . import perf
from crypto_bot.utils.strategy_utils import compute_strategy_weights

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
from crypto_bot.utils.rank_logger import log_second_place
from crypto_bot.strategy import grid_bot
from crypto_bot.volatility_filter import calc_atr
from ta.volatility import BollingerBands
from crypto_bot.utils import zscore
from crypto_bot.utils.telemetry import telemetry


def _fn_name(fn: callable) -> str:
    """Return the underlying function name even for functools.partial."""
    if isinstance(fn, functools.partial):
        return getattr(fn.func, "__name__", str(fn))
    return getattr(fn, "__name__", str(fn))


analysis_logger = setup_logger("strategy_rank", LOG_DIR / "strategy_rank.log")


async def run_candidates(
    df: pd.DataFrame,
    strategies: Iterable,
    symbol: str,
    cfg: Dict,
    regime: str | None = None,
) -> List[Tuple[callable, float, str]]:
    """Evaluate ``strategies`` and rank them by score times edge."""

    strategy_list = list(strategies)
    weights = compute_strategy_weights()
    try:
        evals = await evaluate_async(
            strategy_list,
            df,
            cfg,
            max_parallel=cfg.get("max_parallel", 4),
        )
    except Exception as exc:  # pragma: no cover - safety
        analysis_logger.warning("Batch evaluation failed: %s", exc)
        return []

    results: List[Tuple[float, callable, float, str]] = []
    for strat, (score, direction, _atr) in zip(strategy_list, evals):
        name = _fn_name(strat)
        try:
            edge = perf.edge(name, symbol, cfg.get("drawdown_penalty_coef", 0.0))
        except Exception:  # pragma: no cover - if perf fails use neutral edge
            edge = 1.0
        weight = weights.get(name, 1.0)
        rank = score * edge * weight
        results.append((rank, strat, score, direction))

    results.sort(key=lambda x: x[0], reverse=True)
    ranked = [(s, sc, d) for (rank, s, sc, d) in results]

    if regime is not None and ranked:
        scores = [sc for _fn, sc, _d in ranked]
        dirs = [d for _fn, _sc, d in ranked]
        if (
            len(set(scores)) == 1
            or all(s == 0.0 for s in scores)
            or all(d == "none" for d in dirs)
        ):
            for idx, (fn, sc, d) in enumerate(ranked):
                reg_filter = getattr(fn, "regime_filter", None)
                if reg_filter is None:
                    try:
                        module = importlib.import_module(fn.__module__)
                        reg_filter = getattr(module, "regime_filter", None)
                    except Exception:  # pragma: no cover - safety
                        reg_filter = None
                try:
                    if (
                        reg_filter
                        and hasattr(reg_filter, "matches")
                        and reg_filter.matches(regime)
                    ):
                        ranked.insert(0, ranked.pop(idx))
                        break
                except Exception:  # pragma: no cover - safety
                    pass

    return ranked


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
    lookback = config.get("indicator_lookback", 14) * 2
    for tf, df in df_map.items():
        if df is None or len(df) < lookback:
            analysis_logger.info(
                "Skipping analysis for %s on %s: insufficient data (%d candles)",
                symbol,
                tf,
                0 if df is None else len(df),
            )
            return {"symbol": symbol, "skip": True}
    base_tf = router_cfg.timeframe
    higher_tf = config.get("higher_timeframe", "1d")
    df = df_map.get(base_tf)
    if df is None:
        telemetry.inc("analysis.skipped_no_df")
        return {"symbol": symbol, "skip": "no_ohlcv"}

    if df.empty:
        telemetry.inc("analysis.skipped_no_df")
        analysis_logger.info("Skipping %s: no data for %s", symbol, base_tf)
        return {
            "symbol": symbol,
            "df": df,
            "regime": "unknown",
            "patterns": {},
            "future_return": 0.0,
            "confidence": 0.0,
            "min_confidence": 0.0,
            "direction": "none",
        }
    baseline = float(
        config.get("min_confidence_score", config.get("signal_threshold", 0.0))
    )
    bb_z = 0.0
    if df is not None and len(df) >= 14:
        try:
            bb = BollingerBands(df["close"], window=14)
            width = bb.bollinger_wband()
            z = zscore(width, 14)
            if not z.empty:
                bb_z = float(z.iloc[-1])
        except Exception:
            bb_z = 0.0
    min_conf_adaptive = baseline * (1 + bb_z / 3)
    higher_df = df_map.get("1d")
    regime, probs = await classify_regime_async(df, higher_df)
    sub_regime = regime
    regime = regime.split("_")[-1]
    patterns = detect_patterns(df)
    base_conf = float(probs.get(sub_regime, 0.0))
    bias_cfg = config.get("sentiment_filter", {})
    try:
        from crypto_bot.sentiment_filter import boost_factor

        bias = boost_factor(bias_cfg.get("bull_fng", 50), bias_cfg.get("bull_sentiment", 50))
    except Exception:
        bias = 1.0
    if bias > 1:
        for k in list(probs.keys()):
            if k.startswith("bullish"):
                probs[k] *= bias
        total = sum(probs.values())
        if total > 0:
            probs = {kk: vv / total for kk, vv in probs.items()}
    profile = bool(config.get("profile_regime", False))
    regime, _ = await classify_regime_cached(
        symbol,
        base_tf,
        df,
        higher_df,
        profile,
    )
    regime = regime.split("_")[-1]
    higher_df = df_map.get(higher_tf)

    if df is not None:
        regime_tmp, info = await classify_regime_async(df, higher_df)
        sub_regime = regime_tmp
        regime = regime_tmp.split("_")[-1]
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
        r = r.split("_")[-1]
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
                r = r.split("_")[-1]
                regime_counts[r] = regime_counts.get(r, 0) + 1

    if regime_counts:
        regime, votes = max(regime_counts.items(), key=lambda kv: kv[1])
    else:
        regime, votes = "unknown", 0

    denom = len(regime_tfs)
    if vote_map:
        denom *= 2
    confidence = votes / max(denom, 1)
    confidence *= base_conf
    if votes < min_agree:
        regime = "unknown"

    analysis_logger.info(
        "%s regime=%s conf=%.2f votes=%d",
        symbol,
        regime,
        confidence,
        votes,
    )

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
        "sub_regime": sub_regime,
        "patterns": patterns,
        "future_return": future_return,
        "confidence": confidence,
        "min_confidence": min_conf_adaptive,
        "probabilities": probs,
    }

    if regime != "unknown":
        env = mode if mode != "auto" else "cex"
        eval_mode = config.get("strategy_evaluation_mode", "mapped")
        cfg = {**config, "symbol": symbol}

        atr = None
        higher_df_1h = df_map.get("1h")

        def wrap(fn):
            if fn is grid_bot.generate_signal:
                return functools.partial(fn, higher_df=higher_df_1h)
            return fn

        if eval_mode == "best":
            strategies = [
                wrap(s) for s in get_strategies_for_regime(regime, router_cfg)
            ]
            res = evaluate_strategies(strategies, df, cfg)
            name = res.get("name", strategy_name(regime, env))
            score = float(res.get("score", 0.0))
            direction = res.get("direction", "none")
            if len(strategies) > 1:
                remaining = [s for s in strategies if _fn_name(s) != name]
                if remaining:
                    second = evaluate_strategies(remaining, df, cfg)
                    second_score = float(second.get("score", 0.0))
                    edge = score - second_score
                    log_second_place(
                        symbol, regime, second.get("name", ""), second_score, edge
                    )
        elif eval_mode == "ensemble":
            min_conf = float(config.get("ensemble_min_conf", 0.15))
            candidates = [wrap(strategy_for(regime, router_cfg))]
            extra = meta_selector._scores_for(regime)
            for strat_name, val in extra.items():
                if val >= min_conf:
                    fn = get_strategy_by_name(strat_name)
                    if fn:
                        fn = wrap(fn)
                        if fn not in candidates:
                            candidates.append(fn)
            ranked = await run_candidates(df, candidates, symbol, cfg, regime)
            if ranked:
                best_fn, raw_score, raw_dir = ranked[0]
                name = _fn_name(best_fn)
                score = raw_score
                direction = raw_dir if raw_score >= min_conf else "none"
                if len(ranked) > 1:
                    second = ranked[1]
                    analysis_logger.info(
                        "%s second %s %.4f %s",
                        symbol,
                        _fn_name(second[0]),
                        second[1],
                        second[2],
                    )
            else:
                name = strategy_name(regime, env)
                score = 0.0
                direction = "none"
        else:
            strategy_fn = wrap(route(regime, env, router_cfg, notifier, df_map=df_map))
            name = strategy_name(regime, env)
            score, direction, atr = (await evaluate_async([strategy_fn], df_map, cfg))[
                0
            ]

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

        result.update(
            {
                "env": env,
                "name": name,
                "score": final,
                "direction": direction,
                "atr": atr,
            }
        )

        votes = []
        voting = config.get("voting_strategies", [])
        if isinstance(voting, list):
            for strat_name in voting:
                fn = get_strategy_by_name(strat_name)
                if fn is None:
                    continue
                fn = wrap(fn)
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
