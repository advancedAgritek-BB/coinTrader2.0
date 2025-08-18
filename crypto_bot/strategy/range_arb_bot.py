"""Range arbitrage strategy for low volatility markets using kernel
regression."""

from typing import Optional, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler

import ta
from scipy import stats
from crypto_bot.utils.stats import last_window_zscore
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False
    warn_ml_unavailable_once()

if ML_AVAILABLE:
    MODEL = load_model("range_arb_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def kernel_regression(df: pd.DataFrame, window: int) -> float:
    """Predict next price using Gaussian Process with RBF kernel."""
    if len(df) < window:
        return np.nan
    recent = df.iloc[-window:]
    X = recent[["close", "volume"]].values
    y = recent["close"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kernel = ConstantKernel(
        1.0, constant_value_bounds=(1e-5, 1e5)
    ) * RBF(1.0, length_scale_bounds=(1e-5, 1e5))
    gp = GaussianProcessRegressor(
        kernel=kernel, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=15
    )
    gp.fit(X_scaled, y)
    latest_features = recent[["close", "volume"]].iloc[-1].values.reshape(1, -1)
    pred, _ = gp.predict(scaler.transform(latest_features), return_std=True)
    return float(pred)


def generate_signal(
    df: pd.DataFrame,
    config: dict | None = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    **_,
) -> Tuple[float, str]:
    """Generate arb signal using kernel prediction.

    The strategy can run in any market regime, but trades are only taken
    when low volatility conditions are detected.
    """
    if df.empty:
        return 0.0, "none"

    params = config.get("range_arb_bot", {}) if config else {}
    atr_window = int(params.get("atr_window", 14))
    kr_window = int(params.get("kr_window", 20))
    z_threshold = float(params.get("z_threshold", 1.5))
    vol_z_threshold = float(
        params.get("vol_z_threshold", 1.0)
    )  # Low vol confirm
    volume_mult = float(params.get("volume_mult", 1.5))
    atr_normalization = bool(params.get("atr_normalization", True))

    lookback = max(atr_window, kr_window)
    if len(df) < lookback:
        return 0.0, "none"

    recent = df.iloc[-lookback:].copy()

    atr = ta.volatility.average_true_range(
        recent["high"], recent["low"], recent["close"], window=atr_window
    )
    vol_ma = recent["volume"].rolling(kr_window).mean()
    atr_z = pd.Series(stats.zscore(atr), index=atr.index)
    vol_z = pd.Series(stats.zscore(recent["volume"]), index=recent.index)

    atr = cache_series("atr_range", df, atr, lookback)
    atr_z = cache_series("atr_z_range", df, atr_z, lookback)
    vol_ma = cache_series("vol_ma_range", df, vol_ma, lookback)
    vol_z = cache_series("vol_z_range", df, vol_z, lookback)

    recent["atr"] = atr
    recent["atr_z"] = atr_z
    recent["vol_ma"] = vol_ma
    recent["vol_z"] = vol_z

    latest = recent.iloc[-1]

    # Ignore if not low vol
    if (
        latest["atr_z"] >= vol_z_threshold
        or latest["vol_z"] >= vol_z_threshold
    ):
        return 0.0, "none"

    # Avoid volume spikes
    if latest["volume"] > latest["vol_ma"] * volume_mult:
        return 0.0, "none"

    pred_price = kernel_regression(recent, kr_window)
    if np.isnan(pred_price):
        return 0.0, "none"
    z_val = last_window_zscore(recent["close"], lookback)
    if np.isnan(z_val):
        return 0.0, "none"
    z_dev = z_val

    score = 0.0
    direction = "none"

    if z_dev < -z_threshold:  # Undervalued, buy
        score = min(-z_dev / z_threshold, 1.0)
        direction = "long"
    elif z_dev > z_threshold:  # Overvalued, sell
        score = min(z_dev / z_threshold, 1.0)
        direction = "short"

    if score > 0:
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if atr_normalization:
            score = normalize_score_by_volatility(df, score)

    return score, direction


class regime_filter:
    """Match all market regimes.

    This filter now always returns ``True`` so the strategy is considered in
    every regime.  Additional safeguards within :func:`generate_signal`
    ensure trades only occur during suitable low volatility conditions.
    """

    @staticmethod
    def matches(regime: str) -> bool:
        """Return ``True`` for every supplied regime."""
        return True
