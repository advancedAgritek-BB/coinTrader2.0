import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def _welford_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Return correlation of ``x`` and ``y`` using Welford's algorithm."""
    mean_x = mean_y = var_x = var_y = cov = 0.0
    n = 0
    for a, b in zip(x, y):
        n += 1
        dx = a - mean_x
        mean_x += dx / n
        dy = b - mean_y
        mean_y += dy / n
        cov += dx * (b - mean_y)
        var_x += dx * (a - mean_x)
        var_y += dy * (b - mean_y)
    if n < 2:
        return 0.0
    if var_x == 0 or var_y == 0:
        return 1.0
    return cov / np.sqrt(var_x * var_y)


def compute_pairwise_correlation(
    df_cache: dict[str, pd.DataFrame], max_pairs: int | None = None
) -> dict[tuple[str, str], float]:
    """Return Pearson correlation coefficients for each pair of symbols.

    Parameters
    ----------
    df_cache : dict[str, pd.DataFrame]
        Mapping of symbol to OHLCV DataFrame. ``close`` column must be present.
    max_pairs : int, optional
        Maximum number of pairwise correlations to compute. When ``None`` all
        possible pairs are evaluated.
    """
    symbols = list(df_cache.keys())
    correlations: dict[tuple[str, str], float] = {}
    computed = 0
    for i, sym1 in enumerate(symbols):
        if max_pairs is not None and computed >= max_pairs:
            break
        df1 = df_cache.get(sym1)
        if df1 is None or df1.empty or "close" not in df1.columns:
            continue
        for sym2 in symbols[i + 1 :]:
            if max_pairs is not None and computed >= max_pairs:
                break
            df2 = df_cache.get(sym2)
            if df2 is None or df2.empty or "close" not in df2.columns:
                continue
            n = min(len(df1), len(df2))
            if n < 2:
                corr = 0.0
            else:
                s1 = df1["close"].tail(n).to_numpy()
                s2 = df2["close"].tail(n).to_numpy()
                if np.std(s1) == 0 or np.std(s2) == 0:
                    corr = 1.0
                else:
                    corr = float(np.corrcoef(s1, s2)[0, 1])
            correlations[(sym1, sym2)] = corr
            computed += 1
    return correlations


def compute_correlation_matrix(df_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a symmetric correlation matrix of closing prices.

    Parameters
    ----------
    df_cache : dict[str, pd.DataFrame]
        Mapping of symbol to OHLCV DataFrame. ``close`` column must be present.

    Notes
    -----
    Only pairs with matching lengths are considered. Pairs with mismatched
    lengths are left as ``NaN`` in the resulting matrix.
    """

    closes: dict[str, pd.Series] = {}
    for sym, df in df_cache.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        closes[sym] = df["close"].reset_index(drop=True)

    if not closes:
        return pd.DataFrame()

    symbols = list(closes.keys())
    matrix = pd.DataFrame(np.nan, index=symbols, columns=symbols, dtype=float)

    for i, sym1 in enumerate(symbols):
        s1 = closes[sym1]
        matrix.loc[sym1, sym1] = 1.0
        for sym2 in symbols[i + 1 :]:
            s2 = closes[sym2]
            if len(s1) != len(s2) or len(s1) < 2:
                # skip mismatched or insufficient lengths
                continue
            arr1 = s1.to_numpy()
            arr2 = s2.to_numpy()
            if np.std(arr1) == 0 or np.std(arr2) == 0:
                corr = 1.0
            else:
                corr = float(np.corrcoef(arr1, arr2)[0, 1])
            matrix.loc[sym1, sym2] = corr
            matrix.loc[sym2, sym1] = corr

    return matrix


def incremental_correlation(
    df_cache: dict[str, pd.DataFrame], window: int = 100
) -> dict[tuple[str, str], float]:
    """Return correlation of returns for each pair of symbols.

    Parameters
    ----------
    df_cache : dict[str, pd.DataFrame]
        Mapping of symbol to DataFrame containing a ``return`` column.
    window : int, optional
        Number of most recent returns to consider.
    """

    returns: dict[str, np.ndarray] = {}
    for sym, df in df_cache.items():
        if df is None or df.empty or "return" not in df.columns:
            continue
        arr = df["return"].dropna().to_numpy()
        if arr.size:
            returns[sym] = arr[-window:]

    symbols = list(returns.keys())
    correlations: dict[tuple[str, str], float] = {}
    for i, sym1 in enumerate(symbols):
        r1 = returns[sym1]
        for sym2 in symbols[i + 1 :]:
            r2 = returns[sym2]
            n = min(len(r1), len(r2))
            if n < 2:
                corr = 0.0
            else:
                corr = _welford_corr(r1[-n:], r2[-n:])
            correlations[(sym1, sym2)] = corr
    return correlations
