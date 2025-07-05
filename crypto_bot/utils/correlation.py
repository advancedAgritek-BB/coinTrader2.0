import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def compute_pairwise_correlation(df_cache: dict[str, pd.DataFrame]) -> dict[tuple[str, str], float]:
    """Return Pearson correlation coefficients for each pair of symbols.

    Parameters
    ----------
    df_cache : dict[str, pd.DataFrame]
        Mapping of symbol to OHLCV DataFrame. ``close`` column must be present.
    """
    symbols = list(df_cache.keys())
    correlations: dict[tuple[str, str], float] = {}
    for i, sym1 in enumerate(symbols):
        df1 = df_cache.get(sym1)
        if df1 is None or df1.empty or "close" not in df1.columns:
            continue
        for sym2 in symbols[i + 1 :]:
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
                corr, _ = pearsonr(arr1, arr2)
            matrix.loc[sym1, sym2] = corr
            matrix.loc[sym2, sym1] = corr

    return matrix
