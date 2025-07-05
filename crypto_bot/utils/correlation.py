import pandas as pd
import numpy as np


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
    """Return correlation matrix of closing prices for each symbol."""

    closes: dict[str, pd.Series] = {}
    for sym, df in df_cache.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        closes[sym] = df["close"].reset_index(drop=True)

    if not closes:
        return pd.DataFrame()

    combined = pd.DataFrame(closes)
    return combined.corr(method="pearson")
