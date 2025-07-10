import numpy as np
import pandas as pd
import pytest
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "correlation", Path(__file__).resolve().parents[1] / "crypto_bot" / "utils" / "correlation.py"
)
correlation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(correlation)
compute_correlation_matrix = correlation.compute_correlation_matrix
compute_pairwise_correlation = correlation.compute_pairwise_correlation


def test_compute_correlation_matrix_returns_correct_coefficients():
    df_cache = {
        "A": pd.DataFrame({"close": [1, 2, 3, 4]}),
        "B": pd.DataFrame({"close": [2, 4, 6, 8]}),
        "C": pd.DataFrame({"close": [4, 3, 2, 1]}),
    }
    mat = compute_correlation_matrix(df_cache)
    assert mat.loc["A", "B"] == pytest.approx(1.0)
    assert mat.loc["A", "C"] == pytest.approx(-1.0)
    assert mat.loc["B", "C"] == pytest.approx(-1.0)
    assert mat.loc["A", "A"] == 1.0
    assert mat.loc["B", "B"] == 1.0
    assert mat.loc["C", "C"] == 1.0


def test_compute_correlation_matrix_skips_mismatched_lengths():
    df_cache = {
        "A": pd.DataFrame({"close": [1, 2, 3]}),
        "B": pd.DataFrame({"close": [1, 2]}),
    }
    mat = compute_correlation_matrix(df_cache)
    assert np.isnan(mat.loc["A", "B"])


def test_compute_pairwise_correlation_max_pairs():
    df_cache = {
        "A": pd.DataFrame({"close": [1, 2, 3]}),
        "B": pd.DataFrame({"close": [1, 2, 3]}),
        "C": pd.DataFrame({"close": [1, 2, 3]}),
    }
    result = compute_pairwise_correlation(df_cache, max_pairs=1)
    assert len(result) == 1


def test_compute_pairwise_correlation_zero_pairs():
    df_cache = {
        "A": pd.DataFrame({"close": [1, 2]}),
        "B": pd.DataFrame({"close": [1, 2]}),
    }
    result = compute_pairwise_correlation(df_cache, max_pairs=0)
    assert result == {}
