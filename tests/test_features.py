from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.vrp import add_vix_and_vrp_features
from src.features.volatility import add_price_features


def _make_df(n: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100.0, 130.0, n), index=dates)
    vix = pd.Series(np.linspace(10.0, 30.0, n), index=dates)
    return pd.DataFrame({"close": close, "vix": vix}, index=dates)


def test_vrp_calculation_identity() -> None:
    df = _make_df(400)
    df = add_price_features(df)
    df = add_vix_and_vrp_features(df)

    sample = df.dropna().iloc[-1]
    expected = sample["implied_variance"] - sample["realized_variance_21d"]
    assert np.isclose(sample["VRP_21d"], expected, atol=1e-12)


def test_implied_variance_formula() -> None:
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    df = pd.DataFrame({"close": np.arange(300) + 100.0, "vix": 20.0}, index=dates)
    df = add_price_features(df)
    df = add_vix_and_vrp_features(df)

    iv = df["implied_variance"].dropna().iloc[0]
    assert np.isclose(iv, 0.04, atol=1e-12)


# ---------------------------------------------------------------------------
# K1: Long-window features must be populated when dataset > 252 rows
# ---------------------------------------------------------------------------
def test_long_window_features_populated_with_sufficient_data() -> None:
    """K1: 63-day and 252-day features must not be 100% missing for n=400."""
    df = _make_df(400)
    df = add_price_features(df)
    df = add_vix_and_vrp_features(df)

    assert df["realized_variance_63d"].notna().sum() > 0, "realized_variance_63d is 100% missing"
    assert df["rolling_vol_63d"].notna().sum() > 0, "rolling_vol_63d is 100% missing"
    assert df["realized_vol_annualized_63d"].notna().sum() > 0, "realized_vol_annualized_63d is 100% missing"
    assert df["VRP_63d"].notna().sum() > 0, "VRP_63d is 100% missing"
    assert df["dist_from_ma_200"].notna().sum() > 0, "dist_from_ma_200 is 100% missing"
    assert df["rolling_drawdown"].notna().sum() > 0, "rolling_drawdown is 100% missing"
    assert df["vix_pct_252d"].notna().sum() > 0, "vix_pct_252d is 100% missing"
    assert df["vrp_zscore_252d"].notna().sum() > 0, "vrp_zscore_252d is 100% missing"
    assert df["vrp_pct_252d"].notna().sum() > 0, "vrp_pct_252d is 100% missing"


def test_strict_min_periods_prevents_early_values() -> None:
    """Rolling features with window=63 must have NaN for the first 62 rows."""
    df = _make_df(400)
    df = add_price_features(df)

    # First 63 rows of log_return rolling window should be NaN (first row of
    # log_return itself is NaN, so rolling(63) needs rows 1..63 = first 64).
    assert df["rolling_vol_63d"].iloc[:62].isna().all()
    assert df["dist_from_ma_200"].iloc[:199].isna().all()
