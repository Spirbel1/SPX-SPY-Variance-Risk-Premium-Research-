from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.targets import add_targets
from src.features.volatility import add_price_features


def _make_df(n: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 120, n), index=dates)
    return pd.DataFrame({"close": close}, index=dates)


def test_next_5d_return_shift() -> None:
    df = _make_df(40)
    df = add_price_features(df)
    df = add_targets(df)

    dates = df.index
    t = dates[10]
    expected = df.loc[dates[15], "close"] / df.loc[t, "close"] - 1.0
    assert np.isclose(df.loc[t, "next_5d_return"], expected, atol=1e-12)


def test_no_future_target_on_last_rows() -> None:
    """K2: final unavailable rows must be NaN, not zero."""
    df = _make_df(50)
    df = add_price_features(df)
    df = add_targets(df)

    assert df["next_21d_return"].tail(21).isna().all()
    assert df["next_5d_return"].tail(5).isna().all()


# ---------------------------------------------------------------------------
# K2: Binary targets must be NaN where the underlying return is NaN
# ---------------------------------------------------------------------------
def test_binary_targets_nan_where_return_nan() -> None:
    """K2: next_5d_positive_return must be NaN for the final 5 rows."""
    df = _make_df(50)
    df = add_price_features(df)
    df = add_targets(df)

    assert df["next_5d_positive_return"].tail(5).isna().all(), (
        "next_5d_positive_return must be NaN for final 5 rows, not 0"
    )
    assert df["next_21d_positive_return"].tail(21).isna().all(), (
        "next_21d_positive_return must be NaN for final 21 rows, not 0"
    )


def test_binary_targets_valid_mid_sample() -> None:
    """Binary targets should be 0 or 1 (not NaN) for rows with valid returns."""
    df = _make_df(50)
    df = add_price_features(df)
    df = add_targets(df)

    mid = df["next_5d_positive_return"].dropna()
    assert set(mid.unique()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# K4: Backtest must not trade before valid predictions exist
# ---------------------------------------------------------------------------
def test_backtest_does_not_trade_before_valid_predictions() -> None:
    """K4: backtest frame starts on first valid prediction date."""
    from src.backtest.signal_backtest import BacktestConfig, run_return_signal_backtest

    dates = pd.date_range("2021-01-01", periods=60, freq="B")
    price = pd.DataFrame({"date": dates, "close": np.linspace(100, 120, 60)})

    # First 20 dates have NaN predictions, then valid predictions
    y_pred_values = [np.nan] * 20 + [0.005] * 40
    predictions = pd.DataFrame({"date": dates, "y_pred": y_pred_values})

    cfg = BacktestConfig(signal_threshold=0.0, transaction_cost_bps=2.0, weekly_signal_day="FRI")
    bt, _ = run_return_signal_backtest(price, predictions, cfg)

    first_pred_date = dates[20]
    assert pd.to_datetime(bt["date"]).min() == first_pred_date
    assert bt["y_pred"].notna().all()
