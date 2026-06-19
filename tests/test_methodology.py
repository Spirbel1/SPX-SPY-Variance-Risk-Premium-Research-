from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.signal_backtest import add_strategy_returns, prepare_backtest_frame
from src.features.targets import add_targets
from src.features.volatility import add_price_features
from src.features.vrp import add_vix_and_vrp_features
from src.models.walk_forward import default_feature_sets, make_model_frame


def _make_market_df(n: int = 1000) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    trend = np.linspace(100.0, 180.0, n)
    cycle = 2.0 * np.sin(np.linspace(0, 12 * np.pi, n))
    close = trend + cycle
    vix = 18.0 + 4.0 * np.sin(np.linspace(0, 6 * np.pi, n))
    return pd.DataFrame({"close": close, "vix": vix}, index=dates)


def test_feature_completeness_thresholds_n1000() -> None:
    df = _make_market_df(1000)
    df = add_price_features(df)
    df = add_vix_and_vrp_features(df)

    checks = [
        "realized_variance_63d",
        "dist_from_ma_200",
        "vix_pct_252d",
        "vrp_zscore_252d",
        "vrp_pct_252d",
        "VRP_63d",
    ]

    for col in checks:
        non_missing_ratio = float(df[col].notna().mean())
        assert non_missing_ratio > 0.5, f"{col} non-missing ratio too low: {non_missing_ratio:.3f}"


def test_target_tail_nans_and_binary_values() -> None:
    df = _make_market_df(300)
    df = add_price_features(df)
    df = add_targets(df, price_col="close")

    assert df["next_5d_return"].tail(5).isna().all()
    assert df["next_21d_return"].tail(21).isna().all()

    v5 = set(df["next_5d_positive_return"].dropna().unique())
    v21 = set(df["next_21d_positive_return"].dropna().unique())
    assert v5.issubset({0.0, 1.0})
    assert v21.issubset({0.0, 1.0})


def test_model_frame_is_strictly_non_missing() -> None:
    df = _make_market_df(450)
    df = add_price_features(df)
    df = add_vix_and_vrp_features(df)
    df = add_targets(df, price_col="close")
    df = df.reset_index().rename(columns={"index": "date"})

    features = default_feature_sets().price + default_feature_sets().realized_vol + default_feature_sets().vrp
    model_df = make_model_frame(df, features, "next_5d_return")

    assert model_df[features + ["next_5d_return"]].isna().sum().sum() == 0


def test_backtest_starts_at_first_valid_prediction() -> None:
    dates = pd.date_range("2021-01-01", periods=40, freq="B")
    prices = pd.DataFrame({"date": dates, "adj_close": np.linspace(100, 110, 40)})
    preds = pd.DataFrame({"date": dates, "y_pred": [np.nan] * 7 + [0.01] * 33})

    bt = prepare_backtest_frame(preds, prices)
    assert bt["date"].min() == dates[7]
    assert bt["y_pred"].notna().all()


def test_signal_position_is_shifted_by_one_day() -> None:
    bt = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "y_pred": [1.0, -1.0, 1.0, -1.0, 1.0],
            "ret_1d": [0.01, 0.02, -0.01, 0.03, -0.02],
            "adj_close": [100, 101, 100, 103, 101],
        }
    )

    out = add_strategy_returns(bt, threshold=0.0, transaction_cost=0.0, mode="long_flat")

    expected_signal = [1.0, 0.0, 1.0, 0.0, 1.0]
    expected_position = [0.0, 1.0, 0.0, 1.0, 0.0]

    assert out["signal"].tolist() == expected_signal
    assert out["position"].tolist() == expected_position
