from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.volatility_experiment import (
    VolatilityExperimentConfig,
    avg_vix_rv21_forecast,
    make_volatility_decile_study,
    rv21_direct_forecast,
    run_volatility_experiment,
    vix_direct_forecast,
)
from src.features.targets import add_targets
from src.features.volatility import add_price_and_volatility_features
from src.features.vrp import add_vrp_features


def make_large_test_df(n: int = 300) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.sin(np.linspace(0, 10, n)) + 0.1)
    vix = 18.0 + 2.5 * np.sin(np.linspace(0, 8, n))
    return pd.DataFrame({"date": dates, "close": close, "vix": vix})


def test_volatility_targets_have_expected_nan_tail() -> None:
    df = make_large_test_df(n=300)
    df = add_price_and_volatility_features(df)
    df = add_targets(df)

    assert df["next_5d_realized_volatility"].tail(5).isna().all()
    assert df["next_21d_realized_volatility"].tail(21).isna().all()


def test_vix_direct_forecast_uses_decimal_units() -> None:
    df = pd.DataFrame({"vix": [20.0]})
    forecast = vix_direct_forecast(df)
    assert forecast.iloc[0] == 0.20


def test_vol_model_predictions_include_current_rv21() -> None:
    df = make_large_test_df(n=420)
    df = add_price_and_volatility_features(df)
    df = add_vrp_features(df)
    df = add_targets(df)

    outputs = run_volatility_experiment(df, cfg=VolatilityExperimentConfig(initial_train_years=1, test_window_years=1, step_years=1))
    assert "realized_vol_annualized_21d" in outputs.regression_predictions.columns
    assert "realized_vol_annualized_21d" in outputs.classification_predictions.columns


def test_vol_expansion_targets_are_zero_one_or_nan() -> None:
    df = make_large_test_df(n=300)
    df = add_price_and_volatility_features(df)
    df = add_targets(df)

    for col in ["next_5d_vol_expansion", "next_21d_vol_expansion"]:
        vals = set(df[col].dropna().unique())
        assert vals.issubset({0.0, 1.0})


def test_volatility_decile_study_has_valid_deciles() -> None:
    df = make_large_test_df(n=1000)
    df = add_price_and_volatility_features(df)
    df = add_vrp_features(df)
    df = add_targets(df)

    out = make_volatility_decile_study(df)

    assert out["decile"].between(0, 9).all()
    assert {"vrp_decile", "vix_decile", "rv21_decile"}.issubset(set(out["decile_type"]))