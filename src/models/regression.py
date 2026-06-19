from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def information_coefficient(y_true: pd.Series, y_pred: np.ndarray) -> float:
    s = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if len(s) < 3:
        return np.nan
    return float(s["y_true"].corr(s["y_pred"], method="spearman"))


def oos_r2(y_true: pd.Series, y_pred: np.ndarray, benchmark_pred: np.ndarray) -> float:
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_bench = np.mean((y_true - benchmark_pred) ** 2)
    if mse_bench == 0:
        return np.nan
    return 1.0 - mse_model / mse_bench


def newey_west_tstat(y: pd.Series, x: pd.Series, lags: int = 5) -> float:
    """Newey-West adjusted t-stat for y ~ const + x."""
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 20:
        return np.nan
    yv = d.iloc[:, 0]
    xv = sm.add_constant(d.iloc[:, 1])
    model = sm.OLS(yv, xv).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.tvalues.iloc[1])


def fit_regression_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Fit regression model family and return predictions."""
    preds: Dict[str, np.ndarray] = {}

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    preds["linear"] = lr.predict(x_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)
    preds["ridge"] = ridge.predict(x_test)

    enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
    enet.fit(x_train, y_train)
    preds["elastic_net"] = enet.predict(x_test)

    return preds


def evaluate_regression(
    y_true: pd.Series,
    y_pred: np.ndarray,
    benchmark_pred: np.ndarray,
    dates: pd.Series | None = None,
) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    directional_hit = float((np.sign(y_true) == np.sign(y_pred)).mean()) if len(y_true) > 0 else np.nan
    out: Dict[str, float] = {
        "mse": mse,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "oos_r2": float(oos_r2(y_true, y_pred, benchmark_pred)),
        "information_coefficient": information_coefficient(y_true, y_pred),
        "directional_hit_rate": directional_hit,
        "n_predictions": float(len(y_true)),
    }
    if dates is not None and len(dates) > 0:
        out["first_prediction_date"] = pd.to_datetime(dates).min()
        out["last_prediction_date"] = pd.to_datetime(dates).max()
    return {
        **out,
    }


def fit_statsmodels_linear(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Fit OLS for coefficient and t-stat tables."""
    model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
    return model.params, model.tvalues
