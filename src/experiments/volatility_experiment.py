from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SETTINGS
from src.features.targets import add_targets
from src.features.volatility import add_price_and_volatility_features
from src.features.vrp import add_vrp_features, rolling_percentile

VOL_REGRESSION_TARGETS = ["next_5d_realized_volatility", "next_21d_realized_volatility"]
VOL_CLASSIFICATION_TARGETS = ["next_5d_vol_expansion", "next_21d_vol_expansion"]

RV_ONLY_FEATURES = [
    "realized_vol_annualized_5d",
    "realized_vol_annualized_21d",
    "realized_vol_annualized_63d",
    "rolling_vol_5d",
    "rolling_vol_21d",
    "rolling_vol_63d",
]

VIX_ONLY_FEATURES = ["vix", "vix_change_1d", "vix_change_5d", "vix_pct_252d", "implied_variance"]

VRP_ONLY_FEATURES = ["VRP_5d", "VRP_21d", "VRP_63d", "vrp_ratio", "vrp_pct_252d", "vrp_zscore_252d"]

EXTREME_VRP_FEATURES = ["abs_vrp_zscore_252d", "high_vrp_decile", "low_vrp_decile"]

FEATURE_GROUPS = {
    "rv_only": RV_ONLY_FEATURES,
    "vix_only": VIX_ONLY_FEATURES,
    "vrp_only": VRP_ONLY_FEATURES,
    "extreme_vrp": EXTREME_VRP_FEATURES,
    "combined": list(dict.fromkeys(RV_ONLY_FEATURES + VIX_ONLY_FEATURES + VRP_ONLY_FEATURES + EXTREME_VRP_FEATURES)),
}


@dataclass(frozen=True)
class VolatilityExperimentConfig:
    initial_train_years: int = SETTINGS.initial_train_years
    test_window_years: int = SETTINGS.test_window_years
    step_years: int = SETTINGS.step_years
    random_state: int = 42


@dataclass(frozen=True)
class VolatilityExperimentOutputs:
    regression_metrics: pd.DataFrame
    regression_predictions: pd.DataFrame
    classification_metrics: pd.DataFrame
    classification_predictions: pd.DataFrame
    calibration: pd.DataFrame
    feature_importance: pd.DataFrame
    decile_study: pd.DataFrame


def vix_direct_forecast(df: pd.DataFrame) -> pd.Series:
    values = df.loc[:, "vix"]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce") / 100.0


def rv21_direct_forecast(df: pd.DataFrame) -> pd.Series:
    values = df.loc[:, "realized_vol_annualized_21d"]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce")


def avg_vix_rv21_forecast(df: pd.DataFrame) -> pd.Series:
    return 0.5 * (vix_direct_forecast(df) + rv21_direct_forecast(df))


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or "index": "date"})
        else:
            raise ValueError("Volatility experiment requires a date column or DatetimeIndex.")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_dates(df)
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    if "adj_close" not in out.columns:
        raise ValueError("Missing adj_close/close column.")
    if "vix" not in out.columns:
        raise ValueError("Missing vix column.")

    out = out.dropna(subset=["adj_close", "vix"]).copy()
    out = add_price_and_volatility_features(out, price_col="adj_close")
    out = add_vrp_features(out)
    out = add_targets(out, price_col="adj_close")
    out["rv21_pct_252d"] = rolling_percentile(out["realized_vol_annualized_21d"], 252)
    out["abs_vrp_zscore_252d"] = out["vrp_zscore_252d"].abs()
    out["high_vrp_decile"] = (out["vrp_pct_252d"] >= 0.9).astype(float)
    out["low_vrp_decile"] = (out["vrp_pct_252d"] <= 0.1).astype(float)
    return out


def _make_model_frame(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str) -> pd.DataFrame:
    required = list(dict.fromkeys(["date", target_col, "realized_vol_annualized_21d", "vix", "VRP_21d", "vrp_pct_252d"] + list(feature_cols)))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for model frame: {missing}")
    out = df[required].replace([np.inf, -np.inf], np.nan).dropna(subset=list(feature_cols) + [target_col]).copy()
    return out.sort_values("date").reset_index(drop=True)


def _split_splits(df: pd.DataFrame, cfg: VolatilityExperimentConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if df.empty:
        return []
    start = df["date"].min()
    end = df["date"].max()
    train_end = start + pd.DateOffset(years=cfg.initial_train_years)
    splits: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    while train_end < end:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(years=cfg.test_window_years) - pd.Timedelta(days=1), end)
        if test_start >= test_end:
            break
        splits.append((start, train_end, test_start, test_end))
        train_end = train_end + pd.DateOffset(years=cfg.step_years)
    return splits


def _window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df["date"] >= start) & (df["date"] <= end)].copy()


def _impute(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    train_arr = imputer.fit_transform(train_x)
    test_arr = imputer.transform(test_x)
    return (
        pd.DataFrame(train_arr, columns=train_x.columns, index=train_x.index),
        pd.DataFrame(test_arr, columns=test_x.columns, index=test_x.index),
    )


def _regression_model(model_name: str, random_state: int):
    if model_name == "linear":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    if model_name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=random_state, min_samples_leaf=10, n_jobs=-1)
    raise ValueError(model_name)


def _classification_model(model_name: str, random_state: int):
    if model_name == "logistic":
        return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=4000, solver="lbfgs", random_state=random_state))])
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=random_state, min_samples_leaf=10, n_jobs=-1)
    raise ValueError(model_name)


def _extract_importance(model: object, feature_names: Sequence[str]) -> pd.Series:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        return pd.Series(np.abs(coef), index=feature_names)
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(np.asarray(estimator.feature_importances_), index=feature_names)
    return pd.Series(dtype=float)


def _directional_vol_hit_rate(y_true: pd.Series, y_pred: pd.Series, current_rv21: pd.Series) -> float:
    d = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "current_rv21": current_rv21}).dropna()
    if d.empty:
        return np.nan
    actual = d["y_true"] > d["current_rv21"]
    predicted = d["y_pred"] > d["current_rv21"]
    return float((actual == predicted).mean())


def _oos_r2_vs_rv21(y_true: pd.Series, y_pred: pd.Series, current_rv21: pd.Series) -> float:
    d = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "current_rv21": current_rv21}).dropna()
    if d.empty:
        return np.nan
    sse_model = float(np.sum((d["y_true"] - d["y_pred"]) ** 2))
    sse_rv21 = float(np.sum((d["y_true"] - d["current_rv21"]) ** 2))
    if sse_rv21 == 0:
        return np.nan
    return 1.0 - sse_model / sse_rv21


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series, current_rv21: pd.Series, dates: pd.Series) -> dict:
    d = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "current_rv21": current_rv21, "date": dates}).dropna(subset=["y_true", "y_pred"])
    if d.empty:
        return {
            "n_predictions": 0,
            "first_prediction_date": pd.NaT,
            "last_prediction_date": pd.NaT,
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "OOS_R2_vs_rv21_baseline": np.nan,
            "information_coefficient": np.nan,
            "directional_vol_hit_rate": np.nan,
        }
    ic = d[["y_true", "y_pred"]].corr(method="spearman").iloc[0, 1] if len(d) >= 3 else np.nan
    mse = float(mean_squared_error(d["y_true"], d["y_pred"]))
    return {
        "n_predictions": int(len(d)),
        "first_prediction_date": pd.to_datetime(d["date"]).min(),
        "last_prediction_date": pd.to_datetime(d["date"]).max(),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(d["y_true"], d["y_pred"])),
        "r2": float(r2_score(d["y_true"], d["y_pred"])),
        "OOS_R2_vs_rv21_baseline": float(_oos_r2_vs_rv21(d["y_true"], d["y_pred"], d["current_rv21"])),
        "information_coefficient": float(ic) if pd.notna(ic) else np.nan,
        "directional_vol_hit_rate": float(_directional_vol_hit_rate(d["y_true"], d["y_pred"], d["current_rv21"])),
    }


def _classification_metrics(y_true: pd.Series, y_pred_class: pd.Series, y_pred_proba: pd.Series, dates: pd.Series) -> dict:
    d = pd.DataFrame({"y_true": y_true, "y_pred_class": y_pred_class, "y_pred_proba": y_pred_proba, "date": dates}).dropna(subset=["y_true", "y_pred_class", "y_pred_proba"])
    if d.empty:
        return {
            "n_predictions": 0,
            "first_prediction_date": pd.NaT,
            "last_prediction_date": pd.NaT,
            "positive_rate_actual": np.nan,
            "positive_rate_predicted": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "F1": np.nan,
            "ROC_AUC": np.nan,
            "Brier_score": np.nan,
            "average_predicted_probability": np.nan,
            "degenerate_classifier": True,
        }

    y_true_i = pd.Series(d["y_true"]).astype(int)
    pred_i = pd.Series(d["y_pred_class"]).astype(int)
    proba = pd.Series(d["y_pred_proba"]).astype(float)
    metrics = {
        "n_predictions": int(len(d)),
        "first_prediction_date": pd.to_datetime(d["date"]).min(),
        "last_prediction_date": pd.to_datetime(d["date"]).max(),
        "positive_rate_actual": float(y_true_i.mean()),
        "positive_rate_predicted": float(pred_i.mean()),
        "accuracy": float(accuracy_score(y_true_i, pred_i)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_i, pred_i)),
        "precision": float(precision_score(y_true_i, pred_i, zero_division=0)),
        "recall": float(recall_score(y_true_i, pred_i, zero_division=0)),
        "F1": float(f1_score(y_true_i, pred_i, zero_division=0)),
        "Brier_score": float(brier_score_loss(y_true_i, proba)),
        "average_predicted_probability": float(proba.mean()),
        "degenerate_classifier": bool(pred_i.nunique(dropna=True) <= 1),
    }
    try:
        metrics["ROC_AUC"] = float(roc_auc_score(y_true_i, proba))
    except Exception:
        metrics["ROC_AUC"] = np.nan
    return metrics


def _regression_fold_predictions(
    data: pd.DataFrame,
    target: str,
    feature_group: str,
    model_name: str,
    cfg: VolatilityExperimentConfig,
    benchmark_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = FEATURE_GROUPS[feature_group]
    model_df = _make_model_frame(data, features, target)
    splits = _split_splits(model_df, cfg)
    metric_rows: List[dict] = []
    pred_rows: List[dict] = []
    fi_rows: List[dict] = []

    if benchmark_fn is not None:
        bench = model_df[["date", target, "realized_vol_annualized_21d", "vix", "VRP_21d", "vrp_pct_252d"]].copy()
        bench["y_pred"] = benchmark_fn(bench)
        bench = bench.dropna(subset=[target, "y_pred"])
        metric_rows.append({"target": target, "model_name": model_name, "feature_group": "benchmark", **_regression_metrics(bench[target], bench["y_pred"], bench["realized_vol_annualized_21d"], bench["date"])})
        for row in bench.itertuples(index=False):
            pred_rows.append({
                "date": row.date,
                "target": target,
                "model_name": model_name,
                "feature_group": "benchmark",
                "y_true": getattr(row, target),
                "y_pred": row.y_pred,
                "realized_vol_annualized_21d": row.realized_vol_annualized_21d,
                "vix": row.vix,
                "VRP_21d": row.VRP_21d,
                "vrp_pct_252d": row.vrp_pct_252d,
                "train_start": pd.NaT,
                "train_end": pd.NaT,
                "test_start": pd.NaT,
                "test_end": pd.NaT,
            })
        return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(fi_rows)

    for train_start, train_end, test_start, test_end in splits:
        train = _window(model_df, train_start, train_end)
        test = _window(model_df, test_start, test_end)
        if train.empty or test.empty:
            continue
        x_train, x_test = _impute(train[features], test[features])
        y_train = train[target]
        y_test = test[target]

        model = _regression_model(model_name, cfg.random_state)
        fitted = model.fit(x_train, y_train)
        y_pred = pd.Series(fitted.predict(x_test), index=test.index)

        metric_rows.append({
            "target": target,
            "model_name": model_name,
            "feature_group": feature_group,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            **_regression_metrics(y_test, y_pred, test["realized_vol_annualized_21d"], test["date"]),
        })

        for idx, row in test.iterrows():
            pred_rows.append({
                "date": row["date"],
                "target": target,
                "model_name": model_name,
                "feature_group": feature_group,
                "y_true": row[target],
                "y_pred": float(y_pred.loc[idx]),
                "realized_vol_annualized_21d": row["realized_vol_annualized_21d"],
                "vix": row["vix"],
                "VRP_21d": row["VRP_21d"],
                "vrp_pct_252d": row["vrp_pct_252d"],
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

        importance = _extract_importance(fitted, features)
        for feature, value in importance.items():
            fi_rows.append({"target": target, "model_name": f"{feature_group}_{model_name}", "feature": feature, "importance": float(value), "fold_id": f"{train_start.date()}_{test_end.date()}"})

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(fi_rows)


def _classification_fold_predictions(
    data: pd.DataFrame,
    target: str,
    feature_group: str,
    model_name: str,
    cfg: VolatilityExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = FEATURE_GROUPS[feature_group]
    model_df = _make_model_frame(data, features, target)
    splits = _split_splits(model_df, cfg)
    metric_rows: List[dict] = []
    pred_rows: List[dict] = []
    fi_rows: List[dict] = []

    for train_start, train_end, test_start, test_end in splits:
        train = _window(model_df, train_start, train_end)
        test = _window(model_df, test_start, test_end)
        if train.empty or test.empty:
            continue
        x_train, x_test = _impute(train[features], test[features])
        y_train = train[target].astype(int)
        y_test = test[target].astype(int)

        if y_train.nunique(dropna=True) < 2:
            constant = int(y_train.mode().iloc[0]) if len(y_train.dropna()) else 0
            fitted = None
            y_pred_class = pd.Series(np.full(len(test), constant, dtype=int), index=test.index)
            y_pred_proba = pd.Series(np.full(len(test), float(constant)), index=test.index)
        else:
            model = _classification_model(model_name, cfg.random_state)
            fitted = model.fit(x_train, y_train)
            proba = fitted.predict_proba(x_test)
            if proba.ndim == 2 and proba.shape[1] > 1:
                proba = proba[:, 1]
            else:
                proba = np.full(len(test), float(y_train.mean()))
            y_pred_proba = pd.Series(proba, index=test.index)
            y_pred_class = pd.Series((y_pred_proba >= 0.5).astype(int), index=test.index)

        metric_rows.append({
            "target": target,
            "model_name": model_name,
            "feature_group": feature_group,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            **_classification_metrics(y_test, y_pred_class, y_pred_proba, test["date"]),
        })

        for idx, row in test.iterrows():
            pred_rows.append({
                "date": row["date"],
                "target": target,
                "model_name": model_name,
                "feature_group": feature_group,
                "y_true": row[target],
                "y_pred_class": int(y_pred_class.loc[idx]),
                "y_pred_proba": float(y_pred_proba.loc[idx]),
                "realized_vol_annualized_21d": row["realized_vol_annualized_21d"],
                "vix": row["vix"],
                "VRP_21d": row["VRP_21d"],
                "vrp_pct_252d": row["vrp_pct_252d"],
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

        if fitted is not None:
            importance = _extract_importance(fitted, features)
            for feature, value in importance.items():
                fi_rows.append({"target": target, "model_name": f"{feature_group}_{model_name}", "feature": feature, "importance": float(value), "fold_id": f"{train_start.date()}_{test_end.date()}"})

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(fi_rows)


def _aggregate_feature_importance(fi_df: pd.DataFrame) -> pd.DataFrame:
    if fi_df.empty:
        return pd.DataFrame(columns=["target", "model_name", "feature_group", "feature", "mean_importance", "std_importance", "rank"])
    # model_name in fi_df is formatted as "{feature_group}_{model_name}" — split to separate columns
    if "feature_group" not in fi_df.columns:
        fi_df = fi_df.copy()
        fi_df["feature_group"] = fi_df["model_name"].str.rsplit("_", n=1).str[0]
    agg = fi_df.groupby(["target", "model_name", "feature_group", "feature"], as_index=False).agg(mean_importance=("importance", "mean"), std_importance=("importance", "std"))
    agg["std_importance"] = agg["std_importance"].fillna(0.0)
    agg["rank"] = agg.groupby(["target", "model_name"])["mean_importance"].rank(ascending=False, method="dense").astype(int)
    return agg.sort_values(["target", "model_name", "rank", "feature"])


def make_volatility_decile_study(df: pd.DataFrame) -> pd.DataFrame:
    d = _prepare_dataset(df)
    d = d.dropna(subset=["vrp_pct_252d", "vix_pct_252d", "rv21_pct_252d", "next_5d_realized_volatility", "next_21d_realized_volatility", "next_5d_vol_expansion", "next_21d_vol_expansion", "next_5d_return", "next_21d_return"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["method", "decile_type", "decile", "n_obs", "avg_next_5d_vol", "median_next_5d_vol", "avg_next_21d_vol", "median_next_21d_vol", "vol_expansion_rate_5d", "vol_expansion_rate_21d", "avg_next_5d_return", "avg_next_21d_return"])

    d["vrp_decile"] = np.floor(d["vrp_pct_252d"] * 10).clip(0, 9).astype(int)
    d["vix_decile"] = np.floor(d["vix_pct_252d"] * 10).clip(0, 9).astype(int)
    d["rv21_decile"] = np.floor(d["rv21_pct_252d"] * 10).clip(0, 9).astype(int)
    rows: List[dict] = []
    for decile_type in ["vrp_decile", "vix_decile", "rv21_decile"]:
        for decile in range(10):
            g = d[d[decile_type] == decile]
            rows.append({
                "method": "historical rolling percentiles -> deciles",
                "decile_type": decile_type,
                "decile": decile,
                "n_obs": int(len(g)),
                "avg_next_5d_vol": float(g["next_5d_realized_volatility"].mean()) if not g.empty else np.nan,
                "median_next_5d_vol": float(g["next_5d_realized_volatility"].median()) if not g.empty else np.nan,
                "avg_next_21d_vol": float(g["next_21d_realized_volatility"].mean()) if not g.empty else np.nan,
                "median_next_21d_vol": float(g["next_21d_realized_volatility"].median()) if not g.empty else np.nan,
                "vol_expansion_rate_5d": float(g["next_5d_vol_expansion"].mean()) if not g.empty else np.nan,
                "vol_expansion_rate_21d": float(g["next_21d_vol_expansion"].mean()) if not g.empty else np.nan,
                "avg_next_5d_return": float(g["next_5d_return"].mean()) if not g.empty else np.nan,
                "avg_next_21d_return": float(g["next_21d_return"].mean()) if not g.empty else np.nan,
            })
    return pd.DataFrame(rows)


def run_volatility_experiment(df: pd.DataFrame, cfg: VolatilityExperimentConfig | None = None) -> VolatilityExperimentOutputs:
    cfg = cfg or VolatilityExperimentConfig()
    data = _prepare_dataset(df)

    reg_metric_frames: List[pd.DataFrame] = []
    reg_pred_frames: List[pd.DataFrame] = []
    cls_metric_frames: List[pd.DataFrame] = []
    cls_pred_frames: List[pd.DataFrame] = []
    fi_frames: List[pd.DataFrame] = []
    calibration_rows: List[dict] = []

    for target in VOL_REGRESSION_TARGETS:
        print(f"[volatility] regression target={target}")
        for bench_name, bench_fn in [
            ("vix_direct_forecast", vix_direct_forecast),
            ("rv21_direct_forecast", rv21_direct_forecast),
            ("avg_vix_rv21_forecast", avg_vix_rv21_forecast),
        ]:
            print(f"[volatility] benchmark {bench_name}")
            metrics, preds, _ = _regression_fold_predictions(data, target, "combined", bench_name, cfg, benchmark_fn=bench_fn)
            reg_metric_frames.append(metrics)
            reg_pred_frames.append(preds)

        for feature_group in ["rv_only", "vix_only", "vrp_only", "extreme_vrp", "combined"]:
            for model_name in ["linear", "ridge", "random_forest"]:
                print(f"[volatility] regression model={feature_group}/{model_name}")
                metrics, preds, fi = _regression_fold_predictions(data, target, feature_group, model_name, cfg)
                reg_metric_frames.append(metrics)
                reg_pred_frames.append(preds)
                fi_frames.append(fi)

    for target in VOL_CLASSIFICATION_TARGETS:
        print(f"[volatility] classification target={target}")
        for feature_group in ["rv_only", "vix_only", "vrp_only", "extreme_vrp", "combined"]:
            for model_name in ["logistic", "random_forest"]:
                print(f"[volatility] classification model={feature_group}/{model_name}")
                metrics, preds, fi = _classification_fold_predictions(data, target, feature_group, model_name, cfg)
                cls_metric_frames.append(metrics)
                cls_pred_frames.append(preds)
                fi_frames.append(fi)

                if not preds.empty:
                    bins = pd.cut(preds["y_pred_proba"], bins=np.linspace(0.0, 1.0, 11), include_lowest=True, labels=False)
                    for b in range(10):
                        mask = bins == b
                        calibration_rows.append({
                            "target": target,
                            "model_name": model_name,
                            "feature_group": feature_group,
                            "bin": b,
                            "n_obs": int(mask.sum()),
                            "avg_predicted_probability": float(preds.loc[mask, "y_pred_proba"].mean()) if mask.any() else np.nan,
                            "actual_positive_rate": float(preds.loc[mask, "y_true"].mean()) if mask.any() else np.nan,
                        })

    regression_metrics = pd.concat(reg_metric_frames, ignore_index=True) if reg_metric_frames else pd.DataFrame()
    regression_predictions = pd.concat(reg_pred_frames, ignore_index=True) if reg_pred_frames else pd.DataFrame()
    classification_metrics = pd.concat(cls_metric_frames, ignore_index=True) if cls_metric_frames else pd.DataFrame()
    classification_predictions = pd.concat(cls_pred_frames, ignore_index=True) if cls_pred_frames else pd.DataFrame()
    calibration = pd.DataFrame(calibration_rows)
    feature_importance = _aggregate_feature_importance(pd.concat(fi_frames, ignore_index=True) if fi_frames else pd.DataFrame())
    decile_study = make_volatility_decile_study(data)

    return VolatilityExperimentOutputs(regression_metrics, regression_predictions, classification_metrics, classification_predictions, calibration, feature_importance, decile_study)


def save_volatility_experiment_outputs(outputs: VolatilityExperimentOutputs, output_dir: Path | None = None) -> None:
    outdir = output_dir or SETTINGS.data_processed_dir
    outdir.mkdir(parents=True, exist_ok=True)
    outputs.regression_metrics.to_csv(outdir / "volatility_regression_metrics.csv", index=False)
    outputs.regression_predictions.to_csv(outdir / "volatility_regression_predictions.csv", index=False)
    outputs.classification_metrics.to_csv(outdir / "volatility_classification_metrics.csv", index=False)
    outputs.classification_predictions.to_csv(outdir / "volatility_classification_predictions.csv", index=False)
    outputs.calibration.to_csv(outdir / "volatility_calibration.csv", index=False)
    outputs.feature_importance.to_csv(outdir / "volatility_feature_importance.csv", index=False)
    outputs.decile_study.to_csv(outdir / "volatility_decile_study.csv", index=False)


def main() -> None:
    if not SETTINGS.dataset_path.exists():
        raise FileNotFoundError("Run scripts/run_pipeline.py first to create vrp_dataset.parquet.")
    df = pd.read_parquet(SETTINGS.dataset_path)
    outputs = run_volatility_experiment(df)
    save_volatility_experiment_outputs(outputs)
    print(f"Volatility experiment outputs saved in: {SETTINGS.data_processed_dir}")
