from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.baselines import historical_mean_regression, majority_class_baseline
from src.models.classification import calibration_data, evaluate_classification, fit_classification_models
from src.models.regression import evaluate_regression, fit_regression_models, fit_statsmodels_linear, newey_west_tstat
from src.utils.dates import build_walk_forward_splits


@dataclass(frozen=True)
class FeatureSets:
    price: List[str]
    realized_vol: List[str]
    vrp: List[str]


def default_feature_sets() -> FeatureSets:
    return FeatureSets(
        price=[
            "log_return",
            "return_5d",
            "return_21d",
            "return_63d",
            "rolling_vol_5d",
            "rolling_vol_21d",
            "rolling_vol_63d",
            "rolling_drawdown",
            "dist_from_ma_20",
            "dist_from_ma_50",
            "dist_from_ma_200",
        ],
        realized_vol=[
            "realized_variance_5d",
            "realized_variance_21d",
            "realized_variance_63d",
            "realized_vol_annualized_5d",
            "realized_vol_annualized_21d",
            "realized_vol_annualized_63d",
        ],
        vrp=[
            "vix_level",
            "vix_daily_change",
            "vix_5d_change",
            "vix_pct_252d",
            "implied_variance",
            "VRP_5d",
            "VRP_21d",
            "VRP_63d",
            "vrp_ratio",
            "vrp_zscore_252d",
            "vrp_pct_252d",
        ],
    )


def _window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df["date"] >= start) & (df["date"] <= end)].copy()


def _normalize_df_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or "index": "date"})
        else:
            raise ValueError("Input dataframe must include a 'date' column or DatetimeIndex.")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return out


def _build_expanding_splits(
    df: pd.DataFrame,
    initial_train_years: int,
    test_window_years: int,
    step_years: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if df.empty:
        return []

    start = df["date"].min()
    end = df["date"].max()

    train_start = start
    train_end = train_start + pd.DateOffset(years=initial_train_years)

    splits: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    while train_end < end:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(years=test_window_years) - pd.Timedelta(days=1), end)
        if test_start >= test_end:
            break
        splits.append((train_start, train_end, test_start, test_end))
        train_end = train_end + pd.DateOffset(years=step_years)

    return splits


def make_model_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    required_cols = ["date"] + feature_cols + [target_col]
    base = _normalize_df_dates(df)

    missing_cols = [c for c in required_cols if c not in base.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in model frame: {missing_cols}")

    model_df = base[required_cols].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=feature_cols + [target_col])
    model_df = model_df.sort_values("date").reset_index(drop=True)
    if model_df.empty:
        raise ValueError(
            f"No valid rows after filtering for target={target_col}, features={feature_cols}"
        )
    return model_df


def run_walk_forward_regression(
    df: pd.DataFrame,
    target_col: str,
    initial_train_years: int,
    test_window_years: int,
    step_years: int,
) -> Dict[str, pd.DataFrame]:
    feature_sets = default_feature_sets()
    model_rows: List[dict] = []
    pred_rows: List[dict] = []
    coef_rows: List[dict] = []

    all_sets = {
        "price_only": feature_sets.price,
        "realized_vol_only": feature_sets.realized_vol,
        "vrp_only": feature_sets.vrp,
        "combined": feature_sets.price + feature_sets.realized_vol + feature_sets.vrp,
    }

    sample_rows: List[dict] = []

    for family_name, cols in all_sets.items():
        try:
            model_df = make_model_frame(df, cols, target_col)
        except ValueError:
            continue

        splits = _build_expanding_splits(model_df, initial_train_years, test_window_years, step_years)

        for split_i, split in enumerate(splits):
            train_start, train_end, test_start, test_end = split
            train = _window(model_df, train_start, train_end)
            test = _window(model_df, test_start, test_end)
            if train.empty or test.empty:
                continue

            y_train = train[target_col]
            y_test = test[target_col]
            x_train = train[cols]
            x_test = test[cols]

            sample_rows.append({
                "target": target_col,
                "model_family": family_name,
                "split": split_i,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

            # Baseline (historical mean) computed on this family's clean data
            baseline_pred = historical_mean_regression(y_train, len(y_test))
            base_metrics = evaluate_regression(y_test, baseline_pred, baseline_pred, dates=test["date"])
            model_rows.append({
                "split": split_i,
                "model_family": f"{family_name}_baseline_mean",
                "target": target_col,
                **base_metrics,
                "newey_west_tstat": np.nan,
            })
            for dt, yp, yt in zip(test["date"], baseline_pred, y_test):
                pred_rows.append({"date": dt, "split": split_i, "model": f"{family_name}_baseline_mean", "target": target_col, "y_pred": yp, "y_true": yt})

            preds = fit_regression_models(x_train, y_train, x_test)

            try:
                params, tvals = fit_statsmodels_linear(x_train, y_train)
                for n, v in params.items():
                    coef_rows.append({"split": split_i, "model_family": family_name, "coef_name": n, "coef_value": v, "t_value": tvals.get(n, np.nan)})
            except Exception:
                pass

            for model_name, y_pred in preds.items():
                full_name = f"{family_name}_{model_name}"
                metrics = evaluate_regression(y_test, y_pred, baseline_pred, dates=test["date"])
                try:
                    nw_t = newey_west_tstat(y_train, x_train[cols[0]]) if cols else np.nan
                except Exception:
                    nw_t = np.nan
                model_rows.append({
                    "split": split_i,
                    "model_family": full_name,
                    "target": target_col,
                    **metrics,
                    "newey_west_tstat": nw_t,
                })
                for dt, yp, yt in zip(test["date"], y_pred, y_test):
                    pred_rows.append({"date": dt, "split": split_i, "model": full_name, "target": target_col, "y_pred": yp, "y_true": yt})

    return {
        "metrics": pd.DataFrame(model_rows),
        "predictions": pd.DataFrame(pred_rows),
        "coefficients": pd.DataFrame(coef_rows),
        "sample_report": pd.DataFrame(sample_rows),
    }


def run_walk_forward_classification(
    df: pd.DataFrame,
    target_col: str,
    initial_train_years: int,
    test_window_years: int,
    step_years: int,
) -> Dict[str, pd.DataFrame]:
    feature_sets = default_feature_sets()
    model_rows: List[dict] = []
    pred_rows: List[dict] = []
    calib_rows: List[dict] = []

    all_sets = {
        "price_only": feature_sets.price,
        "vrp_only": feature_sets.vrp,
        "combined": feature_sets.price + feature_sets.realized_vol + feature_sets.vrp,
    }

    sample_rows: List[dict] = []

    for family_name, cols in all_sets.items():
        try:
            model_df = make_model_frame(df, cols, target_col)
        except ValueError:
            continue

        splits = _build_expanding_splits(model_df, initial_train_years, test_window_years, step_years)

        for split_i, split in enumerate(splits):
            train_start, train_end, test_start, test_end = split
            train = _window(model_df, train_start, train_end)
            test = _window(model_df, test_start, test_end)
            if train.empty or test.empty:
                continue

            y_train = train[target_col].astype(int)
            y_test = test[target_col].astype(int)
            x_train = train[cols]
            x_test = test[cols]

            sample_rows.append({
                "target": target_col,
                "model_family": family_name,
                "split": split_i,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

            # Baseline (majority class) computed on this family's clean data
            baseline_pred = majority_class_baseline(y_train, len(y_test))
            baseline_proba = baseline_pred.astype(float)
            base_metrics = evaluate_classification(y_test, baseline_pred, baseline_proba)
            model_rows.append({"split": split_i, "model_family": f"{family_name}_baseline_majority", "target": target_col, **base_metrics})
            for dt, yp, yt, pp in zip(test["date"], baseline_pred, y_test, baseline_proba):
                pred_rows.append({"date": dt, "split": split_i, "model": f"{family_name}_baseline_majority", "target": target_col, "y_pred": yp, "y_true": yt, "proba": pp})

            preds = fit_classification_models(x_train, y_train, x_test)

            for model_name, pred_obj in preds.items():
                full_name = f"{family_name}_{model_name}"
                proba = pred_obj["proba"]
                y_pred = pred_obj["pred"]
                metrics = evaluate_classification(y_test, y_pred, proba)
                metrics["first_prediction_date"] = test["date"].min()
                metrics["last_prediction_date"] = test["date"].max()
                model_rows.append({"split": split_i, "model_family": full_name, "target": target_col, **metrics})

                try:
                    cal = calibration_data(y_test, proba)
                    for _, row in cal.iterrows():
                        calib_rows.append({"split": split_i, "model": full_name, "target": target_col, "mean_pred": row["mean_pred"], "frac_pos": row["frac_pos"]})
                except Exception:
                    pass

                for dt, yp, yt, pp in zip(test["date"], y_pred, y_test, proba):
                    pred_rows.append({"date": dt, "split": split_i, "model": full_name, "target": target_col, "y_pred": yp, "y_true": yt, "proba": pp})

    return {
        "metrics": pd.DataFrame(model_rows),
        "predictions": pd.DataFrame(pred_rows),
        "calibration": pd.DataFrame(calib_rows),
        "sample_report": pd.DataFrame(sample_rows),
    }
