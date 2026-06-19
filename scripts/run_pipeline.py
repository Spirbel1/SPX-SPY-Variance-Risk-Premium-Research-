from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import SETTINGS
from src.data_loaders.fred_loader import load_sp500, load_vixcls
from src.data_loaders.yahoo_loader import load_spy_ohlcv
from src.features.targets import add_targets
from src.features.volatility import add_price_and_volatility_features
from src.features.vrp import add_vrp_features
from src.models.walk_forward import run_walk_forward_classification, run_walk_forward_regression
from src.utils.validation import missing_data_report, validate_no_lookahead


def _debug_row(step: str, df: pd.DataFrame, notes: str = "") -> dict:
    out = df.copy()
    if "date" in out.columns:
        date_min = pd.to_datetime(out["date"]).min()
        date_max = pd.to_datetime(out["date"]).max()
    elif isinstance(out.index, pd.DatetimeIndex):
        date_min = out.index.min()
        date_max = out.index.max()
    else:
        date_min = pd.NaT
        date_max = pd.NaT

    def miss(col: str) -> int:
        return int(out[col].isna().sum()) if col in out.columns else int(len(out))

    return {
        "step": step,
        "n_rows": int(len(out)),
        "date_min": date_min,
        "date_max": date_max,
        "n_missing_adj_close": miss("adj_close"),
        "n_missing_vix": miss("vix"),
        "n_missing_log_return": miss("log_return"),
        "n_missing_VRP_21d": miss("VRP_21d"),
        "n_missing_VRP_63d": miss("VRP_63d"),
        "n_missing_vrp_zscore_252d": miss("vrp_zscore_252d"),
        "n_missing_next_5d_return": miss("next_5d_return"),
        "n_missing_next_21d_return": miss("next_21d_return"),
        "notes": notes,
    }


def merge_market_data(price_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    price = price_df.copy()
    vix = vix_df.copy()

    price["date"] = pd.to_datetime(price["date"]).dt.normalize()
    vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()

    price = price.drop_duplicates(subset=["date"]).sort_values("date")
    vix = vix.drop_duplicates(subset=["date"]).sort_values("date")

    price["adj_close"] = pd.to_numeric(price["adj_close"], errors="coerce")
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")

    df = price.merge(vix[["date", "vix"]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["date", "adj_close", "vix"]).reset_index(drop=True)
    return df


def build_dataset() -> pd.DataFrame:
    SETTINGS.data_raw_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.data_processed_dir.mkdir(parents=True, exist_ok=True)
    debug_rows: list[dict] = []

    spx = load_sp500(SETTINGS.start_date, SETTINGS.end_date, SETTINGS.data_raw_dir)
    vix = load_vixcls(SETTINGS.start_date, SETTINGS.end_date, SETTINGS.data_raw_dir)
    spy = load_spy_ohlcv(SETTINGS.start_date, SETTINGS.end_date, SETTINGS.data_raw_dir)

    for d in (spx, vix, spy):
        if not d.empty and "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    debug_rows.append(_debug_row("after_raw_data_load", pd.DataFrame({"date": pd.concat([spx["date"], vix["date"]], ignore_index=True)}), notes=f"spx_rows={len(spx)}, vix_rows={len(vix)}, spy_rows={len(spy)}"))

    if not spy.empty and "adj_close" in spy.columns:
        price_df = spy[[c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in spy.columns]].copy()
        notes = "primary_universe=SPY"
    else:
        price_df = spx.rename(columns={"spx_close": "adj_close"}).copy()
        price_df["close"] = price_df["adj_close"]
        notes = "primary_universe=SPX_proxy"

    merged = merge_market_data(price_df, vix)
    debug_rows.append(_debug_row("after_merge", merged, notes=notes))

    ds = add_price_and_volatility_features(merged, price_col="adj_close")
    debug_rows.append(_debug_row("after_price_vol_features", ds, notes="price/vol features computed; VRP columns not yet created"))

    ds = add_vrp_features(ds)
    debug_rows.append(_debug_row("after_vrp_features", ds, notes="VRP_21d and VRP_63d now available; warmup NaN is normal"))

    ds = add_targets(ds, price_col="adj_close")
    debug_rows.append(_debug_row("after_targets", ds, notes="future return/vol targets added with NaN tails at end"))

    ds = ds.sort_values("date").reset_index(drop=True)
    ds.to_parquet(SETTINGS.dataset_path)

    pd.DataFrame(debug_rows).to_csv(SETTINGS.data_processed_dir / "debug_pipeline_report.csv", index=False)

    miss = missing_data_report(ds)
    miss.to_csv(SETTINGS.data_processed_dir / "missing_data_report.csv")

    feature_cols = [
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
    ]
    target_cols = [
        "next_5d_return",
        "next_21d_return",
        "next_5d_positive_return",
        "next_21d_positive_return",
        "next_5d_realized_volatility",
        "next_21d_realized_volatility",
        "next_5d_vol_expansion",
        "next_21d_vol_expansion",
    ]
    validate_no_lookahead(ds, feature_cols, target_cols)

    return ds


def run_models(ds: pd.DataFrame) -> None:
    # Keep full dataset and let each walk-forward routine filter per target window.
    clean = ds.copy()

    reg_5 = run_walk_forward_regression(
        clean,
        target_col="next_5d_return",
        initial_train_years=SETTINGS.initial_train_years,
        test_window_years=SETTINGS.test_window_years,
        step_years=SETTINGS.step_years,
    )
    reg_21 = run_walk_forward_regression(
        clean,
        target_col="next_21d_return",
        initial_train_years=SETTINGS.initial_train_years,
        test_window_years=SETTINGS.test_window_years,
        step_years=SETTINGS.step_years,
    )
    cls_5 = run_walk_forward_classification(
        clean,
        target_col="next_5d_positive_return",
        initial_train_years=SETTINGS.initial_train_years,
        test_window_years=SETTINGS.test_window_years,
        step_years=SETTINGS.step_years,
    )
    cls_21 = run_walk_forward_classification(
        clean,
        target_col="next_21d_positive_return",
        initial_train_years=SETTINGS.initial_train_years,
        test_window_years=SETTINGS.test_window_years,
        step_years=SETTINGS.step_years,
    )

    reg_metrics = pd.concat([reg_5["metrics"], reg_21["metrics"]], ignore_index=True)
    reg_metrics.to_csv(
        SETTINGS.data_processed_dir / "regression_metrics.csv", index=False
    )
    reg_preds = pd.concat([reg_5["predictions"], reg_21["predictions"]], ignore_index=True)
    reg_preds.to_csv(
        SETTINGS.data_processed_dir / "regression_predictions.csv", index=False
    )
    pd.concat([reg_5["coefficients"], reg_21["coefficients"]], ignore_index=True).to_csv(
        SETTINGS.data_processed_dir / "regression_coefficients.csv", index=False
    )

    cls_metrics = pd.concat([cls_5["metrics"], cls_21["metrics"]], ignore_index=True)
    cls_metrics.to_csv(
        SETTINGS.data_processed_dir / "classification_metrics.csv", index=False
    )
    cls_preds = pd.concat([cls_5["predictions"], cls_21["predictions"]], ignore_index=True)
    cls_preds.to_csv(
        SETTINGS.data_processed_dir / "classification_predictions.csv", index=False
    )
    pd.concat([cls_5["calibration"], cls_21["calibration"]], ignore_index=True).to_csv(
        SETTINGS.data_processed_dir / "classification_calibration.csv", index=False
    )

    # Model sample report: row counts per target / model-family / split
    sample_parts = [
        reg_5.get("sample_report", pd.DataFrame()),
        reg_21.get("sample_report", pd.DataFrame()),
        cls_5.get("sample_report", pd.DataFrame()),
        cls_21.get("sample_report", pd.DataFrame()),
    ]
    non_empty = [s for s in sample_parts if not s.empty]
    sample_report = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    sample_report.to_csv(SETTINGS.data_processed_dir / "model_sample_report.csv", index=False)

    debug_path = SETTINGS.data_processed_dir / "debug_pipeline_report.csv"
    if debug_path.exists():
        dbg = pd.read_csv(debug_path)
        dbg = pd.concat(
            [
                dbg,
                pd.DataFrame(
                    [
                        {
                            "step": "after_model_frame_filtering",
                            "n_rows": int(sample_report["train_rows"].sum() + sample_report["test_rows"].sum()) if not sample_report.empty else 0,
                            "date_min": sample_report["train_start"].min() if not sample_report.empty else pd.NaT,
                            "date_max": sample_report["test_end"].max() if not sample_report.empty else pd.NaT,
                            "n_missing_adj_close": np.nan,
                            "n_missing_vix": np.nan,
                            "n_missing_log_return": np.nan,
                            "n_missing_VRP_21d": np.nan,
                            "n_missing_VRP_63d": np.nan,
                            "n_missing_vrp_zscore_252d": np.nan,
                            "n_missing_next_5d_return": np.nan,
                            "n_missing_next_21d_return": np.nan,
                            "notes": "aggregate rows across train/test splits in model_sample_report",
                        },
                        {
                            "step": "after_walk_forward_predictions",
                            "n_rows": int(len(reg_preds) + len(cls_preds)),
                            "date_min": min(pd.to_datetime(reg_preds["date"]).min() if not reg_preds.empty else pd.Timestamp.max,
                                            pd.to_datetime(cls_preds["date"]).min() if not cls_preds.empty else pd.Timestamp.max),
                            "date_max": max(pd.to_datetime(reg_preds["date"]).max() if not reg_preds.empty else pd.Timestamp.min,
                                            pd.to_datetime(cls_preds["date"]).max() if not cls_preds.empty else pd.Timestamp.min),
                            "n_missing_adj_close": np.nan,
                            "n_missing_vix": np.nan,
                            "n_missing_log_return": np.nan,
                            "n_missing_VRP_21d": np.nan,
                            "n_missing_VRP_63d": np.nan,
                            "n_missing_vrp_zscore_252d": np.nan,
                            "n_missing_next_5d_return": np.nan,
                            "n_missing_next_21d_return": np.nan,
                            "notes": "total prediction rows across regression and classification",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        dbg.to_csv(debug_path, index=False)


def main() -> None:
    ds = build_dataset()
    run_models(ds)
    print(f"Saved dataset to: {SETTINGS.dataset_path}")
    print(f"Processed outputs in: {SETTINGS.data_processed_dir}")


if __name__ == "__main__":
    main()
