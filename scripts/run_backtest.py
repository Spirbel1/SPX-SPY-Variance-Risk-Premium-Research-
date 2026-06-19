from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.backtest.performance import performance_by_vix_regime, performance_by_vrp_decile, performance_by_year, summarize_performance
from src.backtest.signal_backtest import BacktestConfig, add_strategy_returns, prepare_backtest_frame, run_return_signal_backtest
from src.config import SETTINGS


def _append_debug_backtest_row(df: pd.DataFrame, notes: str) -> None:
    path = SETTINGS.data_processed_dir / "debug_pipeline_report.csv"
    if not path.exists():
        return
    dbg = pd.read_csv(path)
    row = {
        "step": "after_backtest_construction",
        "n_rows": int(len(df)),
        "date_min": pd.to_datetime(df["date"]).min() if not df.empty else pd.NaT,
        "date_max": pd.to_datetime(df["date"]).max() if not df.empty else pd.NaT,
        "n_missing_adj_close": int(df["adj_close"].isna().sum()) if "adj_close" in df.columns else np.nan,
        "n_missing_vix": int(df["vix"].isna().sum()) if "vix" in df.columns else np.nan,
        "n_missing_log_return": np.nan,
        "n_missing_VRP_21d": int(df["VRP_21d"].isna().sum()) if "VRP_21d" in df.columns else np.nan,
        "n_missing_VRP_63d": int(df["VRP_63d"].isna().sum()) if "VRP_63d" in df.columns else np.nan,
        "n_missing_vrp_zscore_252d": int(df["vrp_zscore_252d"].isna().sum()) if "vrp_zscore_252d" in df.columns else np.nan,
        "n_missing_next_5d_return": np.nan,
        "n_missing_next_21d_return": np.nan,
        "notes": notes,
    }
    dbg = pd.concat([dbg, pd.DataFrame([row])], ignore_index=True)
    dbg.to_csv(path, index=False)


def _rule_based_timeseries(ds: pd.DataFrame, tc_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = ds.copy().sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["ret_1d"] = d["adj_close"].pct_change()
    tc = tc_bps / 10000.0

    rules = {
        "buy_hold": pd.Series(1.0, index=d.index),
        "ma_200_filter": (d["dist_from_ma_200"] > 0).astype(float),
        "vix_pct_filter": (d["vix_pct_252d"] < 0.8).astype(float),
        "vrp_pct_filter": (d["vrp_pct_252d"] < 0.8).astype(float),
        "combined_filter": ((d["dist_from_ma_200"] > 0) & (d["vix_pct_252d"] < 0.8) & (d["vrp_pct_252d"] < 0.8)).astype(float),
    }

    ts = d[["date", "adj_close", "ret_1d"]].copy()
    summary_rows: list[dict] = []
    for name, signal in rules.items():
        pos = signal.shift(1).fillna(0.0)
        turnover = pos.diff().abs().fillna(0.0)
        strat = pos * d["ret_1d"] - turnover * tc
        equity = (1.0 + strat.fillna(0.0)).cumprod()

        ts[f"{name}_signal"] = signal
        ts[f"{name}_position"] = pos
        ts[f"{name}_ret"] = strat
        ts[f"equity_{name}"] = equity

        ann_ret = (1.0 + strat.dropna().mean()) ** 252 - 1 if strat.dropna().size else np.nan
        ann_vol = strat.dropna().std() * np.sqrt(252) if strat.dropna().size else np.nan
        sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
        downside = strat[strat < 0].std() * np.sqrt(252) if (strat < 0).any() else np.nan
        sortino = ann_ret / downside if downside and downside > 0 else np.nan
        dd = equity / equity.cummax() - 1
        mdd = dd.min() if not dd.empty else np.nan
        calmar = ann_ret / abs(mdd) if mdd and mdd < 0 else np.nan
        exposure = pos.mean()

        summary_rows.append(
            {
                "strategy": name,
                "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan,
                "annualized_return": float(ann_ret) if pd.notna(ann_ret) else np.nan,
                "annualized_volatility": float(ann_vol) if pd.notna(ann_vol) else np.nan,
                "sharpe_ratio": float(sharpe) if pd.notna(sharpe) else np.nan,
                "sortino_ratio": float(sortino) if pd.notna(sortino) else np.nan,
                "max_drawdown": float(mdd) if pd.notna(mdd) else np.nan,
                "calmar_ratio": float(calmar) if pd.notna(calmar) else np.nan,
                "exposure_percentage": float(exposure * 100.0) if pd.notna(exposure) else np.nan,
                "turnover": float(turnover.sum()),
                "transaction_cost_adjusted_return": float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan,
                "active_start_date": d["date"].min(),
                "active_end_date": d["date"].max(),
            }
        )

    return ts, pd.DataFrame(summary_rows)


def _vrp_decile_forward_returns(ds: pd.DataFrame) -> pd.DataFrame:
    d = ds[["date", "vrp_pct_252d", "next_5d_return", "next_21d_return", "next_5d_realized_volatility", "next_21d_realized_volatility"]].copy()
    d = d.dropna(subset=["vrp_pct_252d", "next_5d_return", "next_21d_return"])
    if d.empty:
        return pd.DataFrame()
    d["vrp_decile"] = np.clip((d["vrp_pct_252d"] * 10).astype(int), 0, 9)
    rows = []
    for decile, g in d.groupby("vrp_decile"):
        rows.append(
            {
                "vrp_decile": int(decile),
                "n_obs": int(len(g)),
                "avg_next_5d_return": float(g["next_5d_return"].mean()),
                "median_next_5d_return": float(g["next_5d_return"].median()),
                "positive_rate_5d": float((g["next_5d_return"] > 0).mean()),
                "avg_next_21d_return": float(g["next_21d_return"].mean()),
                "median_next_21d_return": float(g["next_21d_return"].median()),
                "positive_rate_21d": float((g["next_21d_return"] > 0).mean()),
                "avg_next_5d_vol": float(g["next_5d_realized_volatility"].mean()),
                "avg_next_21d_vol": float(g["next_21d_realized_volatility"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ds_path = SETTINGS.dataset_path
    pred_path = SETTINGS.data_processed_dir / "regression_predictions.csv"
    if not ds_path.exists() or not pred_path.exists():
        raise FileNotFoundError("Run scripts/run_pipeline.py first.")

    ds = pd.read_parquet(ds_path)
    ds = ds.reset_index().rename(columns={"index": "date"}) if "date" not in ds.columns else ds
    ds["date"] = pd.to_datetime(ds["date"]).dt.normalize()
    if "adj_close" not in ds.columns and "close" in ds.columns:
        ds["adj_close"] = ds["close"]

    pred = pd.read_csv(pred_path, parse_dates=["date"])
    pred = pred[(pred["target"] == "next_5d_return") & (pred["model"].str.contains("combined_linear", na=False))]
    if pred.empty:
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        pred = pred[pred["target"] == "next_5d_return"].copy()

    pred = pred.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    pred = pred[pred["y_pred"].notna()].copy()

    cfg = BacktestConfig(
        signal_threshold=0.0,
        transaction_cost_bps=SETTINGS.transaction_cost_bps,
        weekly_signal_day=SETTINGS.weekly_signal_day,
    )

    bt_df, trade_log = run_return_signal_backtest(ds[["date", "adj_close", "close"]], pred[["date", "y_pred"]], cfg)
    bt_df = bt_df.merge(ds[["date", "vix", "VRP_21d", "VRP_63d", "vrp_zscore_252d", "vix_pct_252d", "vrp_pct_252d", "dist_from_ma_200"]], on="date", how="left")

    # Active period starts at first valid prediction date; benchmark evaluated over same period only.
    active_start = pd.to_datetime(pred["date"]).min()
    active_end = pd.to_datetime(pred["date"]).max()
    bt_active = bt_df[(bt_df["date"] >= active_start) & (bt_df["date"] <= active_end)].copy()

    summary_rows = [
        summarize_performance(bt_active, "overlay_ret", "equity_overlay_ret"),
        summarize_performance(bt_active, "long_short_ret", "equity_long_short_ret"),
        summarize_performance(bt_active, "buy_hold_ret", "equity_buy_hold_ret"),
    ]
    summary = pd.DataFrame(summary_rows)
    summary["active_start_date"] = active_start
    summary["active_end_date"] = active_end

    robustness_year = performance_by_year(bt_active, "overlay_ret")
    robustness_vix = performance_by_vix_regime(bt_active, "overlay_ret", "vix")
    robustness_decile = performance_by_vrp_decile(bt_active, "overlay_ret", "VRP_21d")

    rule_ts, rule_summary = _rule_based_timeseries(ds, SETTINGS.transaction_cost_bps)
    vrp_decile = _vrp_decile_forward_returns(ds)

    bt_df.to_csv(SETTINGS.data_processed_dir / "backtest_timeseries.csv", index=False)
    trade_log.to_csv(SETTINGS.data_processed_dir / "trade_log.csv", index=False)
    summary.to_csv(SETTINGS.data_processed_dir / "backtest_summary.csv", index=False)
    robustness_year.to_csv(SETTINGS.data_processed_dir / "robustness_by_year.csv", index=False)
    robustness_vix.to_csv(SETTINGS.data_processed_dir / "robustness_by_vix_regime.csv", index=False)
    robustness_decile.to_csv(SETTINGS.data_processed_dir / "robustness_by_vrp_decile.csv", index=False)
    rule_summary.to_csv(SETTINGS.data_processed_dir / "rule_based_strategy_summary.csv", index=False)
    rule_ts.to_csv(SETTINGS.data_processed_dir / "rule_based_strategy_timeseries.csv", index=False)
    vrp_decile.to_csv(SETTINGS.data_processed_dir / "vrp_decile_forward_returns.csv", index=False)

    _append_debug_backtest_row(
        bt_active,
        notes=f"active_period={active_start.date()}..{active_end.date()}, rows={len(bt_active)}",
    )

    print(f"Backtest outputs saved in: {SETTINGS.data_processed_dir}")


if __name__ == "__main__":
    main()
