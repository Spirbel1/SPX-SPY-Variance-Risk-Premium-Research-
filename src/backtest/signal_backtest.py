from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    signal_threshold: float = 0.0
    transaction_cost_bps: float = 2.0
    weekly_signal_day: str = "FRI"


def _weekly_rebalance_mask(index: pd.DatetimeIndex, day_name: str = "FRI") -> pd.Series:
    day_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
    target = day_map.get(day_name.upper(), 4)
    return pd.Series(index.weekday == target, index=index)


def prepare_backtest_frame(
    predictions: pd.DataFrame,
    price_df: pd.DataFrame,
    prediction_col: str = "y_pred",
) -> pd.DataFrame:
    preds = predictions.copy()
    preds = preds[preds[prediction_col].notna()].copy()
    preds["date"] = pd.to_datetime(preds["date"]).dt.normalize()

    prices = price_df.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.sort_values("date")
    if "adj_close" not in prices.columns:
        prices["adj_close"] = prices["close"]
    prices["ret_1d"] = prices["adj_close"].pct_change()

    bt = preds.merge(prices[["date", "adj_close", "ret_1d"]], on="date", how="inner")
    bt = bt.sort_values("date").reset_index(drop=True)
    if bt.empty:
        raise ValueError("No valid rows in backtest frame after merging predictions and prices.")
    return bt


def add_strategy_returns(
    bt: pd.DataFrame,
    threshold: float,
    transaction_cost: float,
    mode: str = "long_flat",
) -> pd.DataFrame:
    out = bt.copy().sort_values("date").reset_index(drop=True)

    if mode == "long_flat":
        out["signal"] = np.where(out["y_pred"] > threshold, 1.0, 0.0)
    elif mode == "long_short":
        out["signal"] = 0.0
        out.loc[out["y_pred"] > threshold, "signal"] = 1.0
        out.loc[out["y_pred"] <= threshold, "signal"] = -1.0
    else:
        raise ValueError("mode must be one of {'long_flat', 'long_short'}")

    # Signal at t takes effect from next trading day.
    out["position"] = out["signal"].shift(1).fillna(0.0)
    out["turnover"] = out["position"].diff().abs().fillna(0.0)

    out["strategy_ret_gross"] = out["position"] * out["ret_1d"]
    out["tc_paid"] = out["turnover"] * transaction_cost
    out["strategy_ret_net"] = out["strategy_ret_gross"] - out["tc_paid"]

    out["benchmark_ret"] = out["ret_1d"]

    out["equity_strategy"] = (1.0 + out["strategy_ret_net"].fillna(0.0)).cumprod()
    out["equity_benchmark"] = (1.0 + out["benchmark_ret"].fillna(0.0)).cumprod()

    return out


def run_return_signal_backtest(
    price_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = price_df.copy()
    if "date" not in prices.columns:
        prices = prices.reset_index().rename(columns={prices.index.name or "index": "date"})

    preds = prediction_df[["date", "y_pred"]].copy()
    bt = prepare_backtest_frame(preds, prices, prediction_col="y_pred")

    tc = cfg.transaction_cost_bps / 10000.0
    overlay = add_strategy_returns(bt, cfg.signal_threshold, tc, mode="long_flat")
    long_short = add_strategy_returns(bt, cfg.signal_threshold, tc, mode="long_short")

    out = bt.copy()
    out["overlay_signal"] = overlay["signal"]
    out["long_short_signal"] = long_short["signal"]
    out["overlay_ret"] = overlay["strategy_ret_net"]
    out["long_short_ret"] = long_short["strategy_ret_net"]
    out["buy_hold_ret"] = overlay["benchmark_ret"]
    out["equity_overlay_ret"] = overlay["equity_strategy"]
    out["equity_long_short_ret"] = long_short["equity_strategy"]
    out["equity_buy_hold_ret"] = overlay["equity_benchmark"]
    out["overlay_turnover"] = overlay["turnover"]
    out["long_short_turnover"] = long_short["turnover"]

    trade_log = out[["date", "y_pred", "overlay_signal", "long_short_signal"]].copy()
    return out, trade_log
