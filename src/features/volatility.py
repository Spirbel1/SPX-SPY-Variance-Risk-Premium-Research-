from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out = out.sort_values("date").reset_index(drop=True)
    else:
        out = out.sort_index()
    return out


def add_price_and_volatility_features(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Add price, realized-volatility, MA-distance, and drawdown features.

    Features are computed once on the full chronological dataset and never
    inside train/test windows.
    """
    out = _sort_chronologically(df)

    if price_col not in out.columns:
        if "close" in out.columns:
            price_col = "close"
        else:
            raise ValueError(f"Missing required price column: {price_col}")

    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")

    out["log_return"] = np.log(out[price_col] / out[price_col].shift(1))
    out["ret_1d"] = out[price_col].pct_change(1)

    out["return_5d"] = out[price_col].pct_change(5)
    out["return_21d"] = out[price_col].pct_change(21)
    out["return_63d"] = out[price_col].pct_change(63)

    for window in (5, 21, 63):
        out[f"rolling_vol_{window}d"] = out["log_return"].rolling(window=window, min_periods=window).std()
        out[f"realized_vol_annualized_{window}d"] = out[f"rolling_vol_{window}d"] * np.sqrt(TRADING_DAYS)
        out[f"realized_variance_{window}d"] = out[f"realized_vol_annualized_{window}d"] ** 2

    out["ma_20"] = out[price_col].rolling(20, min_periods=20).mean()
    out["ma_50"] = out[price_col].rolling(50, min_periods=50).mean()
    out["ma_200"] = out[price_col].rolling(200, min_periods=200).mean()

    out["dist_from_ma_20"] = out[price_col] / out["ma_20"] - 1.0
    out["dist_from_ma_50"] = out[price_col] / out["ma_50"] - 1.0
    out["dist_from_ma_200"] = out[price_col] / out["ma_200"] - 1.0

    out["rolling_max_252d"] = out[price_col].rolling(252, min_periods=252).max()
    out["rolling_drawdown"] = out[price_col] / out["rolling_max_252d"] - 1.0
    out["rv21_pct_252d"] = out["realized_vol_annualized_21d"].rolling(252, min_periods=252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    return out


def add_price_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Backward-compatible wrapper."""
    return add_price_and_volatility_features(df, price_col=price_col)
