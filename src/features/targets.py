from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def future_realized_volatility(log_returns: pd.Series, horizon: int) -> pd.Series:
    """Compute forward-looking realized volatility for the next `horizon` days.

    At date t, uses log returns for days t+1 through t+horizon.
    Returns NaN for the final `horizon` dates where future data is unavailable.
    """
    future_returns = log_returns.shift(-1)
    future_vol = (
        future_returns
        .rolling(window=horizon, min_periods=horizon)
        .std()
        .shift(-(horizon - 1))
        * np.sqrt(TRADING_DAYS)
    )
    return future_vol


def add_targets(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Create future return/volatility targets with strict no-lookahead rules."""
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out = out.sort_values("date").reset_index(drop=True)
    else:
        out = out.sort_index()

    if price_col not in out.columns:
        if "close" in out.columns:
            price_col = "close"
        else:
            raise ValueError(f"Missing required price column: {price_col}")

    out["next_5d_return"] = out[price_col].shift(-5) / out[price_col] - 1.0
    out["next_21d_return"] = out[price_col].shift(-21) / out[price_col] - 1.0

    out["next_5d_positive_return"] = np.nan
    mask_5 = out["next_5d_return"].notna()
    out.loc[mask_5, "next_5d_positive_return"] = (out.loc[mask_5, "next_5d_return"] > 0).astype(float)

    out["next_21d_positive_return"] = np.nan
    mask_21 = out["next_21d_return"].notna()
    out.loc[mask_21, "next_21d_positive_return"] = (out.loc[mask_21, "next_21d_return"] > 0).astype(float)

    log_ret = out["log_return"] if "log_return" in out.columns else np.log(out[price_col] / out[price_col].shift(1))
    out["next_5d_realized_volatility"] = future_realized_volatility(log_ret, 5)
    out["next_21d_realized_volatility"] = future_realized_volatility(log_ret, 21)

    out["next_5d_vol_expansion"] = np.nan
    if "realized_vol_annualized_21d" in out.columns:
        mask_vol_5 = out["next_5d_realized_volatility"].notna() & out["realized_vol_annualized_21d"].notna()
        out.loc[mask_vol_5, "next_5d_vol_expansion"] = (
            out.loc[mask_vol_5, "next_5d_realized_volatility"] > out.loc[mask_vol_5, "realized_vol_annualized_21d"]
        ).astype(float)

    out["next_21d_vol_expansion"] = np.nan
    if "realized_vol_annualized_21d" in out.columns:
        mask_vol_21 = out["next_21d_realized_volatility"].notna() & out["realized_vol_annualized_21d"].notna()
        out.loc[mask_vol_21, "next_21d_vol_expansion"] = (
            out.loc[mask_vol_21, "next_21d_realized_volatility"] > out.loc[mask_vol_21, "realized_vol_annualized_21d"]
        ).astype(float)

    return out
