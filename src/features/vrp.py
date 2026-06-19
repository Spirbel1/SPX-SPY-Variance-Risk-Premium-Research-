from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def last_percentile(x: pd.Series) -> float:
        s = pd.Series(x)
        return float(s.rank(pct=True).iloc[-1])

    return series.rolling(window=window, min_periods=window).apply(last_percentile, raw=False)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def add_vrp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add implied-variance and variance-risk-premium features."""
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out = out.sort_values("date").reset_index(drop=True)
    else:
        out = out.sort_index()

    out["vix"] = pd.to_numeric(out["vix"], errors="coerce")

    out["vix_level"] = out["vix"]
    out["vix_daily_change"] = out["vix"].diff(1)
    out["vix_5d_change"] = out["vix"].diff(5)
    out["vix_change_1d"] = out["vix"].diff(1)
    out["vix_change_5d"] = out["vix"].diff(5)
    out["vix_pct_252d"] = rolling_percentile(out["vix"], 252)

    out["implied_variance"] = (out["vix"] / 100.0) ** 2

    for window in (5, 21, 63):
        rv_col = f"realized_variance_{window}d"
        vrp_col = f"VRP_{window}d"
        if rv_col not in out.columns:
            raise ValueError(f"Missing required column: {rv_col}")
        out[vrp_col] = out["implied_variance"] - out[rv_col]

    out["vrp_ratio"] = out["implied_variance"] / out["realized_variance_21d"]
    out["vrp_ratio"] = out["vrp_ratio"].replace([np.inf, -np.inf], np.nan)

    out["vrp_zscore_252d"] = rolling_zscore(out["VRP_21d"], 252)
    out["vrp_pct_252d"] = rolling_percentile(out["VRP_21d"], 252)
    out["abs_vrp_zscore_252d"] = out["vrp_zscore_252d"].abs()
    out["high_vrp_decile"] = (out["vrp_pct_252d"] >= 0.9).astype(float)
    out["low_vrp_decile"] = (out["vrp_pct_252d"] <= 0.1).astype(float)

    return out


def add_vix_and_vrp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper."""
    return add_vrp_features(df)
