from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_OPTION_COLUMNS = [
    "date",
    "symbol",
    "underlying_price",
    "dte",
    "expiry",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "iv",
    "delta",
    "gamma",
    "vega",
    "theta",
    "volume",
    "open_interest",
]


def load_options_csv(path: str | Path) -> pd.DataFrame:
    """Load generic option-chain CSV and validate expected columns."""
    df = pd.read_csv(path, parse_dates=["date", "expiry"])
    missing = [c for c in REQUIRED_OPTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_orats_csv(path: str | Path) -> pd.DataFrame:
    return load_options_csv(path)


def load_cboe_datashop_csv(path: str | Path) -> pd.DataFrame:
    return load_options_csv(path)


def load_thetadata_csv(path: str | Path) -> pd.DataFrame:
    return load_options_csv(path)


def load_optionmetrics_csv(path: str | Path) -> pd.DataFrame:
    return load_options_csv(path)


def _nearest_by_dte(df: pd.DataFrame, target_dte: int) -> pd.DataFrame:
    d = df.copy()
    d["dte_distance"] = (d["dte"] - target_dte).abs()
    min_dist = d.groupby(["date", "symbol"])["dte_distance"].transform("min")
    return d[min_dist == d["dte_distance"]]


def compute_atm_30d_iv(df: pd.DataFrame) -> pd.Series:
    """Approximate ATM 30-day IV from chain snapshots."""
    d = _nearest_by_dte(df, 30).copy()
    d["moneyness"] = (d["strike"] / d["underlying_price"] - 1.0).abs()
    idx = d.groupby(["date", "symbol"])["moneyness"].idxmin()
    out = d.loc[idx, ["date", "symbol", "iv"]].set_index(["date", "symbol"])["iv"]
    return out


def compute_25d_put_call_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate 25-delta put/call IV and risk reversal."""
    d = _nearest_by_dte(df, 30).copy()
    d["abs_delta_dist"] = np.where(
        d["option_type"].str.upper() == "P",
        (d["delta"].abs() - 0.25).abs(),
        (d["delta"] - 0.25).abs(),
    )

    def pick(group: pd.DataFrame, side: str) -> float:
        subset = group[group["option_type"].str.upper() == side]
        if subset.empty:
            return np.nan
        return subset.loc[subset["abs_delta_dist"].idxmin(), "iv"]

    rows = []
    for (date, symbol), g in d.groupby(["date", "symbol"]):
        put_iv = pick(g, "P")
        call_iv = pick(g, "C")
        rows.append(
            {
                "date": date,
                "symbol": symbol,
                "iv_25d_put": put_iv,
                "iv_25d_call": call_iv,
                "risk_reversal_25d": call_iv - put_iv if pd.notna(call_iv) and pd.notna(put_iv) else np.nan,
                "put_skew": put_iv - call_iv if pd.notna(call_iv) and pd.notna(put_iv) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_term_structure(df: pd.DataFrame, short_dte: int = 30, long_dte: int = 60) -> pd.DataFrame:
    """Compute IV term structure short minus long tenor."""
    short_df = _nearest_by_dte(df, short_dte)
    long_df = _nearest_by_dte(df, long_dte)

    short_iv = short_df.groupby(["date", "symbol"])["iv"].mean().rename("iv_short")
    long_iv = long_df.groupby(["date", "symbol"])["iv"].mean().rename("iv_long")

    out = pd.concat([short_iv, long_iv], axis=1).reset_index()
    out["term_structure"] = out["iv_short"] - out["iv_long"]
    return out


def compute_chain_implied_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder approximation for chain-based implied variance.

    For production, replace with robust model-free implied variance integration.
    """
    atm = compute_atm_30d_iv(df).rename("atm_30d_iv").reset_index()
    atm["implied_variance_chain"] = (atm["atm_30d_iv"] / 100.0) ** 2
    return atm[["date", "symbol", "implied_variance_chain"]]
