from __future__ import annotations

import pandas as pd


def apply_default_filters(
    df: pd.DataFrame,
    *,
    dte_min: int = 1,
    dte_max: int = 60,
    moneyness_min: float = 0.80,
    moneyness_max: float = 1.20,
    max_bid_ask_spread_pct: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply default options filters and return (filtered, excluded)."""
    if df.empty:
        return df.copy(), df.copy()

    mask = (
        (df["DTE"] >= dte_min)
        & (df["DTE"] <= dte_max)
        & (df["moneyness"] >= moneyness_min)
        & (df["moneyness"] <= moneyness_max)
        & (df["market_price"] > 0)
        & (df["strike"] > 0)
        & (df["underlying_price"] > 0)
        & (df["T_years"] > 0)
    )

    has_bid_ask = {"bid", "ask"}.issubset(df.columns)
    if has_bid_ask:
        # Keep rows with missing quotes; only enforce quote checks where bid/ask exist.
        has_quotes = df["bid"].notna() & df["ask"].notna()
        quote_ok = (~has_quotes) | ((df["bid"] >= 0) & (df["ask"] >= 0) & (df["ask"] >= df["bid"]))
        mask = mask & quote_ok
        if max_bid_ask_spread_pct is not None:
            spread_pct = (df["ask"] - df["bid"]) / ((df["ask"] + df["bid"]) / 2).replace(0, pd.NA)
            spread_ok = (~has_quotes) | (spread_pct <= max_bid_ask_spread_pct)
            mask = mask & spread_ok

    filtered = df[mask].copy()
    excluded = df[~mask].copy()
    return filtered, excluded
