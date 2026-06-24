from __future__ import annotations

import pandas as pd

from src.options.filters import apply_default_filters


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DTE": [20, 25],
            "moneyness": [1.0, 0.99],
            "market_price": [1.5, 2.0],
            "strike": [500.0, 505.0],
            "underlying_price": [502.0, 503.0],
            "T_years": [20 / 365.0, 25 / 365.0],
        }
    )


def test_default_filters_do_not_drop_rows_when_bid_ask_missing() -> None:
    df = _base_df().assign(bid=[pd.NA, pd.NA], ask=[pd.NA, pd.NA])

    filtered, excluded = apply_default_filters(df, dte_min=18, dte_max=38, moneyness_min=0.95, moneyness_max=1.04)

    assert len(filtered) == 2
    assert len(excluded) == 0


def test_default_filters_still_enforce_bid_ask_when_present() -> None:
    df = _base_df().assign(bid=[1.0, 2.5], ask=[1.2, 2.0])

    filtered, excluded = apply_default_filters(df, dte_min=18, dte_max=38, moneyness_min=0.95, moneyness_max=1.04)

    assert len(filtered) == 1
    assert len(excluded) == 1
    assert float(filtered.iloc[0]["bid"]) == 1.0
    assert float(filtered.iloc[0]["ask"]) == 1.2
