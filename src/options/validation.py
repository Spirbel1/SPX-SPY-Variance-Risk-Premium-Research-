from __future__ import annotations

import pandas as pd


REQUIRED_PROCESSED_COLUMNS = [
    "date",
    "underlying",
    "option_symbol",
    "expiration",
    "strike",
    "option_type",
    "market_price",
    "market_price_source",
    "underlying_price",
    "underlying_price_source",
    "DTE",
    "T_years",
    "moneyness",
    "log_moneyness",
    "intrinsic_value",
    "extrinsic_value",
    "implied_vol_market",
    "iv_solver_status",
]


def validate_required_columns(df: pd.DataFrame, required: list[str] | None = None) -> list[str]:
    req = required or REQUIRED_PROCESSED_COLUMNS
    return [c for c in req if c not in df.columns]
