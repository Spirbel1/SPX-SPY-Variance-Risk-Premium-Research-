from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def load_spy_ohlcv(
    start_date: str,
    end_date: Optional[str],
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load SPY OHLCV from Yahoo Finance with local caching."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "yahoo_spy.csv"

    if path.exists() and not force_refresh:
        return pd.read_csv(path, parse_dates=["date"])

    df = yf.download("SPY", start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map).reset_index().rename(columns={"Date": "date"})
    df.to_csv(path, index=False)
    return df
