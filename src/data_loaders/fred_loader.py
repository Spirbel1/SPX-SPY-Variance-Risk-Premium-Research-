from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _fred_csv_url(series_id: str, start_date: str, end_date: Optional[str]) -> str:
    base = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    if end_date:
        return f"{base}?id={series_id}&cosd={start_date}&coed={end_date}"
    return f"{base}?id={series_id}&cosd={start_date}"


def _cache_path(raw_dir: Path, series_id: str) -> Path:
    return raw_dir / f"fred_{series_id}.csv"


def load_fred_series(
    series_id: str,
    start_date: str,
    end_date: Optional[str],
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load a FRED time series with local CSV caching."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(raw_dir, series_id)

    if path.exists() and not force_refresh:
        df = pd.read_csv(path, parse_dates=["date"])
        return df

    url = _fred_csv_url(series_id, start_date, end_date)
    df = pd.read_csv(url)

    date_col = None
    for candidate in ("DATE", "date", "observation_date"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError(f"FRED response missing date column for {series_id}: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date", series_id: "value"})
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(path, index=False)
    return df


def load_vixcls(start_date: str, end_date: Optional[str], raw_dir: Path, force_refresh: bool = False) -> pd.DataFrame:
    df = load_fred_series("VIXCLS", start_date, end_date, raw_dir, force_refresh)
    return df.rename(columns={"value": "vix"})


def load_sp500(start_date: str, end_date: Optional[str], raw_dir: Path, force_refresh: bool = False) -> pd.DataFrame:
    df = load_fred_series("SP500", start_date, end_date, raw_dir, force_refresh)
    return df.rename(columns={"value": "spx_close"})
