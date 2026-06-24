from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SETTINGS


def _normalize_underlying_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower(): c for c in df.columns}
    if "date" not in colmap:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError("Underlying data must contain a date column.")

    price_col = None
    for candidate in ["close", "adj_close", "adj close", "settlement", "last"]:
        if candidate in colmap:
            price_col = colmap[candidate]
            break
    if price_col is None:
        raise ValueError("Underlying data must contain a close/adj_close-like price column.")

    out = df.rename(columns={colmap.get("date", "date"): "date", price_col: "underlying_price"}).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["underlying_price"] = pd.to_numeric(out["underlying_price"], errors="coerce")
    out = out.dropna(subset=["date", "underlying_price"])
    return out[["date", "underlying_price"]].drop_duplicates(subset=["date"])


def load_cached_spy_underlying_prices(
    start_date: str,
    end_date: str,
    *,
    project_root: Path | None = None,
    allow_user_cached_file: bool = True,
) -> pd.DataFrame:
    """Load SPY daily close from existing project cache sources.

    Source priority:
    1) data/processed/vrp_dataset.parquet close or adj_close
    2) data/raw/spy_ohlcv.csv
    3) data/raw/spy_ohlcv.parquet
    """
    root = project_root or SETTINGS.project_root
    sources: list[tuple[Path, str]] = []

    if SETTINGS.dataset_path.exists():
        sources.append((SETTINGS.dataset_path, "project_vrp_dataset"))

    if allow_user_cached_file:
        sources.extend(
            [
                (root / "data" / "raw" / "spy_ohlcv.parquet", "user_cached_parquet"),
                (root / "data" / "raw" / "spy_ohlcv.csv", "user_cached_csv"),
            ]
        )

    for src_path, src_name in sources:
        if not src_path.exists():
            continue
        try:
            if src_path.suffix.lower() == ".parquet":
                raw = pd.read_parquet(src_path)
            else:
                raw = pd.read_csv(src_path)
            data = _normalize_underlying_columns(raw)
            data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))].copy()
            data["underlying"] = "SPY"
            data["underlying_price_source"] = src_name
            return data
        except Exception:
            continue

    return pd.DataFrame(columns=["date", "underlying_price", "underlying", "underlying_price_source"])


def add_realized_volatility_columns(underlying_df: pd.DataFrame) -> pd.DataFrame:
    out = underlying_df.sort_values("date").copy()
    out["ret"] = np.log(out["underlying_price"] / out["underlying_price"].shift(1))
    for window in [10, 20, 30]:
        out[f"realized_vol_{window}d"] = out["ret"].rolling(window).std() * np.sqrt(252)
    return out
