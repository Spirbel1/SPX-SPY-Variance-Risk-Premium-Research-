from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config import SETTINGS
from src.data.underlying_prices import add_realized_volatility_columns, load_cached_spy_underlying_prices
from src.options.cache import (
    cache_file_paths,
    default_cache_paths,
    ensure_cache_dirs,
    get_cache_record,
    load_manifest,
    save_manifest,
    upsert_cache_record,
)
from src.options.option_chain_processing import PricingConfig, build_base_pricing_snapshot
from src.options.validation import validate_required_columns


class CacheMissingError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class DownloadResult:
    pricing_snapshot_path: Path
    raw_paths: dict[str, Path]
    cache_record: dict[str, Any] | None
    used_cache: bool


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _read_api_key(streamlit_secrets: Mapping[str, Any] | None = None) -> str | None:
    key = os.getenv("DATABENTO_API_KEY")
    if key:
        return key
    if streamlit_secrets and "DATABENTO_API_KEY" in streamlit_secrets:
        secret = streamlit_secrets.get("DATABENTO_API_KEY")
        if isinstance(secret, str) and secret.strip():
            return secret.strip()
    return None


def _import_databento() -> Any:
    try:
        import databento as db
    except Exception as exc:
        raise ImportError("databento package is required for download flow.") from exc
    return db


def _fetch_daily_schema(
    client: Any,
    *,
    dataset: str,
    schema: str,
    symbols: str | list[str],
    stype_in: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch one daily schema and return a DataFrame.

    This wrapper intentionally avoids intraday schemas.
    """
    attempts = [
        dict(dataset=dataset, schema=schema, symbols=symbols, start=start_date, end=end_date, stype_in=stype_in),
    ]

    if isinstance(symbols, str):
        attempts.append(dict(dataset=dataset, schema=schema, symbols=[symbols], start=start_date, end=end_date, stype_in=stype_in))

    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            response = client.timeseries.get_range(**kwargs)
            if hasattr(response, "to_df"):
                return response.to_df()
            if isinstance(response, pd.DataFrame):
                return response
            try:
                return pd.DataFrame(response)
            except Exception:
                return pd.DataFrame()
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise RuntimeError(f"Databento schema unavailable or request failed for schema={schema}: {last_exc}") from last_exc
    return pd.DataFrame()


def _materialize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a timestamp is present as a normal column before parquet writes.

    Databento often returns ts_event as the DataFrame index for timeseries schemas.
    If we save with index=False, that timestamp would be lost unless we first reset it.
    """
    out = df.copy()
    if "ts_event" in out.columns or "date" in out.columns:
        return out
    if isinstance(out.index, pd.DatetimeIndex):
        idx_name = out.index.name or "ts_event"
        out = out.reset_index().rename(columns={idx_name: "ts_event"})
    return out


def load_spy_options_pricing_cache(
    start_date: str,
    end_date: str,
    *,
    project_root: Path | None = None,
) -> pd.DataFrame | None:
    """Load processed cached pricing snapshot only, with no Databento calls."""
    root = project_root or SETTINGS.project_root
    paths = default_cache_paths(root)
    files = cache_file_paths(paths, start_date, end_date, symbol="SPY")
    if not files["pricing_snapshot"].exists():
        return None
    return pd.read_parquet(files["pricing_snapshot"])


def _select_limited_contract_symbols(
    definitions: pd.DataFrame,
    underlying: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    dte_min: int,
    dte_max: int,
    moneyness_min: float,
    moneyness_max: float,
    max_contract_symbols: int,
) -> list[str]:
    if definitions.empty or underlying.empty:
        return []

    sym_col = _first_existing(definitions, ["raw_symbol", "symbol", "instrument"])
    strike_col = _first_existing(definitions, ["strike_price", "strike"])
    exp_col = _first_existing(definitions, ["expiration", "expiration_date", "expiry"])
    if not sym_col or not strike_col or not exp_col:
        return []

    defs = definitions.copy()
    defs["symbol_raw"] = defs[sym_col].astype(str)
    defs["strike_norm"] = pd.to_numeric(defs[strike_col], errors="coerce")
    defs["expiration_norm"] = pd.to_datetime(defs[exp_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    defs = defs.dropna(subset=["symbol_raw", "strike_norm", "expiration_norm"])

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    # DTE limiter: keep expiries relevant to requested horizon only.
    exp_low = start_ts + timedelta(days=dte_min)
    exp_high = end_ts + timedelta(days=dte_max)
    defs = defs[(defs["expiration_norm"] >= exp_low) & (defs["expiration_norm"] <= exp_high)]

    u = underlying.copy()
    spot_low = float(u["underlying_price"].min())
    spot_high = float(u["underlying_price"].max())
    spot_mid = float(u["underlying_price"].median())

    strike_low = spot_low * float(moneyness_min)
    strike_high = spot_high * float(moneyness_max)
    defs = defs[(defs["strike_norm"] >= strike_low) & (defs["strike_norm"] <= strike_high)]

    if defs.empty:
        return []

    defs["atm_distance"] = (defs["strike_norm"] / spot_mid - 1.0).abs()
    defs = defs.sort_values(["atm_distance", "expiration_norm", "strike_norm"])
    symbols = defs["symbol_raw"].drop_duplicates().tolist()
    return symbols[:max_contract_symbols]


def download_spy_options_daily(
    start_date: str,
    end_date: str,
    *,
    force_refresh: bool = False,
    streamlit_secrets: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
    dataset: str = "OPRA.PILLAR",
    parent_symbol: str = "SPY.OPT",
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.013,
    limiter_dte_min: int = 7,
    limiter_dte_max: int = 45,
    limiter_moneyness_min: float = 0.90,
    limiter_moneyness_max: float = 1.10,
    max_contract_symbols: int = 1500,
) -> DownloadResult:
    """Download/cache daily SPY options + definitions + statistics and build processed snapshot."""
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    span_days = int((end_ts - start_ts).days) + 1
    if span_days <= 0:
        raise ValueError("end_date must be >= start_date")
    if span_days > 90:
        raise ValueError("Date range above 90 calendar days is blocked by default.")

    root = project_root or SETTINGS.project_root
    paths = default_cache_paths(root)
    ensure_cache_dirs(paths)
    files = cache_file_paths(paths, start_date, end_date, symbol="SPY")

    manifest = load_manifest(files["manifest"])
    rec = get_cache_record(
        manifest,
        symbol="SPY",
        parent_symbol=parent_symbol,
        start_date=start_date,
        end_date=end_date,
        dataset=dataset,
    )

    if not force_refresh and files["pricing_snapshot"].exists() and rec is not None:
        return DownloadResult(
            pricing_snapshot_path=files["pricing_snapshot"],
            raw_paths={k: v for k, v in files.items() if k in {"ohlcv", "definitions", "statistics"}},
            cache_record=rec,
            used_cache=True,
        )

    api_key = _read_api_key(streamlit_secrets)
    if not api_key:
        raise PermissionError("Missing Databento API key. Set DATABENTO_API_KEY in env or Streamlit secrets.")

    db = _import_databento()
    client = db.Historical(api_key)

    underlying = load_cached_spy_underlying_prices(start_date, end_date, project_root=root)
    if underlying.empty:
        raise RuntimeError(
            "Underlying SPY prices are missing. Provide cached SPY daily data in project files before pricing options."
        )
    underlying = add_realized_volatility_columns(underlying)

    # Pull definitions first and create a contract limiter to reduce billed rows.
    definitions = _fetch_daily_schema(
        client,
        dataset=dataset,
        schema="definition",
        symbols=parent_symbol,
        stype_in="parent",
        start_date=start_date,
        end_date=end_date,
    )
    if definitions.empty:
        raise RuntimeError("Databento returned empty option definitions; cannot build limiter.")

    selected_symbols = _select_limited_contract_symbols(
        definitions,
        underlying,
        start_date=start_date,
        end_date=end_date,
        dte_min=limiter_dte_min,
        dte_max=limiter_dte_max,
        moneyness_min=limiter_moneyness_min,
        moneyness_max=limiter_moneyness_max,
        max_contract_symbols=max_contract_symbols,
    )
    if not selected_symbols:
        raise RuntimeError("Limiter selected zero contracts. Adjust limiter bounds.")

    ohlcv = _fetch_daily_schema(
        client,
        dataset=dataset,
        schema="ohlcv-1d",
        symbols=selected_symbols,
        stype_in="raw_symbol",
        start_date=start_date,
        end_date=end_date,
    )
    ohlcv = _materialize_timestamp_column(ohlcv)
    if ohlcv.empty:
        raise RuntimeError("Databento returned empty daily options OHLCV for the selected period.")
    ohlcv.to_parquet(files["ohlcv"], index=False)

    statistics = pd.DataFrame()
    schemas_downloaded = ["definition", "ohlcv-1d"]

    try:
        if not definitions.empty:
            definitions.to_parquet(files["definitions"], index=False)
    except Exception:
        definitions = pd.DataFrame()

    try:
        statistics = _fetch_daily_schema(
            client,
            dataset=dataset,
            schema="statistics",
            symbols=selected_symbols,
            stype_in="raw_symbol",
            start_date=start_date,
            end_date=end_date,
        )
        statistics = _materialize_timestamp_column(statistics)
        if not statistics.empty:
            statistics.to_parquet(files["statistics"], index=False)
            schemas_downloaded.append("statistics")
    except Exception:
        statistics = pd.DataFrame()

    pricing_df = build_base_pricing_snapshot(
        raw_ohlcv=ohlcv,
        definitions=definitions,
        statistics=statistics,
        underlying_df=underlying,
        config=PricingConfig(risk_free_rate=risk_free_rate, dividend_yield=dividend_yield),
    )

    missing_required = validate_required_columns(pricing_df)
    if missing_required:
        raise RuntimeError(f"Processed options dataset missing required columns: {missing_required}")

    pricing_df.to_parquet(files["pricing_snapshot"], index=False)

    row_counts = {
        "ohlcv_rows": int(len(ohlcv)),
        "definitions_rows": int(len(definitions)),
        "statistics_rows": int(len(statistics)),
        "pricing_rows": int(len(pricing_df)),
        "limited_contract_symbols": int(len(selected_symbols)),
    }
    source_files = [str(files["ohlcv"]) ]
    if files["definitions"].exists():
        source_files.append(str(files["definitions"]))
    if files["statistics"].exists():
        source_files.append(str(files["statistics"]))

    rec = upsert_cache_record(
        manifest,
        symbol="SPY",
        parent_symbol=parent_symbol,
        start_date=start_date,
        end_date=end_date,
        dataset=dataset,
        schemas_downloaded=schemas_downloaded,
        row_counts=row_counts,
        processed_file_paths=[str(files["pricing_snapshot"])],
        source_files_used=source_files,
    )
    save_manifest(files["manifest"], manifest)

    return DownloadResult(
        pricing_snapshot_path=files["pricing_snapshot"],
        raw_paths={k: v for k, v in files.items() if k in {"ohlcv", "definitions", "statistics"}},
        cache_record=rec,
        used_cache=False,
    )
