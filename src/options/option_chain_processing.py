from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.options.black_scholes import bsm_price
from src.options.greeks import compute_greeks_frame
from src.options.implied_vol import implied_volatility_with_status


@dataclass(frozen=True)
class PricingConfig:
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.013


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _normalize_option_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    return s.replace({"c": "call", "p": "put", "call": "call", "put": "put"})


def _normalize_option_symbol(series: pd.Series) -> pd.Series:
    # Databento OCC-like symbols can contain internal spaces in some columns.
    return series.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def _select_market_price(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["bid", "ask", "close", "settlement", "last"]:
        if col not in out.columns:
            out[col] = np.nan

    has_bid_ask = out["bid"].notna() & out["ask"].notna()
    out["mid"] = np.where(has_bid_ask, (out["bid"] + out["ask"]) / 2.0, np.nan)

    market_price = out["settlement"].copy()
    source = pd.Series(np.where(out["settlement"].notna(), "settlement", ""), index=out.index)

    close_mask = market_price.isna() & out["close"].notna()
    market_price.loc[close_mask] = out.loc[close_mask, "close"]
    source.loc[close_mask] = "close"

    last_mask = market_price.isna() & out["last"].notna()
    market_price.loc[last_mask] = out.loc[last_mask, "last"]
    source.loc[last_mask] = "last"

    mid_mask = out["mid"].notna()
    market_price.loc[mid_mask] = out.loc[mid_mask, "mid"]
    source.loc[mid_mask] = "mid"

    out["market_price"] = pd.to_numeric(market_price, errors="coerce")
    out["market_price_source"] = source.replace("", np.nan)
    return out


def _extract_contract_fields(raw: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()

    if "ts_event" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        idx_name = out.index.name or "ts_event"
        out = out.reset_index().rename(columns={idx_name: "ts_event"})

    date_col = _first_existing(out, ["date", "ts_event", "ts_recv", "timestamp"])
    if date_col is None:
        raise ValueError("Options OHLCV data missing date/timestamp column.")

    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()

    symbol_col = _first_existing(out, ["symbol", "raw_symbol", "instrument_id", "instrument"])
    out["option_symbol"] = _normalize_option_symbol(out[symbol_col]) if symbol_col else ""

    strike_col = _first_existing(definitions, ["strike", "strike_price"])
    expiry_col = _first_existing(definitions, ["expiration", "expiry", "expiration_date"])
    type_col = _first_existing(definitions, ["option_type", "put_call", "right", "instrument_class"])
    def_symbol_col = _first_existing(definitions, ["symbol", "raw_symbol", "instrument_id", "instrument"])

    if def_symbol_col and strike_col and expiry_col and type_col:
        defs = definitions.copy()
        defs["option_symbol"] = _normalize_option_symbol(defs[def_symbol_col])
        defs["strike"] = pd.to_numeric(defs[strike_col], errors="coerce")
        defs["expiration"] = pd.to_datetime(defs[expiry_col], errors="coerce").dt.tz_localize(None).dt.normalize()
        defs["option_type"] = _normalize_option_type(defs[type_col])
        out = out.merge(
            defs[["option_symbol", "strike", "expiration", "option_type"]].drop_duplicates(),
            on="option_symbol",
            how="left",
        )
    else:
        for col in ["strike", "expiration", "option_type"]:
            if col not in out.columns:
                out[col] = np.nan

    # Backfill from raw columns if present.
    if out["strike"].isna().all():
        c = _first_existing(out, ["strike", "strike_price"])
        if c:
            out["strike"] = pd.to_numeric(out[c], errors="coerce")
    if out["expiration"].isna().all():
        c = _first_existing(out, ["expiration", "expiry", "expiration_date"])
        if c:
            out["expiration"] = pd.to_datetime(out[c], errors="coerce").dt.tz_localize(None).dt.normalize()
    if out["option_type"].isna().all():
        c = _first_existing(out, ["option_type", "put_call", "right"])
        if c:
            out["option_type"] = _normalize_option_type(out[c])

    return out


def _merge_statistics(out: pd.DataFrame, statistics: pd.DataFrame) -> pd.DataFrame:
    if statistics.empty:
        if "open_interest" not in out.columns:
            out["open_interest"] = np.nan
        return out

    stats = statistics.copy()
    date_col = _first_existing(stats, ["date", "ts_event", "timestamp"])
    sym_col = _first_existing(stats, ["symbol", "raw_symbol", "instrument_id", "instrument"])
    oi_col = _first_existing(stats, ["open_interest", "oi"])
    vol_col = _first_existing(stats, ["volume", "vol"])
    settle_col = _first_existing(stats, ["settlement", "settle", "settlement_price"])

    if date_col is None or sym_col is None:
        return out

    s = pd.DataFrame()
    s["date"] = pd.to_datetime(stats[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    s["option_symbol"] = stats[sym_col].astype(str)
    if oi_col:
        s["open_interest"] = pd.to_numeric(stats[oi_col], errors="coerce")
    if vol_col:
        s["volume"] = pd.to_numeric(stats[vol_col], errors="coerce")
    if settle_col and "settlement" not in out.columns:
        s["settlement"] = pd.to_numeric(stats[settle_col], errors="coerce")

    cols = ["date", "option_symbol"] + [c for c in ["open_interest", "volume", "settlement"] if c in s.columns]
    return out.merge(s[cols].drop_duplicates(subset=["date", "option_symbol"]), on=["date", "option_symbol"], how="left")


def build_base_pricing_snapshot(
    raw_ohlcv: pd.DataFrame,
    definitions: pd.DataFrame,
    statistics: pd.DataFrame,
    underlying_df: pd.DataFrame,
    *,
    config: PricingConfig | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Build a processed daily options chain dataset with market and IV fields."""
    if raw_ohlcv.empty:
        return pd.DataFrame()

    cfg = config or PricingConfig()
    out = raw_ohlcv.copy()

    # Normalize commonly used market columns.
    for raw_name, normalized in [
        ("close", "close"),
        ("settlement", "settlement"),
        ("last", "last"),
        ("bid", "bid"),
        ("ask", "ask"),
        ("volume", "volume"),
    ]:
        col = _first_existing(out, [raw_name])
        if col:
            out[normalized] = pd.to_numeric(out[col], errors="coerce")
        elif normalized not in out.columns:
            out[normalized] = np.nan

    out = _extract_contract_fields(out, definitions)
    out = _merge_statistics(out, statistics)
    out = _select_market_price(out)

    if underlying_df.empty:
        return pd.DataFrame()

    und = underlying_df.copy()
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    out = out.merge(
        und[["date", "underlying_price", "underlying_price_source", "realized_vol_10d", "realized_vol_20d", "realized_vol_30d"]],
        on="date",
        how="left",
    )

    out["underlying"] = "SPY"
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out["option_type"] = _normalize_option_type(out["option_type"])

    out["DTE"] = (out["expiration"] - out["date"]).dt.days
    out["T_years"] = out["DTE"] / 365.0
    out["moneyness"] = out["strike"] / out["underlying_price"]
    out["log_moneyness"] = np.log(out["moneyness"])

    call_mask = out["option_type"] == "call"
    put_mask = out["option_type"] == "put"
    out["intrinsic_value"] = np.nan
    out.loc[call_mask, "intrinsic_value"] = np.maximum(out.loc[call_mask, "underlying_price"] - out.loc[call_mask, "strike"], 0.0)
    out.loc[put_mask, "intrinsic_value"] = np.maximum(out.loc[put_mask, "strike"] - out.loc[put_mask, "underlying_price"], 0.0)
    out["extrinsic_value"] = out["market_price"] - out["intrinsic_value"]

    out["risk_free_rate"] = cfg.risk_free_rate
    out["dividend_yield"] = cfg.dividend_yield

    # Solve implied volatility row-wise for robust solver status capture.
    iv_vals: list[float] = []
    iv_status: list[str] = []
    total = len(out)
    if progress and total > 0:
        print("Solving implied volatility...")

    for i, row in enumerate(out.itertuples(index=False), start=1):
        sigma, status = implied_volatility_with_status(
            market_price=float(getattr(row, "market_price", np.nan)),
            S=float(getattr(row, "underlying_price", np.nan)),
            K=float(getattr(row, "strike", np.nan)),
            T=float(getattr(row, "T_years", np.nan)),
            r=float(getattr(row, "risk_free_rate", np.nan)),
            q=float(getattr(row, "dividend_yield", np.nan)),
            option_type=str(getattr(row, "option_type", "")),
        )
        iv_vals.append(sigma)
        iv_status.append(status)

        if progress and (i == 1 or i % 5000 == 0 or i == total):
            pct = i / total
            filled = int(30 * pct)
            bar = "#" * filled + "-" * (30 - filled)
            print(f"\rIV progress [{bar}] {pct * 100:5.1f}% ({i}/{total})", end="")

    if progress and total > 0:
        print()

    out["implied_vol_market"] = iv_vals
    out["iv_solver_status"] = iv_status

    out["iv_minus_realized_vol"] = out["implied_vol_market"] - out["realized_vol_20d"]
    out["extrinsic_value_pct"] = out["extrinsic_value"] / out["market_price"].replace(0, np.nan)

    return out


def apply_pricing_mode(
    df: pd.DataFrame,
    *,
    volatility_mode: str,
    manual_volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> pd.DataFrame:
    """Apply selected volatility mode and compute BSM prices + Greeks."""
    if df.empty:
        return df.copy()

    out = df.copy()
    out["risk_free_rate"] = float(risk_free_rate)
    out["dividend_yield"] = float(dividend_yield)

    mode_key = volatility_mode.lower().strip()
    out["volatility_mode"] = mode_key

    if mode_key == "implied from market price":
        out["selected_sigma"] = out["implied_vol_market"]
    elif mode_key == "10-day realized volatility":
        out["selected_sigma"] = out["realized_vol_10d"]
    elif mode_key == "20-day realized volatility":
        out["selected_sigma"] = out["realized_vol_20d"]
    elif mode_key == "30-day realized volatility":
        out["selected_sigma"] = out["realized_vol_30d"]
    elif mode_key == "manual volatility":
        out["selected_sigma"] = float(manual_volatility)
    else:
        out["selected_sigma"] = out["implied_vol_market"]

    out["bsm_price"] = bsm_price(
        out["underlying_price"].to_numpy(dtype=float),
        out["strike"].to_numpy(dtype=float),
        out["T_years"].to_numpy(dtype=float),
        out["risk_free_rate"].to_numpy(dtype=float),
        out["dividend_yield"].to_numpy(dtype=float),
        out["selected_sigma"].to_numpy(dtype=float),
        out["option_type"].astype(str).to_numpy(),
    )

    out = compute_greeks_frame(out)

    out["price_error"] = out["market_price"] - out["bsm_price"]
    out["abs_price_error"] = out["price_error"].abs()
    out["relative_price_error"] = out["price_error"] / out["bsm_price"].replace(0, np.nan)

    return out
