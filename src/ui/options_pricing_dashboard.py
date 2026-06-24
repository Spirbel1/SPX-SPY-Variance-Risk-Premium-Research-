from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import SETTINGS
from src.data.databento_options_daily import load_spy_options_pricing_cache
from src.options.cache import cache_file_paths, default_cache_paths, load_manifest, get_cache_record
from src.options.filters import apply_default_filters
from src.options.option_chain_processing import apply_pricing_mode


VOL_MODES = [
    "implied from market price",
    "10-day realized volatility",
    "20-day realized volatility",
    "30-day realized volatility",
    "manual volatility",
]

# Temporary fixed period to keep Streamlit strictly cache-only.
FIXED_START_DATE = "2026-05-23"
FIXED_END_DATE = "2026-06-23"


def _get_manifest_record(start_date: str, end_date: str) -> dict | None:
    paths = default_cache_paths(SETTINGS.project_root)
    files = cache_file_paths(paths, start_date, end_date, symbol="SPY")
    manifest = load_manifest(files["manifest"])
    return get_cache_record(
        manifest,
        symbol="SPY",
        parent_symbol="SPY.OPT",
        start_date=start_date,
        end_date=end_date,
        dataset="OPRA.PILLAR",
    )


def _cache_status(start_date: str, end_date: str) -> dict[str, object]:
    paths = default_cache_paths(SETTINGS.project_root)
    files = cache_file_paths(paths, start_date, end_date, symbol="SPY")
    return {
        "files": files,
        "processed_exists": files["pricing_snapshot"].exists(),
        "raw_ohlcv_exists": files["ohlcv"].exists(),
        "raw_definitions_exists": files["definitions"].exists(),
        "raw_statistics_exists": files["statistics"].exists(),
        "record": _get_manifest_record(start_date, end_date),
    }


def _render_cache_panel(status: dict[str, object], start_date: str, end_date: str) -> None:
    st.subheader("Cache Status")
    files = status["files"]
    st.write(f"Selected period: {start_date} to {end_date}")
    st.write(f"Processed cache available: {bool(status['processed_exists'])}")
    st.write(f"Raw OHLCV cache available: {bool(status['raw_ohlcv_exists'])}")
    st.write(f"Raw definitions cache available: {bool(status['raw_definitions_exists'])}")
    st.write(f"Raw statistics cache available: {bool(status['raw_statistics_exists'])}")

    st.caption(f"Processed cache file: {files['pricing_snapshot']}")
    st.caption(f"Raw OHLCV file: {files['ohlcv']}")
    st.caption(f"Raw definitions file: {files['definitions']}")
    st.caption(f"Raw statistics file: {files['statistics']}")

    rec = status.get("record")
    if rec:
        st.json(rec)
    else:
        st.warning("No manifest record for this range. Load cache may fail unless files exist.")

    if not bool(status["processed_exists"]):
        st.warning("Selected date range is not cached yet. Use Download / Update Databento Cache.")


def _select_chain_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "expiration",
        "DTE",
        "option_type",
        "strike",
        "moneyness",
        "market_price",
        "market_price_source",
        "implied_vol_market",
        "selected_sigma",
        "bsm_price",
        "price_error",
        "relative_price_error",
        "bsm_delta",
        "bsm_gamma",
        "bsm_vega_per_1pct",
        "bsm_theta_per_day",
        "bsm_rho_per_1pct",
        "volume",
        "open_interest",
    ]
    return df[[c for c in cols if c in df.columns]]


def render_options_pricing_dashboard() -> None:
    st.subheader("Options Pricing & Greeks")
    st.caption(
        "Daily SPY options pricing education module. Uses Databento raw daily data, local cache, "
        "Black-Scholes-Merton pricing, implied volatility, and Greeks."
    )

    start_s = FIXED_START_DATE
    end_s = FIXED_END_DATE
    st.info(f"Fixed options period (cache-only mode): {start_s} to {end_s}")
    st.text_input("Symbol", value="SPY", disabled=True, key="opt_symbol")

    load_clicked = st.button("Load from cache", key="opt_load_cache")

    status = _cache_status(start_s, end_s)
    _render_cache_panel(status, start_s, end_s)

    if load_clicked:
        cached = load_spy_options_pricing_cache(start_s, end_s)
        if cached is None or cached.empty:
            st.error("No cached pricing snapshot found for the selected range.")
        else:
            st.session_state["options_pricing_base_df"] = cached
            st.session_state["options_pricing_loaded_range"] = (start_s, end_s)
            st.success(f"Loaded {len(cached):,} rows from local cache.")

    st.caption(
        "Dynamic Databento downloads are disabled in Streamlit for now. "
        "Refresh cache from terminal if needed."
    )

    base_df = st.session_state.get("options_pricing_base_df")
    loaded_range = st.session_state.get("options_pricing_loaded_range")
    if base_df is None:
        st.info("No options pricing data loaded in this session yet. Use Load from cache.")
        return

    if loaded_range != (start_s, end_s):
        st.warning(
            "Loaded in-memory data is from a different date range than current controls. "
            "Use Load from cache to switch ranges."
        )

    st.caption(
        "Black-Scholes and Greeks are computed locally after cache load using the selected mode/rates. "
        "Core logic: src/options/black_scholes.py, src/options/implied_vol.py, src/options/greeks.py, "
        "and src/options/option_chain_processing.py."
    )

    st.subheader("Data Controls")
    if st.button("Reset filters to defaults", key="opt_reset_filters"):
        st.session_state["opt_dte_range"] = (1, 60)
        st.session_state["opt_mny_range"] = (0.80, 1.20)
        st.session_state["opt_vol_mode"] = "implied from market price"

    dte_range = st.slider("DTE filter", min_value=1, max_value=60, value=(1, 60), key="opt_dte_range")
    moneyness_range = st.slider("Moneyness filter", min_value=0.80, max_value=1.20, value=(0.80, 1.20), key="opt_mny_range")
    vol_mode = st.selectbox("Volatility mode", options=VOL_MODES, index=0, key="opt_vol_mode")

    mcol1, mcol2, mcol3 = st.columns([1, 1, 1])
    with mcol1:
        risk_free_rate = st.number_input("Risk-free rate (annual)", value=0.04, step=0.005, format="%.4f", key="opt_r")
    with mcol2:
        dividend_yield = st.number_input("Dividend yield (annual)", value=0.013, step=0.001, format="%.4f", key="opt_q")
    with mcol3:
        manual_vol = st.number_input("Manual volatility", value=0.20, step=0.01, format="%.4f", key="opt_manual_sigma")

    modeled = apply_pricing_mode(
        base_df,
        volatility_mode=vol_mode,
        manual_volatility=float(manual_vol),
        risk_free_rate=float(risk_free_rate),
        dividend_yield=float(dividend_yield),
    )

    filtered, excluded = apply_default_filters(
        modeled,
        dte_min=dte_range[0],
        dte_max=dte_range[1],
        moneyness_min=float(moneyness_range[0]),
        moneyness_max=float(moneyness_range[1]),
    )

    st.caption(f"Rows before filters: {len(modeled):,} | after filters: {len(filtered):,}")

    st.subheader("Data Quality Summary")
    if filtered.empty:
        st.warning("No rows after filtering. Relax DTE/moneyness filters or load a different date range.")
        # Show quick diagnostics so users can see which rule is binding.
        checks = {
            "DTE in range": (modeled["DTE"] >= dte_range[0]) & (modeled["DTE"] <= dte_range[1]),
            "Moneyness in range": (modeled["moneyness"] >= float(moneyness_range[0])) & (modeled["moneyness"] <= float(moneyness_range[1])),
            "Market price > 0": modeled["market_price"] > 0,
            "Strike > 0": modeled["strike"] > 0,
            "Underlying price > 0": modeled["underlying_price"] > 0,
            "T_years > 0": modeled["T_years"] > 0,
        }
        if {"bid", "ask"}.issubset(modeled.columns):
            has_quotes = modeled["bid"].notna() & modeled["ask"].notna()
            checks["Bid/ask usable when present"] = (~has_quotes) | (
                (modeled["bid"] >= 0)
                & (modeled["ask"] >= 0)
                & (modeled["ask"] >= modeled["bid"])
            )
        diag = pd.DataFrame(
            {
                "rule": list(checks.keys()),
                "rows_passing": [int(mask.fillna(False).sum()) for mask in checks.values()],
                "rows_failing": [int((~mask.fillna(False)).sum()) for mask in checks.values()],
            }
        )
        st.dataframe(diag, width="stretch")
        return

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Contracts", f"{len(filtered):,}")
    q1.metric("Calls", f"{int((filtered['option_type'] == 'call').sum()):,}")
    q2.metric("Puts", f"{int((filtered['option_type'] == 'put').sum()):,}")
    q2.metric("Expiries", f"{filtered['expiration'].nunique():,}")
    q3.metric("DTE min/max", f"{int(filtered['DTE'].min())} / {int(filtered['DTE'].max())}")
    q3.metric("Rows excluded", f"{len(excluded):,}")
    q4.metric("IV solver failures", f"{int((filtered['iv_solver_status'] != 'ok').sum()):,}")
    q4.metric("Median market price", f"{float(filtered['market_price'].median()):.4f}")

    if {"bid", "ask"}.issubset(filtered.columns):
        spread_pct = (filtered["ask"] - filtered["bid"]) / ((filtered["ask"] + filtered["bid"]) / 2).replace(0, np.nan)
        st.caption(f"Median bid-ask spread %: {float(spread_pct.median() * 100):.2f}%")

    st.caption("Options data source: Databento OPRA.PILLAR (daily schemas only)")
    source_name = str(filtered["underlying_price_source"].dropna().iloc[0]) if filtered["underlying_price_source"].notna().any() else "unknown"
    st.caption(f"Underlying price source: {source_name}")

    st.subheader("Option-chain Table")
    st.dataframe(_select_chain_columns(filtered).sort_values(["date", "expiration", "strike"]), width="stretch")

    st.subheader("Single-contract Inspector")
    f_date = st.selectbox("Date", sorted(filtered["date"].dropna().astype(str).unique()), key="opt_ins_date")
    date_slice = filtered[filtered["date"].astype(str) == f_date]
    f_exp = st.selectbox("Expiration", sorted(date_slice["expiration"].dropna().astype(str).unique()), key="opt_ins_exp")
    exp_slice = date_slice[date_slice["expiration"].astype(str) == f_exp]
    f_type = st.selectbox("Option type", sorted(exp_slice["option_type"].dropna().unique()), key="opt_ins_type")
    type_slice = exp_slice[exp_slice["option_type"] == f_type]
    strikes = sorted(type_slice["strike"].dropna().astype(float).unique())
    f_strike = st.selectbox("Strike", strikes, key="opt_ins_strike")

    row = type_slice[type_slice["strike"].astype(float) == float(f_strike)].head(1)
    if not row.empty:
        r = row.iloc[0]
        i1, i2, i3 = st.columns(3)
        i1.metric("Market price", f"{float(r['market_price']):.4f}")
        i1.caption(f"Source: {r.get('market_price_source', 'n/a')}")
        i1.metric("Intrinsic", f"{float(r['intrinsic_value']):.4f}")
        i1.metric("Extrinsic", f"{float(r['extrinsic_value']):.4f}")

        i2.metric("Implied volatility", f"{float(r['implied_vol_market']):.4f}" if pd.notna(r["implied_vol_market"]) else "nan")
        i2.metric("Selected sigma", f"{float(r['selected_sigma']):.4f}" if pd.notna(r["selected_sigma"]) else "nan")
        i2.metric("BSM price", f"{float(r['bsm_price']):.4f}" if pd.notna(r["bsm_price"]) else "nan")
        i2.metric("Price error", f"{float(r['price_error']):.4f}" if pd.notna(r["price_error"]) else "nan")

        i3.metric("Delta", f"{float(r['bsm_delta']):.4f}" if pd.notna(r["bsm_delta"]) else "nan")
        i3.metric("Gamma", f"{float(r['bsm_gamma']):.6f}" if pd.notna(r["bsm_gamma"]) else "nan")
        i3.metric("Vega per 1%", f"{float(r['bsm_vega_per_1pct']):.4f}" if pd.notna(r["bsm_vega_per_1pct"]) else "nan")
        i3.metric("Theta/day", f"{float(r['bsm_theta_per_day']):.4f}" if pd.notna(r["bsm_theta_per_day"]) else "nan")
        i3.metric("Rho per 1%", f"{float(r['bsm_rho_per_1pct']):.4f}" if pd.notna(r["bsm_rho_per_1pct"]) else "nan")

        st.markdown("Model inputs")
        st.write(
            {
                "S": float(r["underlying_price"]),
                "K": float(r["strike"]),
                "T": float(r["T_years"]),
                "r": float(r["risk_free_rate"]),
                "q": float(r["dividend_yield"]),
                "sigma": float(r["selected_sigma"]) if pd.notna(r["selected_sigma"]) else None,
                "option_type": str(r["option_type"]),
            }
        )

    st.subheader("IV Smile")
    smile_date = st.selectbox("Smile date", sorted(filtered["date"].dropna().astype(str).unique()), key="opt_smile_date")
    smile_x = st.radio("Smile x-axis", ["strike", "moneyness"], index=1, horizontal=True, key="opt_smile_x")
    smile_side = st.radio("Smile side", ["calls", "puts", "both"], index=2, horizontal=True, key="opt_smile_side")

    smile = filtered[(filtered["date"].astype(str) == smile_date) & filtered["implied_vol_market"].notna()].copy()
    if smile_side == "calls":
        smile = smile[smile["option_type"] == "call"]
    elif smile_side == "puts":
        smile = smile[smile["option_type"] == "put"]
    if not smile.empty:
        st.plotly_chart(
            px.scatter(smile, x=smile_x, y="implied_vol_market", color=smile["expiration"].astype(str), title="IV Smile by Expiration"),
            width="stretch",
        )

    st.subheader("IV Term Structure")
    term_date = st.selectbox("Term date", sorted(filtered["date"].dropna().astype(str).unique()), key="opt_term_date")
    term_side = st.radio("Term side", ["calls", "puts", "combined"], index=2, horizontal=True, key="opt_term_side")

    term = filtered[(filtered["date"].astype(str) == term_date) & filtered["implied_vol_market"].notna()].copy()
    if term_side == "calls":
        term = term[term["option_type"] == "call"]
    elif term_side == "puts":
        term = term[term["option_type"] == "put"]
    if not term.empty:
        term["atm_distance"] = (term["moneyness"] - 1.0).abs()
        atm = term.sort_values(["DTE", "atm_distance"]).groupby("DTE", as_index=False).first()
        st.plotly_chart(px.line(atm, x="DTE", y="implied_vol_market", markers=True, title="ATM IV Term Structure"), width="stretch")

    st.subheader("Greeks by Strike")
    g_date = st.selectbox("Greeks date", sorted(filtered["date"].dropna().astype(str).unique()), key="opt_greeks_date")
    g_exp = st.selectbox(
        "Greeks expiration",
        sorted(filtered[filtered["date"].astype(str) == g_date]["expiration"].dropna().astype(str).unique()),
        key="opt_greeks_exp",
    )
    g_type = st.selectbox("Greeks option type", ["call", "put"], key="opt_greeks_type")
    gdf = filtered[
        (filtered["date"].astype(str) == g_date)
        & (filtered["expiration"].astype(str) == g_exp)
        & (filtered["option_type"] == g_type)
    ].copy()
    if not gdf.empty:
        for greek_col, title in [
            ("bsm_delta", "Delta by Strike"),
            ("bsm_gamma", "Gamma by Strike"),
            ("bsm_vega_per_1pct", "Vega per 1% by Strike"),
            ("bsm_theta_per_day", "Theta per Day by Strike"),
            ("bsm_rho_per_1pct", "Rho per 1% by Strike"),
        ]:
            st.plotly_chart(px.line(gdf.sort_values("strike"), x="strike", y=greek_col, markers=True, title=title), width="stretch")

    st.subheader("Market versus Theoretical Price")
    p_x = st.radio("Price chart x-axis", ["strike", "moneyness"], index=0, horizontal=True, key="opt_px_x")
    p_df = gdf.sort_values(p_x)
    if not p_df.empty:
        st.plotly_chart(px.line(p_df, x=p_x, y=["market_price", "bsm_price"], title="Market vs BSM Price"), width="stretch")

    st.subheader("Pricing Discrepancy")
    if vol_mode == "implied from market price":
        st.warning(
            "Price error is not meaningful in implied-volatility mode because the model volatility is solved from the market price."
        )
    e_y = st.radio("Discrepancy metric", ["price_error", "relative_price_error"], index=0, horizontal=True, key="opt_err_y")
    if not p_df.empty:
        st.plotly_chart(px.bar(p_df, x=p_x, y=e_y, title="Pricing Discrepancy"), width="stretch")

    st.subheader("Explanation")
    st.markdown(
        "- Black-Scholes-Merton is a benchmark model, not the true price.\n"
        "- Market option prices imply volatility.\n"
        "- Greeks are model-derived sensitivities, not directly observed values.\n"
        "- If implied volatility is solved from market price, theoretical price matches market price by construction.\n"
        "- Pricing discrepancies are meaningful when using historical or manual volatility.\n"
        "- Differences can come from smile, term structure, dividends, rates, spreads, settlement conventions, liquidity, and supply/demand."
    )
