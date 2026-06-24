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


def _render_step_explainer(step_title: str, markdown_text: str) -> None:
    with st.expander(f"Method details: {step_title}", expanded=False):
        st.markdown(markdown_text)


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

    status = _cache_status(start_s, end_s)
    _render_cache_panel(status, start_s, end_s)

    hist_label = f"Use historical data from period {start_s} to {end_s} (current data)"
    c_load, c_hist = st.columns(2)
    load_clicked = c_load.button("Load from cache", key="opt_load_cache")
    use_hist_clicked = c_hist.button(hist_label, key="opt_use_hist_data", disabled=not bool(status["processed_exists"]))

    def _load_fixed_snapshot(success_prefix: str) -> None:
        cached = load_spy_options_pricing_cache(start_s, end_s)
        if cached is None or cached.empty:
            st.error("No cached pricing snapshot found for the selected range.")
            return
        st.session_state["options_pricing_base_df"] = cached
        st.session_state["options_pricing_loaded_range"] = (start_s, end_s)
        st.success(f"{success_prefix} Loaded {len(cached):,} rows.")

    if load_clicked:
        _load_fixed_snapshot("Loaded from local cache.")
    if use_hist_clicked:
        _load_fixed_snapshot("Loaded fixed historical dataset.")

    st.caption(
        "Dynamic Databento downloads are disabled in Streamlit for now. "
        "Refresh cache from terminal if needed."
    )

    base_df = st.session_state.get("options_pricing_base_df")
    loaded_range = st.session_state.get("options_pricing_loaded_range")
    if base_df is None and bool(status["processed_exists"]):
        _load_fixed_snapshot("Auto-loaded fixed historical dataset.")
        base_df = st.session_state.get("options_pricing_base_df")
        loaded_range = st.session_state.get("options_pricing_loaded_range")

    if base_df is None:
        st.info("No options pricing data loaded in this session yet. Use Load from cache or Use historical data.")
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
    _render_step_explainer(
        "Data Controls",
        r"""
**What this step does**
- Selects the analysis subset and model configuration before any charting.

**Method**
- Filters contracts by DTE and moneyness.
- Re-prices every selected contract under the chosen volatility mode.

**Core definitions**
- $DTE = (\text{expiration date} - \text{valuation date})$ in calendar days.
- $T = DTE / 365$ years.
- $moneyness = K / S$ where $K$ is strike and $S$ is underlying spot.

**Volatility modes**
- implied from market price: solves $\sigma$ such that model price equals observed market price.
- realized volatility modes: use trailing realized vol as proxy for $\sigma$.
- manual volatility: set fixed $\sigma$ to run scenarios.

**Assumptions**
- Continuous compounding for rates/dividends.
- Single volatility input per contract for each valuation pass.

**How to interpret results**
- Narrow moneyness around 1.0 focuses near-ATM behavior.
- Wider DTE captures stronger term-structure effects.
- Price discrepancy analysis is most informative when volatility is not implied-from-market mode.
""",
    )
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
    _render_step_explainer(
        "Data Quality Summary",
        """
**What this step does**
- Reports how much data survives quality/eligibility filters and why rows are dropped.

**Method**
- Applies hard validity checks: positive price, strike, underlying, and time-to-expiry.
- Applies range checks: selected DTE and moneyness intervals.
- Applies quote consistency checks (when bid/ask is present).

**Assumptions**
- Contracts with invalid core inputs are excluded because BSM/Greeks become undefined or unstable.

**How to read key metrics**
- Contracts / Calls / Puts: sample size and composition.
- Rows excluded: strictness of current filters.
- IV solver failures: where implied-vol root-finding could not produce a valid solution.
- Median market price: central scale for option premium in filtered set.

**Practical meaning**
- If exclusions are high, conclusions can become selection-sensitive.
- If solver failures cluster by tenor/strike, market data quality or no-arbitrage boundaries may be driving instability.
""",
    )
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
    _render_step_explainer(
        "Option-chain Table",
        r"""
**What this step does**
- Displays row-level outputs used by all downstream charts.

**Method**
- Each row corresponds to one option contract on one valuation date.
- Includes observed fields (market price, volume/OI) and model outputs (IV, BSM price, Greeks).

**Column meaning highlights**
- implied_vol_market: volatility that matches market price under BSM.
- selected_sigma: volatility actually used for pricing mode.
- price_error: $\text{market} - \text{model}$.
- relative_price_error: $(\text{market} - \text{model}) / \max(\text{market}, \epsilon)$.

**How to interpret**
- Sort by date/expiry/strike to inspect smile and term effects directly in tabular form.
- Large absolute error at very low option prices can still imply small economic significance.
""",
    )
    st.dataframe(_select_chain_columns(filtered).sort_values(["date", "expiration", "strike"]), width="stretch")

    st.subheader("Single-contract Inspector")
    _render_step_explainer(
        "Single-contract Inspector",
        r"""
**What this step does**
- Performs point-in-time, point-on-surface diagnostics for one selected contract.

**Method**
- Pulls one row and reports decomposition + sensitivities.
- Intrinsic/extrinsic split:
    - Call intrinsic: $\max(S-K, 0)$
    - Put intrinsic: $\max(K-S, 0)$
    - Extrinsic: market price minus intrinsic.

**Black-Scholes-Merton (BSM) Model vs Black-Scholes (BS)**

The **Black-Scholes-Merton** model extends the classical Black-Scholes framework by incorporating a **continuous dividend yield** ($q$). This is critical for:
- Dividend-paying stocks (continuous q = annual dividend / spot)
- ETFs like SPY that distribute dividends regularly

The classical **Black-Scholes** assumes $q = 0$ (no dividends), which underprices calls and overprices puts for dividend-paying underlyings.

**Black-Scholes-Merton equations (with dividend yield $q$)**
$$
d_1 = \frac{\ln(S/K) + (r-q+\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},\quad
d_2 = d_1 - \sigma\sqrt{T}
$$
$$
C = Se^{-qT}N(d_1) - Ke^{-rT}N(d_2),\quad
P = Ke^{-rT}N(-d_2) - Se^{-qT}N(-d_1)
$$

**Greeks: Partial Derivatives and Black-Scholes-Merton Formulas**

- **Delta** ($\Delta$): First-order sensitivity to spot moves.
  - Call: $\Delta_C = e^{-qT}N(d_1)$
  - Put: $\Delta_P = -e^{-qT}N(-d_1)$

- **Gamma** ($\Gamma$): Curvature of delta; second derivative w.r.t. spot. High near ATM/short tenor.
  - $\Gamma = \frac{e^{-qT}}{S\sigma\sqrt{T}}n(d_1)$ where $n$ is the standard normal PDF

- **Vega** ($\nu$): Sensitivity to volatility (per 1% change). Reports $\nu / 100$.
  - $\nu = Se^{-qT}\sqrt{T}n(d_1)$

- **Theta** ($\Theta$): Carry/decay effect per day. Time decay of option value.
  - Call: $\Theta_C = -Se^{-qT}n(d_1)\frac{\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)$
  - Put: $\Theta_P = -Se^{-qT}n(d_1)\frac{\sigma}{2\sqrt{T}} + rKe^{-rT}N(-d_2) - qSe^{-qT}N(-d_1)$

- **Rho** ($\rho$): Sensitivity to risk-free rate (per 1% change). Reports $\rho / 100$.
  - Call: $\rho_C = KTe^{-rT}N(d_2)$
  - Put: $\rho_P = -KTe^{-rT}N(-d_2)$

**Price Comparison: Theoretical vs Actual**

- **Actual (Market) Price**: Observed price in the market.
- **Theoretical (BSM) Price**: Model price computed using Black-Scholes-Merton with selected volatility.
- **Difference**: Actual minus Theoretical. Positive = market overpriced relative to model; negative = underpriced.
""",
    )
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
        
        # Price comparison: Theoretical vs Actual vs Difference
        actual_price = float(r['market_price']) if pd.notna(r['market_price']) else None
        theoretical_price = float(r['bsm_price']) if pd.notna(r['bsm_price']) else None
        difference = actual_price - theoretical_price if (actual_price is not None and theoretical_price is not None) else None
        
        i1.metric("Actual (Market) Price", f"{actual_price:.4f}" if actual_price is not None else "nan")
        i1.caption(f"Source: {r.get('market_price_source', 'n/a')}")
        i1.metric("Theoretical (BSM) Price", f"{theoretical_price:.4f}" if theoretical_price is not None else "nan")
        i1.metric("Difference (Actual − Theoretical)", f"{difference:.4f}" if difference is not None else "nan")
        i1.caption("Positive = market overpriced; negative = underpriced")
        
        # Intrinsic/Extrinsic breakdown
        i2.metric("Intrinsic", f"{float(r['intrinsic_value']):.4f}")
        i2.metric("Extrinsic", f"{float(r['extrinsic_value']):.4f}")
        i2.metric("Implied volatility", f"{float(r['implied_vol_market']):.4f}" if pd.notna(r["implied_vol_market"]) else "nan")
        i2.metric("Selected sigma", f"{float(r['selected_sigma']):.4f}" if pd.notna(r["selected_sigma"]) else "nan")

        # Greeks
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
        
        with st.expander("Model inputs explained", expanded=False):
            st.markdown(r"""
**BSM Model Parameters**

- **$S$ (Spot price)**: Current price of the underlying asset (SPY ETF in dollars). Example: $S = 450$ means SPY is trading at $450.

- **$K$ (Strike price)**: The fixed price at which the option holder can buy (call) or sell (put) the underlying. Example: $K = 455$ is an out-of-the-money (OTM) call if $S = 450$.

- **$T$ (Time to expiration in years)**: Days remaining until expiration divided by 365. Example: $T = 0.05$ ≈ 18 calendar days. Smaller $T$ → faster time decay (larger theta in magnitude).

- **$r$ (Risk-free rate)**: Annualized discount rate for cashflows. Typically the 3-month or 1-year Treasury yield. Higher $r$ → calls more valuable, puts less valuable (higher rho).

- **$q$ (Dividend yield)**: Annualized continuous dividend paid by the underlying. For SPY: typical $q \\approx 1.5\%$ to $2\%$ per year. Higher $q$ → calls less valuable, puts more valuable (reduces carry benefit of buying calls).

- **$\\sigma$ (Volatility)**: Annualized standard deviation of log returns. Controls option price and Greeks. Example: $\\sigma = 0.18$ means ~18% annual realized/implied volatility. Higher $\\sigma$ → all options more expensive (vega > 0).

- **option_type**: Either **call** or **put**. Call = right to buy at $K$; Put = right to sell at $K$.

**Quick reference**
- Moneyness: $S/K$ near 1.0 → ATM (at-the-money). $S/K > 1$ → Call ITM/Put OTM. $S/K < 1$ → Call OTM/Put ITM.
- Time value decay: Smaller $T$ → Theta magnitude increases; gamma peaks near expiration.
""")


    st.subheader("Implied Volatility Skew")
    _render_step_explainer(
        "Implied Volatility Skew",
        """
**What this step does**
- Visualizes cross-sectional implied volatility vs strike, moneyness, or log-moneyness for a single date.

**Method**
- Uses solved market IV values only (rows where implied vol exists).
- Optional filters: DTE range, moneyness state (OTM/ATM/ITM), IV solver status, and positive extrinsic value.
- Expiration can be viewed one-at-a-time or as a selected set.

**Assumptions**
- IV is model-implied under BSM; smile therefore captures deviations from constant-vol assumptions.

**How to interpret**
- U-shape/smirk indicates skewed tail pricing and non-lognormal market beliefs.
- Call/put asymmetry can indicate demand imbalance and downside crash premium.

**Example interpretation**
This chart shows the implied volatility surface collapsed into a two-dimensional view. The x-axis shows moneyness, the y-axis shows market-implied volatility, and each color represents a different expiration. The downward slope across moneyness indicates typical equity-index volatility skew: lower strikes are priced with higher implied volatility than higher strikes. The vertical dispersion at similar moneyness levels shows that implied volatility also varies by expiration, meaning the market prices a full volatility surface rather than one constant volatility input. This directly illustrates a key limitation of the Black-Scholes-Merton model: while the model assumes a single volatility input, real option prices imply different volatilities across strikes and maturities.

**Why lower strikes usually have higher implied volatility for SPY**
- SPY inherits index-like downside jump risk: equity drawdowns tend to be faster and larger than upside rallies.
- Investors and funds buy downside put protection (portfolio insurance), increasing demand for lower-strike puts.
- Market makers who sell those puts hedge dynamically and charge more for downside convexity and gap risk.
- Negative return-volatility relationship (leverage effect) means volatility tends to rise when SPY falls, so left-tail scenarios embed higher future vol.
- Together, these effects raise option prices for lower strikes; when inverted through BSM, that appears as higher implied volatility at lower moneyness.
""",
    )
    smile_date = st.selectbox("Skew date", sorted(filtered["date"].dropna().astype(str).unique()), key="opt_smile_date")
    smile_x = st.radio("Skew x-axis", ["strike", "moneyness", "log_moneyness"], index=1, horizontal=True, key="opt_smile_x")
    smile_side = st.radio("Skew side", ["calls", "puts", "both"], index=2, horizontal=True, key="opt_smile_side")
    smile_mny_state = st.radio("Moneyness state", ["all", "OTM only", "ATM only", "ITM only"], index=0, horizontal=True, key="opt_smile_mny_state")

    smile = filtered[(filtered["date"].astype(str) == smile_date) & filtered["implied_vol_market"].notna()].copy()
    if "log_moneyness" not in smile.columns:
        smile["log_moneyness"] = np.log(smile["strike"] / smile["underlying_price"])
    smile = smile.replace([np.inf, -np.inf], np.nan)

    if smile_side == "calls":
        smile = smile[smile["option_type"] == "call"]
    elif smile_side == "puts":
        smile = smile[smile["option_type"] == "put"]

    if not smile.empty and "moneyness" in smile.columns:
        atm_tolerance = 0.01
        is_call = smile["option_type"].astype(str).str.lower() == "call"
        is_put = smile["option_type"].astype(str).str.lower() == "put"
        is_atm = (smile["moneyness"] - 1.0).abs() <= atm_tolerance
        is_itm = (is_call & (smile["moneyness"] < 1.0 - atm_tolerance)) | (is_put & (smile["moneyness"] > 1.0 + atm_tolerance))
        is_otm = (is_call & (smile["moneyness"] > 1.0 + atm_tolerance)) | (is_put & (smile["moneyness"] < 1.0 - atm_tolerance))

        if smile_mny_state == "OTM only":
            smile = smile[is_otm]
        elif smile_mny_state == "ATM only":
            smile = smile[is_atm]
        elif smile_mny_state == "ITM only":
            smile = smile[is_itm]
        st.caption("ATM is defined as |K/S - 1| <= 0.01 for this filter.")

    only_ok_status = st.checkbox("Only IV solver status == ok", value=True, key="opt_smile_only_ok")
    only_positive_extrinsic = st.checkbox("Only extrinsic_value > 0", value=True, key="opt_smile_extrinsic_pos")

    if only_ok_status and "iv_solver_status" in smile.columns:
        smile = smile[smile["iv_solver_status"].astype(str).str.lower() == "ok"]
    if only_positive_extrinsic and "extrinsic_value" in smile.columns:
        smile = smile[smile["extrinsic_value"] > 0]

    if not smile.empty and smile["DTE"].notna().any():
        dte_min = int(smile["DTE"].min())
        dte_max = int(smile["DTE"].max())
        skew_dte_range = st.slider("Skew DTE filter", min_value=dte_min, max_value=dte_max, value=(dte_min, dte_max), key="opt_smile_dte")
        smile = smile[(smile["DTE"] >= skew_dte_range[0]) & (smile["DTE"] <= skew_dte_range[1])]

    if not smile.empty:
        exp_choices = sorted(smile["expiration"].dropna().astype(str).unique())
        exp_mode = st.radio("Expiration selection", ["single", "multi"], index=0, horizontal=True, key="opt_smile_exp_mode")

        if exp_mode == "single":
            selected_exp = st.selectbox("Expiration", exp_choices, key="opt_smile_exp_single")
            smile = smile[smile["expiration"].astype(str) == selected_exp]
        else:
            selected_exps = st.multiselect(
                "Expirations (multi-select)",
                exp_choices,
                default=exp_choices,
                key="opt_smile_exp_multi",
                help="Use Ctrl/Cmd+click to select multiple expirations.",
            )
            if st.button("Select all expirations", key="opt_smile_select_all"):
                selected_exps = exp_choices
                st.session_state["opt_smile_exp_multi"] = exp_choices
            smile = smile[smile["expiration"].astype(str).isin(selected_exps)]

    if not smile.empty:
        st.plotly_chart(
            px.scatter(
                smile,
                x=smile_x,
                y="implied_vol_market",
                color=smile["expiration"].astype(str),
                title="Implied Volatility Skew by Expiration",
            ),
            width="stretch",
        )
    else:
        st.info("No data available for the current skew filters.")

    st.subheader("IV Term Structure")
    _render_step_explainer(
        "IV Term Structure",
        """
**What this step does**
- Tracks how near-ATM implied volatility varies across maturities (DTE).

**Method**
- For each DTE bucket, picks the contract closest to ATM by minimizing $|moneyness - 1|$.

**How to interpret**
- Upward slope (contango-like): longer-dated uncertainty priced higher.
- Downward slope (backwardation-like): near-term stress/event risk elevated.
- Kinks can indicate event dates (macro releases, earnings clusters, policy decisions).
""",
    )
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
    _render_step_explainer(
        "Greeks by Strike",
        r"""
**What this step does**
- Shows shape of each Greek across strikes for fixed date/expiry/type.

**Method**
- Computes Greeks from BSM using selected model inputs.

**Greeks definitions**
- $\Delta = \partial V/\partial S$
- $\Gamma = \partial^2 V/\partial S^2$
- $\nu$ (vega) $= \partial V/\partial \sigma$
- $\Theta = \partial V/\partial t$
- $\rho = \partial V/\partial r$

**How to interpret**
- Delta transitions across strikes reveal directional exposure profile.
- Gamma peaks generally near ATM and short maturities.
- Vega tends to concentrate around ATM and medium tenors.
- Theta often most negative near ATM for long options.

**Delta interpretation example (calls)**
| Strike region  | Delta      | Interpretation                                       |
| -------------- | ---------- | ---------------------------------------------------- |
| Low strikes    | ~0.8-1.0   | Deep ITM calls, behave almost like SPY              |
| Middle strikes | ~0.45-0.60 | Near ATM calls, most sensitive transition zone      |
| High strikes   | ~0.1-0.3   | OTM calls, low probability / low stock-like exposure |

**Vega per 1% by strike (calls): key insight**
- Vega per 1% means option price change for a 1 percentage-point IV move.
- Example: Vega per 1% = 0.35 means IV from 15% to 16% adds about $0.35 to option value; 15% to 14% subtracts about $0.35.
- Vega is usually highest near ATM because uncertainty about finish location is largest there.
- Deep ITM calls behave more stock-like (less volatility sensitivity), and far OTM calls have low finish probability, so both typically show lower vega than ATM.
- Practical read: the strike region around the local vega peak is most exposed to IV repricing risk.

**Theta per day by strike (calls): key insight**
- Theta per day is time decay holding spot/IV/rates/dividends constant.
- For long calls, theta is usually negative: option value decays as expiry approaches.
- Most negative theta tends to occur near ATM where extrinsic value is highest.
- Deep ITM options contain more intrinsic value and usually lower absolute theta; far OTM options are cheaper so absolute theta is often smaller.
- Practical read: near-ATM long calls can be high-convexity but pay the highest daily carry cost.

**Rho per 1% by strike (calls): key insight**
- Rho per 1% is call price sensitivity to a 1 percentage-point change in risk-free rate.
- Call rho is typically positive: higher rates increase call value (all else equal).
- Rho often declines with strike across short-dated chains because rate sensitivity is usually stronger for lower-strike, more in-the-money calls.
- Practical read: for short maturities, rho is usually a second-order risk versus delta/gamma/vega/theta.

**Combined practical interpretation (near-ATM calls)**
- Near-ATM calls often combine high gamma, high vega, and strongly negative theta.
- They are powerful when spot moves or IV expands, but fragile if spot is stagnant and IV compresses.
- If the position does not get directional follow-through or vol expansion quickly, theta drag can dominate.
""",
    )
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
        key_cols = ["date", "expiration", "option_type", "strike"]
        duplicates_removed = int(len(gdf) - len(gdf.drop_duplicates(subset=key_cols)))
        if duplicates_removed > 0:
            st.caption(
                f"Deduplicated Greeks rows by date+expiration+option_type+strike (removed {duplicates_removed} duplicate rows)."
            )
        gdf = (
            gdf.sort_values(["date", "expiration", "option_type", "strike"])
            .groupby(key_cols, as_index=False)
            .last()
        )

        for greek_col, title in [
            ("bsm_delta", "Delta by Strike"),
            ("bsm_gamma", "Gamma by Strike"),
            ("bsm_vega_per_1pct", "Vega per 1% by Strike"),
            ("bsm_theta_per_day", "Theta per Day by Strike"),
            ("bsm_rho_per_1pct", "Rho per 1% by Strike"),
        ]:
            st.plotly_chart(px.line(gdf.sort_values("strike"), x="strike", y=greek_col, markers=True, title=title), width="stretch")

    st.subheader("Market versus Theoretical Price")
    _render_step_explainer(
        "Market versus Theoretical Price",
        """
**What this step does**
- Compares observed market premium against BSM premium under current volatility mode.

**Method**
- Plots two curves over strike or moneyness:
    - market_price (observed)
    - bsm_price (model output)

**Assumptions**
- Any divergence is interpreted relative to chosen volatility input, not as pure mispricing.

**How to interpret**
- Tight overlap in implied-vol mode is expected (calibration-by-construction).
- Persistent gaps in realized/manual mode indicate model misspecification or risk premium effects.
""",
    )
    p_x = st.radio("Price chart x-axis", ["strike", "moneyness"], index=0, horizontal=True, key="opt_px_x")
    p_df = gdf.sort_values(p_x)
    if not p_df.empty:
        st.plotly_chart(px.line(p_df, x=p_x, y=["market_price", "bsm_price"], title="Market vs BSM Price"), width="stretch")

    st.subheader("Pricing Discrepancy")
    _render_step_explainer(
        "Pricing Discrepancy",
        """
**What this step does**
- Quantifies residual difference between market and model prices.

**Metrics**
- Absolute error: $\text{price_error} = \text{market} - \text{BSM}$
- Relative error: $\text{relative_error} = (\text{market} - \text{BSM}) / \text{market}$ (stabilized for tiny prices)

**How to interpret sign**
- Positive error: market richer than model (model underpricing).
- Negative error: market cheaper than model (model overpricing).

**Caveat**
- In implied-vol mode, discrepancy has limited diagnostic value because volatility is solved to fit market prices.

**Economic meaning**
- Large systematic residuals by strike/tenor can reflect skew, jumps, stochastic volatility, liquidity frictions, or supply-demand imbalance.
""",
    )
    if vol_mode == "implied from market price":
        st.warning(
            "Price error is not meaningful in implied-volatility mode because the model volatility is solved from the market price."
        )
    e_y = st.radio("Discrepancy metric", ["price_error", "relative_price_error"], index=0, horizontal=True, key="opt_err_y")
    if not p_df.empty:
        st.plotly_chart(px.bar(p_df, x=p_x, y=e_y, title="Pricing Discrepancy"), width="stretch")

    st.subheader("Explanation")
    _render_step_explainer(
        "End-to-end workflow and assumptions",
        """
**Pipeline overview**
1. Load cached daily options snapshot.
2. Apply filters (quality + analysis scope).
3. Choose volatility regime (implied/realized/manual).
4. Recompute BSM theoretical values and Greeks.
5. Visualize smile, term structure, sensitivities, and residuals.

**Global model assumptions**
- Frictionless continuous-time benchmark (BSM).
- Continuous dividend yield and risk-free carry.
- European-style valuation formula as analytical baseline.
- Constant volatility per priced contract in each model pass.

**What results mean operationally**
- This module is primarily diagnostic/educational and for relative comparison across regimes.
- It is strongest for identifying structure (smile/term/skew/sensitivity), not for asserting executable alpha by itself.
""",
    )
    st.markdown(
        "- Black-Scholes-Merton is a benchmark model, not the true price.\n"
        "- Market option prices imply volatility.\n"
        "- Greeks are model-derived sensitivities, not directly observed values.\n"
        "- If implied volatility is solved from market price, theoretical price matches market price by construction.\n"
        "- Pricing discrepancies are meaningful when using historical or manual volatility.\n"
        "- Differences can come from smile, term structure, dividends, rates, spreads, settlement conventions, liquidity, and supply/demand."
    )
