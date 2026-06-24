from __future__ import annotations

import numpy as np
import pandas as pd

from src.options.black_scholes import (
    bsm_delta,
    bsm_gamma,
    bsm_rho,
    bsm_theta,
    bsm_vega,
)


def compute_greeks_frame(
    frame: pd.DataFrame,
    *,
    s_col: str = "underlying_price",
    k_col: str = "strike",
    t_col: str = "T_years",
    r_col: str = "risk_free_rate",
    q_col: str = "dividend_yield",
    sigma_col: str = "selected_sigma",
    type_col: str = "option_type",
) -> pd.DataFrame:
    """Compute BSM Greeks and convention-normalized variants."""
    out = frame.copy()
    S = out[s_col].to_numpy(dtype=float)
    K = out[k_col].to_numpy(dtype=float)
    T = out[t_col].to_numpy(dtype=float)
    r = out[r_col].to_numpy(dtype=float)
    q = out[q_col].to_numpy(dtype=float)
    sigma = out[sigma_col].to_numpy(dtype=float)
    option_type = out[type_col].astype(str).to_numpy()

    out["bsm_delta"] = bsm_delta(S, K, T, r, q, sigma, option_type)
    out["bsm_gamma"] = bsm_gamma(S, K, T, r, q, sigma)
    out["bsm_vega_raw"] = bsm_vega(S, K, T, r, q, sigma)
    out["bsm_vega_per_1pct"] = out["bsm_vega_raw"] * 0.01
    out["bsm_theta_annual"] = bsm_theta(S, K, T, r, q, sigma, option_type)
    out["bsm_theta_per_day"] = out["bsm_theta_annual"] / 365.0
    out["bsm_rho_raw"] = bsm_rho(S, K, T, r, q, sigma, option_type)
    out["bsm_rho_per_1pct"] = out["bsm_rho_raw"] * 0.01

    # Guard against infs from malformed inputs in edge rows.
    for col in [
        "bsm_delta",
        "bsm_gamma",
        "bsm_vega_raw",
        "bsm_vega_per_1pct",
        "bsm_theta_annual",
        "bsm_theta_per_day",
        "bsm_rho_raw",
        "bsm_rho_per_1pct",
    ]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out
