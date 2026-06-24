from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.optimize import brentq

from src.options.black_scholes import bsm_price


def _no_arb_bounds(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
) -> Tuple[float, float]:
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if option_type == "call":
        lower = max(0.0, S * disc_q - K * disc_r)
        upper = S * disc_q
    else:
        lower = max(0.0, K * disc_r - S * disc_q)
        upper = K * disc_r
    return lower, upper


def implied_volatility_with_status(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    lower_bound: float = 1e-6,
    upper_bound: float = 5.0,
) -> tuple[float, str]:
    """Solve implied volatility with Brent's method and return solver status."""
    if option_type not in {"call", "put"}:
        return np.nan, "invalid_option_type"
    if not np.isfinite(market_price) or market_price <= 0.0:
        return np.nan, "invalid_market_price"
    if not np.isfinite(S) or not np.isfinite(K) or not np.isfinite(T):
        return np.nan, "invalid_inputs"
    if S <= 0.0 or K <= 0.0 or T <= 0.0:
        return np.nan, "invalid_inputs"

    no_arb_low, no_arb_high = _no_arb_bounds(S, K, T, r, q, option_type)
    if market_price < no_arb_low - 1e-10 or market_price > no_arb_high + 1e-10:
        return np.nan, "no_arb_violation"

    def objective(sig: float) -> float:
        model_price = float(bsm_price(S, K, T, r, q, sig, option_type))
        return model_price - market_price

    try:
        f_lo = objective(lower_bound)
        f_hi = objective(upper_bound)
        if np.isnan(f_lo) or np.isnan(f_hi):
            return np.nan, "nan_objective"
        if f_lo == 0.0:
            return lower_bound, "ok"
        if f_hi == 0.0:
            return upper_bound, "ok"
        if np.sign(f_lo) == np.sign(f_hi):
            return np.nan, "no_bracket"
        sigma = brentq(objective, lower_bound, upper_bound, maxiter=100)
        return float(sigma), "ok"
    except Exception:
        return np.nan, "solver_failure"


def implied_volatility_from_market_price(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
) -> float:
    """Requested public API that returns only the implied volatility value."""
    sigma, _ = implied_volatility_with_status(market_price, S, K, T, r, q, option_type)
    return sigma
