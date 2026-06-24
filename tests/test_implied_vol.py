from __future__ import annotations

import numpy as np

from src.options.black_scholes import bsm_price
from src.options.implied_vol import implied_volatility_from_market_price


def test_implied_vol_recovers_known_sigma() -> None:
    S = 100.0
    K = 95.0
    T = 0.6
    r = 0.03
    q = 0.01
    sigma_true = 0.28

    market_price = float(bsm_price(S, K, T, r, q, sigma_true, "call"))
    sigma_est = implied_volatility_from_market_price(market_price, S, K, T, r, q, "call")
    assert np.isclose(sigma_est, sigma_true, atol=1e-6)


def test_invalid_market_price_returns_nan() -> None:
    iv = implied_volatility_from_market_price(-1.0, 100.0, 100.0, 0.5, 0.02, 0.01, "call")
    assert np.isnan(iv)


def test_no_arb_violating_price_returns_nan() -> None:
    # For a call, price cannot exceed discounted spot upper bound.
    iv = implied_volatility_from_market_price(500.0, 100.0, 100.0, 1.0, 0.03, 0.01, "call")
    assert np.isnan(iv)
