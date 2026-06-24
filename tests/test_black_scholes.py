from __future__ import annotations

import numpy as np

from src.options.black_scholes import bsm_delta, bsm_gamma, bsm_price


def test_bsm_call_put_benchmark_prices() -> None:
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20

    call = float(bsm_price(S, K, T, r, q, sigma, "call"))
    put = float(bsm_price(S, K, T, r, q, sigma, "put"))

    assert np.isclose(call, 9.2270, atol=1e-3)
    assert np.isclose(put, 6.3301, atol=1e-3)


def test_put_call_parity_with_dividend_yield() -> None:
    S = 105.0
    K = 100.0
    T = 0.75
    r = 0.03
    q = 0.01
    sigma = 0.25

    call = float(bsm_price(S, K, T, r, q, sigma, "call"))
    put = float(bsm_price(S, K, T, r, q, sigma, "put"))

    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert np.isclose(lhs, rhs, atol=1e-8)


def test_delta_signs_and_gamma_positive() -> None:
    S = 100.0
    K = 102.0
    T = 0.5
    r = 0.02
    q = 0.01
    sigma = 0.30

    c_delta = float(bsm_delta(S, K, T, r, q, sigma, "call"))
    p_delta = float(bsm_delta(S, K, T, r, q, sigma, "put"))
    gamma = float(bsm_gamma(S, K, T, r, q, sigma))

    assert c_delta > 0.0
    assert p_delta < 0.0
    assert gamma > 0.0
